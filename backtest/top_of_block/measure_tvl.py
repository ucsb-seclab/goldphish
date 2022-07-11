"""
Measures maximum Total Value Locked (TVL) for a given
token. TVL is computed as the sum of Ether locked in
each exchange. In situations where exchanges have
multiple tokens, we compute TVL by the proportional
weight of the token in question.
"""
import argparse
import asyncio
import collections
import datetime
import decimal
import itertools
import json
import time
import pika
import pika.spec
import typing
import psycopg2.extensions
import web3
import web3.types
import web3._utils.filters
import logging

from backtest.utils import connect_db, connect_rabbit
from pricers.balancer import BalancerPricer
from pricers.pricer_pool import PricerPool
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer
from utils import BALANCER_VAULT_ADDRESS, WETH_ADDRESS

l = logging.getLogger(__name__)


def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'measure-tvl'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (run before mass scan)')

    return parser_name, measure_tvl


def measure_tvl(w3: web3.Web3, args: argparse.Namespace):
    l.info('Starting TVL measurement')

    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        fill_queue(w3, curr)
        db.commit()
        return

    scrape(w3, curr)


def setup_db(curr: psycopg2.extensions.cursor):
    l.debug('setup db')
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS max_tvl_records (
            token_id        INTEGER NOT NULL REFERENCES tokens (id) ON DELETE CASCADE,
            max_tvl_wei     NUMERIC(78, 0) NOT NULL,
            max_tvl_block   INTEGER NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_max_tvl_records ON max_tvl_records (token_id);

        CREATE TABLE IF NOT EXISTS max_tvl_queue (
            id          SERIAL NOT NULL PRIMARY KEY,
            start_block INTEGER NOT NULL,
            end_block   INTEGER NOT NULL,
            taken       BOOLEAN NOT NULL DEFAULT FALSE
        );
        '''
    )
    curr.execute(
        '''
        INSERT INTO max_tvl_records (token_id, max_tvl_wei, max_tvl_dollars)
        SELECT t.id, 0, 0
        FROM tokens t
        WHERE NOT EXISTS(SELECT token_id FROM max_tvl_records);
        '''
    )
    l.debug(f'inserted {curr.rowcount:,} new default token tvl records')


def fill_queue(w3: web3.Web3, curr: psycopg2.extensions.cursor):
    curr.execute('SELECT COUNT(*) FROM max_tvl_queue')
    (n_queued,) = curr.fetchone()

    if n_queued > 0:
        l.debug('not filling queue')
        return

    start_block = scan_start(curr)
    end_block = w3.eth.get_block('latest')['number']

    l.info(f'filling queue from {start_block:,} to {end_block:,}')

    n_segments = 1_000
    segment_width = (end_block - start_block) // n_segments
    for i in itertools.count():
        segment_start = start_block + i * segment_width
        segment_end = min(end_block, segment_start + segment_width - 1)

        if segment_start > end_block:
            break

        curr.execute(
            '''
            INSERT INTO max_tvl_queue (start_block, end_block)
            VALUES (%s, %s)
            ''',
            (segment_start, segment_end),
        )
        assert curr.rowcount == 1


def scrape(w3: web3.Web3, curr: psycopg2.extensions.cursor):
    while True:
        curr.execute(
            '''
            SELECT id, start_block, end_block FROM max_tvl_queue WHERE taken = false FOR UPDATE SKIP LOCKED
            '''
        )
        if curr.rowcount == 0:
            l.info('No more work, exiting')
        
        (id_, start_block, end_block) = curr.fetchone()

        curr.execute(
            '''
            UPDATE max_tvl_queue SET taken = true WHERE id = %s
            ''',
            (id_,)
        )
        assert curr.rowcount == 1

        curr.connection.commit()

        process_work(w3, curr, start_block, end_block)


def process_work(w3: web3.Web3, curr: psycopg2.extensions.cursor, start_block: int, end_block: int):
    pool = load_pool(w3, curr)

    l.info(f'Scanning from {start_block:,} to {end_block:,}')
    pool.warm(start_block - 1)

    batch_size = 100

    observed_maxs: typing.Dict[str, typing.Tuple[decimal.Decimal, int]] = collections.defaultdict(lambda: (0, 0))
    maxs_updated: typing.Set[str] = set()

    for i in itertools.count():
        batch_start_block = start_block + i * batch_size
        batch_end_block = min(end_block, batch_start_block + batch_size - 1)

        if batch_start_block > end_block:
            break

        l.debug('start get logs')

        f: web3._utils.filters.Filter = w3.eth.filter({
            'address': list(pool._uniswap_v2_pools.keys()),
            'topics': [['0x' + x.hex() for x in UniswapV2Pricer.RELEVANT_LOGS]],
            'fromBlock': batch_start_block,
            'toBlock': batch_end_block,
        })

        logs = f.get_all_entries()

        l.debug('got uniswap v2 logs')

        f: web3._utils.filters.Filter = w3.eth.filter({
            'address': list(pool._uniswap_v3_pools.keys()),
            'topics': [['0x' + x.hex() for x in UniswapV3Pricer.RELEVANT_LOGS]],
            'fromBlock': batch_start_block,
            'toBlock': batch_end_block,
        })

        logs.extend(f.get_all_entries())

        l.debug('got uniswap v3 logs')

        f: web3._utils.filters.Filter = w3.eth.filter({
            'address': list(pool._balancer_v1_pools.keys()),
            'topics': [['0x' + x.hex() for x in BalancerPricer.RELEVANT_LOGS]],
            'fromBlock': batch_start_block,
            'toBlock': batch_end_block,
        })

        logs.extend(f.get_all_entries())

        l.debug('got balancer v1 logs')

        f: web3._utils.filters.Filter = w3.eth.filter({
            'address': list(pool._balancer_v2_pools.keys()) + [BALANCER_VAULT_ADDRESS],
            'fromBlock': batch_start_block,
            'toBlock': batch_end_block,
        })

        logs.extend(f.get_all_entries())

        l.debug('got balancer v2 logs')

        logs = sorted(logs, key=lambda x: (x['blockNumber'], x['logIndex']))

        l.debug(f'got {len(logs):,} logs this batch')

        # group by block
        logs_by_block: typing.Dict[int, typing.List[web3.types.LogReceipt]] = collections.defaultdict(lambda: [])
        for log in logs:
            block_number = log['blockNumber']
            if block_number in logs_by_block:
                assert log['logIndex'] > logs_by_block[block_number][-1]['logIndex']

            logs_by_block[log['blockNumber']].append(log)

        # scan each block
        for block_number in sorted(logs_by_block.keys()):
            block_logs = logs_by_block[block_number]

            # update the pricer pool
            result = pool.observe_block(block_logs)

            # find out which tokens we need to query TVL for
            modified_tokens: typing.Set[str] = set()
            for t1, t2 in result:
                if WETH_ADDRESS not in [t1, t2]:
                    continue
                for address in result[(t1, t2)]:
                    p = pool.get_pricer_for(address)
                    modified_tokens.update(p.get_tokens(block_number))

            try:
                modified_tokens.remove(WETH_ADDRESS)
            except KeyError:
                pass # weth not present

            # query TVL for each modified token
            for token in sorted(modified_tokens):
                if bytes.fromhex(token[2:]) < bytes.fromhex(WETH_ADDRESS[2:]):
                    token0 = token
                    token1 = WETH_ADDRESS
                else:
                    token0 = WETH_ADDRESS
                    token1 = token

                sum_tvl = 0

                for exchange in pool.get_exchanges_for_pair(token0, token1, block_number):
                    p = pool.get_pricer_for(exchange)
                    weth_weight = p.get_token_weight(WETH_ADDRESS, block_number)
                    weth_amt = p.get_value_locked(WETH_ADDRESS, block_number)
                    for token in p.get_tokens(block_number).difference([WETH_ADDRESS]):
                        token_weight = p.get_token_weight(token, block_number)
                        token_weight_minus_weth = token_weight / (decimal.Decimal(1) - weth_weight)
                        token_weth_locked = weth_amt * token_weight_minus_weth
                        
                        sum_tvl += token_weth_locked

                old_max, _ = observed_maxs[token]
                if sum_tvl > old_max:
                    observed_maxs[token] = (token_weth_locked, block_number)
                    maxs_updated.add(token)

        #
        # Merge results into db
        #
        for token in maxs_updated:
            # get token id
            curr.execute('SELECT id FROM tokens WHERE address = %s', (bytes.fromhex(token[2:]),))
            (token_id,) = curr.fetchone()

            raise NotImplementedError('didnt finish this')

            curr.execute(
                '''
                UPDATE max_tvl_records
                SET
                    max_tvl_wei = greatest(max_tvl_wei, %(tvl)s),
                SELECT 1
                FROM max_tvl_records mtvl
                WHERE token_id = %s
                FOR UPDATE
                ''',
                (token_id,)
            )
            assert curr.rowcount > 0


        maxs_updated.clear()


def scan_start(curr: psycopg2.extensions.cursor) -> int:
    return 13_000_000

    curr.execute(
        '''
        SELECT LEAST(
            (SELECT MIN(origin_block) FROM uniswap_v2_exchanges),
            (SELECT MIN(origin_block) FROM uniswap_v3_exchanges),
            (SELECT MIN(origin_block) FROM sushiv2_swap_exchanges),
            (SELECT MIN(origin_block) FROM balancer_exchanges),
            (SELECT MIN(origin_block) FROM balancer_v2_exchanges)
        )
        '''
    )
    (ret,) = curr.fetchone()
    return ret


def load_pool(w3: web3.Web3, curr: psycopg2.extensions.cursor) -> PricerPool:
    #
    # load known pricer pool
    #
    pool = PricerPool(w3)

    # count total number of exchanges we need to load
    curr.execute(
        '''
        SELECT
            (SELECT COUNT(*) FROM uniswap_v2_exchanges) + 
            (SELECT COUNT(*) FROM uniswap_v3_exchanges) + 
            (SELECT COUNT(*) FROM sushiv2_swap_exchanges) + 
            (SELECT COUNT(*) FROM balancer_exchanges) + 
            (SELECT COUNT(*) FROM balancer_v2_exchanges)
        '''
    )
    (n_exchanges,) = curr.fetchone()

    l.debug(f'Loading a total of {n_exchanges:,} exchanges into pricer pool')

    # a quick-and-dirty progress reporter
    t_start = time.time()
    last_report = t_start
    n_loaded = 0
    def report_progress():
        nonlocal last_report
        if n_loaded % 4 != 0:
            return
        now = time.time()
        if now - last_report >= 10:
            last_report = now
            elapsed = now - t_start
            nps = n_loaded / elapsed
            remain = n_exchanges - n_loaded
            eta_sec = remain / nps
            eta = datetime.timedelta(seconds=eta_sec)
            l.info(f'Loaded {n_loaded:,} of {n_exchanges:,} ({n_loaded / n_exchanges * 100:.2f}%) ETA {eta}')

    l.debug('Loading uniswap v2 ...')

    curr.execute(
        '''
        SELECT uv2.address, uv2.origin_block, t0.address, t1.address
        FROM uniswap_v2_exchanges uv2
        JOIN tokens t0 ON uv2.token0_id = t0.id
        JOIN tokens t1 ON uv2.token1_id = t1.id
        '''
    )
    for n_loaded, (address, origin_block, token0, token1) in zip(itertools.count(n_loaded), curr):
        address = w3.toChecksumAddress(address.tobytes())
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        pool.add_uniswap_v2(address, token0, token1, origin_block)
        report_progress()

    l.debug(f'Loading uniswap v3 ...')

    curr.execute(
        '''
        SELECT uv3.address, uv3.origin_block, uv3.originalfee, t0.address, t1.address
        FROM uniswap_v3_exchanges uv3
        JOIN tokens t0 ON uv3.token0_id = t0.id
        JOIN tokens t1 ON uv3.token1_id = t1.id
        '''
    )
    for n_loaded, (address, origin_block, fee, token0, token1) in zip(itertools.count(n_loaded), curr):
        address = w3.toChecksumAddress(address.tobytes())
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        pool.add_uniswap_v3(address, token0, token1, fee, origin_block)
        report_progress()

    l.debug('Loading sushiswap v2 ...')

    curr.execute(
        '''
        SELECT sv2.address, sv2.origin_block, t0.address, t1.address
        FROM sushiv2_swap_exchanges sv2
        JOIN tokens t0 ON sv2.token0_id = t0.id
        JOIN tokens t1 ON sv2.token1_id = t1.id
        '''
    )
    for n_loaded, (address, origin_block, token0, token1) in zip(itertools.count(n_loaded), curr):
        address = w3.toChecksumAddress(address.tobytes())
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        pool.add_sushiswap_v2(address, token0, token1, origin_block)
        report_progress()

    curr.execute(
        '''
        SELECT address, origin_block
        FROM balancer_exchanges
        '''
    )
    for n_loaded, (address, origin_block) in zip(itertools.count(n_loaded), curr):
        address = w3.toChecksumAddress(address.tobytes())
        pool.add_balancer_v1(address, origin_block)

    curr.execute(
        '''
        SELECT address, pool_id, pool_type, origin_block
        FROM balancer_v2_exchanges
        WHERE pool_type = 'WeightedPool2Tokens' OR
              pool_type = 'WeightedPool' OR
              pool_type = 'LiquidityBootstrappingPool' OR
              pool_type = 'NoProtocolFeeLiquidityBootstrappingPool'
        '''
    )
    for n_loaded, (address, pool_id, pool_type, origin_block) in zip(itertools.count(n_loaded), curr):
        address = w3.toChecksumAddress(address.tobytes())
        pool_id = pool_id.tobytes()
        pool.add_balancer_v2(address, pool_id, pool_type, origin_block)

    return pool
