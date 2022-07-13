import argparse
import collections
import gzip
import itertools
import logging
import os
import time
import typing
import web3
import web3.types
import web3._utils.filters
import psycopg2
import psycopg2.extensions

from backtest.top_of_block.common import load_pool
from backtest.top_of_block.constants import FNAME_EXCHANGES_WITH_BALANCES, IMPORTANT_TOPICS_HEX, MIN_PROFIT_PREFILTER, univ2_fname, univ3_fname
from backtest.utils import CancellationToken, connect_db
import pricers
import find_circuit
from pricers.balancer import BalancerPricer
from pricers.pricer_pool import PricerPool
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer
from utils import BALANCER_VAULT_ADDRESS, ProgressReporter, get_block_timestamp
import utils.profiling


l = logging.getLogger(__name__)

LOG_BATCH_SIZE = 200
RESERVATION_SIZE = 1 * 24 * 60 * 60 // 13 # about 1 days' worth


DEBUG = False

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'seek-candidates'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (run before mass scan)')

    return parser_name, seek_candidates


def seek_candidates(w3: web3.Web3, args: argparse.Namespace):
    l.info('Starting candidate searching')
    db = connect_db()
    curr = db.cursor()
    time.sleep(4)

    if args.setup_db:
        setup_db(curr)
        fill_queue(w3, curr)
        db.commit()
        return

    #
    # Load all candidate profitable arbitrages
    batch_size_blocks = 100 # batch size for getting logs
    last_processed_block = None
    while True:
        maybe_rez = get_reservation(curr)
        if maybe_rez is None:
            # we're at the end
            break

        reservation_id, reservation_start, reservation_end = maybe_rez
        pricer: PricerPool = load_pool(w3, curr)
        if last_processed_block is not None:
            assert last_processed_block < reservation_start

        curr_block = reservation_start
        while curr_block <= reservation_end:
            this_end_block = min(curr_block + batch_size_blocks - 1, reservation_end)
            for block_number, logs in get_relevant_logs(w3, pricer, curr_block, this_end_block):
                update = pricer.observe_block(block_number, logs)
                utils.profiling.maybe_log()
                while True:
                    try:
                        process_candidates(w3, pricer, block_number, update, curr)
                        if not DEBUG:
                            db.commit()
                        break
                    except Exception as e:
                        db.rollback()
                        if 'execution aborted (timeout = 5s)' in str(e):
                            l.exception('Encountered timeout, trying again in a little bit')
                            time.sleep(30)
                        else:
                            raise e

            curr_block = this_end_block + 1
            last_processed_block = this_end_block

        # mark reservation as completed
        if not DEBUG:
            if True: # not cancellation_token.cancel_requested():
                assert this_end_block == reservation_end
                l.debug(f'Completed reservation id={reservation_id:,}')
                curr.execute(
                    'UPDATE candidate_arbitrage_reservations SET completed_on = NOW()::timestamp WHERE id = %s',
                    (reservation_id,)
                )
                db.commit()
            else:
                # cancellation was requested
                if curr_block <= reservation_end:
                    l.info('Splitting off unifinished reservation into new one')
                    curr.execute(
                        '''
                        UPDATE candidate_arbitrage_reservations SET block_number_end = %s, completed_on = NOW()::timestamp WHERE id = %s
                        ''',
                        (this_end_block, reservation_id),
                    )
                    curr.execute(
                        '''
                        INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end)
                        VALUES (%s, %s)
                        RETURNING id
                        ''',
                        (curr_block, reservation_end)
                    )
                    (new_id,) = curr.fetchone()
                    l.debug(f'Created new reservation id={new_id:,} {curr_block:,} -> {reservation_end:,}')
                    db.commit()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_arbitrages (
            id              SERIAL PRIMARY KEY NOT NULL,
            block_number    INTEGER NOT NULL,
            exchanges       bytea[] NOT NULL,
            directions      bytea[] NOT NULL,
            amount_in       NUMERIC(78, 0) NOT NULL,
            profit_no_fee   NUMERIC(78, 0) NOT NULL,
            verify_started  TIMESTAMP WITHOUT TIME ZONE,
            verify_finished TIMESTAMP WITHOUT TIME ZONE,
            verify_run      BOOLEAN DEFAULT FALSE
        );

        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrages_block_number ON candidate_arbitrages (block_number);
        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrages_verify_run ON candidate_arbitrages (verify_run);

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_reservations (
            id                 SERIAL PRIMARY KEY NOT NULL,
            block_number_start INTEGER NOT NULL,
            block_number_end   INTEGER NOT NULL,
            claimed_on         TIMESTAMP WITHOUT TIME ZONE,
            completed_on       TIMESTAMP WITHOUT TIME ZONE 
        );

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_blocks_to_verify (
            block_number INTEGER PRIMARY KEY NOT NULL,
            max_profit_no_fee NUMERIC(78, 0) NOT NULL,
            verify_started  TIMESTAMP WITHOUT TIME ZONE,
            verify_finished TIMESTAMP WITHOUT TIME ZONE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_blocks_to_verify_block_number ON candidate_arbitrage_blocks_to_verify(block_number);
        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrage_blocks_to_verify_max_profit_no_fee ON candidate_arbitrage_blocks_to_verify (max_profit_no_fee);
        """
    )

    curr.execute('SELECT count(*) FROM candidate_arbitrage_reservations')


def fill_queue(w3: web3.Web3, curr: psycopg2.extensions.cursor):
    curr.execute('SELECT COUNT(*) FROM candidate_arbitrage_reservations')
    (n_queued,) = curr.fetchone()

    if n_queued > 0:
        l.debug('not filling queue')
        return

    start_block = scan_start(curr)
    end_block = 15_111_766 # w3.eth.get_block('latest')['number']

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
            INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end)
            VALUES (%s, %s)
            ''',
            (segment_start, segment_end),
        )
        assert curr.rowcount == 1


def scan_start(curr: psycopg2.extensions.cursor) -> int:
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


def get_reservation(curr: psycopg2.extensions.cursor) -> typing.Optional[typing.Tuple[int, int, int]]:
    curr.execute('BEGIN TRANSACTION')
    curr.execute('LOCK TABLE candidate_arbitrage_reservations') # for safety
    query_do_reservation = '''
        UPDATE candidate_arbitrage_reservations car
        SET claimed_on = NOW()::timestamp
        FROM (
            SELECT id
            FROM candidate_arbitrage_reservations
            WHERE claimed_on IS NULL AND completed_on IS NULL AND block_number_start > 13000000
            ORDER BY block_number_start ASC
            LIMIT 1
        ) x
        WHERE car.id = x.id
        RETURNING car.id, car.block_number_start, car.block_number_end
    '''
    curr.execute(query_do_reservation)
    maybe_row = curr.fetchall()

    if len(maybe_row) == 0:
        # no reservations left
        l.info('Finished queue')
        return

    assert len(maybe_row) == 1
    id_, start, end = maybe_row[0]
    assert start <= end

    l.info(f'Processing reservation id={id_:,} from={start:,} to end={end:,} ({end - start:,} blocks)')

    if not DEBUG:
        curr.execute('COMMIT')

    return id_, start, end


# def load_exchange_balances(w3: web3.Web3):
#     if os.path.exists(FNAME_EXCHANGES_WITH_BALANCES):
#         l.debug(f'already prefetched balances (\'{FNAME_EXCHANGES_WITH_BALANCES}\' exists), no need to redo')
#         return

#     tip = w3.eth.get_block('latest')
#     block_number = tip['number'] - 10
#     l.debug(f'Prefiltering based off balances in block {block_number:,}')

#     with open(FNAME_EXCHANGES_WITH_BALANCES + '.tmp', mode='w') as fout:
#         with gzip.open(univ2_fname, mode='rt') as fin:
#             for i, line in enumerate(fin):
#                 if i % 100 == 0:
#                     l.debug(f'Processed {i:,} uniswap v2 exchanges for prefilter')
#                     fout.flush()
#                 address, origin_block, token0, token1 = line.strip().split(',')
#                 address = w3.toChecksumAddress(address)
#                 origin_block = int(origin_block)
#                 token0 = w3.toChecksumAddress(token0)
#                 token1 = w3.toChecksumAddress(token1)

#                 # record liquidity balance of both token0 and token1
#                 try:
#                     bal_token0 = w3.eth.contract(
#                         address=token0,
#                         abi=get_abi('erc20.abi.json'),
#                     ).functions.balanceOf(address).call()
#                     bal_token1 = w3.eth.contract(
#                         address=token1,
#                         abi=get_abi('erc20.abi.json'),
#                     ).functions.balanceOf(address).call()
#                     fout.write(f'2,{address},{origin_block},{token0},{token1},{bal_token0},{bal_token1}\n')
#                 except:
#                     l.exception('could not get balance, ignoring')


#         l.debug('loaded all uniswap v2 exchanges')

#         with gzip.open(univ3_fname, mode='rt') as fin:
#             for i, line in enumerate(fin):
#                 if i % 100 == 0:
#                     l.debug(f'Processed {i:,} uniswap v3 exchanges for prefilter')
#                     fout.flush()
#                 address, origin_block, token0, token1, fee = line.strip().split(',')
#                 address = w3.toChecksumAddress(address)
#                 origin_block = int(origin_block)
#                 token0 = w3.toChecksumAddress(token0)
#                 token1 = w3.toChecksumAddress(token1)
#                 fee = int(fee)

#                 # record liquidity balance of both token0 and token1
#                 try:
#                     bal_token0 = w3.eth.contract(
#                         address=token0,
#                         abi=get_abi('erc20.abi.json'),
#                     ).functions.balanceOf(address).call()
#                     bal_token1 = w3.eth.contract(
#                         address=token1,
#                         abi=get_abi('erc20.abi.json'),
#                     ).functions.balanceOf(address).call()
#                     fout.write(f'3,{address},{origin_block},{token0},{token1},{fee},{bal_token0},{bal_token1}\n')
#                 except:
#                     l.exception('could not get balance, ignoring')

#         l.debug('finished load of exchange graph')
#     os.rename(FNAME_EXCHANGES_WITH_BALANCES + '.tmp', FNAME_EXCHANGES_WITH_BALANCES)


def get_relevant_logs(
        w3: web3.Web3,
        pool: PricerPool,
        batch_start_block: int,
        batch_end_block: int
    ) -> typing.Iterator[typing.Tuple[int, typing.List[web3.types.LogReceipt]]]:

    assert batch_start_block <= batch_end_block

    l.debug(f'start get logs from {batch_start_block:,} to {batch_end_block:,}')

    with utils.profiling.profile('get_logs'):

        assert '0x5Fa464CEfe8901d66C09b85d5Fcdc55b3738c688' in pool._uniswap_v2_pools

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
            'address': list(pool._sushiswap_v2_pools.keys()),
            'topics': [['0x' + x.hex() for x in UniswapV2Pricer.RELEVANT_LOGS]],
            'fromBlock': batch_start_block,
            'toBlock': batch_end_block,
        })

        logs.extend(f.get_all_entries())

        l.debug('got sushiswap v2 logs')

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

    gather = collections.defaultdict(lambda: [])
    for log in logs:
        gather[log['blockNumber']].append(log)

    for i in range(batch_start_block, batch_end_block + 1):
        yield (i, gather[i])


def process_candidates(
        w3: web3.Web3,
        pool: pricers.PricerPool,
        block_number: int,
        updated_exchanges: typing.Dict[typing.Tuple[str, str], typing.List[str]],
        curr: psycopg2.extensions.cursor
    ):
    l.debug(f'{len(updated_exchanges)} exchanges updated in block {block_number:,}')

    next_block_ts = get_block_timestamp(w3, block_number + 1)

    n_ignored = 0
    n_found = 0
    max_profit_no_fee = -1
    for p in find_circuit.profitable_circuits(updated_exchanges, pool, block_number, timestamp=next_block_ts, only_weth_pivot=True):
        if p.profit < MIN_PROFIT_PREFILTER:
            n_ignored += 1
            continue

        if False:
            # some debugging
            exchange_outs = {}
            amount = p.amount_in
            for exc, (token_in, token_out) in zip(p.circuit, p.directions):
                amount_out = exc.token_out_for_exact_in(token_in, token_out, amount, block_identifier=block_number, timestamp=next_block_ts)
                amount_out = pricers.token_transfer.out_from_transfer(token_out, amount_out)
                l.debug(f'{exc.address} out={amount_out}')
                amount = amount_out
                exchange_outs[exc.address] = amount_out
            if amount != p.amount_in + p.profit:
                l.warning(f'did not match up with profit! computed amount_out={amount_out} expected {p.amount_in + p.profit}')

            # run it again using fresh pricers
            amount = p.amount_in
            for exc, (token_in, token_out) in zip(p.circuit, p.directions):
                pricer = exc.copy_without_cache()
                amount_out = pricer.token_out_for_exact_in(token_in, token_out, amount, block_identifier=block_number, timestamp=next_block_ts)
                amount_out = pricers.token_transfer.out_from_transfer(token_out, amount_out)
                amount = amount_out
                l.debug(f'fresh {exc.address} out={amount_out}')
                if exchange_outs[exc.address] != amount_out:
                    l.debug(f'Amount_out {exc.address} changed from {exchange_outs[exc.address]} to {amount_out}')

            if amount != p.amount_in + p.profit:
                l.debug('This did not match after fresh pricer!!!!!!!! weird!!!!')
                l.debug('----------------------------')
                l.debug(f'Found arbitrage')
                l.debug(f'block_number ..... {block_number}')
                l.debug(f'amount_in ........ {p.amount_in}')
                l.debug(f'expected profit .. {p.profit}')
                for p, dxn in zip(p.circuit, p.directions):
                    l.debug(f'    address={p.address} direction={dxn}')
                l.debug('----------------------------')
                raise Exception('what is this')


        # this is potentially profitable, log as a candidate
        dxns = [bytes.fromhex(t[2:]) for t, _ in p.directions]
        curr.execute(
            """
            INSERT INTO candidate_arbitrages (block_number, exchanges, directions, amount_in, profit_no_fee)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING (id)
            """,
            (block_number, [bytes.fromhex(x.address[2:]) for x in p.circuit], dxns, p.amount_in, p.profit)
        )
        # (inserted_id,) = curr.fetchone()
        # l.debug(f'inserted candidate id={inserted_id}')
        n_found += 1
        
        max_profit_no_fee = max(max_profit_no_fee, p.profit)

    if n_ignored > 0:
        l.debug(f'Ignored {n_ignored} arbitrages due to not meeting profit threshold in block {block_number:,}')
    if n_found > 0:
        l.debug(f'Found {n_found} candidate arbitrages in block {block_number:,}')
