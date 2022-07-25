import itertools
import json
import os
import pathlib
import subprocess
import sys
import time
import web3
import web3.types
import web3.contract
import typing
import logging
import argparse
import psycopg2.extensions
import threading
import queue
import pricers

from backtest.utils import connect_db
from eth_account import Account
from eth_account.signers.local import LocalAccount
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.liquidity_bootstrapping_pool import BalancerV2LiquidityBootstrappingPoolPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer

from utils import BALANCER_VAULT_ADDRESS, WETH_ADDRESS, get_abi

l = logging.getLogger(__name__)

DEBUG = True

SHOOTER_ARTIFACT_PATH = pathlib.Path(__file__).parent.parent.parent / 'artifacts' / 'contracts' / 'shooter.sol' / 'Shooter.json'

assert os.path.isfile(SHOOTER_ARTIFACT_PATH)

with open(SHOOTER_ARTIFACT_PATH) as fin:
    SHOOTER_ARTIFACT = json.load(fin)

QUEUE_FINISHED = 0xDEADBEEF

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'connect'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (before run)')

    return parser_name, connect


def connect(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        fill_queue(curr)
        db.commit()
        return

    if args.worker_name is None:
        print('Must supply worker_name', file=sys.stderr)
        exit(1)

    l.info('Starting arbitrage campaign connection')

    time.sleep(4)

    while True:
        maybe_rez = get_reservation(curr, args.worker_name)

        if maybe_rez is None:
            break
        
        reservation_id, start_block, end_block = maybe_rez
        process_reservation(curr, reservation_id, start_block, end_block)


def process_reservation(curr: psycopg2.extensions.cursor, reservation_id: int, start_block: int, end_block: int):
    q = queue.Queue(maxsize=3)

    t = start_ganache_spawner(q, 12_000_000, 12_000_000 + 100)

    while True:
        queue_item = q.get()
        if queue_item == QUEUE_FINISHED:
            l.info(f'Reservation id={reservation_id} completed')
            raise NotImplementedError('idk what do to here')

        if queue_item == None:
            # something broke
            break

        block_number, proc, w3_ganache, acct, shooter_address = queue_item
        block_number: int
        proc: subprocess.Popen
        w3_ganache: web3.Web3
        acct: LocalAccount
        shooter_address: str

        l.debug(f'Processing block {block_number:,}')
        arbs = get_candidates_in_block(curr, block_number)
        l.debug(f'Have {len(arbs):,} candidate arbitrages in block {block_number:,}')


        proc.kill()
        proc.wait()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        """
        CREATE TABLE IF NOT EXISTS connected_candidate_arbitrage_reservations (
            id                 SERIAL PRIMARY KEY NOT NULL,
            block_number_start INTEGER NOT NULL,
            block_number_end   INTEGER NOT NULL,
            progress           INTEGER,
            worker             TEXT,
            updated_on         TIMESTAMP WITHOUT TIME ZONE,
            claimed_on         TIMESTAMP WITHOUT TIME ZONE,
            completed_on       TIMESTAMP WITHOUT TIME ZONE            
        );

        CREATE TABLE IF NOT EXISTS inferred_token_fee_on_transfer (
            id SERIAL             PRIMARY KEY NOT NULL,
            token_id              INTEGER NOT NULL REFERENCES tokens (id) ON DELETE CASCADE,
            fee                   NUMERIC(20, 20),
            block_number_inferred INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_inferred_token_fee_on_transfer_token_id ON inferred_token_fee_on_transfer (token_id);

        CREATE TABLE IF NOT EXISTS connected_arbitrages (
            id                      SERIAL PRIMARY KEY NOT NULL,
            block_number_start      INTEGER NOT NULL,
            block_number_end        INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS connected_arbitrage_member (
            id                      SERIAL NOT NULL,
            connected_arbitrage_id  INTEGER NOT NULL REFERENCES connected_arbitrages (id) ON DELETE CASCADE,
            block_number_start      INTEGER NOT NULL,
            block_number_end        INTEGER NOT NULL,
            gas_usage               INTEGER NOT NULL,
            real_profit_before_fee  INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_connected_arbitrage_member_connected_arbitrage_id ON connected_arbitrage_member (connected_arbitrage_id);
        """
    )
    pass

def fill_queue(curr: psycopg2.extensions.cursor):
    curr.execute('SELECT COUNT(*) FROM connected_candidate_arbitrage_reservations;')
    (n_queued,) = curr.fetchone()

    if n_queued > 0:
        l.debug('not filling queue')
        return

    curr.execute(
        '''
        SELECT MIN(block_number_start), MAX(block_number_end)
        FROM candidate_arbitrage_reservations
        '''
    )
    start_block, end_block = curr.fetchone()

    l.info(f'filling queue from {start_block:,} to {end_block:,}')

    n_segments = 2_000
    segment_width = (end_block - start_block) // n_segments
    for i in itertools.count():
        segment_start = start_block + i * segment_width
        segment_end = min(end_block, segment_start + segment_width - 1)

        if segment_start > end_block:
            break

        curr.execute(
            '''
            INSERT INTO connected_candidate_arbitrage_reservations (block_number_start, block_number_end)
            VALUES (%s, %s)
            ''',
            (segment_start, segment_end),
        )
        assert curr.rowcount == 1

def get_reservation(curr: psycopg2.extensions.cursor, worker_name: str) -> typing.Optional[typing.Tuple[int, int, int]]:
    curr.execute('BEGIN TRANSACTION')

    curr.execute(
        '''
        SELECT id, block_number_start, block_number_end
        FROM connected_candidate_arbitrage_reservations
        WHERE claimed_on IS NULL AND completed_on IS NULL AND block_number_start >= 13100000
        ORDER BY block_number_start ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        '''
    )
    if curr.rowcount < 1:
        l.info('Finished queue')
        return

    id_, start, end = curr.fetchone()
    curr.execute(
        '''
        UPDATE connected_candidate_arbitrage_reservations ccar
        SET claimed_on = NOW()::timestamp, worker = %s
        WHERE id = %s
        ''',
        (worker_name, id_),
    )
    assert curr.rowcount == 1

    assert start <= end

    if not DEBUG:
        curr.connection.commit()

    l.info(f'Processing reservation id={id_:,} from={start:,} to end={end:,} ({end - start:,} blocks)')

    return id_, start, end


_port = 0
def open_ganache(block_number: int) -> typing.Tuple[subprocess.Popen, web3.Web3, LocalAccount, str]:
    global _port
    acct: LocalAccount = Account.from_key(bytes.fromhex('f96003b86ed95cb86eae15653bf4b0bc88691506141a1a9ae23afd383415c268'))

    bin_loc = '/opt/ganache-fork/src/packages/ganache/dist/node/cli.js'
    cwd_loc = '/opt/ganache-fork/'

    my_pid = os.getpid()
    my_slice = my_pid % 500 # I assume this is enough that none of our workers will have duplicate slices
    n_per_slice = 10 # number of ports allocated per slice before they must be reused
    slice_start = 10000 + my_slice * n_per_slice
    ganache_port = slice_start + _port
    _port = (_port + 1) % n_per_slice

    web3_host = os.getenv('WEB3_HOST', 'ws://172.17.0.1:8546')
    p = subprocess.Popen(
        [
            'node',
            bin_loc,
            '--fork.url', web3_host,
            '--fork.blockNumber', str(block_number),
            '--server.port', str(ganache_port),
            '--chain.chainId', '1',
            # '--chain.hardfork', 'arrowGlacier',
            '--wallet.accounts', f'{acct.key.hex()},{web3.Web3.toWei(1000, "ether")}',
        ],
        stdout=subprocess.DEVNULL,
        cwd=cwd_loc,
    )

    l.debug(f'spawned ganache on PID={p.pid} port={ganache_port}')

    w3 = web3.Web3(web3.WebsocketProvider(
            f'ws://localhost:{ganache_port}',
            websocket_timeout=60 * 5,
            websocket_kwargs={
                'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
            },
        )
    )

    def patch_make_batch_request(requests: typing.Tuple[str, typing.Any]):
        ret = []
        for method, args in requests:
            ret.append(w3.provider.make_request(method, args))
        return ret

    w3.provider.make_request_batch = patch_make_batch_request

    while not w3.isConnected():
        time.sleep(0.1)

    for _ in range(10):
        if w3.eth.get_balance(acct.address) == web3.Web3.toWei(1000, 'ether'):
            break
        l.debug('Backing off on balance check...')
        time.sleep(0.01)

    #
    # deploy the shooter
    #
    shooter = w3.eth.contract(
        bytecode = SHOOTER_ARTIFACT['bytecode'],
        abi = SHOOTER_ARTIFACT['abi'],
    )

    constructor_txn = shooter.constructor().buildTransaction({'from': acct.address})
    txn_hash = w3.eth.send_transaction(constructor_txn)
    receipt = w3.eth.wait_for_transaction_receipt(txn_hash)

    shooter_addr = receipt['contractAddress']
    l.info(f'deployed shooter to {shooter_addr} with admin key {acct.address}')

    shooter = w3.eth.contract(
        address = shooter_addr,
        abi = SHOOTER_ARTIFACT['abi'],
    )

    #
    # fund the shooter with some wrapped ether
    #
    amt_to_send = w3.toWei(100, 'ether')
    weth: web3.contract.Contract = w3.eth.contract(
        address=WETH_ADDRESS,
        abi=get_abi('weth9/WETH9.json')['abi'],
    )
    wrap = weth.functions.deposit().buildTransaction({'value': amt_to_send, 'from': acct.address})
    wrap_hash = w3.eth.send_transaction(wrap)
    wrap_receipt = w3.eth.wait_for_transaction_receipt(wrap_hash)
    assert wrap_receipt['status'] == 1

    # transfer to shooter
    xfer = weth.functions.transfer(shooter_addr, amt_to_send).buildTransaction({'from': acct.address})
    xfer_hash = w3.eth.send_transaction(xfer)
    xfer_receipt = w3.eth.wait_for_transaction_receipt(xfer_hash)
    assert xfer_receipt['status'] == 1

    l.info(f'Transferred {amt_to_send / (10 ** 18):.2f} ETH to shooter')

    return p, w3, acct, shooter_addr


def start_ganache_spawner(q: queue.Queue, start_block: int, end_block: int) -> threading.Thread:
    def fill_queue():
        for block_number in range(start_block, end_block + 1):
            try:
                tup = open_ganache(block_number)
            except:
                l.exception('ganache spawner exception')
                q.put(None)
                return

            q.put((block_number, *tup))
        q.put(QUEUE_FINISHED)

    t = threading.Thread(
        target=fill_queue,
        name='queue-filler',
        daemon=True,
    )
    t.start()
    return t


class CandidateArbitrage(typing.NamedTuple):
    id_: int
    exchanges: typing.List[str]
    directions: typing.List[typing.Tuple[str, str]]
    amount_in: int
    profit_before_fee: int
    block_number: int


def get_candidates_in_block(
        curr: psycopg2.extensions.cursor,
        block_number: int
    ) -> typing.List[
        typing.Tuple[int, typing.Tuple[str], typing.List[typing.Tuple[str, str]]]
    ]:
    curr.execute(
        '''
        SELECT id, exchanges, directions, amount_in, profit_no_fee
        FROM candidate_arbitrages
        WHERE block_number = %s
        ''',
        (block_number,)
    )

    ret = []
    for id_, bexchanges, bdirections, amount_in, profit in curr:
        exchanges = [web3.Web3.toChecksumAddress(x.tobytes()) for x in bexchanges]
        directions = [web3.Web3.toChecksumAddress(x.tobytes()) for x in bdirections]
        directions = list(zip(directions, directions[1:] + [directions[0]]))
        
        assert len(exchanges) == len(directions)

        ret.append(CandidateArbitrage(
            id_ = id_,
            exchanges = exchanges,
            directions = directions,
            amount_in = int(amount_in),
            profit_before_fee = int(profit),
            block_number=block_number
        ))
    
    return ret

def attempt_shoot_candidate(
        w3_ganache: web3.Web3,
        curr: psycopg2.extensions.cursor,
        account: LocalAccount,
        shooter_address: str,
        candidate: CandidateArbitrage,
    ):
    exchange_pricers = [
        load_pricer_for(w3_ganache, curr, x)
        for x in candidate.exchanges
    ]
    pass


def load_pricer_for(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        exchange: str,
    ) -> typing.Optional[pricers.BaseExchangePricer]:
    bexchange = bytes.fromhex(exchange[2:])

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM uniswap_v2_exchanges uv2
        JOIN tokens t0 ON uv2.token0_id = t0.id
        JOIN tokens t1 ON uv2.token1_id = t1.id
        WHERE uv2.address = %s
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM sushiv2_swap_exchanges sv2
        JOIN tokens t0 ON sv2.token0_id = t0.id
        JOIN tokens t1 ON sv2.token1_id = t1.id
        WHERE sv2.address = %s
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM shibaswap_exchanges ss
        JOIN tokens t0 ON ss.token0_id = t0.id
        JOIN tokens t1 ON ss.token1_id = t1.id
        WHERE ss.address = %s
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address, originalfee
        FROM uniswap_v3_exchanges uv3
        JOIN tokens t0 ON uv3.token0_id = t0.id
        JOIN tokens t1 ON uv3.token1_id = t1.id
        WHERE uv3.address = %s            
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1
        token0, token1, fee = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV3Pricer(w3, exchange, token0, token1, fee)
        return p

    curr.execute(
        '''
        SELECT EXISTS(SELECT 1 FROM balancer_exchanges WHERE address = %s)
        ''',
        (bexchange,)
    )
    (is_balancerv1,) = curr.fetchone()
    if is_balancerv1:
        p = BalancerPricer(w3, exchange)
        return p

    curr.execute(
        '''
        SELECT pool_id, pool_type
        FROM balancer_v2_exchanges
        WHERE address = %s
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1
        pool_id, pool_type = curr.fetchone()
        pool_id = pool_id.tobytes()

        vault = w3.eth.contract(
            address = BALANCER_VAULT_ADDRESS,
            abi = get_abi('balancer_v2/Vault.json'),
        )

        if pool_type in ['WeightedPool', 'WeightedPool2Tokens']:
            return BalancerV2WeightedPoolPricer(w3, vault, exchange, pool_id)
        elif pool_type in ['LiquidityBootstrappingPool', 'NoProtocolFeeLiquidityBootstrappingPool']:
            return BalancerV2LiquidityBootstrappingPoolPricer(w3, vault, exchange, pool_id)

    l.error(f'Could not find exchange for address {exchange}')
    return None
