"""
Fills information about which arbitrages must have been backrunning.

Detects backrunning by re-ordering arbitrages to top-of-block
and checking against our model for the same or different profit before fees.
"""

import argparse
import datetime
import itertools
import os
import subprocess
import sys
import time
import typing
import psycopg2
import psycopg2.extensions
import web3
import web3.contract
import web3.types
import web3._utils.filters
import logging
import numpy as np

import backtest.gather_samples.analyses
from backtest.utils import ERC20_TRANSFER_TOPIC, connect_db
from utils import get_abi, setup_logging
from utils import erc20

l = logging.getLogger(__name__)

uv2_factory: web3.contract.Contract = web3.Web3().eth.contract(address=b'\x00'*20, abi=get_abi('uniswap_v2/IUniswapV2Factory.json'))
uv3_factory: web3.contract.Contract = web3.Web3().eth.contract(address=b'\x00'*20, abi=get_abi('uniswap_v3/IUniswapV3Factory.json'))

DEBUG = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fill', action='store_true', help='fill the work queue', dest='fill')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.add_argument('--worker-name', type=str, default=None, help='worker name for log, must be POSIX path-safe')
    args = parser.parse_args()

    setup_logging('fill_backrunners', worker_name=args.worker_name, stdout_level=logging.DEBUG if args.verbose else logging.INFO)

    db = connect_db()
    curr = db.cursor()


    # curr.execute(
    #     '''
    #     SELECT id, start_block, end_block_exclusive
    #     FROM backrun_detection_reservations
    #     WHERE started_on IS NULL AND finished_on IS NULL
    #     '''
    # )
    # for id_, start_block, end_block in list(curr):
    #     curr.execute(
    #         '''
    #         DELETE FROM sample_arbitrage_backrun_detections bd
    #         WHERE EXISTS(
    #             SELECT 1
    #             FROM sample_arbitrages sa
    #             WHERE sa.id = bd.sample_arbitrage_id AND %s <= sa.block_number AND sa.block_number < %s
    #         )
    #         ''',
    #         (start_block, end_block),
    #     )
    #     l.info(f'Deleted {curr.rowcount:,} from {start_block:,} to {end_block:,}')

    # input('continue?')

    # return

    # # quick fixup
    # l.debug(f'fixing up')

    # d = datetime.datetime(year=2022, month=8, day=8, hour=22, minute=55)

    # curr.execute(
    #     '''
    #     SELECT id, start_block, end_block_exclusive
    #     FROM backrun_detection_reservations
    #     WHERE (finished_on > %s)
    #     ''',
    #     (d,)
    # )
    # l.info(f'Have {curr.rowcount} reservations to fix-up')
    # for id_, start_block, end_block in list(curr):
    #     curr.execute(
    #         '''
    #         DELETE FROM sample_arbitrage_backrun_detections bd
    #         WHERE EXISTS(
    #             SELECT 1
    #             FROM sample_arbitrages sa
    #             WHERE sa.id = bd.sample_arbitrage_id AND %s <= sa.block_number AND sa.block_number < %s
    #         )
    #         ''',
    #         (start_block, end_block),
    #     )
    #     l.info(f'Deleted {curr.rowcount:,} from {start_block:,} to {end_block:,}')

    #     curr.execute(
    #         '''
    #         UPDATE backrun_detection_reservations SET started_on = NULL, finished_on = NULL WHERE id = %s
    #         ''',
    #         (id_,)
    #     )
    #     assert curr.rowcount == 1

    # input('Continue?')

    # db.commit()

    # return

    web3_host = os.getenv('WEB3_HOST', 'ws://172.17.0.1:8546')
    w3 = web3.Web3(web3.WebsocketProvider(
        web3_host,
        websocket_timeout=60 * 5,
        websocket_kwargs={
            'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
        },
    ))
    
    if not w3.isConnected():
        l.error(f'Could not connect to web3')
        exit(1)

    l.debug(f'Connected to web3, chainId={w3.eth.chain_id}')

    # ganache_proc, ganache_w3 = open_ganache()
    # exit(1)

    ganache_proc = None
    try:
        setup_db(curr)
        if args.fill:
            fill_queue(curr)
        else:
            ganache_proc, ganache_w3 = open_ganache()
            for res in get_reservations(curr):
                do_reorder(w3, ganache_w3, curr, res)
    except:
        l.exception('top-level exception')
        raise
    finally:
        if ganache_proc is not None:
            ganache_proc.kill()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS sample_arbitrage_backrun_detections (
            sample_arbitrage_id INTEGER NOT NULL PRIMARY KEY REFERENCES sample_arbitrages (id) ON DELETE CASCADE,
            rerun_exactly BOOLEAN,
            rerun_reverted BOOLEAN,
            rerun_no_arbitrage BOOLEAN,
            rerun_not_comparable BOOLEAN,
            rerun_profit_token_changed BOOLEAN,
            rerun_profit NUMERIC(78, 0)
        );
        '''
    )
    
    # create queue
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS backrun_detection_reservations (
            id SERIAL PRIMARY KEY NOT NULL,
            start_block         INTEGER NOT NULL,
            end_block_exclusive  INTEGER NOT NULL,
            started_on          TIMESTAMP WITHOUT TIME ZONE,
            finished_on         TIMESTAMP WITHOUT TIME ZONE
        );
        '''
    )
    curr.connection.commit()

    # see if we need to fill queue    
    curr.execute('SELECT count(*) FROM backrun_detection_reservations')


    (n_rows,) = curr.fetchone()
    if n_rows == 0:
        l.warning('Queue not created. Please run the fill procedure.')


def fill_queue(curr: psycopg2.extensions.cursor):
    # see if we need to fill queue    
    curr.execute('SELECT count(*) FROM backrun_detection_reservations')

    (n_rows,) = curr.fetchone()

    # find lowest / highest blocks where we have sample arbitrage transactions
    curr.execute(
        '''
        SELECT min(block_number), max(block_number)
        FROM sample_arbitrages
        '''
    )
    min_block, max_block = curr.fetchone()
    assert min_block < max_block
    assert (max_block - min_block) > 1000

    if n_rows > 0:
        l.error('Queue already created.')

        # check for expanding back and forward
        curr.execute('SELECT MIN(start_block), MAX(end_block_exclusive) FROM backrun_detection_reservations')
        reservations_start_at, reservations_end_at_exclusive = curr.fetchone()

        batch_size = 1000

        n_inserted = 0
        if reservations_start_at > min_block:
            l.info('expanding back in history')
            # fill from start to end
            for i in itertools.count():
                this_start_block = i * batch_size + min_block
                this_end_block = min(reservations_start_at, (i + 1) * batch_size + min_block)
                if this_start_block > reservations_start_at:
                    break
                curr.execute(
                    '''
                    INSERT INTO backrun_detection_reservations (start_block, end_block_exclusive)
                    VALUES (%s, %s)
                    ''',
                    (this_start_block, this_end_block)
                )
                assert curr.rowcount == 1
                n_inserted += 1

        if reservations_end_at_exclusive < max_block:
            l.info('expanding forward in history')
            # fill from start to end
            for i in itertools.count():
                this_start_block = i * batch_size + reservations_end_at_exclusive
                this_end_block = min(max_block + 1, (i + 1) * batch_size + reservations_end_at_exclusive)
                if this_start_block > max_block + 1:
                    break
                curr.execute(
                    '''
                    INSERT INTO backrun_detection_reservations (start_block, end_block_exclusive)
                    VALUES (%s, %s)
                    ''',
                    (this_start_block, this_end_block)
                )
                assert curr.rowcount == 1
                n_inserted += 1

        l.info(f'Inserted {n_inserted} reservations')
        if n_inserted > 0:
            input('ENTER to commit')
            curr.connection.commit()
            input('ENTER to continue')
            raise Exception('stop')

        exit(1)


    l.debug(f'filling queue...')

    # break into 10,000 reservations
    lower_bounds, step = np.linspace(min_block, max_block, 10_000, dtype=int, retstep=True)
    l.debug(f'reservation step size = {int(step):,} blocks')

    for res_lower, res_upper_excl in zip(lower_bounds, list(lower_bounds[1:]) + [max_block + 1]):
        assert res_lower < res_upper_excl

        curr.execute(
            '''
            INSERT INTO backrun_detection_reservations (start_block, end_block_exclusive)
            VALUES (%s, %s)
            ''',
            (int(res_lower), int(res_upper_excl))
        )

    curr.connection.commit()
    l.debug(f'Inserted {len(lower_bounds)} reservations')


def get_reservations(curr: psycopg2.extensions.cursor) -> typing.Iterator[typing.Tuple[int, int, int]]:
    """
    Gets reorder reservations
    Returns (reservation_id, start, end_exclusive)
    """
    while True:
        if not DEBUG:
            curr.connection.commit()
        curr.execute(
            '''
            SELECT id, start_block, end_block_exclusive
            FROM backrun_detection_reservations
            WHERE started_on is null
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            '''
        )

        if curr.rowcount != 1:
            assert curr.rowcount <= 0
            l.info('Out of work')
            return

        (reservation_id, start_block, end_block_exclusive) = curr.fetchone()

        curr.execute('UPDATE backrun_detection_reservations SET started_on = now()::timestamp WHERE id = %s', (reservation_id,))
        assert curr.rowcount == 1
        if not DEBUG:
            curr.connection.commit()
        yield (reservation_id, start_block, end_block_exclusive)


def open_ganache() -> typing.Tuple[subprocess.Popen, web3.Web3]:
    bin_loc = '/opt/ganache-fork/src/packages/ganache/dist/node/cli.js'
    cwd_loc = '/opt/ganache-fork/'

    my_pid = os.getpid()
    ganache_port = 34451 + (my_pid % 10_000)

    web3_host = os.getenv('WEB3_HOST', 'ws://172.17.0.1:8546')
    p = subprocess.Popen(
        [
            'node',
            bin_loc,
            '--fork.url', web3_host,
            '--server.port', str(ganache_port),
            '--chain.chainId', '1',
        ],
        stdout=subprocess.DEVNULL,
        cwd=cwd_loc,
    )

    l.debug(f'spawned ganache on PID={p.pid} port={ganache_port}')

    w3 = web3.Web3(
        web3.WebsocketProvider(
            f'ws://localhost:{ganache_port}',
            websocket_timeout=60 * 5,
            websocket_kwargs={
                'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
            },
        )
    )

    while not w3.isConnected():
        time.sleep(0.1)

    return p, w3


def do_reorder(w3_mainnet: web3.Web3, w3_ganache: web3.Web3, curr: psycopg2.extensions.cursor, reservation):
    reservation_id, reservation_start, reservation_end_exclusive = reservation

    curr.execute(
        '''
        SELECT id, txn_hash
        FROM sample_arbitrages_no_fp sa
        WHERE %s <= block_number AND block_number < %s AND NOT EXISTS(SELECT FROM sample_arbitrage_backrun_detections WHERE sample_arbitrage_id = sa.id)
        ''',
        (reservation_start, reservation_end_exclusive),
    )
    l.debug(f'have {curr.rowcount:,} items to process in this reservation (id={reservation_id:,})')
    queue: typing.List[typing.Tuple[int, bytes]] = [(i, txn.tobytes()) for i, txn in curr]

    for id_, txn_hash in queue:
        result = test_a_reorder(curr, w3_mainnet, w3_ganache, id_, txn_hash)
        insert_reorder_result(curr, id_, txn_hash, result)
        if id_ & 0x7 == 0:
            if not DEBUG:
                curr.connection.commit()

    l.info(f'finished with reservation id={reservation_id:,}')
    curr.execute(
        'UPDATE backrun_detection_reservations SET finished_on = now()::timestamp WHERE id = %s',
        (reservation_id,)
    )
    if not DEBUG:
        curr.connection.commit()


class ReshootResult:
    pass

class ReshootReverted(ReshootResult):
    pass

class ReshootIdentical(ReshootResult):
    """
    The reshoot was a success, the transaction logs are identical
    """
    pass

class ReshootNoArbitrage(ReshootResult):
    pass

class NotOneCycle(ReshootResult):
    pass

class ProfitTokenChanged(ReshootResult):
    pass

class ProfitChanged(ReshootResult):
    new_profit: int

    def __init__(self, new_profit: int) -> None:
        self.new_profit = new_profit

def test_a_reorder(
        curr: psycopg2.extensions.cursor,
        w3_mainnet: web3.Web3,
        w3_ganache: web3.Web3,
        sample_id: int,
        txn_hash: bytes
    ) -> ReshootResult:
    l.debug(f'reordering 0x{txn_hash.hex()} (id={sample_id:,})')

    original_receipt = w3_mainnet.eth.get_transaction_receipt('0x' + txn_hash.hex())

    resp = w3_ganache.provider.make_request('eth_callAtFront', ['0x' + txn_hash.hex()])

    if 'result' not in resp:
        print(resp)

    # if the reorder shot reverted then we can quickly discard it anyway
    if resp['result']['exception'] != None:
        assert resp['result']['exception'] in ['revert', 'out of gas', 'invalid opcode'], f"unexpected exception value {resp['result']['exception']}"
        return ReshootReverted()

    # gather relevant logs for both (into tuples)
    logs_reshoot  = []
    logs_original = []

    for log in resp['result']['logs']:
        topics = tuple(bytes.fromhex(topic) for topic in log['indexes'])
        if len(topics) == 0 or topics[0] != ERC20_TRANSFER_TOPIC:
            continue
        address = w3_mainnet.toChecksumAddress(log['address'])
        payload = log['payload']
        logs_reshoot.append((address, topics, payload))

    for log in original_receipt['logs']:
        if len(log['topics']) == 0 or log['topics'][0] != ERC20_TRANSFER_TOPIC:
            continue
        logs_original.append((
            log['address'],
            tuple(log['topics']),
            log['data'][2:]
        ))

    assert len(logs_original) >= 3, 'need 3 transfers to make an arbitrage'

    if set(logs_reshoot) == set(logs_original):
        return ReshootIdentical()

    # logs are not identical -- re-run analysis and see what's up

    # parse everything to erc20 transfer events
    erc20_xfers_reshoot = []
    for i, (address, topics, payload) in enumerate(logs_reshoot):
        lr = web3.types.LogReceipt(
            address = address,
            topic = topics[0],
            topics = topics,
            data = payload,
            logIndex = i,
            transactionIndex = i,
            blockNumber = 0,
            blockHash = None,
            transactionHash = txn_hash,
        )
        erc20_xfers_reshoot.append(erc20.events.Transfer().processLog(lr))

    analysis = backtest.gather_samples.analyses.get_arbitrage_from_receipt_if_exists(original_receipt, erc20_xfers_reshoot)

    if analysis is None:
        return ReshootNoArbitrage()

    #
    # is this too advanced to diagnose what happened
    #
    if analysis.n_cycles > 1:
        return NotOneCycle()

    curr.execute('SELECT n_cycles FROM sample_arbitrages WHERE id = %s', (sample_id,))
    assert curr.rowcount == 1
    (original_n_cycles,) = curr.fetchone()

    if original_n_cycles > 1:
        return NotOneCycle()

    #
    # is profit still in the same token? if so we can do apples-to-apples comparison
    #
    curr.execute(
        '''
        SELECT t1.address
        FROM sample_arbitrage_cycles sac
        JOIN tokens t1 ON sac.profit_token = t1.id
        WHERE sac.sample_arbitrage_id = %s
        ''',
        (sample_id,)
    )
    assert curr.rowcount == 1
    (original_profit_token,) = curr.fetchone()
    original_profit_token = w3_mainnet.toChecksumAddress(original_profit_token.tobytes())

    if original_profit_token != analysis.only_cycle.profit_token:
        return ProfitTokenChanged()

    return ProfitChanged(analysis.only_cycle.profit_amount)


def insert_reorder_result(curr: psycopg2.extensions.cursor, id_: int, txn_hash: bytes, result: ReshootResult):
    l.debug(f'reshoot result 0x{txn_hash.hex()} -> {type(result).__name__}')

    if isinstance(result, ReshootIdentical):
        curr.execute(
            '''
            INSERT INTO sample_arbitrage_backrun_detections
            (sample_arbitrage_id, rerun_exactly)
            VALUES (%s, true)
            ''',
            (id_,)
        )
    elif isinstance(result, ReshootReverted):
        curr.execute(
            '''
            INSERT INTO sample_arbitrage_backrun_detections
            (sample_arbitrage_id, rerun_reverted)
            VALUES (%s, true)
            ''',
            (id_,)
        )
    elif isinstance(result, ReshootNoArbitrage):
        curr.execute(
            '''
            INSERT INTO sample_arbitrage_backrun_detections
            (sample_arbitrage_id, rerun_no_arbitrage)
            VALUES (%s, true)
            ''',
            (id_,)
        )
    elif isinstance(result, NotOneCycle):
        curr.execute(
            '''
            INSERT INTO sample_arbitrage_backrun_detections
            (sample_arbitrage_id, rerun_not_comparable)
            VALUES (%s, true)
            ''',
            (id_,)
        )
    elif isinstance(result, ProfitTokenChanged):
        curr.execute(
            '''
            INSERT INTO sample_arbitrage_backrun_detections
            (sample_arbitrage_id, rerun_profit_token_changed)
            VALUES (%s, true)
            ''',
            (id_,)
        )
    elif isinstance(result, ProfitChanged):
        assert isinstance(result.new_profit, int)
        curr.execute(
            '''
            INSERT INTO sample_arbitrage_backrun_detections
            (sample_arbitrage_id, rerun_profit)
            VALUES (%s, %s)
            ''',
            (id_, result.new_profit)
        )
    else:
        raise NotImplementedError('unreachable')


if __name__ == '__main__':
    main()

