import random
import collections
import itertools
import logging
import argparse
import os
import socket
import networkx as nx
import typing
import web3
import web3.types
import web3._utils.filters
import psycopg2.extensions
from backtest.gather_samples.analyses import get_arbitrage_if_exists
from backtest.gather_samples.database import insert_arbs, setup_db

from backtest.utils import ERC20_TRANSFER_TOPIC_HEX, ERC20_TRANSFER_TOPIC, CancellationToken, connect_db
from utils import setup_logging, erc20
from utils.throttler import BlockThrottle

l = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-name', type=str, default=None, help='worker name for log, must be POSIX path-safe')
    parser.add_argument('--setup-db', action='store_true', dest='setup_db')


    args = parser.parse_args()
    if args.worker_name is None:
        args.worker_name = socket.gethostname()
    job_name = 'gather_samples'

    setup_logging(job_name, worker_name = args.worker_name, stdout_level=logging.INFO)

    try:
        l.info('booting up sample arbitrage scraper')

        db = connect_db()
        curr = db.cursor()

        curr.execute('SELECT MIN(start_block), MAX(end_block) FROM block_samples')
        start_block, end_block = curr.fetchone()

        if args.setup_db:
            setup_db(curr)

            # start_block = 15721079 - (6_646) * 14
            # end_block = 15721079
            setup_reservations(curr, start_block, end_block)
            return

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

        # # debug a transaction
        # txn_hash = '0x4c4fd405de8f88d33570b2a27013e95f8ab8a5394cfe4a5fd9efea0120434f6f'
        # btxn_hash = bytes.fromhex(txn_hash[2:])
        # receipt = w3.eth.get_transaction_receipt(txn_hash)
        # txns = []
        # for r in receipt['logs']:
        #     print(r)
        #     if r['topics'][0] == ERC20_TRANSFER_TOPIC:
        #         txns.append(erc20.events.Transfer().processLog(r))

        # print(txns)
        # get_arbitrage_if_exists(
        #     w3,
        #     bytes.fromhex('4c4fd405de8f88d33570b2a27013e95f8ab8a5394cfe4a5fd9efea0120434f6f'),
        #     txns,
        # )

        # return


        l.debug(f'Connected to web3, chainId={w3.eth.chain_id}')

        cancellation_token = CancellationToken(job_name, args.worker_name, connect_db())

        while not cancellation_token.cancel_requested():
            maybe_rez = get_reservation(curr, start_block, end_block)
            if maybe_rez is None:
                # we're at the end
                break

            try:
                with maybe_rez as (reservation_start, reservation_end):
                    process_reservation(w3, curr, reservation_start, reservation_end, cancellation_token)
            except ReservationCancelRequestedException:
                pass
    except Exception:
        l.exception('exiting from root-level exception')
        exit(1)


class ReservationContextManager:

    def __init__(self, id_: int, curr: psycopg2.extensions.cursor, start_block: int, end_block_exclusive: int) -> None:
        assert isinstance(id_, int)
        assert isinstance(start_block, int)
        assert isinstance(end_block_exclusive, int)
        assert start_block < end_block_exclusive
        self.id = id_
        self.start_block = start_block
        self.end_block_exclusive = end_block_exclusive
        self.curr = curr

    def __enter__(self) -> typing.Tuple[int, int]:
        return (self.start_block, self.end_block_exclusive)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is None:
            self.curr.execute(
                '''
                UPDATE gather_sample_arbitrages_reservations
                SET finished_on = now()::timestamp
                WHERE id = %s
                ''',
                (self.id,)
            )
            assert self.curr.rowcount == 1
            self.curr.connection.commit()
        elif exc_type == ReservationCancelRequestedException:
            exc_value = typing.cast(ReservationCancelRequestedException, exc_value)
            if exc_value.remaining_start_block == self.start_block:
                # no work was done, just mark as not started
                self.curr.execute(
                    '''
                    UPDATE gather_sample_arbitrages_reservations
                    SET started_on = NULL
                    WHERE id = %s
                    ''',
                    (self.id,),
                )
            else:
                # break the remaining work off
                self.curr.execute(
                    '''
                    UPDATE gather_sample_arbitrages_reservations
                    SET finished_on = now()::timestamp, to_block_exclusive = %s
                    WHERE id = %s
                    ''',
                    (exc_value.remaining_start_block, self.id)
                )
                self.curr.execute(
                    '''
                    INSERT INTO gather_sample_arbitrages_reservations (from_block, to_block_exclusive)
                    VALUES (%s, %s)
                    RETURNING id
                    ''',
                    (exc_value.remaining_start_block, self.end_block_exclusive),
                )
                assert self.curr.rowcount == 1
                (new_reservation_id,) = self.curr.fetchone()
                l.debug(f'split remaining work of reservation id={self.id} into id={new_reservation_id}')
            self.curr.connection.commit()

BATCH_SIZE_BLOCKS = 1_000
def setup_reservations(curr: psycopg2.extensions.cursor, start_block: int, end_block_inclusive: int):
    curr.execute('LOCK TABLE gather_sample_arbitrages_reservations')

    curr.execute('SELECT COUNT(*) FROM gather_sample_arbitrages_reservations')
    (res_cnt,) = curr.fetchone()
    if res_cnt > 0:
        # already have some reservations

        # test for expanding back or forward in history

        curr.execute('SELECT MIN(from_block), MAX(to_block_exclusive) FROM gather_sample_arbitrages_reservations')
        reservations_start_at, reservations_end_at_exclusive = curr.fetchone()

        n_inserted = 0
        if reservations_start_at > start_block:
            l.info('expanding back in history')
            # fill from start to end
            for i in itertools.count():
                this_start_block = i * BATCH_SIZE_BLOCKS + start_block
                this_end_block = min(reservations_start_at, (i + 1) * BATCH_SIZE_BLOCKS + start_block)
                if this_start_block > reservations_start_at:
                    break
                curr.execute(
                    '''
                    INSERT INTO gather_sample_arbitrages_reservations (from_block, to_block_exclusive)
                    VALUES (%s, %s)
                    RETURNING id
                    ''',
                    (this_start_block, this_end_block)
                )
                assert curr.rowcount == 1
                n_inserted += 1

        if reservations_end_at_exclusive < end_block_inclusive + 1:
            l.info('expanding forward in history')
            # fill from start to end
            for i in itertools.count():
                this_start_block = i * BATCH_SIZE_BLOCKS + reservations_end_at_exclusive
                this_end_block = min(end_block_inclusive + 1, (i + 1) * BATCH_SIZE_BLOCKS + reservations_end_at_exclusive)
                if this_start_block > end_block_inclusive + 1:
                    break
                curr.execute(
                    '''
                    INSERT INTO gather_sample_arbitrages_reservations (from_block, to_block_exclusive)
                    VALUES (%s, %s)
                    RETURNING id
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
    else:
        l.debug(f'inserting reservations')
        block_ranges = []
        for i in itertools.count():
            this_start_block = i * BATCH_SIZE_BLOCKS + start_block
            this_end_block = min(end_block_inclusive + 1, (i + 1) * BATCH_SIZE_BLOCKS + start_block)
            if this_start_block > end_block_inclusive:
                break
            block_ranges.append((this_start_block, this_end_block))
        
        random.shuffle(block_ranges)
        for this_start_block, this_end_block in block_ranges:
            curr.execute(
                '''
                INSERT INTO gather_sample_arbitrages_reservations (from_block, to_block_exclusive)
                VALUES (%s, %s)
                RETURNING id
                ''',
                (this_start_block, this_end_block)
            )
            assert curr.rowcount == 1
        curr.connection.commit()


def get_reservation(curr: psycopg2.extensions.cursor, start_block: int, end_block_inclusive: int) -> typing.Optional[ReservationContextManager]:
    """
    Gets the next group of blocks to process.

    Returns None when there are no more groups to process.
    """
    curr.execute('BEGIN TRANSACTION')
    curr.execute(
        '''
            SELECT id, from_block, to_block_exclusive
            FROM gather_sample_arbitrages_reservations
            WHERE started_on IS NULL
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        '''
    )

    maybe_row = curr.fetchall()

    reservation_id = None
    reservation_start = None
    reservation_end = None

    if len(maybe_row) > 0:
        assert len(maybe_row) == 1
        reservation_id, reservation_start, reservation_end = maybe_row[0]
        curr.execute('UPDATE gather_sample_arbitrages_reservations SET started_on = now()::timestamp WHERE id = %s', (reservation_id,))
        assert curr.rowcount == 1
        curr.connection.commit()
        l.debug(f'working on reservation id={reservation_id} start={reservation_start:,} to {reservation_end:,}')
        return ReservationContextManager(
            reservation_id,
            curr,
            reservation_start,
            reservation_end
        )
    else:
        # release locks
        curr.connection.commit()
        l.info('work done')
        return None


class ReservationCancelRequestedException(Exception):
    remaining_start_block: int

    def __init__(self, remaining_start_block, *args: object) -> None:
        super().__init__(*args)
        self.remaining_start_block = remaining_start_block


def process_reservation(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        reservation_start: int,
        reservation_end_exclusive: int,
        cancellation_token: CancellationToken
    ):
    l.info(f'processing blocks {reservation_start:,} to {reservation_end_exclusive:,} ({reservation_end_exclusive-reservation_start:,} blocks)')

    throttler = BlockThrottle(
        setpoint = 5_000, # log events per round-trip query
        additive_increase = 2,
        initial = 10, # starting guess
    )

    this_block_start = reservation_start
    while this_block_start < reservation_end_exclusive:
        if cancellation_token.cancel_requested():
            raise ReservationCancelRequestedException(this_block_start)

        n_blocks = throttler.val_int_clamp(1, 10_000)
        this_end_block_inclusive = min(reservation_end_exclusive - 1, this_block_start + n_blocks - 1)
        assert this_block_start <= this_end_block_inclusive

        f: web3._utils.filters.Filter = w3.eth.filter({
            'fromBlock': this_block_start,
            'toBlock': this_end_block_inclusive,
            'topics': [ERC20_TRANSFER_TOPIC_HEX],
        })
        logs = f.get_all_entries()
        throttler.observe(len(logs))

        process_batch(w3, curr, logs)
        curr.connection.commit()

        this_block_start = this_end_block_inclusive + 1


def process_batch(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        logs: typing.List[web3.types.LogReceipt]
    ):
    # gather logs into per-transaction
    tx_to_logs = collections.defaultdict(lambda: [])
    for log in logs:
        tx_to_logs[bytes(log['transactionHash'])].append(log)

    # ensure logs are sorted properly
    for tx_hash in tx_to_logs.keys():
        tx_to_logs[tx_hash] = sorted(
            tx_to_logs[tx_hash],
            key = lambda x: (x['blockNumber'], x['logIndex']),
        )

    # drop transactions with transfer logs that don't parse
    tx_to_parsed_txns = {}
    for tx_hash, logs in tx_to_logs.items():
        broken = False
        parsed_txns = []
        for log in logs:
            try:
                txn = erc20.events.Transfer().processLog(log)
                parsed_txns.append(txn)
            except web3.exceptions.LogTopicError:
                # broken
                broken = True
                break
        if not broken:
            if len(parsed_txns) >= 3:
                tx_to_parsed_txns[tx_hash] = parsed_txns

    l.debug(f'Have {len(tx_to_parsed_txns)} transactions to investigate')

    # Process each transaction
    arbs = []
    processed_already = set()
    for tx_hash, txns in tx_to_parsed_txns.items():
        assert tx_hash not in processed_already
        processed_already.add(tx_hash)
        if len(txns) >= 3:
            arb = get_arbitrage_if_exists(
                w3, tx_hash, txns
            )
            if arb is not None:
                arbs.append(arb)
    
    insert_arbs(w3, curr, arbs)

    curr.connection.commit()


if __name__ == '__main__':
    main()
