"""
Finds the top-value candidate arbitrages
"""


import argparse
import collections
import datetime
import decimal
import itertools
import time
import typing
import logging
import psycopg2.extensions

import web3

from backtest.utils import connect_db

l = logging.getLogger(__name__)

LARGE_ARBITRAGE_THRESHOLD = 1 * (10 ** 18)

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'fill-top-arbs'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (before run)')

    parser.add_argument('--fill-samples', type=bool, help='fill top sample arbitrages (step 1)')

    parser.add_argument('--id', type=int, help='worker id, required for processing', default=0)
    parser.add_argument('--n-workers', type=int, help='number of workers', default=1)

    return parser_name, fill_top_arbs

def fill_top_arbs(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        db.commit()
        l.info('setup db')
        return

    l.info(f'Starting fill top arbitrages')

    batch_size = 10_000

    assert args.id < args.n_workers

    curr.execute('SELECT MIN(start_block), MAX(end_block) FROM block_samples')
    start_block, end_block = curr.fetchone()
    marks = collections.deque(maxlen=100)
    last_update = time.time()

    for i in itertools.count():
        if i % args.n_workers != args.id:
            continue

        batch_start = start_block + i * batch_size
        batch_end = min(end_block, batch_start + batch_size - 1)

        marks.append((batch_start, time.time()))
        now = time.time()
        if now > last_update + 60:
            last_update = now
            prev_batch_start, prev_time = marks[0]
            elapsed = now - prev_time
            processed = batch_start - prev_batch_start
            nps = processed / elapsed
            remaining = end_block - batch_start
            eta_seconds = remaining / nps
            eta = datetime.timedelta(seconds=eta_seconds)

            l.info(f'Progress {(batch_start - start_block) / (end_block - start_block) * 100:.2f}% ETA {eta}')

        if batch_start > end_block:
            l.info('Done')
            break

        curr.execute(
            '''
            INSERT INTO large_candidate_arbitrages (candidate_arbitrage_id, profit, block_number)
            SELECT id, profit_no_fee, block_number
            FROM candidate_arbitrages
            WHERE %s <= block_number AND block_number <= %s AND profit_no_fee >= %s
            ''',
            (batch_start, batch_end, decimal.Decimal(LARGE_ARBITRAGE_THRESHOLD))
        )
        l.debug(f'Have {curr.rowcount:,} large arbitrages from block {batch_start:,} to {batch_end:,}')

    db.commit()

def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS large_candidate_arbitrages (
            candidate_arbitrage_id BIGINT NOT NULL,
            profit NUMERIC(78, 0) NOT NULL,
            block_number INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_large_candidate_arbitrages_candidate_arbitrage_id ON large_candidate_arbitrages (candidate_arbitrage_id);
        CREATE INDEX IF NOT EXISTS idx_large_candidate_arbitrages_block_number ON large_candidate_arbitrages (block_number);
        '''
    )

