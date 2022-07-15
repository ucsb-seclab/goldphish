"""
Attempts to re-find arbitrages that actually occurred on the blockchain.
"""

import argparse
import time
import typing
import web3
import web3.types
import web3._utils.filters
import psycopg2
import psycopg2.extensions
import logging

from backtest.utils import connect_db

l = logging.getLogger(__name__)

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'replicate'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (run before mass scan)')

    return parser_name, replicate


def replicate(w3: web3.Web3, args: argparse.Namespace):
    l.info('Starting replication')
    db = connect_db()
    curr = db.cursor()
    time.sleep(4)

    if args.setup_db:
        setup_db(curr)
        return

    l.info('done')


def setup_db(curr: psycopg2.extensions.cursor):
    l.info('setting up database')
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS sample_arbitrage_replications (
            sample_arbitrage_id   INTEGER NOT NULL REFERENCES sample_arbitrages (id) ON DELETE CASCADE,
            verification_started  BOOLEAN NOT NULL DEFAULT FALSE,
            verification_finished BOOLEAN,
            supported             BOOLEAN,
            replication_diff      NUMERIC(78, 0),
            replication_percent   DOUBLE PRECISION
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_sample_arbitrage_replication_id ON sample_arbitrage_replications (sample_arbitrage_id);
        CREATE INDEX IF NOT EXISTS idx_sample_arbitrage_replication_started ON sample_arbitrage_replications (verification_started);
        '''
    )
    curr.execute(
        '''
        INSERT INTO sample_arbitrage_replications (sample_arbitrage_id)
        SELECT sample_arbitrage_id
        FROM sample_arbitrage_cycles sac
        WHERE
            NOT EXISTS(SELECT 1 FROM sample_arbitrage_replications sar WHERE sar.sample_arbitrage_id = sac.sample_arbitrage_id) AND
            EXISTS(SELECT 1 FROM sample_arbitrages sa WHERE sa.id = sac.sample_arbitrage_id)
        '''
    )
    l.info(f'inserted {curr.rowcount:,} replication rows')

    curr.connection.commit()
    l.info('done setting up database')
    pass


def get_candidate(curr: psycopg2.extensions.cursor):
    """
    Get the next candidate ID from the queue
    """
    curr.execute(
        '''
        SELECT sample_arbitrage_id
        FROM sample_arbitrage_replications sar
        WHERE verification_started = false
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        '''
    )
    if curr.rowcount == 0:
        curr.connection.revert()
        l.info('No more work')
        return None

    assert curr.rowcount == 1

    (id_,) = curr.fetchone()
    curr.execute('UPDATE sample_arbitrage_replications SET verification_started = true WHERE id = %s', (id_,))
    assert curr.rowcount == 1
    curr.connection.commit()

    l.debug(f'Processing id_={id_}')

    return id_
