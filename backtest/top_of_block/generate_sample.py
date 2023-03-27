"""
Generates the random sampling priority.

Assigns 1-day block ranges a random, unique integer id from 0 to NBLOCKS
"""

import argparse
import random
import typing

import web3

from backtest.utils import connect_db
import logging


BLOCKS_PER_DAY = 6_646

l = logging.getLogger(__name__)

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'generate-sample'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    return parser_name, generate_sample


def generate_sample(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()

    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS block_samples (
            id SERIAL PRIMARY KEY NOT NULL,
            start_block INTEGER NOT NULL,
            end_block INTEGER NOT NULL,
            priority INTEGER NOT NULL
        );
        '''
    )
    curr.execute('SELECT COUNT(*) FROM block_samples')
    (n_samples,) = curr.fetchone()
    if n_samples > 0:
        l.info(f'not filling sample labels')
        return

    MIN_BLOCK =  9_569_113
    MAX_BLOCK = 15_965_926

    l.info('assigning sample labels')
    # assign sample groups
    sample_groups = []
    for start_block in range(MIN_BLOCK, MAX_BLOCK, BLOCKS_PER_DAY):
        sample_groups.append((start_block, start_block + BLOCKS_PER_DAY - 1))
    
    random.shuffle(sample_groups)

    for i, (start_block, end_block) in enumerate(sample_groups):
        curr.execute(
            '''
            INSERT INTO block_samples (start_block, end_block, priority)
            VALUES (%s, %s, %s)
            ''',
            (start_block, end_block, i),
        )
        assert curr.rowcount <= BLOCKS_PER_DAY

    db.commit()
    l.info('Generated sample')
