"""
fill ethereum price over time
"""

import argparse
import collections
import datetime
import decimal
import itertools
import logging
import os
import socket
import time
import typing
from eth_utils import event_abi_to_log_topic
import psycopg2
import psycopg2.extensions
import psycopg2.extras
import numpy as np


import web3
import web3.contract
import web3.exceptions
import web3._utils.filters
from backtest.utils import connect_db

from utils import BALANCER_VAULT_ADDRESS, connect_web3, get_abi, setup_logging

ROLLING_WINDOW_SIZE_BLOCKS = 60 * 60 // 13 # about 1 hour

l = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.add_argument('--setup-db', action='store_true', dest='setup_db')
    parser.add_argument('--n-workers', type=int)
    parser.add_argument('--id', type=int)
    parser.add_argument('--finalize', action='store_true', dest='finalize')

    args = parser.parse_args()

    setup_logging('fill_naive_gas_price', stdout_level=logging.DEBUG if args.verbose else logging.INFO)

    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        db.commit()
        l.info('setup db')
        return

    if args.finalize:
        finalize(curr)
        db.commit()
        l.info(f'finalized')
        return

    w3 = connect_web3()

    assert args.id < args.n_workers

    curr.execute(
        'SELECT start_block, end_block FROM block_samples WHERE MOD(priority, %s) = %s ORDER BY priority ASC',
        (args.n_workers, args.id)
    )
    assignments = curr.fetchall()

    simple_abi = [{
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {
                "internalType": "uint80",
                "name": "roundId",
                "type": "uint80"
            },
            {
                "internalType": "int256",
                "name": "answer",
                "type": "int256"
            },
            {
                "internalType": "uint256",
                "name": "startedAt",
                "type": "uint256"
            },
            {
                "internalType": "uint256",
                "name": "updatedAt",
                "type": "uint256"
            },
            {
                "internalType": "uint80",
                "name": "answeredInRound",
                "type": "uint80"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }]
    chainlink_contract = w3.eth.contract(address='0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419', abi=simple_abi)

    simple_abi = [{
        "constant": True,
        "inputs": [],
        "name": "read",
        "outputs": [
            {
                "name": "",
                "type": "bytes32"
            }
        ],
        "payable": False,
        "type": "function"
    }]
    maker_contract = w3.eth.contract(address='0x729D19f657BD0614b4985Cf1D82531c67569197B', abi=simple_abi)


    curr2 = db.cursor()
    last_update = time.time()
    t_start = last_update
    last_complete_reservation = None
    for i_res, (start_block, end_block) in enumerate(assignments):
        if i_res > 0:
            elapsed = time.time() - t_start
            nps = i_res / elapsed
            remain = len(assignments) - i_res
            eta_s = remain / nps
            eta = datetime.timedelta(seconds=eta_s)
            print(F'Finished {i_res / len(assignments) * 100:.2f}% ETA {eta}')

        to_insert = []
        for block_number in range(start_block, end_block + 1):
            if block_number % 50 != 0:
                continue
            try:
                (_, answer, _, _, _) = chainlink_contract.functions.latestRoundData().call(block_identifier=block_number)
                price_usd = decimal.Decimal(answer) / (10 ** 8)
            except (web3.exceptions.BadFunctionCallOutput, web3.exceptions.ContractLogicError):
                l.debug(f'Failed at block {block_number:,}, using backup method...')
                try:
                    answer = maker_contract.functions.read().call(block_identifier=block_number)
                    answer = int.from_bytes(answer, byteorder='big', signed=False)
                    price_usd = decimal.Decimal(answer) / (10 ** 18)
                except web3.exceptions.BadFunctionCallOutput:
                    l.critical(f'Failed at block {block_number:,}')
                    raise

            to_insert.append((block_number, price_usd))
        
        psycopg2.extras.execute_values(
            curr,
            'INSERT INTO eth_prices (block_number, eth_price_usd) VALUES %s',
            to_insert
        )
    
    db.commit()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS eth_prices (
            block_number  INTEGER NOT NULL,
            eth_price_usd NUMERIC NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_eth_prices ON eth_prices(block_number);
        '''
    )


def finalize(curr: psycopg2.extensions.cursor):
    curr.execute('SELECT MIN(start_block), MAX(end_block) FROM block_samples')
    start_block, end_block = curr.fetchone()
    # curr.execute(
    #     '''
    #     CREATE TABLE eth_price_blocks AS
    #     SELECT block_number, (SELECT eth_price_usd FROM eth_prices ep WHERE ep.block_number < s.block_number ORDER BY block_number DESC LIMIT 1) eth_price_usd
    #     FROM generate_series(%s, %s) AS s(block_number)
    #     ''',
    #     (start_block, end_block),
    # )
    # assert curr.rowcount > 0

    curr.execute('SELECT eth_price_usd FROM eth_prices ORDER BY block_number ASC LIMIT 1')
    (first_eth_price,) = curr.fetchone()

    # sanity
    curr.execute('SELECT COUNT(*) FROM eth_price_blocks WHERE block_number > %s AND eth_price_usd IS NULL', (start_block + 100,))
    (n_broken,) = curr.fetchone()
    assert n_broken == 0

    curr.execute(
        '''
        UPDATE eth_price_blocks
        SET eth_price_usd = %s
        WHERE eth_price_usd IS NULL
        ''',
        (first_eth_price,)
    )
    l.debug(f'fixed bottom {curr.rowcount} entries')

    # curr.execute('CREATE INDEX idx_eth_pr_blocks_block_number ON eth_price_blocks (block_number)')


if __name__ == '__main__':
    main()
