import argparse
import collections
import datetime
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
import web3._utils.filters
from backtest.utils import ERC20_TRANSFER_TOPIC, connect_db

from utils import BALANCER_VAULT_ADDRESS, connect_web3, get_abi, setup_logging, erc20

ROLLING_WINDOW_SIZE_BLOCKS = 60 * 60 // 13 # about 1 hour

l = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.add_argument('--setup-db', action='store_true', dest='setup_db')
    parser.add_argument('--n-workers', type=int)
    parser.add_argument('--id', type=int)

    args = parser.parse_args()

    setup_logging('fill_odd_token_xfers', stdout_level=logging.DEBUG if args.verbose else logging.INFO)

    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        db.commit()
        l.info('setup db')
        return

    w3 = connect_web3()

    assert args.id < args.n_workers

    curr.execute(
        'SELECT start_block, end_block FROM block_samples WHERE MOD(priority, %s) = %s ORDER BY priority ASC',
        (args.n_workers, args.id)
    )
    assignments = curr.fetchall()

    curr2 = db.cursor()
    last_update = time.time()
    t_start = last_update
    last_complete_reservation = None
    for i_res, (start_block, end_block) in enumerate(assignments):
        curr.execute(
            '''
            SELECT id, txn_hash
            FROM sample_arbitrages
            WHERE %s <= block_number AND block_number <= %s
            ''',
            (start_block, end_block),
        )
        n_to_process = curr.rowcount
        n_broken = 0
        for i, (id_, txn_hash) in enumerate(curr):
            txn_hash: bytes = txn_hash.tobytes()
            receipt = w3.eth.get_transaction_receipt(txn_hash)

            if last_update + 10 < time.time() and i > 0:
                last_update = time.time()
                elapsed = last_update - t_start
                nps = i / elapsed
                remain = n_to_process - i
                eta_s = remain / nps
                eta = datetime.timedelta(seconds=eta_s)

                if last_complete_reservation is not None:
                    res_per_second = i_res / (last_complete_reservation - t_start)
                    remain_res = len(assignments) - i_res
                    eta_res_s = remain_res / res_per_second
                    eta_res = datetime.timedelta(seconds=eta_res_s)
                    print(f'Processed {i:,} of {n_to_process:,} in reservation ({i / n_to_process * 100:.2f}%) - broken percent {n_broken / i * 100:.2f}% - reservation eta {eta} -- (global {i_res / (len(assignments))*100:.2f}%) -- eta {eta_res}')
                else:
                    print(f'Processed {i:,} of {n_to_process:,} in reservation ({i / n_to_process * 100:.2f}%) - broken percent {n_broken / i * 100:.2f}% - reservation eta {eta} -- (global {i_res / (len(assignments))*100:.2f}%)')

            parsed_txns = []
            for log in receipt['logs']:
                if len(log['topics']) > 0 and log['topics'][0] == ERC20_TRANSFER_TOPIC:
                    txn = erc20.events.Transfer().processLog(log)
                    parsed_txns.append(txn)

            assert len(parsed_txns) >= 3
            all_tokens = set(x['address'] for x in parsed_txns)
            weird_tokens_received = set()
            weird_tokens_sent = set()
            for txn in parsed_txns:
                if txn['args']['to'] in all_tokens:
                    l.warning(f'Found weird token in transaction 0x{txn_hash.hex()}: {txn["args"]["to"]}')
                    weird_tokens_received.add((txn['args']['to'], txn['address']))
                if txn['args']['from'] in all_tokens:
                    l.warning(f'Found weird token in transaction 0x{txn_hash.hex()}: {txn["args"]["from"]}')
                    weird_tokens_sent.add((txn['args']['from'], txn['address']))
            
            if len(weird_tokens_received) > 0 or len(weird_tokens_sent) > 0:
                n_broken += 1

            for weird_token, outside_token in sorted(weird_tokens_received):
                curr2.execute(
                    '''
                    INSERT INTO sample_arbitrages_odd_tokens (sample_arbitrage_id, odd_token, outside_token, type)
                    VALUES (%s, %s, %s, 'received')
                    ''',
                    (id_, bytes.fromhex(weird_token[2:]), bytes.fromhex(outside_token[2:]))
                )
            for weird_token, outside_token in sorted(weird_tokens_sent):
                curr2.execute(
                    '''
                    INSERT INTO sample_arbitrages_odd_tokens (sample_arbitrage_id, odd_token, outside_token, type)
                    VALUES (%s, %s, %s, 'sent')
                    ''',
                    (id_, bytes.fromhex(weird_token[2:]), bytes.fromhex(outside_token[2:]))
                )

        last_complete_reservation = time.time()
        db.commit()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS sample_arbitrages_odd_tokens (
            sample_arbitrage_id INTEGER NOT NULL REFERENCES sample_arbitrages (id) ON DELETE CASCADE,
            odd_token BYTEA NOT NULL,
            outside_token BYTEA NOT NULL,
            type TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_saot_sample_arbitrage_id ON sample_arbitrages_odd_tokens (sample_arbitrage_id);
        '''
    )

if __name__ == '__main__':
    main()
