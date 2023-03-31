import datetime
import json
import logging
import os
import socket
import subprocess
import time
import psycopg2
import psycopg2.extensions
import psycopg2.extras
import urllib3

from backtest.utils import connect_db

from utils import setup_logging


l = logging.getLogger(__name__)

def main():
    # fname = '/mnt/goldphish/flashbots_blocks.json'
    # assert os.path.isfile(fname)


    setup_logging('load_flashbots')
    l.info('loading flashbots')

    db = connect_db()
    curr = db.cursor()

    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS flashbots_transactions (
            id                 SERIAL PRIMARY KEY NOT NULL,
            block_number       INTEGER NOT NULL,
            transaction_hash   bytea NOT NULL,
            tx_index           INTEGER NOT NULL,
            bundle_type        TEXT NOT NULL,
            bundle_index       INTEGER NOT NULL,
            gas_used           NUMERIC(78, 0) NOT NULL,
            gas_price          NUMERIC(78, 0) NOT NULL,
            coinbase_transfer  NUMERIC(78, 0) NOT NULL,
            total_miner_reward NUMERIC(78, 0) NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_flashbots_transactions_block_number ON flashbots_transactions (block_number);
        CREATE INDEX IF NOT EXISTS idx_flashbots_transactions_transaction_hash ON flashbots_transactions USING HASH (transaction_hash);
        '''
    )

    curr.execute('SELECT MIN(start_block), MAX(end_block) FROM block_samples')
    (start_block, end_block) = curr.fetchone()

    pool = urllib3.PoolManager()

    curr.execute(
        '''
        PREPARE stmt AS INSERT INTO flashbots_transactions
        (block_number, transaction_hash, tx_index, bundle_type, bundle_index, gas_used, gas_price, coinbase_transfer, total_miner_reward)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        '''
    )

    t_start = time.time()
    t_last_update = time.time()
    for batch_start in range(start_block, end_block + 1, 10_000):
        if t_last_update + 30 < time.time():
            t_last_update = time.time()
            elapsed = t_last_update - t_start
            blocks_processed = batch_start - start_block
            nps = blocks_processed / elapsed
            remain = end_block - batch_start
            eta_s = remain / nps
            eta = datetime.timedelta(seconds=eta_s)
            print(f'Processing {batch_start:,} ({(batch_start - start_block) / (end_block - start_block)*100:.2f}%) ETA {eta}')

        sub_batch_size = 10_000
        sub_batch_start = batch_start
        rows = []
        while sub_batch_start < batch_start + 10_000:
            resp: urllib3.response.HTTPResponse = pool.urlopen(
                'GET',
                f'https://blocks.flashbots.net/v1/blocks?before={sub_batch_start + sub_batch_size - 1}&limit={sub_batch_size}'
            )
            if 500 <= resp.status <= 599:
                # probably need to reduce the batch size, do that and try again
                sub_batch_size = sub_batch_size // 2
                l.debug(f'reducing batch size to {sub_batch_size}')
                continue

            sub_batch_start = sub_batch_start + sub_batch_size

            blob = json.loads(resp.data.decode('utf8'))

            for block in blob['blocks']:
                for transaction in block['transactions']:
                    btxn_hash = bytes.fromhex(transaction['transaction_hash'][2:])
                    if transaction.get('is_megabundle') == True:
                        bundle_type = 'megabundle'
                    else:
                        bundle_type = transaction['bundle_type']
                    rows.append(
                        (
                            transaction['block_number'],
                            btxn_hash,
                            transaction['tx_index'],
                            bundle_type,
                            transaction['bundle_index'],
                            transaction['gas_used'],
                            int(transaction['gas_price']),
                            transaction['coinbase_transfer'],
                            int(transaction['total_miner_reward'])
                        ),
                    )
        
        psycopg2.extras.execute_batch(curr, "EXECUTE stmt (%s, %s, %s, %s, %s, %s, %s, %s, %s)", rows)
    curr.connection.commit()

    # l.info('starting parse')
    # with open(fname) as fin:
    #     parsed = json.load(fin)

    # rows = []
    # for block in parsed:
    #     for transaction in block['transactions']:
    #         btxn_hash = bytes.fromhex(transaction['transaction_hash'][2:])
    #         rows.append(
    #             (
    #                 transaction['block_number'],
    #                 btxn_hash,
    #                 transaction['tx_index'],
    #                 transaction['bundle_type'],
    #                 transaction['bundle_index'],
    #                 transaction['gas_used'],
    #                 int(transaction['gas_price']),
    #                 transaction['coinbase_transfer'],
    #                 int(transaction['total_miner_reward'])
    #             ),
    #         )

    # l.info('executing statements')
    # curr.execute(
    #     '''
    #     PREPARE stmt AS INSERT INTO flashbots_transactions
    #     (block_number, transaction_hash, tx_index, bundle_type, bundle_index, gas_used, gas_price, coinbase_transfer, total_miner_reward)
    #     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    #     '''
    # )
    # psycopg2.extras.execute_batch(curr, "EXECUTE stmt (%s, %s, %s, %s, %s, %s, %s, %s, %s)", rows)
    # curr.execute("DEALLOCATE stmt")
    # curr.connection.commit()
    # l.info('done')


if __name__ == '__main__':
    main()
