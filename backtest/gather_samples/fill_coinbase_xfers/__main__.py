import argparse
import logging
import os
import socket
import time
import psycopg2
import psycopg2.extensions

import web3
from backtest.utils import connect_db

from utils import setup_logging


l = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-name', type=str, default=None, help='worker name for log, must be POSIX path-safe')

    args = parser.parse_args()
    if args.worker_name is None:
        args.worker_name = socket.gethostname()
    job_name = 'fill_coinbase_xfers'


    setup_logging(job_name, worker_name = args.worker_name, root_dir='/data/robert/ethereum-arb/storage')

    db = connect_db()
    curr = db.cursor()

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

    pg_user = os.getenv('PG_USER')
    pg_pass = os.getenv('PG_PASS')
    db_mainnet = psycopg2.connect(
        host = '127.0.0.1',
        port = 5432,
        user = pg_user,
        password = pg_pass,
        database = 'mainnet',
    )
    db.autocommit = False
    l.debug(f'connected to postgresql (mainnet)')

    fill_txn_coinbase_transfers(w3, db, db_mainnet)


def fill_txn_coinbase_transfers(w3: web3.Web3, db_mine: psycopg2.extensions.connection, db_mainnet: psycopg2.extensions.connection):
    curr_mine = db_mine.cursor()
    curr_mainnet = db_mainnet.cursor()

    curr_mine.execute(
        '''
        SELECT distinct block_number
        FROM sample_arbitrages sa
        WHERE coinbase_xfer IS NULL
        '''
    )
    n_blocks_to_process = curr_mine.rowcount
    l.info(f'Have {n_blocks_to_process:,} blocks to process')

    blocks_to_process = sorted(x for (x,) in curr_mine)

    for block_number in blocks_to_process:
        curr_mainnet.execute('SELECT miner FROM blocks WHERE block_number = %s', (block_number,))
        if curr_mainnet.rowcount == 1:
            assert curr_mainnet.rowcount == 1, f'expected to find one miner for block_number = {block_number}'
            (miner,) = curr_mainnet.fetchone()
        else:
            block = w3.eth.get_block(block_number)
            miner = w3.toChecksumAddress(block['miner'])
        assert w3.isChecksumAddress(miner)
        bminer = bytes.fromhex(miner[2:])
        l.debug(f'block {block_number} was mined by {miner}')

        start = time.time()
        curr_mine.execute(
            '''
            SELECT sa.id, txn_hash
            FROM sample_arbitrages sa
            WHERE coinbase_xfer IS NULL AND block_number = %s
            ''',
            (block_number,)
        )
        elapsed = time.time() - start
        l.debug(f'have {curr_mine.rowcount} transactions to check in this block (took {elapsed:.2f}s)')

        arbs = [(aid, '0x' + txn_hash.tobytes().hex()) for aid, txn_hash in curr_mine]
        for arbitrage_id, txn_hash in arbs:

            # use block_number so it can make use of the index ?          
            curr_mainnet.execute('SELECT EXISTS(SELECT 1 FROM traces WHERE block_number = %s AND transaction_hash = %s)', (block_number, txn_hash,))
            (has_txn,) = curr_mainnet.fetchone()
            if has_txn == True:
                curr_mainnet.execute(
                    '''
                    SELECT value
                    FROM traces
                    WHERE block_number = %s AND transaction_hash = %s AND receiver = %s
                    ''',
                    (block_number, txn_hash, miner),
                )

                xfer_amt = 0
                for (v,) in curr_mainnet:
                    xfer_amt += v
            else:
                assert has_txn == False
                l.warning(f'Using backup call dumper')
                resp = w3.provider.make_request('debug_traceTransaction', [txn_hash, {'tracer': 'callTracer'}])

                xfer_amt = 0
                queue = [resp['result']]
                while len(queue) > 0:
                    item = queue.pop()
                    if w3.toChecksumAddress(item['to']) == miner:
                        xfer_amt += int(item['value'][2:], base=16)
                    queue += item.get('calls', [])

            curr_mine.execute(
                '''
                UPDATE sample_arbitrages SET coinbase_xfer = %s, miner = %s WHERE id = %s
                ''',
                (xfer_amt, bminer, arbitrage_id)
            )

        curr_mine.connection.commit()


if __name__ == '__main__':
    main()
