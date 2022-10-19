import datetime
import functools
import subprocess
import time
import typing
import psycopg2
import psycopg2.extensions
import psycopg2.extras
import os
import argparse
from backtest.utils import connect_db
from utils import WETH_ADDRESS, connect_web3

db = connect_db()
curr = db.cursor()

w3 = connect_web3()

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
print('Connected to mainnet')
curr_mainnet = db_mainnet.cursor()


parser = argparse.ArgumentParser()
parser.add_argument('ID', type=int)
parser.add_argument('--debug', action='store_true', dest='debug')

args = parser.parse_args()

print(f'processing priority {args.ID}')

# iterate over all sample transactions and see if they had internal transfers

curr.execute('SELECT start_block, end_block FROM block_samples WHERE priority = %s', (args.ID,))
start_block, end_block = curr.fetchone()


@functools.lru_cache(maxsize=100_000)
def get_block_miner(block_number) -> str:
    curr_mainnet.execute('SELECT miner FROM blocks WHERE block_number = %s', (block_number,))
    if curr_mainnet.rowcount == 1:
        assert curr_mainnet.rowcount == 1, f'expected to find one miner for block_number = {block_number}'
        (miner,) = curr_mainnet.fetchone()
        miner = w3.toChecksumAddress(miner)
    else:
        block = w3.eth.get_block(block_number)
        miner = w3.toChecksumAddress(block['miner'])
    return miner

def get_has_noncoinbase_xfers(txn_hash, block_number: int) -> typing.Tuple[bool, bool]:
    miner = get_block_miner(block_number)
    sz_txn_hash = '0x' + txn_hash.hex()

    if block_number >= 15005545:
        # query directly from our database
        curr.execute(
            '''
            SELECT EXISTS(
                SELECT
                FROM internal_eth_xfers
                WHERE block_number = %s
                    AND txn_hash = %s
                    AND to_address != %s
            )
            ''',
            (block_number, txn_hash, bytes.fromhex(miner[2:]))
        )
        (has_xfer,) = curr.fetchone()

        curr.execute(
            '''
            SELECT EXISTS(
                SELECT
                FROM internal_eth_xfers
                WHERE block_number = %s
                    AND txn_hash = %s
                    AND to_address != %s
                    AND from_address != '\\xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'::bytea
                    AND to_address != '\\xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'::bytea
            )
            ''',
            (block_number, txn_hash, bytes.fromhex(miner[2:]))
        )
        (has_xfer_not_miner_not_weth,) = curr.fetchone()

        return has_xfer, has_xfer_not_miner_not_weth

    curr_mainnet.execute('SELECT EXISTS(SELECT 1 FROM traces WHERE block_number = %s AND transaction_hash = %s)', (block_number, sz_txn_hash,))
    (has_txn,) = curr_mainnet.fetchone()
    if has_txn == True:
        if args.debug:
            print('fetching from mainnet db')
        curr_mainnet.execute(
            '''
            SELECT sender, receiver
            FROM traces
            WHERE block_number = %s AND transaction_hash = %s AND receiver <> %s AND value > 0 AND trace_address <> ''
            ''',
            (block_number, sz_txn_hash, miner),
        )
        has_xfer = curr.rowcount > 0
        # print(f'Has xfer: {sz_txn_hash}')
        has_xfer_not_miner_not_weth = False
        for sender, receiver in curr_mainnet:
            assert receiver != miner
            if sender != WETH_ADDRESS and receiver != WETH_ADDRESS:
                if args.debug:
                    print(f'Saw xfer: {sender} -> {receiver}')
                has_xfer_not_miner_not_weth = True
                break

        return has_xfer, has_xfer_not_miner_not_weth
    else:
        # assert has_txn == False
        if args.debug:
            print('using fallback')
        # fallback -- query our database
        resp = w3.provider.make_request('debug_traceTransaction', [sz_txn_hash, {'tracer': 'callTracer'}])

        has_xfer = False
        has_xfer_not_miner_not_weth = False
        queue = [resp['result']]
        while len(queue) > 0:
            item = queue.pop()
            if item.get('value', '0x0') != '0x0':
                from_ = w3.toChecksumAddress(item['from'])
                to = w3.toChecksumAddress(item['to'])
                if to != miner:
                    has_xfer = True
                    if from_ == WETH_ADDRESS or to == WETH_ADDRESS:
                        has_xfer_not_miner_not_weth = True
                        if args.debug:
                            print(f'found {from_} to {to}')
                        break

            queue += item.get('calls', [])
        return has_xfer, has_xfer_not_miner_not_weth

if args.debug:
    get_has_noncoinbase_xfers(bytes.fromhex('0x55a745cca223e9e42be7f9196fd2d074c58856e872994bfed33c5bafd9d585e3'[2:]), 13702985)
    print('done!!!')
    exit()

curr2 = db.cursor()
curr2.execute(
    '''
    SELECT id, txn_hash, block_number
    FROM sample_arbitrages
    WHERE %s <= block_number and block_number <= %s
    ORDER BY block_number ASC
    ''',
    (start_block, end_block)
)
print(f'Have {curr2.rowcount:,} transactions to investigate')

last_update = time.time()
t_start = time.time()
for i, (id_, txn_hash, block_number) in enumerate(curr2):
    txn_hash = txn_hash.tobytes()

    if last_update + 20 < time.time():
        last_update = time.time()
        elapsed = time.time() - t_start
        nps = i / elapsed
        remain = curr2.rowcount - i
        eta_s = remain / nps
        eta = datetime.timedelta(seconds=eta_s)
        db.commit()
        print(f'{i / curr2.rowcount * 100:.2f}% -- ETA {eta}')

    has_xfer, has_nmw_xfer = get_has_noncoinbase_xfers(txn_hash, block_number)
    curr.execute(
        'INSERT INTO samples_weird_xfer (sample_arbitrage_id, has_xfer, has_xfer_not_weth_not_miner) VALUES (%s, %s, %s)',
        (id_, has_xfer, has_nmw_xfer)
    )

db.commit()
