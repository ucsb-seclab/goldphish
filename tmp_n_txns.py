import os
import psycopg2
import web3
from backtest.utils import connect_db

from utils import connect_web3

db = connect_db()
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
curr = db.cursor()
print('Connected to mainnet')
curr_mainnet = db_mainnet.cursor()


curr.execute('SELECT min(start_block), max(end_block) FROM block_samples')
start_block, end_block = curr.fetchone()

curr_mainnet.execute(
    '''
    SELECT count(*), SUM(transaction_count)
    FROM blocks WHERE %s <= block_number AND block_number <= %s
    ''',
    (start_block, end_block)
)
(n_in_db, txns,) = curr_mainnet.fetchone()
print(f'Have {txns:,} in {n_in_db:,}')

n_per_b = txns / n_in_db
est = n_per_b * (end_block - start_block)
print(f'est {est:,}')
