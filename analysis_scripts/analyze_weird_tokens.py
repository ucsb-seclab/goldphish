from collections import deque
import itertools
import time
import psycopg2
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import scipy.stats

from common import gen_false_positives, setup_only_uniswap_tables, setup_weth_arb_tables


db = psycopg2.connect(
    host='10.10.111.111',
    port=5432,
    user = 'measure',
    password = 'password',
    database = 'eth_measure_db',
)
print('connected to postgresql')
db.autocommit = False

curr = db.cursor()

# curr.execute('SET TRANSACTION READ ONLY')
curr.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')

curr.execute(
    '''
    SELECT COUNT(DISTINCT sample_arbitrage_id)
    FROM sample_arbitrages_odd_tokens
    '''
)
(n_samples_odd,) = curr.fetchone()

curr.execute(
    '''
    SELECT COUNT(*) FROM sample_arbitrages_no_fp
    '''
)
(tot_samples,) = curr.fetchone()

print(f'Have {n_samples_odd:,} odd sample arbitrages: {(n_samples_odd / tot_samples * 100):.2f}%')

curr.execute(
    '''
    SELECT COUNT(distinct sample_arbitrage_id), type
    FROM sample_arbitrages_odd_tokens
    GROUP BY type
    '''
)

tab = curr.fetchall()
print(tabulate.tabulate(tab, headers=['count', 'weird type']))

# sample a few

curr.execute(
    '''
    SELECT sa.txn_hash
    FROM (
        SELECT distinct sample_arbitrage_id
        FROM sample_arbitrages_odd_tokens
    ) a
    JOIN sample_arbitrages sa ON sa.id = a.sample_arbitrage_id
    ORDER BY RANDOM()
    LIMIT 10
    '''
)

for (txn_hash,) in curr:
    txn_hash = txn_hash.tobytes()
    print(f'https://etherscan.io/tx/0x{txn_hash.hex()}')


