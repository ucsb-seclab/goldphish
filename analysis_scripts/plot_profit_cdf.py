from collections import deque
import collections
import datetime
import itertools
import time
import psycopg2
import psycopg2.extras
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import scipy.stats

from common import setup_only_uniswap_tables, setup_weth_arb_tables


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

curr.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')

curr.execute(
    '''
    SELECT profit_no_fee
    FROM candidate_arbitrages
    TABLESAMPLE SYSTEM (0.05)
    ''',
)
print(f'Got {curr.rowcount:,} sample rows')

profits = [int(x) for (x,) in curr]
print(f'Got {len(profits):,} samples')


max_profit = max(profits)
min_profit = min(profits)

print(f'Max profit in sample: {max_profit / (10 ** 18):.5f} ETH')
print(f'Min profit in sample: {min_profit / (10 ** 18):.5f} ETH')

percentile_marks = [95, 80, 75, 65, 50, 40, 30, 25, 10]

percentiles = np.percentile(profits, percentile_marks)

tab = []
for mark, percentile in zip(percentile_marks, percentiles):
    tab.append((f'{mark:02,d}%', f'{percentile / (10 ** 18):.4f}'))

print(tabulate.tabulate(tab, headers=['Percentile', 'ETH']))
