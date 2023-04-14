from collections import deque
import collections
import datetime
import time
import psycopg2
import psycopg2.extras
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import scipy.stats
import sqlite3


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
    SELECT COUNT(*)
    FROM (
        SELECT DISTINCT exchanges, directions, ca.block_number
        FROM large_candidate_arbitrages lca
        JOIN candidate_arbitrages ca ON ca.id = lca.candidate_arbitrage_id
    ) x
    '''
)
(n_large,) = curr.fetchone()

curr.execute('SELECT priority, start_block, end_block FROM block_samples order by priority asc limit 30')

samples = []

for priority, start_block, end_block in curr:
    samples.append((start_block, end_block))

samples = sorted(samples)
for start_block, end_block in samples:
    print(f'${start_block:,}$ & ${end_block:,}$ \\\\')
