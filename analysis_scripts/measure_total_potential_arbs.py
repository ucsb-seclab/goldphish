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

curr.execute('SELECT priority, start_block, end_block FROM block_samples order by priority asc')
samples = curr.fetchall()

tot = 0
t_start = time.time()
for i, (priority, start_block, end_block) in enumerate(samples):
    assert i == priority
    if i > 0:
        elapsed = time.time() - t_start
        nps = i / elapsed
        remain = len(samples) - i
        eta_s = remain / nps
        eta = datetime.timedelta(seconds=eta_s)
        print(f'ETA {eta}')

    curr.execute(
        '''
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT exchanges, directions, block_number
            FROM candidate_arbitrages
            WHERE %s <= block_number AND block_number <= %s
        ) x
        ''',
        (start_block, end_block)
    )
    (n_in_window,) = curr.fetchone()
    tot += n_in_window
    print(f'Samples in first {i+1} samples: {tot:,}')
