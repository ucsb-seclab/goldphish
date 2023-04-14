"""
Dumps info about mev
"""

import collections
import psycopg2
import tabulate
import web3
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from common import gen_false_positives, label_zerox_exchanges


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

if False:

    curr.execute(
        '''
        SELECT mev
        FROM candidate_arbitrages_mev
        '''
    )
    mevs = np.array([int(x) / (10 ** 18) for (x,) in curr])

    # mevs = list(filter(lambda x: x < 0.2, mevs))
    plt.hist(mevs, bins=30)
    plt.yscale('log')
    plt.show()

    exit()

if False:
    curr.execute(
        '''
        SELECT
            block_number,
            PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY mev),
            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY mev),
            PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY mev)
        FROM (
            SELECT ((bs.end_block + bs.start_block) / 2)::integer block_number, mev
            FROM candidate_arbitrages_mev m
            JOIN block_samples bs ON bs.start_block <= block_number AND block_number <= bs.end_block
            ORDER BY block_number ASC
        ) a
        GROUP BY block_number
        ORDER BY block_number asc
        '''
    )

    xs = []
    ys_lower = []
    ys_middle = []
    ys_upper = []

    for block_number, a, b, c in curr:
        xs.append(block_number)
        ys_lower.append(a / (10 ** 18))
        ys_middle.append(b / (10 ** 18))
        ys_upper.append(c / (10 ** 18))

    print(f'Have {len(xs)} pts')

    errors_lower = np.array(ys_middle) - np.array(ys_lower)
    errors_upper = np.array(ys_upper) - np.array(ys_middle)

    # plt.fill_between(xs, ys_lower, ys_upper, color=(0,0,0,0.3))
    plt.errorbar(xs, ys_middle, yerr=[errors_lower, errors_upper], capsize=3, elinewidth=1, ecolor='black')
    plt.xlabel('Block Number')
    plt.ylabel('MEV (ETH)')
    plt.yscale('log')
    plt.show()


    exit(1)

if False:
    curr.execute(
        '''
        SELECT block_number, mev
        FROM candidate_arbitrages_mev
        ORDER BY block_number ASC
        '''
    )

    xs = []
    ys = []

    for x, y in curr:
        xs.append(x)
        ys.append(y / (10 ** 18))

    plt.scatter(xs, ys, s=1, color = (0,0,0,0.5))
    plt.xlabel('Block Number')
    plt.ylabel('MEV (ETH)')
    plt.yscale('log')
    plt.show()


curr.execute(
    '''
    SELECT mev
    FROM candidate_arbitrages_mev
    '''
)
mevs = np.array([int(x) / (10 ** 18) for (x,) in curr])
print(f'Max MEV {max(mevs):,.3f} ETH')
print(f'Median MEV {np.percentile(mevs, 50):,.5f} ETH')
print(f'Average MEV {np.average(mevs):,.5f} ETH')


