"""
Dumps info about gas pricing
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

curr.execute(
    '''
    SELECT niche, block_number, gas_price_median
    FROM naive_gas_price_estimate
    WHERE niche = 'fb|2|uv2|uv3|'
    ORDER BY block_number asc
    '''
)

niche_pts = collections.defaultdict(lambda: ([], []))

for niche, bn, gpm in curr:
    niche_pts[niche][0].append(bn)
    niche_pts[niche][1].append(gpm / (10 ** 9))

for niche, pts in niche_pts.items():
    plt.plot(
        pts[0],
        pts[1],
        label=niche,
        lw=1,
        color='black'
    )

plt.xlabel('block number')
plt.ylabel('gwei')

plt.ylim(0, 1_000)

plt.show()
