from collections import deque
import collections
import itertools
import pickle
import sys
import time
import psycopg2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors
import tabulate
import web3
import scipy.stats
import networkx as nx

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

# Gather the top arbitrages from rich addresses

rich_addresses = [
    bytes.fromhex('0x000000000035B5e5ad9019092C665357240f594e'[2:]),
    bytes.fromhex('0x00000000003b3cc22aF3aE1EAc0440BcEe416B40'[2:]),
    bytes.fromhex('0xE8c060F8052E07423f71D445277c61AC5138A2e5'[2:]),
]

setup_weth_arb_tables(curr)

curr.execute(
    '''
    CREATE TEMP TABLE rich_address_arbs as
    SELECT twa.*
    FROM tmp_weth_arbs twa
    JOIN sample_arbitrages sa ON sa.id = twa.id
    WHERE sa.shooter = any (%s)
    ''',
    (rich_addresses,)
)
print(f'Have {curr.rowcount} arbitrages from rich addresses')

# get profits from the rich ones to plot cdf

curr.execute(
    '''
    SELECT net_profit
    FROM rich_address_arbs
    '''
)

net_profits = [int(x) / (10 ** 18) for (x,)in curr]

print(f'Sum all net profits from rich: {sum(net_profits):,.0f} ETH')

top_95 = np.percentile(net_profits, 95)
print(f'95th percentile: {top_95:.2f} Eth')

profits_above_95 = sum(filter(lambda x: x >= top_95, net_profits))
print(f'{profits_above_95:,.0f} ETH made in top 5% of txns ({profits_above_95 / sum(net_profits) * 100:.4f}%)')


curr.execute(
    '''
    SELECT raa.net_profit, sa.txn_hash
    FROM rich_address_arbs raa
    JOIN sample_arbitrages sa ON sa.id = raa.id
    WHERE raa.net_profit > %s
    ORDER BY RANDOM()
    LIMIT 20
    ''',
    (int(top_95) * (10 ** 18),)
)

tab = []
for p, txn_hash in curr:
    p = int(p) / (10 ** 18)
    txn_hash = '0x' + txn_hash.tobytes().hex()
    tab.append((txn_hash, p))

tab = sorted(tab, key=lambda x: x[1], reverse=True)

print(tabulate.tabulate(tab, headers=['Transaction', 'Profit']))

# plt.hist(net_profits)
# plt.yscale('log')
# plt.show()
