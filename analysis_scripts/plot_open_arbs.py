"""
Plot open arbitrage opportunities
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm
import psycopg2
import psycopg2.extensions


db = psycopg2.connect(
    host='10.10.111.111',
    port=5432,
    user='measure',
    password='password',
    database='eth_measure_db',
)
curr = db.cursor()
print('connected')

xs_blocks = []
exchange_hashes = []
revenues = []
net_profits = []

block_ts_nums = []
block_ts_timestamps = []

with open('tmp/block_timestamps_new.txt') as fin:
    for line in fin:
        block_num, timestamp = line.strip().split(',')
        block_num = int(block_num)
        timestamp = float(timestamp)

        block_ts_nums.append(block_num)
        block_ts_timestamps.append(timestamp)

curr.execute(
    '''
    SELECT MIN(ca.block_number), MAX(ca.block_number)
    FROM verified_arbitrages va
    JOIN candidate_arbitrages ca ON ca.id = va.candidate_id
    '''
)

block_lower, block_upper = curr.fetchone()
block_max_profits = np.zeros(block_upper - block_lower + 1, dtype=float)

curr.execute(
    """
    SELECT ca.block_number, MAX(va.net_profit) 
    FROM verified_arbitrages va
    JOIN candidate_arbitrages ca ON ca.id = va.candidate_id
    GROUP BY ca.block_number
    """
)
for block_number, net_profit in curr:
    block_number = int(block_number)
    net_profit = int(net_profit)
    net_profit_eth = net_profit / (10 ** 18)
    block_max_profits[block_number - block_lower] = net_profit_eth


xs_blocks = np.arange(block_lower, block_upper + 1, dtype=int)

xs_block_num_ts_interp = np.interp(xs_blocks, block_ts_nums, block_ts_timestamps)
xs_dts = [datetime.datetime.fromtimestamp(x) for x in xs_block_num_ts_interp]

net_profits = [x / (10 ** 18) for x in net_profits]

only_profit_xs = []
only_profit_net = []
for dt, profit in zip(xs_dts, block_max_profits):
    if profit > 0:
        only_profit_xs.append(dt)
        only_profit_net.append(profit)

only_profit_colors = []
only_loss_xs = []
only_loss_net = []

cmap = matplotlib.cm.get_cmap('brg')

plt.gca().get_xaxis().set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().tick_params('x', labelrotation=45)
plt.scatter(only_profit_xs, only_profit_net, s=1, color='black')
# plt.scatter(xs_dts, block_max_profits)
# plt.scatter(only_loss_xs, only_loss_net, s=1, color='black', alpha=0.05)
# plt.scatter(only_profit_xs, only_profit_net, c=only_profit_colors, s=4)
plt.ylim(bottom = -0.08)
plt.ylabel('Ether profit (net)')
plt.xlabel('Date appeared')
plt.show()

