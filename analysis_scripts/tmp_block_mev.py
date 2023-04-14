from cProfile import label
import datetime
import math
import sqlite3
import psycopg2
import tabulate
import numpy as np
import matplotlib.pyplot as plt

local_conn = sqlite3.connect('./tmp.db')
local_curr = local_conn.cursor()

local_curr.execute('SELECT block_number, timestamp FROM block_timestamps ORDER BY block_number ASC')

ts_block_numbers = []
ts_timestamps = []
for bn, t in local_curr:
    ts_block_numbers.append(bn)
    ts_timestamps.append(t)

interp_timestamps = []
median_mev = []

with open('tmp_block_mev.csv') as fin:
    for line in fin:
        ts, mmv = line.strip().split(',')
        ts = float(ts)
        mmv = int(mmv) / (10 ** 18)
        interp_timestamps.append(ts)
        median_mev.append(mmv)


bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams['hatch.linewidth'] = 0.2

# plt.axhline(y=0, c='k')
plt.plot(bin_rights_datetimes, median_mev, label='Median MEV')
plt.title('Total Block Fees, Median, Over Time (ETH)')
plt.ylabel('Median Fees to Block Producer')
# plt.yscale('log')
plt.xlabel('Date')
plt.legend()
# plt.ylim(top=0.1, bottom=0)
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig('block_mev.png', format='png', dpi=300)
plt.show()

