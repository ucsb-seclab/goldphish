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
median_profits = []
median_fees = []

with open('tmp_profits.csv') as fin:
    for line in fin:
        ts, mp = line.strip().split(',')
        ts = float(ts)
        mp = float(mp)
        interp_timestamps.append(ts)
        median_profits.append(mp)

with open('tmp_fees.csv') as fin:
    for line in fin:
        _, mf = line.strip().split(',')
        # ts = float(ts)
        mf = float(mf)
        # interp_timestamps.append(ts)
        median_fees.append(mf)


bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams['hatch.linewidth'] = 0.2

# plt.axhline(y=0, c='k')
plt.plot(bin_rights_datetimes, median_profits, label='Profits')
plt.plot(bin_rights_datetimes, median_fees, label='Fees')
plt.title('Profits and Fees Over Time')
plt.ylabel('Median (ETH)')
plt.yscale('log')
plt.axvline(x=datetime.datetime(year=2021, month=4, day=1))
plt.xlabel('Date')
plt.legend()
# plt.ylim(top=0.1, bottom=0)
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig('profits_and_fees.png', format='png', dpi=300)
plt.show()

