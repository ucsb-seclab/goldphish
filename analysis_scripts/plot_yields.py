import collections
import datetime
import math
import sqlite3
import psycopg2
import tabulate
import os
import numpy as np
import matplotlib.pyplot as plt

from common import setup_weth_arb_tables

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

bns = []
ys = []

with open('tmp_yields.csv') as fin:
    for line in fin:
        bn, y = line.strip().split(',')
        bns.append(int(bn))
        ys.append(float(y))

local_conn = sqlite3.connect('./tmp.db')
local_curr = local_conn.cursor()

local_curr.execute('SELECT block_number, timestamp FROM block_timestamps ORDER BY block_number ASC')

ts_block_numbers = []
ts_timestamps = []
for bn, t in local_curr:
    ts_block_numbers.append(bn)
    ts_timestamps.append(t)

interp_timestamps = np.interp(bns, ts_block_numbers, ts_timestamps)

bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

plt.plot(bin_rights_datetimes, 100 - np.array(ys))
plt.title('Block Producer\'s Share of Profits')
plt.ylim(bottom=0, top=100)
plt.xticks(rotation=75)
plt.ylabel('Share of Profit (%)')
plt.tight_layout()
plt.savefig('miner_share_profit.png', format='png', dpi=300)
plt.show()

exit()

setup_weth_arb_tables(curr)

curr.execute(
    '''
    CREATE TEMP TABLE tmp_profits AS
    SELECT
        sa.*,
        sac.profit_amount revenue,
        sa.gas_used * sa.gas_price + sa.coinbase_xfer fee,
        (sac.profit_amount - sa.coinbase_xfer) - (sa.gas_used * sa.gas_price) net_profit
    FROM sample_arbitrages_no_fp sa
    JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id;

    CREATE TEMP VIEW tmp_yields AS
    SELECT *, net_profit::float / revenue * 100 as yield
    FROM tmp_profits
    WHERE revenue > 0;
    '''
)

curr.execute(
    '''
        SELECT
            bs.end_block,
            PERCENTILE_DISC(0.5) WITHIN GROUP (order by yield)
        FROM tmp_yields
        JOIN block_samples bs ON bs.start_block <= block_number AND block_number <= bs.end_block
        GROUP BY bs.end_block
        ORDER BY bs.end_block asc
    '''
)

with open('tmp_yields.csv', mode='w') as fout:
    for b, y in curr:
        fout.write(f'{b},{y}\n')

print('done')
