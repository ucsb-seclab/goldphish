import datetime
import math
import sqlite3
import psycopg2
import tabulate
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
print(f'Computed profits')


curr.execute(
    '''
    SELECT end_block, PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee_sum)
    FROM (
        SELECT block_number, SUM(fee) fee_sum
        FROM tmp_profits
        GROUP BY block_number
    ) x
    JOIN block_samples bs ON bs.start_block <= x.block_number AND x.block_number <= bs.end_block
    GROUP BY bs.end_block
    '''
)

with open('tmp_block_mev.csv', mode='w') as fout:
    for b, p in curr:
        fout.write(f'{b},{p}\n')
exit()


curr.execute(
    '''
    SELECT
        bs.end_block,
        PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee)
    FROM tmp_profits tp
    JOIN block_samples bs ON bs.start_block <= tp.block_number AND tp.block_number <= bs.end_block
    GROUP BY end_block
    ORDER BY end_block ASC
    '''
)

block_numbers = []
median_profits = []
for bn, p in curr:
    block_numbers.append(bn)
    median_profits.append(p / (10 ** 18))

local_conn = sqlite3.connect('./tmp.db')
local_curr = local_conn.cursor()

local_curr.execute('SELECT block_number, timestamp FROM block_timestamps ORDER BY block_number ASC')

ts_block_numbers = []
ts_timestamps = []
for bn, t in local_curr:
    ts_block_numbers.append(bn)
    ts_timestamps.append(t)

interp_timestamps = np.interp(block_numbers, ts_block_numbers, ts_timestamps)

bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]

with open('tmp_fees.csv', mode='w') as fout:
    for ts, mp in zip(interp_timestamps, median_profits):
        fout.write(f'{ts},{mp}\n')

plt.plot(bin_rights_datetimes, median_profits)
plt.ylabel('Median Fee (ETH)')
plt.xlabel('Date')
plt.show()

