"""
Dumps info about arbitrage duration
"""

import collections
import psycopg2
import tabulate
import web3
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


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
    SELECT
        block_number,
        PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY duration),
        PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY duration),
        PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY duration)
    FROM (
        SELECT (bs.start_block + bs.end_block) / 2 block_number, (block_number_end - block_number_start + 1) duration
        FROM candidate_arbitrage_campaigns cac
        JOIN block_samples bs ON bs.start_block <= cac.block_number_start AND cac.block_number_start <= bs.end_block
        WHERE cac.niche LIKE 'nfb|%%' AND cac.gas_pricer = 'median'
    ) a
    GROUP BY block_number
    ORDER BY block_number asc
    '''
)

xs = []
ys_low = []
ys_median = []
ys_hi = []

for bn, a, b, c in curr:
    xs.append(bn)
    ys_low.append(a)
    ys_median.append(b)
    ys_hi.append(c)

errors_lower = np.array(ys_median) - np.array(ys_low)
errors_upper = np.array(ys_hi) - np.array(ys_median)
plt.errorbar(xs, ys_median, yerr=[errors_lower, errors_upper], capsize=3, elinewidth=1, ecolor='black')
plt.xlabel('Block Number')
plt.ylabel('Duration (Blocks)')
plt.show()


        # CREATE TABLE IF NOT EXISTS candidate_arbitrage_campaigns (
        #     id                       SERIAL NOT NULL PRIMARY KEY,
        #     niche                    TEXT NOT NULL,
        #     gas_pricer               TEXT NOT NULL,
        #     block_number_start       INTEGER NOT NULL,
        #     block_number_end         INTEGER NOT NULL,
        #     max_profit_after_fee_wei NUMERIC(78, 0) NOT NULL,
        #     min_profit_after_fee_wei NUMERIC(78, 0) NOT NULL
        # );


curr.execute(
    '''
    SELECT PERCENTILE_DISC(0.5) WITHIN GROUP (order by profit_difference)
    FROM (
        SELECT max_profit_after_fee_wei - min_profit_after_fee_wei profit_difference
        FROM candidate_arbitrage_campaigns
        WHERE gas_pricer = 'median'
    ) a
    '''
)
(med,) = curr.fetchone()
print(f'Median difference from minimum profit to maximum profit {med / (10 ** 18):.5f} ETH')
print()

tab = []
for gas_pricer in ['minimum', '25th percentile', 'median', '75th percentile', 'maximum']:
    curr.execute(
        '''
        SELECT
            COUNT(*),
            PERCENTILE_DISC(0.5) WITHIN GROUP (order by (block_number_end - block_number_start + 1)),
            MAX((block_number_end - block_number_start + 1)),
            PERCENTILE_DISC(0.5) WITHIN GROUP (order by (max_profit_after_fee_wei - min_profit_after_fee_wei))
            FROM candidate_arbitrage_campaigns
            WHERE gas_pricer = %s
        ''',
        (gas_pricer,)
    )
    (n, median_duration, max_duration, median_profit_difference) = curr.fetchone()
    tab.append((gas_pricer, n, median_duration, max_duration, f'{median_profit_difference / (10 ** 18):.05f}'))


print(tabulate.tabulate(tab, headers=['Gas Oracle', 'Count of Arbitrage Campaigns', 'Median Duration (blocks)', 'Max Duration (blocks)', 'Median, difference in max and min profit']))
print()

for gas_pricer in ['median']: # ['minimum', '25th percentile', 'median', '75th percentile', 'maximum']:
    curr.execute(
        '''
        SELECT max_profit_after_fee_wei, (block_number_end - block_number_start + 1)
        FROM candidate_arbitrage_campaigns
        WHERE gas_pricer = %s AND niche LIKE 'nfb|%%'
        ''',
        (gas_pricer,)
    )
    xs = []
    ys = []

    for x, y in curr:
        xs.append(x / (10 ** 18))
        ys.append(y)
    
    plt.scatter(xs, ys, s=1, color = (0,0,0,0.5))
    plt.ylabel('Duration, blocks')
    plt.xlabel('Max campaign profit, ETH')

    plt.yscale('log')

    plt.title(f'Profit vs Duration, Gas Oracle "{gas_pricer.title()}"')
    plt.show()

tab = []
for gas_pricer in ['minimum', '25th percentile', 'median', '75th percentile', 'maximum']:
    curr.execute(
        '''
        SELECT COUNT(*), SUM(CASE WHEN duration <= 1 THEN 1 ELSE 0 END) duration_1, SUM(CASE WHEN duration <= 2 THEN 1 ELSE 0 END) duration_2
        FROM (
            SELECT (block_number_end - block_number_start + 1) duration
            FROM candidate_arbitrage_campaigns
            WHERE gas_pricer = %s AND max_profit_after_fee_wei > %s
        ) a
        ''',
        (gas_pricer, (10 ** 16))
    )
    tot, duration_1, duration_2 = curr.fetchone()
    tab.append((gas_pricer.title(), tot, f'{duration_1} ({duration_1 / tot * 100:.2f}%)', f'{duration_2} ({duration_2 / tot * 100:.2f}%)'))

print(tabulate.tabulate(tab, headers=['Gas Oracle', 'Total over 0.01 ETH', 'Count 1 block duration or less (%)', 'Count 2 blocks duration or less (%)']))


# curr.execute(
#     '''
#     SELECT (block_number_end - block_number_start + 1)
#     FROM candidate_arbitrage_campaigns
#     WHERE gas_pricer = %s AND max_profit_after_fee_wei > %s
#     ''',
#     ('median', (10 ** 16))
# )
# xs = sorted([x for (x,) in curr])

# plt.hist(xs, bins=10)

# # plt.yscale('log')
# plt.xlabel('Duration, blocks')
# plt.ylabel('Count')

# plt.show()
