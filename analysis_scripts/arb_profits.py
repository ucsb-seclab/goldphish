"""
Plots arbitrage net before fee vs after fee
"""

import psycopg2
from matplotlib import pyplot as plt


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

profit_before_fees = []
profit_after_fees  = []

curr.execute(
    '''
    SELECT va.measured_profit_no_fee, va.net_profit
    FROM verified_arbitrages va
    ORDER BY RANDOM()
    LIMIT 200000
    '''
)

for profit_before_fee, profit_after_fee in curr:
    profit_before_fee = int(profit_before_fee)
    profit_after_fee  = int(profit_after_fee)

    profit_before_fees.append(profit_before_fee)
    profit_after_fees.append(profit_after_fee)

assert len(profit_after_fees) == len(profit_before_fees)

profit_before_fees_eth = [x / (10 ** 18) for x in profit_before_fees]
profit_after_fees_eth = [x / (10 ** 18) for x in profit_after_fees]

plt.scatter(profit_before_fees_eth, profit_after_fees_eth, s=1)
plt.xlabel('profit before fees')
plt.ylabel('profit after fees')
plt.axhline(0, color='black', lw=1, ls='-')
plt.axvline((130_000) * (10 * (10 ** 9)) / (10 ** 18), color='black', lw=0.5, ls=':')
plt.show()
