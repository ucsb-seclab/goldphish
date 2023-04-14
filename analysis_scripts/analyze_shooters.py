from collections import deque
import itertools
import time
import psycopg2
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import scipy.stats

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

# curr.execute('SET TRANSACTION READ ONLY')
curr.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')

setup_weth_arb_tables(curr, only_new=True)


curr.execute(
    '''
    SELECT
        SUM(net_profit)
--        SUM(net_profit / power(10, 18) * eth_price_usd)
    FROM tmp_weth_arbs twa
--    JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
    where twa.net_profit is not null
    '''
)

(tot_profit_eth,) = curr.fetchone()
print(f'Had {tot_profit_eth / (10 ** 18):,.0f} ETH profit')
# print(f'Had {tot_profit_usd:,.2f} USD profit')

curr.execute(
    '''
    SELECT
        SUM(net_profit),
        SUM(net_profit / power(10, 18) * eth_price_usd)
    FROM tmp_weth_arbs twa
    JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
    '''
)

curr.execute(
    '''
    SELECT *
    FROM (
        SELECT
            SUM(net_profit) sum_profit_eth,
--            SUM(net_profit / power(10, 18) * eth_price_usd) sum_profit_usd,
            sa.shooter
        FROM tmp_weth_arbs twa
        JOIN sample_arbitrages sa ON sa.id = twa.id
--        JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
        where net_profit is not null
        GROUP BY sa.shooter
    ) x
    ORDER BY sum_profit_eth DESC
    LIMIT 10
    '''
)
tab = []
# for npe, npu, addr in curr:
for npe, addr in curr:
    addr = web3.Web3.toChecksumAddress(addr.tobytes())
    print(addr[:5] + f'\ldots' + addr[-4:] + f' & ${npe / (10 ** 18):,.0f}$ & {npe / tot_profit_eth * 100:.1f} & {npu:,.0f} & {npu / tot_profit_usd * 100:.1f} \\\\')

    tab.append((
        addr,
        f'{npe / (10 ** 18):,.0f}',
        f'{npe / tot_profit_eth * 100:.2f}%',
        # f'{npu:.0f}',
        # f'{npu / tot_profit_usd * 100:.2f}%',
    ))

print()
print()
print(tabulate.tabulate(tab, headers=['Shooter contract', 'Profit (ETH)', 'ETH Profit (% of total)', 'shooter']))
# print(tabulate.tabulate(tab, headers=['Shooter contract', 'Profit (ETH)', 'ETH Profit (% of total)', 'Profit (USD)', 'USD Profit (% of total)', 'shooter']))
