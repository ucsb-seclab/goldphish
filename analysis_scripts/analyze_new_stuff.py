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

# gen_false_positives(curr)

setup_weth_arb_tables(curr, only_new=True)

curr.execute(
    '''
    SELECT txn_hash, net_profit
    FROM tmp_weth_arbs
    where net_profit > 0
    ORDER BY net_profit DESC LIMIT 20
    '''
)

tab = []
for txn_hash, net_profit in curr:
    tab.append((txn_hash.tobytes().hex(), f'{net_profit / (10 ** 18):,.4f} ETH'))

print(tabulate.tabulate(tab, headers=['transaction hash', 'profit']))

# exit()

block_numbers = []
arb_ids = []
profits = []
revenues = []
yields = []

curr.execute(
    '''
    SELECT id, block_number, net_profit, revenue
    FROM tmp_weth_arbs twa
    WHERE coinbase_xfer IS NOT NULL AND EXISTS(SELECT 1 FROM sample_arbitrages_no_fp sa WHERE sa.id = twa.id)
    ORDER BY block_number ASC
    '''
)

for id_, block_number, net_profit, revenue in curr:
    net_profit = int(net_profit)
    revenue = int(revenue)
    if revenue > 0:
        yield_ = net_profit / int(revenue) * 100
        yields.append(yield_)
    else:
        yields.append(None)
    profits.append(net_profit)
    block_numbers.append(block_number)
    arb_ids.append(id_)
    revenues.append(revenue)

min_profit = min(profits)
max_profit = max(profits)
percentile_marks = [5, 25, 50, 75, 95]
percentiles = np.percentile(profits, percentile_marks)
tab = []
# tab.append(('min', f'{min_profit / (10 ** 18):,.5f}'))
for mark, perc in zip(percentile_marks, percentiles):
    tab.append((f'{mark}%', f'{perc / (10 ** 18):,.5f}'))
# tab.append(('max', f'{max_profit / (10 ** 18):,.5f}'))
print('profit after fees')
print(tabulate.tabulate(tab, headers=['percentile', 'net profit ETH']))
print()

# sum profit after fees in WETH and then USD
curr.execute(
    '''
    SELECT
        SUM(net_profit)
--        SUM(net_profit / power(10, 18) * eth_price_usd)
    FROM tmp_weth_arbs twa
    ''',
)
(tot_eth_profit,) = curr.fetchone()
print(f'Total ETH profit: {tot_eth_profit / (10 ** 18):,.2f} ETH')
# print(f'Total USD profit: {tot_usd_profit:,.2f} ETH')
print()


yields_only_present = list(filter(lambda x: x is not None, yields))
percentile_marks = [5, 25, 50, 75, 95]
percentiles = np.percentile(yields_only_present, percentile_marks)
tab = []
# tab.append(('min', f'{min_yield:.2f}'))
for mark, perc in zip(percentile_marks, percentiles):
    tab.append((f'{mark}%', f'{perc:.2f}'))
# tab.append(('max', f'{max_yield:,.2f}%'))
print('net yield')
print(tabulate.tabulate(tab, headers=['percentile', 'net yield percent']))
print()


# # print the txns with the highest revenue after fees
# curr.execute(
#     '''
#     SELECT txn_hash, net_profit
#     FROM tmp_weth_arbs ta
#     JOIN sample_arbitrages sa ON ta.id = sa.id
#     WHERE ta.coinbase_xfer IS NOT NULL AND EXISTS(SELECT 1 FROM sample_arbitrages_no_fp sa WHERE sa.id = ta.id)
#     ORDER BY net_profit desc
#     LIMIT 5
#     '''
# )
# tab = []
# for txn_hash, net_profit in curr:
#     txn_hash = '0x' + txn_hash.tobytes().hex()
#     tab.append((txn_hash, f'{net_profit / (10 ** 18):,.2f}'))
# print('most profitable arbitrages')
# print(tabulate.tabulate(tab, headers=['transaction hash', 'profit (ETH)']))
# print()


exit()

# sliding window analyses
WINDOW_SIZE = int((1 / 13) * 60 * 60 * 24) # about 1 day
trailing_block_numbers = deque()
trailing_yields = deque()
trailing_profits = deque()
trailing_profits_usd = deque()

# trailing_block_numbers_uniswap_only = deque()
# trailing_yields_uniswap_only = deque()
# trailing_profits_uniswap_only = deque()

assert len(block_numbers) == len(profits)
assert len(profits) == len(yields)


# sliding window produced points
window_blocks = []
percentile_marks = [25, 50, 75]
window_yields = {x: [] for x in percentile_marks}
window_profits = {x: [] for x in percentile_marks}
window_profits_usd = {x: [] for x in percentile_marks}
window_sum_profits = []
window_sum_profits_usd = []

# window_yields_only_uniswap = {x: [] for x in percentile_marks}
# window_profits_only_uniswap = {x: [] for x in percentile_marks}
# window_sum_profits_only_uniswap = []

start_block = min(block_numbers)
end_block_inclusive = max(block_numbers)
i = 0
print(f'[*] computing sliding window analyses from {start_block:,} to {end_block_inclusive:,}...')
curr2 = db.cursor()
curr2.execute(
    '''
    SELECT eth_price_usd, block_number
    FROM eth_price_blocks
    WHERE %s <= block_number AND block_number <= %s
    ORDER BY block_number ASC
    ''',
    (start_block, end_block_inclusive),
)
for usd_price, block_number in curr2:
    assert usd_price is not None, f'Could not get price for block_number={block_number:,}'
    usd_price = float(usd_price)

    # remove everything beyond sliding window
    while len(trailing_block_numbers) > 0 and trailing_block_numbers[0] <= block_number - WINDOW_SIZE:
        trailing_block_numbers.popleft()
        trailing_yields.popleft()
        trailing_profits.popleft()
        trailing_profits_usd.popleft()

    # while len(trailing_block_numbers_uniswap_only) > 0 and trailing_block_numbers_uniswap_only[0] <= block_number - WINDOW_SIZE:
    #     trailing_block_numbers_uniswap_only.popleft()
    #     trailing_yields_uniswap_only.popleft()
    #     trailing_profits_uniswap_only.popleft()

    if block_number % 100_000 == 0:
        print(f'[*] {block_number:,} ({(block_number - start_block) / (end_block_inclusive - start_block) * 100:.0f}%)')


    # add everything now in the sliding window
    # `i` is reused each loop as a cursor
    for i in range(i, len(block_numbers)):
        if block_numbers[i] != block_number:
            # reached end of things to add
            break
        trailing_block_numbers.append(block_numbers[i])
        trailing_yields.append(yields[i])
        trailing_profits.append(profits[i])
        trailing_profits_usd.append(profits[i] / (10 ** 18) * usd_price)

        # if arb_ids[i] in only_uniswap_arb_ids:
        #     trailing_block_numbers_uniswap_only.append(block_numbers[i])
        #     trailing_yields_uniswap_only.append(yields[i])
        #     trailing_profits_uniswap_only.append(profits[i])

    assert len(trailing_block_numbers) > 0, f'expected to find data in the sliding window at block {block_number:,} but found none'

    # if we haven't slid at least one window length inward, don't produce point-results
    if block_number < start_block + WINDOW_SIZE:
        continue

    if block_number % 5000 != 0:
        continue

    # produce a data-point (for all)
    only_pos_yields = list(filter(lambda x: x is not None, trailing_yields))

    assert len(only_pos_yields) > 0, f'expected to find some positive yields in window at block {block_number:,}'
    window_blocks.append(block_number)

    for mark, val in zip(percentile_marks, np.percentile(only_pos_yields, percentile_marks)):
        window_yields[mark].append(val)
    for mark, val in zip(percentile_marks, np.percentile(trailing_profits, percentile_marks)):
        window_profits[mark].append(val)
    for mark, val in zip(percentile_marks, np.percentile(trailing_profits_usd, percentile_marks)):
        window_profits_usd[mark].append(val)

    window_sum_profits.append(sum(trailing_profits))
    window_sum_profits_usd.append(sum(trailing_profits_usd))

    # # produce a data-point (for uniswap)
    # only_pos_yields = list(filter(lambda x: x is not None, trailing_yields_uniswap_only))
    # assert len(only_pos_yields) > 0, f'expected to find some positive uniswap yields in window at block {block_number:,}'

    # for mark, val in zip(percentile_marks, np.percentile(only_pos_yields, percentile_marks)):
    #     window_yields_only_uniswap[mark].append(val)
    # for mark, val in zip(percentile_marks, np.percentile(trailing_profits_uniswap_only, percentile_marks)):
    #     window_profits_only_uniswap[mark].append(val)
    # window_sum_profits.append(sum(trailing_profits_uniswap_only))


for k in sorted(window_profits.keys()):
    plt.plot(window_blocks, np.array(window_profits[k]) / (10 ** 18), label=f'{k}th percentile')

plt.legend()
plt.xlabel('block')
plt.ylabel('ETH profit (after fees)')
plt.title(f'Arbitrage profit by percentile\ntrailing window of {WINDOW_SIZE:,} blocks')
plt.show()

for k in sorted(window_profits_usd.keys()):
    plt.plot(window_blocks, np.array(window_profits_usd[k]), label=f'{k}th percentile')

plt.legend()
plt.xlabel('block')
plt.ylabel('USD profit (after fees)')
plt.title(f'Arbitrage profit by percentile\ntrailing window of {WINDOW_SIZE:,} blocks')
plt.show()


for k in sorted(window_yields.keys()):
    plt.plot(window_blocks, window_yields[k], label=f'{k}th percentile')

plt.legend()
plt.xlabel('block')
plt.ylabel('ETH percentage yield')
plt.title(f'Arbitrage percent yield\ntrailing window of {WINDOW_SIZE:,} blocks')
plt.show()

