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


curr.execute('select cnt from (SELECT n_cycles, COUNT(*) cnt FROM sample_arbitrages_no_fp GROUP BY n_cycles ORDER BY n_cycles desc) x where n_cycles = 1')
(count_single_cycle,) = curr.fetchone()

curr.execute(
    '''
    SELECT tokens.address, tokens.name, tokens.symbol, cnt
    FROM (
        SELECT sac.profit_token, COUNT(*) cnt
        FROM sample_arbitrages_no_fp sa
        JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
        WHERE sa.block_number < 15628035
        GROUP BY sac.profit_token
    ) a
    LEFT JOIN tokens ON tokens.id = profit_token
    ORDER BY cnt DESC
    '''
)

tab = []
n_above_4 = 0
tot = 0
for i, (token_address, name, symbol, cnt) in enumerate(curr):
    tot += cnt
    if i > 4:
        n_above_4 += cnt
    else:
        token_address = web3.Web3.toChecksumAddress(token_address.tobytes())
        percent = cnt / count_single_cycle * 100
        tab.append((f'{name} ({symbol})', token_address, cnt, f'{percent:.2f}%'))
print('tot', tot)
tab.append(('Other', '', n_above_4, f'{n_above_4 / count_single_cycle * 100:.2f}%'))
print('top tokens taken profit')
print(tabulate.tabulate(tab, headers=['name', 'address', 'count', 'percent']))
print()

exit()

gen_false_positives(curr)

# Some stats out of curiosity

curr.execute('SELECT COUNT(*) FROM sample_arbitrages_no_fp')
(cnt_samples,) = curr.fetchone()
print(f'Have {cnt_samples:,} sample arbitrages')
print()

curr.execute('SELECT SUM(coinbase_xfer) FROM sample_arbitrages_no_fp')
(total_bribes,) = curr.fetchone()
print(f'Total coinbase transfer: {total_bribes / (10 ** 18):,.2f} ETH')
print()

if False:
    # Gas usage quartiles:
    # Quartile 1: 174981
    # Quartile 2: 225392
    # Quartile 3: 293189

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY gas_used),
            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY gas_used),
            PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY gas_used)
        FROM sample_arbitrages_no_fp
        '''
    )
    qt1_gas_used, qt2_gas_used, qt3_gas_used = curr.fetchone()
    print('Gas usage quartiles:')
    print(f'Quartile 1: {qt1_gas_used}')
    print(f'Quartile 2: {qt2_gas_used}')
    print(f'Quartile 3: {qt3_gas_used}')
    print()

curr.execute(
    '''
    SELECT recipient, sm
    FROM (
        SELECT miner recipient, SUM(coinbase_xfer) sm
        FROM sample_arbitrages_no_fp
        GROUP BY miner
    ) a
    ORDER BY sm DESC
    LIMIT 5
    '''
)

tab = []
for recipient, amt in curr:
    if recipient is None:
        continue
    recipient = web3.Web3.toChecksumAddress(recipient.tobytes())
    percent = amt / total_bribes * 100
    amt = amt / (10 ** 18)
    tab.append((recipient, f'{amt:,.2f}', f'{percent:.2f}%'))
print('top recipients of direct miner bribes')
print(tabulate.tabulate(tab, headers=['recipient', 'Ether', 'percent']))
print()

# sum miner's profits
curr.execute(
    '''
    CREATE TEMP TABLE miner_fees_received AS
    SELECT
        miner recipient,
        SUM(coinbase_xfer + gas_price * gas_used) ether_received,
        SUM((coinbase_xfer + gas_price * gas_used) / power(10, 18) * eth_price_usd) usd_received
    FROM sample_arbitrages_no_fp sa
    JOIN eth_price_blocks ep ON sa.block_number = ep.block_number
    GROUP BY miner
    '''
)
print(f'Have {curr.rowcount:,} distinct miners')

curr.execute('SELECT SUM(ether_received), SUM(usd_received) FROM miner_fees_received')
eth_received, usd_received = curr.fetchone()
print(f'Miners received a total of {eth_received / (10 ** 18):,.5f} ETH')
print(f'Miners received a total of {usd_received:,.2f} USD')


curr.execute(
    '''
    SELECT recipient, ether_received
    FROM miner_fees_received
    ORDER BY ether_received DESC
    LIMIT 5
    '''
)
tab = []
for recipient, amt in curr:
    if recipient is None:
        continue
    recipient = web3.Web3.toChecksumAddress(recipient.tobytes())
    percent = amt / eth_received * 100
    amt = amt / (10 ** 18)
    tab.append((recipient, f'{amt:,.2f}', f'{percent:.2f}%'))
print('top recipients of miner fees from arbitrageurs')
print(tabulate.tabulate(tab, headers=['recipient', 'Ether', 'percent']))
print()

curr.execute(
    '''
    SELECT recipient, usd_received
    FROM miner_fees_received
    ORDER BY usd_received DESC
    LIMIT 5
    '''
)
tab = []
for recipient, amt in curr:
    if recipient is None:
        continue
    recipient = web3.Web3.toChecksumAddress(recipient.tobytes())
    percent = float(amt) / float(usd_received) * 100
    tab.append((recipient, f'{amt:,.2f}', f'{percent:.2f}%'))
print('top recipients of miner fees from arbitrageurs')
print(tabulate.tabulate(tab, headers=['recipient', 'USD', 'percent']))
print()


curr.execute('SELECT COUNT(distinct shooter) FROM sample_arbitrages_no_fp')
(n_shooters,) = curr.fetchone()
print(f'Have {n_shooters:,} shooters')
print()

curr.execute(
    '''
    SELECT shooter, cnt
    FROM (
        SELECT shooter, COUNT(*) cnt
        FROM sample_arbitrages_no_fp
        GROUP BY shooter
    ) a
    ORDER BY cnt DESC
    LIMIT 5
    '''
)
tab = []
for shooter, cnt in curr:
    shooter = web3.Web3.toChecksumAddress(shooter.tobytes())
    percent = cnt / cnt_samples * 100
    tab.append((shooter, cnt, f'{percent:.2f}%'))
print('top shooters:')
print(tabulate.tabulate(tab, headers=['shooter', 'num arbitrages', 'percent']))
print()

curr.execute('SELECT n_cycles, COUNT(*) FROM sample_arbitrages_no_fp GROUP BY n_cycles ORDER BY n_cycles desc')

tab = []
for n_cycles, cnt in curr:
    if n_cycles == 1:
        count_single_cycle = cnt
    percent = cnt / cnt_samples * 100
    tab.append((n_cycles, cnt, f'{percent:.2f}%'))

print(tabulate.tabulate(tab, headers=['n cycles', 'count', 'percent']))
print()

curr.execute('SELECT txn_hash FROM sample_arbitrages_no_fp WHERE n_cycles = 2 ORDER BY RANDOM() LIMIT 1')
(sample_txn_hash,) = curr.fetchone()
print(f'Sample with n_cycles=2: https://etherscan.io/tx/0x{sample_txn_hash.tobytes().hex()}')
print()


# find circuits that split liquidity between multiple exchanges
curr.execute(
    '''
    CREATE TEMP TABLE tmp_split_route_samples AS
    SELECT sa.*
    FROM (
        SELECT distinct cycle_id
        FROM (
            SELECT sace.cycle_id, sace.id, count(sacei.id) cnt
            FROM sample_arbitrage_cycle_exchanges_no_fp sace
            JOIN sample_arbitrage_cycle_exchange_items_no_fp sacei ON sacei.cycle_exchange_id = sace.id
            GROUP BY sace.cycle_id, sace.id
        ) x
        WHERE x.cnt >= 2
    ) y
    JOIN sample_arbitrage_cycles sac ON y.cycle_id = sac.id
    JOIN sample_arbitrages sa ON sa.id = sac.sample_arbitrage_id
    '''
)
curr.execute('SELECT COUNT(*), COUNT(distinct id) FROM tmp_split_route_samples')
cnt, dcnt = curr.fetchone()
assert cnt == dcnt
print(f'Have {cnt:,} sample arbitrages that split their liquidity between exchanges')
curr.execute('SELECT txn_hash FROM tmp_split_route_samples ORDER BY RANDOM() LIMIT 10')
for (txn_hash,) in curr:
    print(f'Sample: https://etherscan.io/tx/0x{txn_hash.hex()}')
print()


# find the profit tokens

curr.execute(
    '''
    SELECT tokens.address, tokens.name, tokens.symbol, cnt
    FROM (
        SELECT sac.profit_token, COUNT(*) cnt
        FROM sample_arbitrages_no_fp sa
        JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
        GROUP BY sac.profit_token
    ) a
    LEFT JOIN tokens ON tokens.id = profit_token
    ORDER BY cnt DESC
    '''
)

tab = []
n_above_4 = 0
tot = 0
for i, (token_address, name, symbol, cnt) in enumerate(curr):
    tot += cnt
    if i > 4:
        n_above_4 += cnt
    else:
        token_address = web3.Web3.toChecksumAddress(token_address.tobytes())
        percent = cnt / count_single_cycle * 100
        tab.append((f'{name} ({symbol})', token_address, cnt, f'{percent:.2f}%'))
print('tot', tot)
tab.append(('Other', '', n_above_4, f'{n_above_4 / count_single_cycle * 100:.2f}%'))
print('top tokens taken profit')
print(tabulate.tabulate(tab, headers=['name', 'address', 'count', 'percent']))
print()


curr.execute(
    '''
    SELECT n_cycles, count
    FROM (
        SELECT n_cycles, count(*)
        FROM (
            SELECT sa.id, count(distinct sace.id) n_cycles
            FROM sample_arbitrages_no_fp sa
            JOIN sample_arbitrage_cycles_no_fp sac ON sac.sample_arbitrage_id = sa.id
            JOIN sample_arbitrage_cycle_exchanges_no_fp sace ON sace.cycle_id = sac.id
            GROUP BY sa.id
        ) a
        GROUP BY n_cycles
    ) a
    ORDER BY n_cycles
    '''
)

tab = []
n_over_four = 0
for n_cycles, count in curr:
    if n_cycles > 4:
        n_over_four += count
    else:
        percent = count / count_single_cycle * 100
        tab.append((n_cycles, count, f'{percent:.3f}%'))
tab.append((f'5+', n_over_four, f'{n_over_four / count_single_cycle * 100:.3f}%'))
print('Count by cycle length')
print(tabulate.tabulate(tab, headers=['length', 'count', 'percentage']))
print()

setup_weth_arb_tables(curr)

block_numbers = []
arb_ids = []
profits = []
revenues = []
yields = []

curr.execute('SELECT COUNT(*) FROM tmp_weth_arbs WHERE coinbase_xfer IS NULL')
print(f'Have {curr.fetchone()[0]} weth arbs with null coinbase xfer')

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
tab.append(('min', f'{min_profit / (10 ** 18):,.5f}'))
for mark, perc in zip(percentile_marks, percentiles):
    tab.append((f'{mark}%', f'{perc / (10 ** 18):,.5f}'))
tab.append(('max', f'{max_profit / (10 ** 18):,.5f}'))
print('net profit after fees')
print(tabulate.tabulate(tab, headers=['percentile', 'net profit ETH']))
print()


# sum profit after fees in WETH and then USD
curr.execute(
    '''
    SELECT
        SUM(net_profit),
        SUM(net_profit / power(10, 18) * eth_price_usd)
    FROM tmp_weth_arbs twa
    JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
    ''',
)
tot_eth_profit, tot_usd_profit = curr.fetchone()
print(f'Total ETH profit: {tot_eth_profit / (10 ** 18):,.2f} ETH')
print(f'Total USD profit: {tot_usd_profit:,.2f} ETH')
print()


yields_only_present = list(filter(lambda x: x is not None, yields))
min_yield = min(yields_only_present)
max_yield = max(yields_only_present)
percentile_marks = [5, 25, 50, 75, 95]
percentiles = np.percentile(yields_only_present, percentile_marks)
tab = []
tab.append(('min', f'{min_yield:.2f}'))
for mark, perc in zip(percentile_marks, percentiles):
    tab.append((f'{mark}%', f'{perc:.2f}'))
tab.append(('max', f'{max_yield:,.2f}%'))
print('net yield')
print(tabulate.tabulate(tab, headers=['percentile', 'net yield percent']))
print()



# print the txns with the highest revenue after fees
curr.execute(
    '''
    SELECT txn_hash, net_profit
    FROM tmp_weth_arbs ta
    JOIN sample_arbitrages sa ON ta.id = sa.id
    WHERE ta.coinbase_xfer IS NOT NULL AND EXISTS(SELECT 1 FROM sample_arbitrages_no_fp sa WHERE sa.id = ta.id)
    ORDER BY net_profit desc
    LIMIT 5
    '''
)
tab = []
for txn_hash, net_profit in curr:
    txn_hash = '0x' + txn_hash.tobytes().hex()
    tab.append((txn_hash, f'{net_profit / (10 ** 18):,.2f}'))
print('most profitable arbitrages')
print(tabulate.tabulate(tab, headers=['transaction hash', 'profit (ETH)']))
print()

if True:
    # print the txns with the highest revenue after fees
    curr.execute(
        '''
        SELECT txn_hash, net_profit
        FROM tmp_weth_arbs ta
        JOIN sample_arbitrages sa ON ta.id = sa.id
        WHERE ta.coinbase_xfer IS NOT NULL AND EXISTS(SELECT 1 FROM sample_arbitrages_no_fp sa WHERE sa.id = ta.id)
        ORDER BY net_profit ASC
        LIMIT 5
        '''
    )
    tab = []
    for txn_hash, net_profit in curr:
        txn_hash = '0x' + txn_hash.tobytes().hex()
        tab.append((txn_hash, f'{net_profit / (10 ** 18):,.2f}'))
    print('least profitable arbitrages')
    print(tabulate.tabulate(tab, headers=['transaction hash', 'profit (ETH)']))
    print()


# curr.execute(
#     '''
#     SELECT oua.id
#     FROM only_uniswap_arbitrages oua
#     WHERE oua.id IN (SELECT id FROM tmp_weth_arbs) AND EXISTS(SELECT 1 FROM sample_arbitrages_no_fp sa WHERE sa.id = oua.id)
#     '''
# )
# only_weth_only_uniswap = curr.rowcount
# print(f'Uniswap (with profit in WETH) was used for {only_weth_only_uniswap:,} exchanges ({only_weth_only_uniswap / n_weth * 100:.2f}% of WETH-only)')
# only_uniswap_arb_ids = set(x for (x,) in curr)


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


# for k in sorted(window_yields_only_uniswap.keys()):
#     plt.plot(window_blocks, window_yields_only_uniswap[k], label=f'{k}th percentile')

# plt.legend()
# plt.xlabel('block')
# plt.ylabel('ETH percentage yield')
# plt.title(f'Arbitrage percent yield (only uniswap used)\ntrailing window of {WINDOW_SIZE:,} blocks')
# plt.show()


revenues_only_positive = []
profits_only_positive = []
for revenue, profit in zip(revenues, profits):
    if revenue > 0:
        revenues_only_positive.append(revenue)
        profits_only_positive.append(profit)

revenues_only_positive_eth = np.array(revenues_only_positive) / (10 ** 18)
profits_only_positive_eth = np.array(profits_only_positive) / (10 ** 18)
assert len(revenues_only_positive_eth) == len(profits_only_positive_eth)
fit = scipy.stats.linregress(list(revenues_only_positive_eth), list(profits_only_positive_eth))

min_revenue = min(revenues_only_positive_eth)
max_revenue = max(revenues_only_positive_eth)
range_ = max_revenue - min_revenue
buffer = 0.1
fit_xs = [min_revenue - (buffer * range_), max_revenue + (buffer * range_)]
fit_ys = np.array(np.array(fit_xs) * fit.slope + fit.intercept)

print(f'fit for revenue vs profit (all weth)')
print(f'r-squared: {fit.rvalue ** 2}')
print(f'y = {fit.slope} * x + {fit.intercept}')
plt.plot(fit_xs, fit_ys, alpha=0.3)
plt.scatter(revenues_only_positive_eth, profits_only_positive_eth, s=1, c='black')
plt.xlabel('ETH revenue')
plt.ylabel('ETH profit after fees')
plt.show()


revenues_only_positive_uniswap = []
profits_only_positive_uniswap = []
for revenue, profit in zip(revenues, profits):
    if revenue > 0:
        revenues_only_positive_uniswap.append(revenue)
        profits_only_positive_uniswap.append(profit)

revenues_only_positive_uniswap_eth = np.array(revenues_only_positive_uniswap) / (10 ** 18)
profits_only_positive_uniswap_eth = np.array(profits_only_positive_uniswap) / (10 ** 18)
assert len(revenues_only_positive_uniswap_eth) == len(profits_only_positive_uniswap_eth)
fit = scipy.stats.linregress(list(revenues_only_positive_uniswap_eth), list(profits_only_positive_uniswap_eth))

min_revenue = min(revenues_only_positive_uniswap_eth)
max_revenue = max(revenues_only_positive_uniswap_eth)
range_ = max_revenue - min_revenue
buffer = 0.1
fit_xs = [min_revenue - (buffer * range_), max_revenue + (buffer * range_)]
fit_ys = np.array(np.array(fit_xs) * fit.slope + fit.intercept)

print(f'fit for revenue vs profit (all weth)\nUniswap only')
print(f'r-squared: {fit.rvalue ** 2}')
print(f'y = {fit.slope} * x + {fit.intercept}')
plt.plot(fit_xs, fit_ys, alpha=0.3)
plt.scatter(revenues_only_positive_uniswap_eth, profits_only_positive_uniswap_eth, s=1, c='black')
plt.xlabel('ETH revenue')
plt.ylabel('ETH profit after fees')
plt.show()

