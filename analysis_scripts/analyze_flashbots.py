import math
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

curr.execute('SELECT COUNT(*) FROM flashbots_transactions')
(n_flashbots,) = curr.fetchone()
print(f'Have {n_flashbots:,} flashbots transactions')

curr.execute('SELECT COUNT(distinct block_number) FROM flashbots_transactions')
(n_flashbots_blocks,) = curr.fetchone()
print(f'Have {n_flashbots_blocks:,} blocks with flashbots tagged')


curr.execute('SELECT MIN(block_number), MAX(block_number) FROM flashbots_transactions')
(min_block_number, max_block_number) = curr.fetchone()
tot_covered_blocks = max_block_number - min_block_number + 1
print(f'Have flashbots from blocks {min_block_number:,} to {max_block_number:,} ({tot_covered_blocks:,} blocks)')
print()

curr.execute(
    '''
    SELECT cnt, count(*)
    FROM (
        SELECT block_number, COUNT(*) cnt FROM flashbots_transactions GROUP BY block_number
    ) a
    GROUP BY cnt
    ORDER BY cnt
    '''
)


tab = [(0, tot_covered_blocks - n_flashbots_blocks, f'{(tot_covered_blocks - n_flashbots_blocks) / tot_covered_blocks * 100:.2f}%')]
for n_flashbot_txns, cnt_flashbots_txns in curr.fetchall():
    percent = cnt_flashbots_txns / tot_covered_blocks * 100
    tab.append((n_flashbot_txns, cnt_flashbots_txns, f'{percent:.2f}%'))

print('')
print(tabulate.tabulate(tab[:20], headers=['flashbots txns / block', 'count', 'percent']))
print()

# show bundle size
curr.execute(
    '''
    SELECT cnt, count(*)
    FROM (
        SELECT COUNT(*) cnt FROM flashbots_transactions GROUP BY block_number, bundle_index
    ) a
    GROUP BY cnt
    ORDER BY cnt
    '''
)
n_bundles = 0
tab = []
for bundle_size, count in curr:
    n_bundles += count
    tab.append([bundle_size, count])

n_bundles = sum(x for (_, x) in tab)
tab = [(x, y, f'{y / n_bundles * 100:.2f}%') for x, y in tab]
print(tabulate.tabulate(tab[:20], headers=['bundle size', 'count', 'percent']))
print()

# compute percentage of my sample arbitrages that were marked flashbots
curr.execute(
    '''
    CREATE TEMP TABLE tmp_flashbots_sample_arbs AS
    SELECT sa.id sample_arbitrage_id, sa.txn_hash transaction_hash, ft.id flashbots_id
    FROM sample_arbitrages_no_fp sa
    JOIN flashbots_transactions ft ON sa.block_number = ft.block_number AND sa.txn_hash = ft.transaction_hash;
    '''
)
n_flashbots_sample_arbs = curr.rowcount

curr.execute('CREATE INDEX IF NOT EXISTS idx_tmp_flashbots_sample_arbs_sample_id ON tmp_flashbots_sample_arbs (sample_arbitrage_id)')

curr.execute('SELECT COUNT(*) FROM sample_arbitrages_no_fp')
(n_sample_arbitrages,) = curr.fetchone()

print(f'Of {n_sample_arbitrages:,} scraped arbitrages, {n_flashbots_sample_arbs:,} are marked flashbots ' + \
      f'({n_flashbots_sample_arbs / n_sample_arbitrages * 100:.2f}%)')
print()

setup_weth_arb_tables(curr)

# # sanity check that coinbase transfer info comports with flashbots reporting
# curr.execute(
#     '''
#     SELECT sa.block_number, sa.txn_hash, ft.coinbase_transfer, sa.coinbase_xfer
#     FROM tmp_flashbots_sample_arbs tfsa
#     JOIN sample_arbitrages sa ON tfsa.sample_arbitrage_id = sa.id
#     JOIN flashbots_transactions ft ON tfsa.flashbots_id = ft.id
#     WHERE ft.coinbase_transfer != sa.coinbase_xfer
#     ORDER BY ABS(ft.coinbase_transfer - sa.coinbase_xfer) DESC
#     '''
# )
# n_differ = curr.rowcount
# print(f'Have {n_differ:,} transactions where our coinbase transfer differs')
# print('Some differing transactions:')
# tab = []
# for block_number, txn_hash, flashbots_coinbase, sa_coinbase in curr:
#     tab.append((block_number, '0x' + txn_hash.tobytes().hex(), flashbots_coinbase, sa_coinbase))
# print(tabulate.tabulate(tab[:30], headers=['block number', 'transaction hash', 'flashbots reported coinbase transfer', 'our coinbase transfer']))

#
# plot flashbots prevalence over time
#
#
# NOTE there's a weird artifact at the end
markers, stepsize = np.linspace(min_block_number, max_block_number, num=500, endpoint=True, retstep=True)
cnt_flashbots = np.zeros(markers.shape, dtype=int)
curr.execute('SELECT DISTINCT block_number FROM flashbots_transactions ORDER BY block_number')
for (block_number,) in curr:
    marker_index = math.floor((block_number - min_block_number) / stepsize)
    cnt_flashbots[marker_index] += 1

percent_flashbots = cnt_flashbots / stepsize * 100
percent_not_flashbots = np.negative(percent_flashbots) + 100


print(f'DEBUG: stepsize={stepsize}')
plt.bar(markers, height=percent_flashbots, width=stepsize, label='% flashbots')
plt.bar(markers, height=percent_not_flashbots, width=stepsize, bottom=percent_flashbots, label='% no flashbots')
plt.xlabel('block number')
plt.ylabel('percentage')
plt.title('Percent blocks with flashbots transaction')
plt.legend()
plt.show()

#
# plot flashbots prevalence among scraped arbitrages
#
print('computing flashbots percent in sample arbitrages')

curr.execute('SELECT MIN(block_number), MAX(block_number) FROM sample_arbitrages_no_fp')
min_block_number_with_arbs, max_block_number_with_arbs = curr.fetchone()

markers, stepsize = np.linspace(min_block_number_with_arbs, max_block_number_with_arbs, num=500, endpoint=True, retstep=True)

tot_arbs = np.zeros(markers.shape, dtype=int)
tot_flashbots_arbs = np.zeros(markers.shape, dtype=int)

curr.execute(
    '''
    SELECT
        block_number,
        EXISTS(SELECT 1 FROM tmp_flashbots_sample_arbs tfsa WHERE tfsa.sample_arbitrage_id = sa.id)
    FROM sample_arbitrages_no_fp sa
    '''
)
for block_number, is_flashbots in curr:
    assert block_number <= max_block_number_with_arbs
    marker_index = math.floor((block_number - min_block_number_with_arbs) / stepsize)
    assert markers[marker_index] <= block_number, f'expected {markers[marker_index]:,} <= {block_number:,}'
    tot_arbs[marker_index] += 1
    if is_flashbots:
        tot_flashbots_arbs[marker_index] += 1

percent_flashbots = np.divide(tot_flashbots_arbs, tot_arbs) * 100
percent_not_flashbots = np.negative(percent_flashbots) + 100

plt.bar(markers, height=percent_flashbots, width=stepsize, label='% flashbots')
plt.bar(markers, height=percent_not_flashbots, width=stepsize, bottom=percent_flashbots, label='% not flashbots')
plt.xlabel('block number')
plt.ylabel('percentage')
plt.title('Percent scraped arbs in flashbots')
plt.legend()
plt.show()

# yields of flashbots vs non-flashbots transactions
print('comparing yields of flashbots vs non-flashbots')
curr.execute(
    '''
    SELECT block_number, revenue, net_profit, EXISTS(SELECT 1 FROM tmp_flashbots_sample_arbs tfsa WHERE tfsa.sample_arbitrage_id = twa.id)
    FROM tmp_weth_arbs twa
    '''
)
yields_flashbots = [[] for _ in range(len(markers))]
yields_non_flashbots = [[] for _ in range(len(markers))]
profits_flashbots = [[] for _ in range(len(markers))]
profits_non_flashbots = [[] for _ in range(len(markers))]
for block_number, revenue, net_profit, is_flashbots in curr:
    if revenue <= 0:
        continue
    yield_ = net_profit / revenue * 100
    marker_index = math.floor((block_number - min_block_number_with_arbs) / stepsize)
    if is_flashbots:
        yields_flashbots[marker_index].append(yield_)
        profits_flashbots[marker_index].append(net_profit)
    else:
        yields_non_flashbots[marker_index].append(yield_)
        profits_non_flashbots[marker_index].append(net_profit)

median_yields_flashbots = [np.median(xs) for xs in yields_flashbots]
median_yields_non_flashbots = [np.median(xs) for xs in yields_non_flashbots]

plt.plot(markers, median_yields_flashbots, label='flashbots')
plt.plot(markers, median_yields_non_flashbots, label='non-flashbots')
plt.ylabel('median yield %')
plt.xlabel('block number')
plt.title('median arbitrage yield, flashbots vs non-flashbots')
plt.legend()
plt.show()

median_profit_flashbots = [np.median(xs) / (10 ** 18) for xs in profits_flashbots]
median_profit_non_flashbots = [np.median(xs) / (10 ** 18) for xs in profits_non_flashbots]

plt.plot(markers, median_profit_flashbots, label='flashbots')
plt.plot(markers, median_profit_non_flashbots, label='non-flashbots')
plt.ylabel('median profit')
plt.xlabel('block number')
plt.title('median arbitrage profits, flashbots vs non-flashbots')
plt.legend()
plt.show()


