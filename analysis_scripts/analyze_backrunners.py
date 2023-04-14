import collections
import datetime
import math
import sqlite3
import psycopg2
import tabulate
import os
import numpy as np
import matplotlib.pyplot as plt


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

# #
# # Some basic facts
# #

#
# I think the below was used for a last-minute calculation before the deadline? Not sure.
# - R
#

# # count how many samples we have
# curr.execute('SELECT COUNT(*) FROM sample_arbitrages_no_fp where n_cycles = 1 and block_number < 15628035')
# (n_samples,) = curr.fetchone()
# print(f'Have {n_samples:,} sample arbitrages')

# print('himom')

# # how many have we rerun
# curr.execute(
#     '''
#     SELECT
#         SUM(
#             CASE WHEN EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = sa.txn_hash)
#             THEN 1 ELSE 0 END
#         ),
#         COUNT(*)
#     FROM sample_arbitrages_no_fp sa
#     WHERE sa.n_cycles = 1 and sa.block_number < 15628035 AND EXISTS (
#         SELECT FROM sample_arbitrage_backrun_detections sabd WHERE sa.id = sabd.sample_arbitrage_id AND (sabd.rerun_reverted = true OR sabd.rerun_no_arbitrage = true)
#     )
#     '''
# )
# (n_backrun_flashbots, n_total) = curr.fetchone()
# print(f'Have {n_backrun_flashbots:,} back-runners that used flashbots and {n_total:,} total back-runners')
# exit()

if False:
    #
    # Analyze 'new stuff'
    #

    # how many have we rerun
    curr.execute(
        '''
        SELECT COUNT(*)
        FROM sample_arbitrage_backrun_detections sabd
        JOIN sample_arbitrages sa on sa.id = sabd.sample_arbitrage_id
        WHERE EXISTS(SELECT FROM sample_arbitrages_no_fp WHERE id = sample_arbitrage_id and n_cycles = 1) and 15628035 <= block_number
        '''
    )
    (n_backrun_reorders_run,) = curr.fetchone()
    print(f'Have {n_backrun_reorders_run:,} arbitrages re-run at top-of-block: {n_backrun_reorders_run / n_samples * 100:.1f}%')
    print()

    # break down re-runs by diagnosis
    curr.execute(
        '''
        SELECT
            SUM(CASE rerun_exactly WHEN true THEN 1 ELSE 0 END),
            SUM(CASE rerun_reverted WHEN true THEN 1 ELSE 0 END),
            SUM(CASE rerun_no_arbitrage WHEN true THEN 1 ELSE 0 END),
            SUM(CASE rerun_not_comparable WHEN true THEN 1 ELSE 0 END),
            SUM(CASE rerun_profit_token_changed WHEN true THEN 1 ELSE 0 END),
            SUM(CASE rerun_profit IS NULL WHEN true THEN 0 ELSE 1 END)
        FROM sample_arbitrage_backrun_detections sabd
        JOIN sample_arbitrages sa on sa.id = sabd.sample_arbitrage_id
        WHERE EXISTS(SELECT FROM sample_arbitrages_no_fp WHERE id = sample_arbitrage_id and n_cycles = 1) and 15628035 <= block_number
        '''
    )
    assert curr.rowcount == 1
    (
        n_rerun_exactly,
        n_rerun_reverted,
        n_rerun_no_arb,
        n_rerun_not_compatible,
        n_rerun_profit_token_changed,
        n_rerun_new_profit,
    ) = curr.fetchone()

    reports = [
        ('No change',        n_rerun_exactly),
        ('Reverted',         n_rerun_reverted),
        ('No arbitrage',     n_rerun_no_arb),
        ('Not supported',    n_rerun_not_compatible),
        ('New profit token', n_rerun_profit_token_changed),
        ('Profit changed',   n_rerun_new_profit),
    ]

    tot = sum(x[1] for x in reports)
    print(f'tot {tot:,}')

    reports = sorted(reports, key=lambda x: x[1], reverse=True)
    reports_with_percent = [(s, n, f'{n / n_backrun_reorders_run * 100:.1f}%') for s, n in reports]

    print()
    print(tabulate.tabulate(reports_with_percent, headers=['Reorder result', 'Count', 'Percent']))

# for t, n in reports:
#     print(f'{t} & ${n:,}$ & ${n / tot * 100:.1f}\\%$ \\\\')

# Reorder result      Count  Percent
# ----------------  -------  ---------
# No change         2108655  55.5%
# No arbitrage       878849  23.1%
# Reverted           566118  14.9%
# Profit changed     238347  6.3%
# New profit token     5260  0.1%
# Not supported          30  0.0%


# exit()

# curr.execute(
#     '''
#     CREATE TEMP TABLE tmp_profit_changed_diff (
#         sample_arbitrage_id INTEGER NOT NULL,
#         old_profit NUMERIC(78, 0) NOT NULL,
#         new_profit NUMERIC(78, 0) NOT NULL,
#         profit_token INTEGER NOT NULL
#     );

#     INSERT INTO tmp_profit_changed_diff (sample_arbitrage_id, old_profit, new_profit, profit_token)
#     SELECT sabd.sample_arbitrage_id, sac.profit_amount, sabd.rerun_profit, sac.profit_token
#     FROM sample_arbitrage_backrun_detections sabd
#     JOIN sample_arbitrage_cycles sac ON sabd.sample_arbitrage_id = sac.sample_arbitrage_id
#     WHERE sabd.rerun_profit IS NOT NULL;
#     '''
# )
# n_inserted = curr.rowcount
# assert n_inserted == n_rerun_new_profit, f'expected {n_inserted} == {n_rerun_new_profit}'

# curr.execute('SELECT SUM(CASE old_profit < new_profit WHEN true THEN 1 ELSE 0 END) FROM tmp_profit_changed_diff')
# (n_profit_increased,) = curr.fetchone()
# print()
# print(f'Saw {n_profit_increased} that increased profit ({n_profit_increased / n_rerun_new_profit * 100:.2f}%)')

# curr.execute(
#     '''
#     SELECT txn_hash
#     FROM sample_arbitrages sa
#     JOIN tmp_profit_changed_diff pcd ON sa.id = pcd.sample_arbitrage_id
#     WHERE old_profit < new_profit
#     ORDER BY RANDOM()
#     LIMIT 5
#     '''
# )
# print('Examples of increased profit:')
# for (txn_hash,) in curr:
#     print('0x' + txn_hash.tobytes().hex())


#
# Compare profitability
#

# gather definite backrunners and definite not backrunners
curr.execute(
    '''
    CREATE TEMP TABLE tmp_backrunners (
        sample_arbitrage_id INTEGER NOT NULL,
        txn_hash BYTEA NOT NULL
    );

    CREATE TEMP TABLE tmp_not_backrunners (
        sample_arbitrage_id INTEGER NOT NULL,
        txn_hash BYTEA NOT NULL
    );

    INSERT INTO tmp_not_backrunners (sample_arbitrage_id, txn_hash)
    SELECT sample_arbitrage_id, sa.txn_hash
    FROM sample_arbitrage_backrun_detections sabd
    JOIN sample_arbitrages_no_fp sa ON sa.id = sabd.sample_arbitrage_id
    WHERE sabd.rerun_exactly = true AND sa.n_cycles = 1 and block_number < 15628035;

    INSERT INTO tmp_backrunners (sample_arbitrage_id, txn_hash)
    SELECT sample_arbitrage_id, sa.txn_hash
    FROM sample_arbitrage_backrun_detections sabd
    JOIN sample_arbitrages_no_fp sa ON sa.id = sabd.sample_arbitrage_id
    WHERE (sabd.rerun_reverted = true OR sabd.rerun_no_arbitrage = true) AND sa.n_cycles = 1 and block_number < 15628035;
    '''
)

curr.execute('SELECT COUNT(*) FROM tmp_not_backrunners')
(n_not_backrunners,) = curr.fetchone()
print(f'Have {n_not_backrunners:,} not-backrunners')
# assert n_not_backrunners == n_rerun_exactly

curr.execute('SELECT COUNT(*) FROM tmp_backrunners')
(n_backrunners,) = curr.fetchone()
print(f'Have {n_backrunners:,} backrunners')
# assert n_backrunners == n_rerun_reverted + n_rerun_no_arb


# # determine how many back-runners used flashbots, vs non-backrunners
# curr.execute(
#     '''
#     SELECT
#         SUM(
#             CASE WHEN EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = tb.txn_hash)
#             THEN 1 ELSE 0 END
#         ),
#         COUNT(*)
#     FROM tmp_backrunners tb
#     '''
# )
# flashbot_br, all_br = curr.fetchone()
# print(f'Saw {flashbot_br:,} backrunners of {all_br:,} ({flashbot_br / all_br * 100:.2f}%) use flashbots')
# print()

# curr.execute(
#     '''
#     SELECT
#         SUM(
#             CASE WHEN EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = tb.txn_hash)
#             THEN 1 ELSE 0 END
#         ),
#         COUNT(*)
#     FROM tmp_not_backrunners tb
#     '''
# )
# non_flashbot_br, all_br = curr.fetchone()
# print(f'Saw {non_flashbot_br:,} backrunners of {all_br:,} ({non_flashbot_br / all_br * 100:.2f}%) use flashbots')
# print()


# Have 2,108,655 not-backrunners
# Have 1,444,967 backrunners
# Saw 468,431 backrunners of 1,444,967 (32.42%) use flashbots

# Saw 668,624 backrunners of 2,108,655 (31.71%) use flashbots


# exit()

# print(f'Counting {n_not_backrunners:,} not backrunning and {n_backrunners:,} backrunning')


curr.execute(
    '''
    CREATE TEMP TABLE tmp_profits AS
    SELECT
        sa.*,
        sac.profit_amount revenue,
        sa.gas_used * sa.gas_price + sa.coinbase_xfer fee,
        (sac.profit_amount - sa.coinbase_xfer) - (sa.gas_used * sa.gas_price) net_profit
    FROM sample_arbitrages_no_fp sa
    JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
    where block_number < 15628035;

    CREATE TEMP VIEW tmp_yields AS
    SELECT *, net_profit::float / revenue * 100 as yield
    FROM tmp_profits
    WHERE revenue > 0;
    '''
)
print('[*] created profit table')
print()

# get percentile profit of both groups
if True:
    #   Percentile    Backrunner profit (ETH)    non-backrunner profit (ETH)
    # ------------  -------------------------  -----------------------------
    #            5                    0                             -0.01121
    #           25                    0.00378                        0.00071
    #           50                    0.02026                        0.0037
    #           75                    0.07636                        0.01468
    #           95                    0.719                          0.28899

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_profits tp
        '''
    )
    row1a = curr.fetchone()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
        FROM tmp_yields tp
        '''
    )
    row1b = curr.fetchone()
    print(f'\\textbf{{All}} & ', end = '')
    for p in row1a:
        print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    for p in row1b:
        print(f' ${round(p, 2)}$ &', end = '')
    print()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_profits tp
        WHERE EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = tp.txn_hash)
        '''
    )
    row1a = curr.fetchone()

    # curr.execute(
    #     '''
    #     SELECT
    #         PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
    #         PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
    #         PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
    #     FROM tmp_yields tp
    #     WHERE EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = tp.txn_hash)
    #     '''
    # )
    # row1b = curr.fetchone()
    # print(f'\\flashbots & ', end = '')
    # for p in row1a:
    #     print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    # for p in row1b:
    #     print(f' ${round(p, 2)}$ &', end = '')
    # print()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_profits tp
        WHERE NOT EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = tp.txn_hash)
        '''
    )
    row1a = curr.fetchone()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
        FROM tmp_yields tp
        WHERE NOT EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = tp.txn_hash)
        '''
    )
    row1b = curr.fetchone()
    print(f'Not \\flashbots & ', end = '')
    for p in row1a:
        print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    for p in row1b:
        print(f' ${round(p, 2)}$ &', end = '')
    print()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_profits tp
        JOIN tmp_backrunners tb ON tp.id = tb.sample_arbitrage_id;
        '''
    )
    row1a = curr.fetchone()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
        FROM tmp_yields tp
        JOIN tmp_backrunners tb ON tp.id = tb.sample_arbitrage_id;
        '''
    )
    row1b = curr.fetchone()
    print(f'Back-running & ', end = '')
    for p in row1a:
        print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    for p in row1b:
        print(f' ${round(p, 2)}$ &', end = '')
    print()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_profits tp
        JOIN tmp_not_backrunners tb ON tp.id = tb.sample_arbitrage_id;
        '''
    )
    row1a = curr.fetchone()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
        FROM tmp_yields tp
        JOIN tmp_not_backrunners tb ON tp.id = tb.sample_arbitrage_id;
        '''
    )
    row1b = curr.fetchone()
    print(f'Not back-running & ', end = '')
    for p in row1a:
        print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    for p in row1b:
        print(f' ${round(p, 2)}$ &', end = '')
    print()


    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.05) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.95) WITHIN GROUP(ORDER BY net_profit)
        FROM tmp_profits tp
        JOIN tmp_not_backrunners tb ON tp.id = tb.sample_arbitrage_id;
        '''
    )
    pn5, pn25, pn50, pn75, pn95 = curr.fetchone()

    tab = []
    tab.append((5, f'{pb5 / (10 ** 18):.5f}', f'{pn5 / (10 ** 18):.5f}'))
    tab.append((25, f'{pb25 / (10 ** 18):.5f}', f'{pn25 / (10 ** 18):.5f}'))
    tab.append((50, f'{pb50 / (10 ** 18):.5f}', f'{pn50 / (10 ** 18):.5f}'))
    tab.append((75, f'{pb75 / (10 ** 18):.5f}', f'{pn75 / (10 ** 18):.5f}'))
    tab.append((95, f'{pb95 / (10 ** 18):.5f}', f'{pn95 / (10 ** 18):.5f}'))
    print(tabulate.tabulate(tab, headers=['Percentile', 'Backrunner profit (ETH)', 'non-backrunner profit (ETH)']))
    print()
    exit()


# # get percentile yield of both groups
# percentiles = [5, 25, 50, 75, 95]

# percentile_yields_backrunners = []
# curr.execute(
#     '''
#     SELECT yield, block_number
#     FROM tmp_backrunners tb
#     JOIN tmp_yields ty ON ty.id = tb.sample_arbitrage_id
#     '''
# )
# yields_backrunners = []
# blocks_backrunners = []
# for y, b in curr:
#     yields_backrunners.append(y)
#     blocks_backrunners.append(b)

# percentile_yields_backrunners.append(np.min(yields_backrunners))
# percentile_yields_backrunners += list(np.percentile(yields_backrunners, percentiles))
# percentile_yields_backrunners.append(np.max(yields_backrunners))

# percentile_yields_not_backrunners = []
# curr.execute(
#     '''
#     SELECT yield, block_number
#     FROM tmp_not_backrunners tb
#     JOIN tmp_yields ty ON ty.id = tb.sample_arbitrage_id
#     '''
# )
# yields_not_backrunners = []
# blocks_not_backrunners = []
# for y, b in curr:
#     yields_not_backrunners.append(y)
#     blocks_not_backrunners.append(b)


# percentile_yields_not_backrunners.append(np.min(yields_not_backrunners))
# percentile_yields_not_backrunners += list(np.percentile(yields_not_backrunners, percentiles))
# percentile_yields_not_backrunners.append(np.max(yields_not_backrunners))


# tab = []
# tab.append(('min', percentile_yields_backrunners[0], percentile_yields_not_backrunners[0]))
# for p, val_br, val_not_br in zip(percentiles, percentile_yields_backrunners[1:-1], percentile_yields_not_backrunners[1:-1]):
#     tab.append((f'{p}%', val_br, val_not_br))
# tab.append(('min', percentile_yields_backrunners[-1], percentile_yields_not_backrunners[-1]))

# print()
# print(tabulate.tabulate(tab, headers=('Percentile', 'Backrunners', 'Non-backrunners')))


# # bin everything by block
# n_bins = 100
# min_block = min(min(blocks_backrunners), min(blocks_not_backrunners))
# max_block = max(max(blocks_backrunners), max(blocks_not_backrunners))

# bin_lefts, sep = np.linspace(min_block, max_block, n_bins, retstep=True, dtype=int)
# print(f'plotting binsize = {sep:.1f} blocks')

# bin_rights = list(bin_lefts)[1:] + [max_block + 1]

# bins_backrunners = [[] for _ in range(n_bins)]
# for y, b in zip(yields_backrunners, blocks_backrunners):
#     bin = int((b - min_block) / sep)
#     if b == bin_rights[bin]:
#         bin += 1
#     assert bin_lefts[bin] <= b < bin_rights[bin], f'expect {bin_lefts[bin]} <= {b} < {bin_rights[bin]}'
#     bins_backrunners[bin].append(y)

# bins_not_backrunners = [[] for _ in range(n_bins)]
# for y, b in zip(yields_not_backrunners, blocks_not_backrunners):
#     bin = int((b - min_block) / sep)
#     if b == bin_rights[bin]:
#         bin += 1
#     assert bin_lefts[bin] <= b < bin_rights[bin], f'expect {bin_lefts[bin]} <= {b} < {bin_rights[bin]}'
#     bins_not_backrunners[bin].append(y)

# marks_backrunners = [(np.percentile(xs, [25, 50, 75]) if len(xs) > 4 else None) for xs in bins_backrunners]
# marks_not_backrunners = [(np.percentile(xs, [25, 50, 75]) if len(xs) > 4 else None) for xs in bins_not_backrunners]


# bin everything by the block_samples
curr.execute(
    '''
    CREATE TEMP TABLE sa_res AS
    SELECT sa.id, bs.priority
    FROM sample_arbitrages_no_fp sa
    JOIN block_samples bs ON bs.start_block <= sa.block_number AND sa.block_number <= bs.end_block;
    '''
)

cnt_backrunners = {}
cnt_not_backrunners = {}
curr.execute(
    '''
        SELECT sa_res.priority, COUNT(*) cnt
        FROM tmp_backrunners tb
        JOIN sa_res ON sa_res.id = tb.sample_arbitrage_id 
        GROUP BY sa_res.priority    
    '''
)
for p, cnt in curr:
    cnt_backrunners[p] = cnt
print(f'Total backrunners? {sum(cnt_backrunners.values()):,}')

curr.execute(
    '''
        SELECT sa_res.priority, COUNT(*) cnt
        FROM tmp_not_backrunners tb
        JOIN sa_res ON sa_res.id = tb.sample_arbitrage_id 
        GROUP BY sa_res.priority    
    '''
)
for p, cnt in curr:
    cnt_not_backrunners[p] = cnt
print(f'Total not backrunners? {sum(cnt_not_backrunners.values()):,}')
print()

cnt_all = {}
curr.execute(
    '''
        SELECT sa_res.priority, COUNT(*) cnt
        from sa_res
        GROUP BY sa_res.priority    
    '''
)
for p, cnt in curr:
    cnt_all[p] = cnt


curr.execute('SELECT start_block, end_block, priority FROM block_samples ORDER BY start_block ASC')

bin_rights = []
cnt_backrunners_lst = []
cnt_not_backrunners_lst = []
cnt_all_lst = []

for sb, eb, p in curr:
    bin_rights.append(eb)
    cnt_backrunners_lst.append(cnt_backrunners.get(p, 0))
    cnt_not_backrunners_lst.append(cnt_not_backrunners.get(p, 0))
    cnt_all_lst.append(cnt_all.get(p, 0))


curr_mainnet = db_mainnet.cursor()

local_conn = sqlite3.connect('./tmp.db')
local_curr = local_conn.cursor()

local_curr.execute('SELECT block_number, timestamp FROM block_timestamps ORDER BY block_number ASC')

ts_block_numbers = []
ts_timestamps = []
for bn, t in local_curr:
    ts_block_numbers.append(bn)
    ts_timestamps.append(t)

interp_timestamps = np.interp(bin_rights, ts_block_numbers, ts_timestamps)

bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]

plt.rcParams["font.family"] = "Times New Roman Cyr"
plt.rcParams["font.size"] = 12

# plot backrunners
buf_ys1 = []
buf_ys2 = []
buf_ys3 = []
for ca, cb, cnb in zip(cnt_all_lst, cnt_backrunners_lst, cnt_not_backrunners_lst):
    buf_ys1.append(0)
    buf_ys2.append(cnb / ca * 100)
    buf_ys3.append((cnb + cb) / ca * 100)

with open('tmp_backrun.csv', mode='w') as fout:
    for ts, ys2, ys3 in zip(interp_timestamps, buf_ys2, buf_ys3):
        fout.write(f'{ts},{ys2},{ys3}\n')

plt.fill_between(bin_rights_datetimes, buf_ys1, buf_ys3, label='Back-runners')
plt.fill_between(bin_rights_datetimes, buf_ys1, buf_ys2, label='Non-backrunners')
plt.title('Back-running Behavior Over Time')
plt.ylabel('Prevalence of Behavior (%)')
plt.xlabel('Date')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.xticks(rotation=75)
plt.legend(loc='lower left')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.savefig('backrunners.png', format='png', dpi=300)
plt.show()



exit()


# plot bounds
buf_xs = []
buf_ys1 = []
buf_ys2 = []
for bin, mark in zip(bin_rights_datetimes, marks_backrunners):
    if mark is not None:
        buf_xs.append(bin)
        buf_ys1.append(mark[0])
        buf_ys2.append(mark[2])
plt.fill_between(buf_xs, buf_ys1, buf_ys2, alpha=0.2)

buf_xs = []
buf_ys1 = []
buf_ys2 = []
for bin, mark in zip(bin_rights_datetimes, marks_not_backrunners):
    if mark is not None:
        buf_xs.append(bin)
        buf_ys1.append(mark[0])
        buf_ys2.append(mark[2])
plt.fill_between(buf_xs, buf_ys1, buf_ys2, alpha=0.2)

# plot medians
buf_xs = []
buf_ys = []
for bin, mark in zip(bin_rights_datetimes, marks_backrunners):
    if mark is not None:
        buf_xs.append(bin)
        buf_ys.append(mark[1])

plt.plot(buf_xs, buf_ys, label='backrunners')

buf_xs = []
buf_ys = []
for bin, mark in zip(bin_rights_datetimes, marks_not_backrunners):
    if mark is not None:
        buf_xs.append(bin)
        buf_ys.append(mark[1])

plt.plot(buf_xs, buf_ys, label='non-backrunners')


plt.legend()
plt.ylabel('Percentage yield')
plt.xlabel('Timestamp')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

