#
# Some re-calculations to square up the numbers
# now that we know that arbitrages are used for sandwich attacking.
#

from collections import deque
import itertools
import time
import psycopg2
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import sqlite3
import scipy.stats
import datetime

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

# gen_false_positives(curr)


# curr.execute('SET TRANSACTION READ ONLY')
curr.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')

if True:
    # print out top arbitrage profiters
    setup_weth_arb_tables(curr)

    # sum profit after fees in WETH and then USD
    curr.execute(
        '''
        SELECT
            SUM(net_profit),
            SUM(net_profit / power(10, 18) * eth_price_usd)
        FROM tmp_weth_arbs twa
        JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld
            WHERE jld.sample_arbitrage_id = twa.id
        )
        ''',
    )
    tot_eth_profit, tot_usd_profit = curr.fetchone()
    print(f'Total ETH profit: {tot_eth_profit / (10 ** 18)} ETH')
    print(f'Total USD profit: {tot_usd_profit:,} USD')


    curr.execute(
        '''
        SELECT *
        FROM (
            SELECT
                shooter,
                SUM(net_profit) sum_eth,
                SUM(net_profit / power(10, 18) * eth_price_usd) sum_usd
            FROM tmp_weth_arbs twa
            JOIN sample_arbitrages_no_fp sa ON sa.id = twa.id
            JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
            WHERE NOT EXISTS(
                SELECT FROM arb_sandwich_detections jld
                WHERE jld.sample_arbitrage_id = twa.id
            )
            GROUP BY shooter
        ) x
        ORDER BY sum_eth DESC LIMIT 10
        '''
    )

    tab = []
    for shooter, eth, usd in curr:
        tab.append((
            web3.Web3.toChecksumAddress(shooter.tobytes().hex()),
            eth / 10 ** 18,
            eth / tot_eth_profit * 100,
            usd
        ))
    print('Top arbitrage shooters...')
    print(tabulate.tabulate(tab, floatfmt=',f', headers=['Address', 'ETH', '(%)', 'USD']))
    exit()


if True:
    # find most profitable arbitrages
    setup_weth_arb_tables(curr)
    curr.execute(
        '''
        SELECT txn_hash, net_profit
        FROM tmp_weth_arbs twa
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld
            WHERE jld.sample_arbitrage_id = twa.id
        )
        ORDER BY net_profit DESC
        LIMIT 10
        '''
    )
    for txn_hash, profit in curr:
        print(txn_hash.tobytes().hex(), profit / (10 ** 18))
    exit()


if False:
    if True:
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
            WHERE sabd.rerun_exactly = true AND sa.n_cycles = 1 and block_number < 15628035
                and not EXISTS(
                    SELECT FROM arb_sandwich_detections jld
                    WHERE jld.sample_arbitrage_id = sa.id
                );

            INSERT INTO tmp_backrunners (sample_arbitrage_id, txn_hash)
            SELECT sample_arbitrage_id, sa.txn_hash
            FROM sample_arbitrage_backrun_detections sabd
            JOIN sample_arbitrages_no_fp sa ON sa.id = sabd.sample_arbitrage_id
            WHERE (sabd.rerun_reverted = true OR sabd.rerun_no_arbitrage = true) AND sa.n_cycles = 1 and block_number < 15628035
                and not EXISTS(
                        SELECT FROM arb_sandwich_detections jld
                        WHERE jld.sample_arbitrage_id = sa.id
                    );
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

    # how many have we rerun
    curr.execute(
        '''
        SELECT COUNT(*)
        FROM sample_arbitrage_backrun_detections sabd
        JOIN sample_arbitrages_no_fp sa on sa.id = sabd.sample_arbitrage_id
        WHERE n_cycles = 1 and 15628035 > block_number
        '''
    )
    (n_backrun_reorders_run,) = curr.fetchone()
    print(f'Have {n_backrun_reorders_run:,} arbitrages re-run at top-of-block')
    print()

    curr.execute(
        '''
        SELECT COUNT(*)
        FROM sample_arbitrages_no_fp sa
        WHERE n_cycles = 1 and 15628035 > block_number and EXISTS(
            SELECT FROM arb_sandwich_detections jld
            WHERE jld.sample_arbitrage_id = sa.id
        )
        '''
    )
    (n_sandwich,) = curr.fetchone()
    print(f'Have {n_sandwich:,} sandwich')


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
        JOIN sample_arbitrages_no_fp sa on sa.id = sabd.sample_arbitrage_id
        WHERE n_cycles = 1 and 15628035 > block_number and not EXISTS(
            SELECT FROM arb_sandwich_detections jld
            WHERE jld.sample_arbitrage_id = sa.id
        )
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
        ('Sandwich',         n_sandwich),
    ]

    tot = sum(x[1] for x in reports)
    print(f'tot {tot:,}')

    reports = sorted(reports, key=lambda x: x[1], reverse=True)
    reports_with_percent = [(s, n, f'{n / n_backrun_reorders_run * 100:.1f}%') for s, n in reports]

    print()
    print(tabulate.tabulate(reports_with_percent, headers=['Reorder result', 'Count', 'Percent']))


if False:
    # count how many sandwich arbitrages there are
    print('Counting number of sandwich-arbs')
    setup_weth_arb_tables(curr)

    curr.execute(
        '''
        SELECT COUNT(*)
        FROM sample_arbitrages_no_fp sa
        WHERE EXISTS(
            SELECT FROM arb_sandwich_detections jld
            WHERE jld.sample_arbitrage_id = sa.id
        ) and 15628035 > block_number
        '''
    )
    print(f'Found {curr.fetchone()[0]:,} sandwich arbitrages')

    curr.execute(
        '''
        SELECT
            SUM(net_profit),
            SUM(net_profit / power(10, 18) * eth_price_usd)
        FROM tmp_weth_arbs twa
        JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
        WHERE EXISTS(
            SELECT FROM arb_sandwich_detections jld
            WHERE jld.sample_arbitrage_id = twa.id
        ) and 15628035 > twa.block_number
        ''',
    )
    tot_eth_profit, tot_usd_profit = curr.fetchone()
    print(f'Total (fake, sandwich) ETH profit: {tot_eth_profit / (10 ** 18):,.2f} ETH')
    print(f'Total (fake, sandwich) USD profit: {tot_usd_profit:,.2f} ETH')
    print()


if False:
    # calculate total amount of profit made, in WETH and USD
    print('Computing total profit in WETH and USD')
    setup_weth_arb_tables(curr)

    # sum profit after fees in WETH and then USD
    curr.execute(
        '''
        SELECT
            SUM(net_profit),
            SUM(net_profit / power(10, 18) * eth_price_usd)
        FROM tmp_weth_arbs twa
        JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld
            WHERE jld.sample_arbitrage_id = twa.id
        )
        ''',
    )
    tot_eth_profit, tot_usd_profit = curr.fetchone()
    print(f'Total ETH profit: {tot_eth_profit / (10 ** 18):,.2f} ETH')
    print(f'Total USD profit: {tot_usd_profit:,.2f} ETH')
    print()

if False:
    # draw the backrun over time graph
    #
    # Produce stacked graph
    #

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams['hatch.linewidth'] = 0.2


    local_conn = sqlite3.connect('./tmp.db')
    local_curr = local_conn.cursor()

    # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    local_curr.execute('SELECT block_number, timestamp FROM block_timestamps ORDER BY block_number ASC')

    ts_block_numbers = []
    ts_timestamps = []
    for bn, t in local_curr:
        ts_block_numbers.append(bn)
        ts_timestamps.append(t)


    profits_s = {}
    profits_br = {}
    profits_o = {}
    profits_nbr = {}

    with open('tmp_marks_sandwich.csv') as fin:
        for line in fin:
            cat, bn, profit = line.strip().split(',')
            bn = int(bn)
            profit = int(profit)

            if cat == 'b':
                profits_br[bn] = profit
            elif cat == 'nb':
                profits_nbr[bn] = profit
            elif cat == 's':
                profits_s[bn] = profit
            elif cat == 'o':
                profits_o[bn] = profit

    all_blocks = sorted(set(profits_s.keys()).union(profits_br.keys()).union(profits_nbr.keys()))
    profits_all = {} # unused i think??? kept just in case we need to switch back
    profits_all_ex_sandwich = {}
    for block in all_blocks:
        profits_all[block] = (
            profits_s.get(block, 0) +
            profits_br.get(block, 0) + 
            profits_nbr.get(block, 0) +
            profits_o.get(block, 0)
        )
        profits_all_ex_sandwich[block] = profits_all[block] - profits_s.get(block, 0)


    interp_timestamps = np.interp(all_blocks, ts_block_numbers, ts_timestamps)

    bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]

    buf_ys1 = []
    buf_ys2 = []
    buf_ys3 = []
    # buf_ys4 = []

    xs = []
    for i, (bin, block) in enumerate(zip(bin_rights_datetimes, all_blocks)):
        if i % 4 != 0:
            continue
        xs.append(bin)
        buf_ys1.append(0)
        buf_ys2.append(profits_nbr.get(block, 0) / profits_all_ex_sandwich.get(block, 1) * 100)
        buf_ys3.append((profits_nbr.get(block, 0) + profits_br.get(block, 0)) / profits_all_ex_sandwich.get(block, 1) * 100)
        # buf_ys2.append(profits_nbr.get(block, 0) / profits_all.get(block, 1) * 100)
        # buf_ys3.append((profits_nbr.get(block, 0) + profits_br.get(block, 0)) / profits_all.get(block, 1) * 100)
        # buf_ys4.append((profits_s.get(block, 0) + profits_nbr.get(block, 0) + profits_br.get(block, 0)) / profits_all.get(block, 1) * 100)

    plt.title('Back-running Percent of Profit Over Time')
    plt.fill_between([min(xs), max(xs)], [0], [100], color='#bbbbbb', label='Unknown')


    # plt.fill_between(xs, buf_ys1, buf_ys4, label='Sandwich', hatch='//')
    plt.fill_between(xs, buf_ys1, buf_ys3, label='Back-running', hatch='//')
    plt.fill_between(xs, buf_ys1, buf_ys2, label='Non-back-running', hatch='\\\\')
    # ax1.set_title('Back-running Behavior Over Time\nBy Profit')
    plt.ylabel('Share of Profit (%)')
    # ax1.xlabel('Date')
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks(rotation=75)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,0]


    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower left')
    plt.margins(x=0, y=0)
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig('backrun_ex_sandwich.png', format='png', dpi=300)
    plt.show()

    ys_not_sampled = [(profits_all.get(block, 0) - profits_s.get(block, 0)) / (10 ** 18) for block in all_blocks]
    ys = []
    xs = []
    for i, (x, y) in enumerate(zip(bin_rights_datetimes, ys_not_sampled)):
        xs.append(x)
        ys.append(y)
    plt.plot(xs, ys)
    plt.xticks(rotation=75)
    plt.title('Daily profit')
    plt.tight_layout()
    # plt.savefig('sandwich_daily_profit.png', format='png', dpi=300)
    plt.show()


    exit()


    interp_timestamps = []
    buf_ys1 = []
    buf_ys2 = []
    buf_ys3 = []

    with open('tmp_backrun.csv') as fin:
        for line in fin:
            ts, ys2, ys3 = line.strip().split(',')
            interp_timestamps.append(float(ts))
            buf_ys1.append(0)
            buf_ys2.append(float(ys2))
            buf_ys3.append(float(ys3))

    bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]

    ax2.fill_between([min(bin_rights_datetimes), max(bin_rights_datetimes)], [0], [100], color='#bbbbbb', label='Unknown')


    ax2.fill_between(bin_rights_datetimes, buf_ys1, buf_ys3, label='Back-running', hatch='//')
    ax2.fill_between(bin_rights_datetimes, buf_ys1, buf_ys2, label='Not back-running', hatch='\\\\')
    ax2.set_ylabel('Count (%)')
    ax2.set_xlabel('Date')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # for spine in plt.gca().spines.values():
    #     spine.set_visible(False)
    plt.xticks(rotation=75)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,0]
    ax2.margins(x=0, y=0)

    fig.suptitle('Back-running Behavior Over Time')
    ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right')#, ncol=3, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout()
    # plt.tight_layout()
    plt.savefig('backrunners.png', format='png', dpi=300)
    plt.show()


