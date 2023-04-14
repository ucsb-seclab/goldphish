from collections import deque
import collections
import datetime
from genericpath import isfile
import itertools
import os
import sys
import psycopg2
import psycopg2.extras
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import scipy.stats
import sqlite3
import networkx as nx

from common import setup_only_uniswap_tables, setup_weth_arb_tables

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


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams['hatch.linewidth'] = 0.2

g = nx.Graph()
campaign_max_profits = {}
campaign_bounds = {}
campaign_exchanges = {}

with open('lc_campaign_bounds.csv') as fin:
    for line in fin:
        id_, start, end, max_profit = line.strip().split(',')
        id_ = int(id_)
        start = int(start)
        end = int(end)
        max_profit = int(max_profit)
        campaign_max_profits[id_] = max_profit
        campaign_bounds[id_] = (start, end)
        g.add_node(id_, weight=max_profit)

curr.execute(
    '''
    SELECT id, exchanges
    FROM top_candidate_arbitrage_campaigns tcac
    '''
)
for id_, exchanges in curr:
    exchanges = set(x.tobytes() for x in exchanges)
    campaign_exchanges[id_] = exchanges

# # very very simple n^2 conflict-id, sorry
# curr.execute(
#     '''
#     CREATE TEMP TABLE campaign_ranges (
#         id integer not null,
#         start_block integer not null,
#         end_block integer not null
#     );
#     '''
# )

# psycopg2.extras.execute_values(
#     curr,
#     '''
#     INSERT INTO campaign_ranges (id, start_block, end_block) VALUES %s
#     ''',
#     [(id_, start, end) for id_, (start, end) in campaign_bounds.items()]
# )
# print('inserted, making indexes')
# curr.execute('CREATE INDEX idx_tmp_cr_s ON campaign_ranges (start_block)')
# curr.execute('CREATE INDEX idx_tmp_cr_e ON campaign_ranges (end_block)')

# print('querying')
# curr.execute(
#     '''
#     SELECT cr1.id, cr2.id
#     FROM campaign_ranges cr1
#     JOIN campaign_ranges cr2 ON cr1.id != cr2.id AND
#         cr1.id < cr2.id AND
#         int4range(cr1.start_block, cr1.end_block + 1) && int4range(cr2.start_block, cr2.end_block + 1)
#     '''
# )

selected_ids = set()

if not os.path.isfile('tmp_selected_ids.csv'):
    ids = sorted(campaign_max_profits.keys())
    pairs = len(ids) * (len(ids) - 1) // 2
    for i, (i1, i2) in enumerate(itertools.combinations(ids, 2)):
        if i % 10_000 == 0:
            print(f'{i / pairs * 100:.2f}%')
        start1, end1 = campaign_bounds[i1]
        start2, end2 = campaign_bounds[i2]

        overlap = False
        if start1 <= start2 <= end1:
            overlap = True
        elif start1 <= end2 <= end1:
            overlap = True
        elif start2 <= start1 <= end2:
            overlap = True
        elif start2 <= end1 <= end2:
            overlap = True

        if overlap and len(campaign_exchanges[i1].intersection(campaign_exchanges[i2])) > 0:
        # if len(campaign_exchanges[i1].intersection(campaign_exchanges[i2])) > 0:
            # conflict
            g.add_edge(i1, i2)

    print('computing connected components')

    components = list(nx.connected_components(g))

    for c in components:
        c = set(c)
        print('.', end='')
        sys.stdout.flush()
        while len(c) > 1:
            if len(c) == 1:
                selected_ids.add(c)
                break

            c_lst = list(c)
            vals = [campaign_max_profits[id_] for id_ in c_lst]
            max_id, max_val = max(zip(c_lst, vals), key=lambda x: x[1])

            selected_ids.add(max_id)
            for neighbor in list(g.neighbors(max_id)):
                if neighbor in c:
                    c.remove(neighbor)
            c.remove(max_id)

    print(f'Selected {len(selected_ids):,}')

    with open('tmp_selected_ids.csv', mode='w') as fout:
        for id_ in selected_ids:
            fout.write(f'{id_}\n')
else:
    with open('tmp_selected_ids.csv') as fin:
        for line in fin:
            id_ = int(line.strip())
            selected_ids.add(id_)


# sum their Ether
tot_eth_val = sum(campaign_max_profits[id_] for id_ in selected_ids)

print(f'Total max: {tot_eth_val / (10 ** 18):,.0f} ETH')

# sum their USD value
usd_val = 0
for id_ in selected_ids:
    start_block = campaign_bounds[id_][0]
    curr.execute('SELECT eth_price_usd FROM eth_price_blocks where block_number = %s', (start_block,))
    (price,) = curr.fetchone()
    usd_val += float(price) * (campaign_max_profits[id_] / (10 ** 18))

print(f'Max usd val: {usd_val}')

exit()


durations = []
with open('lc_campaign_durations.csv') as fin:
    for line in fin:
        _, d = line.strip().split(',')
        d = int(d)
        durations.append(d)

marks = [25, 50, 75, 95]
percs = np.percentile(durations, marks)
tab = zip(marks, percs)
print(tabulate.tabulate(tab, headers=['Percentile', 'Val']))

plt.hist(durations, bins=30)
plt.show()

exit()


profits = []

with open('tmp_lca_results.csv') as fin:
    for line in fin:
        _, bn, p = line.strip().split(',')
        bn = int(bn)
        p = int(p) / (10 ** 18)
        if p < 1:
            continue
        profits.append(p)

plt.hist(profits, bins=30)
plt.yscale('log')
plt.title('Profit After Fees\nSuccessful Large Potential Arbitrages')
plt.ylabel('Count')
plt.xlabel('ETH Profit')
plt.tight_layout()
plt.savefig('profit_distribution_large_arbs.png', format='png', dpi=300)
plt.show()

# exit()

if False:

    block_numbers = []
    profit_after_fees = []

    with open('large_profits_per_block.csv') as fin:
        for line in fin:
            bn, p = line.strip().split(',')
            p = float(p)
            if p < 1:
                continue
            block_numbers.append(int(bn))
            profit_after_fees.append(float(p))

    local_conn = sqlite3.connect('./tmp.db')
    local_curr = local_conn.cursor()

    local_curr.execute('SELECT block_number, timestamp FROM block_timestamps ORDER BY block_number ASC')

    ts_block_numbers = []
    ts_timestamps = []
    for bn, t in local_curr:
        ts_block_numbers.append(bn)
        ts_timestamps.append(t)

    unique_blocks = sorted(set(block_numbers))
    interp_timestamps = np.interp(unique_blocks, ts_block_numbers, ts_timestamps)
    block_to_interp_ts = {
        b: datetime.datetime.fromtimestamp(ts)
        for b, ts in zip(unique_blocks, interp_timestamps)
    }

    timestamps = [block_to_interp_ts[bn] for bn in block_numbers]

    assert len(timestamps) == len(profit_after_fees)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams['hatch.linewidth'] = 0.2

    plt.scatter(timestamps, profit_after_fees, s=0.5, color='black')
    plt.xlabel('Date')
    plt.ylabel('ETH profit')
    plt.yscale('log')
    plt.title('Attainable ETH Profit Per Block\nLarge Potential Arbitrages Only\nMedian Gas-Price Oracle')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.show()

