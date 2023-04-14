from collections import deque
import collections
import itertools
import pickle
import sys
import time
import psycopg2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors
import tabulate
import web3
import scipy.stats
import networkx as nx

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

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams['hatch.linewidth'] = 0.2

campaign_durations = collections.defaultdict(dict)
campaign_periods = collections.defaultdict(dict)
campaign_max_profits = collections.defaultdict(dict)
blocks_to_campaigns = collections.defaultdict(list)

print('loading...')

with open('tmp_campaign_durations_out.csv') as fin:
    for line in fin:
        id, gas_pricer, dur, start, end, max_profit = line.strip().split(',')
        id = int(id)

        campaign_durations[gas_pricer][id] = int(dur)
        campaign_periods[gas_pricer][id] = (int(start), int(end))
        campaign_max_profits[gas_pricer][id] = int(max_profit)

        if gas_pricer == 'median':
            blocks_to_campaigns[int(start)].append(id)
            blocks_to_campaigns[int(end) + 1].append(id)


tot_eth = 0
with open('tmp_selected_ids_sample.csv') as fin:
    for line in fin:
        id_ = int(line.strip())

        tot_eth += campaign_max_profits['median'][id_]

print(f'Total ETH {tot_eth / (10 ** 18):,}')

exit()

with open('tmp_g.dat', mode='rb') as fin:
    g = pickle.load(fin)

print('loaded')

selected_ids = set()
for c in nx.connected_components(g):
    c = set(c)
    print('.', end='')
    sys.stdout.flush()
    while len(c) > 1:
        if len(c) == 1:
            selected_ids.add(c)
            break

        c_lst = list(c)
        vals = [campaign_max_profits['median'][id_] for id_ in c_lst]
        max_id, max_val = max(zip(c_lst, vals), key=lambda x: x[1])

        selected_ids.add(max_id)
        for neighbor in list(g.neighbors(max_id)):
            if neighbor in c:
                c.remove(neighbor)
        c.remove(max_id)

print(f'Selected {len(selected_ids):,}')

with open('tmp_selected_ids_sample.csv', mode='w') as fout:
    for id_ in selected_ids:
        fout.write(f'{id_}\n')

exit()

if False:
    curr.execute(
        '''
        UPDATE candidate_arbitrage_campaigns cac
        SET exchanges = (
            SELECT exchanges
            FROM candidate_arbitrage_campaign_member cacm
            JOIN candidate_arbitrages ca on ca.id = cacm.candidate_arbitrage_id
            WHERE cacm.candidate_arbitrage_campaign = cac.id
            LIMIT 1
        )
        WHERE cac.gas_pricer = 'median'
        '''
    )
    input(f'ENTER to commit exchanges to {curr.rowcount:,} rows')
    db.commit()

# # get campaign exchanges
# campaign_exchanges = {}
# for i, id_ in enumerate(campaign_durations['median'].keys()):
#     if i % 1_000 == 0:
#         print(f'{i / len(campaign_durations["median"]) * 100:.2f}%')

#     curr.execute(
#         '''
#         SELECT exchanges
#         FROM candidate_arbitrage_campaigns cac
#         JOIN candidate_arbitrage_campaign_member cacm ON cac.id = cacm.candidate_arbitrage_campaign
#         JOIN candidate_arbitrages ca on ca.id = cacm.candidate_arbitrage_id
#         WHERE cac.id = %s
#         LIMIT 1
#         ''',
#         (id_,)
#     )
#     (exchanges,) = curr.fetchone()
#     exchanges = set(x.tobytes() for x in exchanges)
#     campaign_exchanges[id_] = exchanges
# print('loaded exchanges')


g = nx.Graph()
for id_, max_profit in campaign_max_profits['median'].items():
    g.add_node(id_, weight=max_profit)

print('building graph')

# curr.execute(
#     '''
#     SELECT c1.id, c2.id
#     FROM candidate_arbitrage_campaigns c1
#     JOIN candidate_arbitrage_campaigns c2 ON
#         int4range(c1.block_number_start, c1.block_number_end + 1) &&
#         int4range(c2.block_number_start, c2.block_number_end + 1)
#         AND
#         c1.id != c2.id
#         AND
#         c1.exchanges && c2.exchanges
#     WHERE c1.gas_pricer = 'median' AND c2.gas_pricer = 'median'
#     '''
# )


if True:
    # compute the conflict-free set of arbitrages
    active_campaigns = set()
    for i, block_number in enumerate(sorted(blocks_to_campaigns.keys())):
        if i % 10 == 0:
            print(f'{block_number:,} active_campaigns {len(active_campaigns)}')
        changed_ids = blocks_to_campaigns[block_number]
        
        # update active_campaigns
        new_ids = list()
        for id_ in changed_ids:
            start, end = campaign_periods['median'][id_]
            if block_number == start:
                active_campaigns.add(id_)
                new_ids.append(id_)
            else:
                assert block_number == end + 1
                active_campaigns.remove(id_)

        # draw any conflicts arising with new changed id and others
        curr.execute(
            '''
            SELECT c1.id, c2.id
            FROM (SELECT * from candidate_arbitrage_campaigns where id = any (%s)) c1
            JOIN (SELECT * from candidate_arbitrage_campaigns where id = any (%s)) c2
                ON c1.exchanges && c2.exchanges and c1.id != c2.id
            ''',
            (
                new_ids,
                list(active_campaigns),
            )
        )
        for i1, i2 in curr:
            if not g.has_edge(i1, i2):
                g.add_edge(i1, i2)

        # for i1, i2 in itertools.product(changed_ids, active_campaigns):
        #     if i1 == i2:
        #         continue

        #     if g.has_edge(i1, i2):
        #         continue

        #     curr.execute(
        #         '''
        #         SELECT c1.exchanges && c2.exchanges
        #         FROM (SELECT exchanges from candidate_arbitrage_campaigns where id = %s) c1
        #         CROSS JOIN (SELECT exchanges from candidate_arbitrage_campaigns where id = %s) c2
        #         ''',
        #         (i1, i2)
        #     )
        #     (overlap,) = curr.fetchone()
        #     if overlap:
        #         g.add_edge(i1, i2)

    with open('tmp_g.dat', mode='wb') as fout:
        pickle.dump(g, fout)
    print('wrote to pickle')


    print('computed graph')

    exit()


if False:
    tab = []
    for gas_pricer in campaign_durations.keys():
        marks = [25, 50, 75]
        percs = np.percentile(list(campaign_durations[gas_pricer].values()), marks)
        tab.append((gas_pricer, *percs))

    print(tabulate.tabulate(tab, headers=['Gas Pricer', '25th', 'median', '75th']))

    plt.hist(campaign_durations['median'].values())
    plt.show()

xs = []
ys = []
for id_ in campaign_durations['median'].keys():
    xs.append(campaign_max_profits['median'][id_] / (10 ** 18))
    ys.append(campaign_durations['median'][id_])

max_dur = np.percentile(list(campaign_durations['median'].values()), 85)
max_profit = np.percentile(list(campaign_max_profits['median'].values()), 85) / (10 ** 18)

# plt.scatter(xs, ys, c='k', s=1)
# plt.xlim(left=0, right=max_profit)
h = plt.hist2d(xs, ys, range=[[0, 0.05], [1, 20]], bins=20, cmap='inferno', norm=matplotlib.colors.LogNorm())
plt.colorbar(h[3], label='Count')
plt.title('Distribution of Arbitrages by Duration and Profit')
plt.xlabel('ETH Profit')
plt.ylabel('Opportunity Duration (blocks)')
plt.yticks(list(range(1, 21, 2)))
plt.savefig('arb_duration_2dhist.png', format='png', dpi=300)
plt.show()
