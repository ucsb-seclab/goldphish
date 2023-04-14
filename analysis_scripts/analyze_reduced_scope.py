"""
Estimates profitability with reduced exchange scope
"""

import pickle
import sys
import typing
import collections
import datetime
import math
import sqlite3
import psycopg2
import psycopg2.extras
import tabulate
import os
import numpy as np
import matplotlib.pyplot as plt
import web3
import web3.contract
import random
import networkx as nx

from common import gen_false_positives, setup_backrun_arb_tables, setup_weth_arb_tables

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


# select the top profitable exchanges
if not os.path.exists('_selected_exchanges.pickle'):

    print('limiting to exchanges....')

    curr2 = db.cursor()

    curr.execute(
        '''
        SELECT lca.candidate_arbitrage_id
        FROM large_candidate_arbitrages lca
        JOIN top_candidate_arbitrage_relay_results tcarr
            ON tcarr.candidate_arbitrage_id = lca.candidate_arbitrage_id
        WHERE shoot_success
        ORDER BY real_profit_before_fee DESC
        '''
    )
    selected_exchanges: typing.Set[bytes] = set()
    n_candidates = 0
    while len(selected_exchanges) < 100:
        (id_,) = curr.fetchone()
        curr2.execute(
            '''
            SELECT ca.exchanges
            FROM candidate_arbitrages ca
            WHERE id = %s AND EXISTS(
                SELECT FROM block_samples
                WHERE start_block <= ca.block_number AND ca.block_number <= end_block
                AND priority < 30
            )
            ''',
            (id_,)
        )

        if curr2.rowcount < 1:
            # not in sample range?
            continue

        n_candidates += 1
        (exchanges,) = curr2.fetchone()
        for e in exchanges:
            selected_exchanges.add(e.tobytes())
        print(f'{len(selected_exchanges)}')

    print(f'selected {len(selected_exchanges)} from {n_candidates} candidates')

    with open('_selected_exchanges.pickle', mode='wb') as fout:
        pickle.dump(selected_exchanges, fout)
else:
    with open('_selected_exchanges.pickle', mode='rb') as fin:
        selected_exchanges = pickle.load(fin)

campaign_durations = collections.defaultdict(dict)
campaign_periods = collections.defaultdict(dict)
campaign_max_profits = collections.defaultdict(dict)
blocks_to_campaigns = collections.defaultdict(list)

print('loading campaign durations...')

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

print('loading campaign exchanges')
curr.execute(
    '''
    SELECT id
    FROM candidate_arbitrage_campaigns
    WHERE exchanges <@ %s
    ''',
    (list(selected_exchanges),)
)
campaign_in_scope = set(x for (x,) in curr)
curr.execute('SELECT COUNT(*)')
n_campaigns = len(campaign_max_profits['median'])
print(f'Have {len(campaign_in_scope) / n_campaigns * 100:.2f}% campaigns in scope ')

if not os.path.exists('_reduced_scope_g.dat'):
    g = nx.Graph()
    for id_, max_profit in campaign_max_profits['median'].items():
        g.add_node(id_, weight=max_profit)

    # compute the conflict-free set of arbitrages
    active_campaigns = set()

    for i, block_number in enumerate(sorted(blocks_to_campaigns.keys())):
        if i % 10 == 0:
            print(f'{block_number:,} active_campaigns {len(active_campaigns)}')
        changed_ids = set(blocks_to_campaigns[block_number]).intersection(campaign_in_scope)

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

        if len(active_campaigns) <= 1:
            continue

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

    with open('_reduced_scope_g.dat', mode='wb') as fout:
        pickle.dump(g, fout)
    print('wrote to pickle')


    print('computed graph')
else:
    with open('_reduced_scope_g.dat', mode='rb') as fin:
        g = pickle.load(fin)

selected_ids = set()
if not os.path.exists('__reduced_selected_ids.csv'):
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

    with open('__reduced_selected_ids.csv', mode='w') as fout:
        for id_ in selected_ids:
            fout.write(f'{id_}\n')
else:
    with open('__reduced_selected_ids.csv') as fin:
        for line in fin:
            selected_ids.add(int(line.strip()))

tot_eth = 0
for id_ in selected_ids:
    tot_eth += campaign_max_profits['median'][id_]


print(f'Total ETH {tot_eth / (10 ** 18):,}')


exit()

curr.execute(
    '''
    SELECT cacm.candidate_arbitrage_id, cacm.profit_after_fee_wei
    FROM candidate_arbitrages_mev_selected cams
    JOIN candidate_arbitrage_campaign_member cacm ON cacm.candidate_arbitrage_id = cams.candidate_arbitrage_id
    '''
)

candidate_to_profit = {}
n_dupe = 0
for id_, profit_wei in curr:
    if id_ in candidate_to_profit:
        n_dupe += 1
    candidate_to_profit[id_] = int(profit_wei)

# sort candidates by most profitable
most_profitable = sorted([(v, id_) for id_, v in candidate_to_profit.items()], reverse=True)

print(f'Have {len(candidate_to_profit):,} selected arbitrages')

# walk until we have 100 exchanges
selected_exchanges = set()
for n_candidates, (_, candidate_id) in enumerate(most_profitable):
    curr.execute(
        '''
        SELECT exchanges FROM candidate_arbitrages WHERE id = %s
        ''',
        (candidate_id,)
    )
    (exchanges,) = curr.fetchone()
    for e in exchanges:
        selected_exchanges.add(e.tobytes())

    if len(selected_exchanges) >= 100:
        break

print('selected', len(selected_exchanges), 'exchanges from', n_candidates, 'candidates')

# reduce down to just those
filtered_candidates = set()
curr.execute(
    '''
    SELECT exchanges, id
    FROM candidate_arbitrages
    WHERE EXISTS (
        SELECT FROM candidate_arbitrages_mev_selected cacm
        WHERE cacm.candidate_arbitrage_id = id
    ) AND EXISTS (
        SELECT FROM block_samples
        WHERE
            start_block <= block_number AND
            block_number <= end_block AND
            priority <= 29
    )
    ''',
    (candidate_id,)
)

for i, (exchanges, candidate_id) in enumerate(curr):
    if i % 10_000 == 0:
        print(f'{i/len(candidate_to_profit)*100:.2f}%')

    these_exchanges = set()
    for e in exchanges:
        these_exchanges.add(e.tobytes())

    if these_exchanges.issubset(selected_exchanges):
        filtered_candidates.add(candidate_id)


filtered_profit = 0
for id_ in filtered_candidates:
    filtered_profit += candidate_to_profit[id_]

print(f'Ended up with {len(filtered_candidates):,} candidates after filter')


print('Daily profit ETH', sum(candidate_to_profit.values()) / 30 / (10 ** 18))
print('Filtered daily profit ETH', filtered_profit / 30 / (10 ** 18))

