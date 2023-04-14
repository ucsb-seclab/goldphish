from collections import deque
import collections
import datetime
from genericpath import isfile
import os
import psycopg2
import psycopg2.extras
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import scipy.stats
import sqlite3

from common import setup_only_uniswap_tables, setup_weth_arb_tables

SAMPLE_SIZE = 5_000
print(f'SAMPLING WITH SIZE {SAMPLE_SIZE}')

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

#
# Compute success rate
#
curr.execute(
    '''
    SELECT SUM(CASE WHEN shoot_success THEN 1 ELSE 0 END), COUNT(*)
    from lca_dedup lca
    join top_candidate_arbitrage_relay_results tcarr on tcarr.candidate_arbitrage_id = lca.candidate_arbitrage_id
    '''
)
n_success, n_total = curr.fetchone()
print(f'Top arbitrages have {n_success:,} of {n_total} ({n_success / n_total * 100:.2f}%) relay success rate')


curr.execute('SELECT COUNT(*) FROM lca_dedup')
(n_dedup_total,) = curr.fetchone()

if False:
    #
    # Compute failure diagnosis reasons
    #

    # curr.execute('SET max_parallel_workers_per_gather = 40')

    curr.execute(
        '''
        SELECT
            SUM(CASE WHEN broken_token THEN 1 ELSE 0 END),
            SUM(CASE WHEN incompatible THEN 1 ELSE 0 END),
            SUM(CASE WHEN other THEN 1 ELSE 0 END),
            SUM(CASE WHEN bad_exchange THEN 1 ELSE 0 END),
            SUM(CASE WHEN no_after_fee THEN 1 ELSE 0 END),
            SUM(CASE WHEN interference THEN 1 ELSE 0 END),
            SUM(CASE WHEN not_avail THEN 1 ELSE 0 END)
        FROM (
            SELECT
                failure_reason LIKE 'Broken token: %' broken_token,
                failure_reason LIKE 'incompatible %' incompatible,
                failure_reason LIKE 'Other%' other,
                failure_reason LIKE 'Bad exchange%' bad_exchange,
                failure_reason LIKE 'No arbitrage after fee' no_after_fee,
                failure_reason LIKE 'token-exchange interference' interference,
                failure_reason LIKE 'Balancer v1: Token % not available %' not_avail,
                *
            FROM top_candidate_arbitrage_relay_results tca
            JOIN lca_dedup lc ON lc.candidate_arbitrage_id = tca.candidate_arbitrage_id
        ) x
        '''
    )

    nums = curr.fetchone()
    descs = [
        'Broken token',
        'Incompatible token',
        'Other',
        'Bad exchange',
        'No arbitrage after fee',
        'Token-exchange interference',
        'Balancer v1 thing',
    ]
    assert len(nums) == len(descs)
    n_failures = sum(nums)
    tab = sorted(zip(descs, nums), key=lambda x: x[1], reverse=True)

    print()
    print(tabulate.tabulate(tab, headers=['category', 'count']))
    print()

    for d, n in tab:
        print(f'{d} & ${n:,}$ & ${round(n / n_dedup_total * 100, 1):.1f}\\%$ \\\\')
    print('\\midrule')
    print(f'All & ${n_failures:,}$ & ${round(n_failures / n_dedup_total * 100, 1):.1f}\\%$ \\\\')


# compute number of campaigns

curr.execute(
    '''
    SELECT COUNT(*) FROM top_candidate_arbitrage_campaigns WHERE removed is not True
    '''
)
(n_campaigns,) = curr.fetchone()
print(f'Have {n_campaigns:,} top candidate arbitrage campaigns')


# properties of the campaigns

# 
# Subtract gas usage from top arbs
#

print('Ingesting gas price oracle values')

if not os.path.isfile('tmp_gpo.csv'):
    print('consuming from postgresql')
    with open('tmp_gpo.csv', mode='w') as fout:
        curr.execute(
            '''
            SELECT niche, block_number, gas_price_median FROM naive_gas_price_estimate
            ''',
        )
        for niche, bn, gpm in curr:
            fout.write(f'{niche},{bn},{gpm}\n')

gas_price_blocks_by_niche = collections.defaultdict(lambda: [])
gas_price_by_niche = collections.defaultdict(lambda: [])

with open('tmp_gpo.csv') as fin:
    for line in fin:
        niche, bn, gpm = line.strip().split(',')
        bn = int(bn)
        gpm = int(gpm)
        gas_price_blocks_by_niche[niche].append(bn)
        gas_price_by_niche[niche].append(gpm)

print('Got gas price oracle')

if False:

    curr.execute(
        '''
        CREATE TABLE tmp_campaign_niches(campaign_id, niche) AS
        SELECT
            x.id,
            CONCAT(
                'fb|',
                array_length(exchanges, 1),
                '|',
                CASE WHEN has_uniswap_v2 THEN 'uv2|' ELSE '' END,
                CASE WHEN has_uniswap_v3 THEN 'uv3|' ELSE '' END,
                CASE WHEN has_sushiswap THEN 'sushi|' ELSE '' END,
                CASE WHEN has_shibaswap THEN 'shiba|' ELSE '' END,
                CASE WHEN has_balancer_v1 THEN 'balv1|' ELSE '' END,
                CASE WHEN has_balancer_v2 THEN 'balv2|' ELSE '' END
            ) as niche
        FROM (
            SELECT
                *,
                EXISTS(SELECT 1 FROM uniswap_v2_exchanges WHERE address = ANY (exchanges)) has_uniswap_v2,
                EXISTS(SELECT 1 FROM uniswap_v3_exchanges WHERE address = ANY (exchanges)) has_uniswap_v3,
                EXISTS(SELECT 1 FROM sushiv2_swap_exchanges WHERE address = ANY (exchanges)) has_sushiswap,
                EXISTS(SELECT 1 FROM shibaswap_exchanges WHERE address = ANY (exchanges)) has_shibaswap,
                EXISTS(SELECT 1 FROM balancer_exchanges WHERE address = ANY (exchanges)) has_balancer_v1,
                EXISTS(SELECT 1 FROM balancer_v2_exchanges WHERE address = ANY (exchanges)) has_balancer_v2
            FROM top_candidate_arbitrage_campaigns
            WHERE removed is not True
        ) x
        '''
    )

    print(f'Labeled niches for {curr.rowcount:,} campaigns')

    curr.execute(
        '''
        UPDATE top_candidate_arbitrage_campaigns tcac
        SET niche = (SELECT tcn.niche FROM tmp_campaign_niches tcn WHERE tcn.campaign_id = tcac.id)
        '''
    )

    db.commit()

curr.execute(
    '''
    SELECT DISTINCT block_number, niche
    FROM (SELECT * FROM top_candidate_arbitrage_campaigns WHERE removed is not True) tcac
    JOIN top_candidate_arbitrage_relay_results tcarr ON tcarr.campaign_id = tcac.id
    JOIN candidate_arbitrages ca on ca.id = tcarr.candidate_arbitrage_id
    '''
)
print(f'Have {curr.rowcount} block-niche gas price queries required')

gas_prices_needed = collections.defaultdict(lambda: [])
for block_number, niche in curr:
    gas_prices_needed[niche].append(block_number)

gas_price_oracle = collections.defaultdict(lambda: {})

for niche, blocks in gas_prices_needed.items():
    if niche.endswith('|2|balv2|') or niche.endswith('|3|balv2|'):
        replacement_niche_sz = niche.replace('|balv2|', '|uv3|')
        xs = gas_price_blocks_by_niche[replacement_niche_sz]
        fs = gas_price_by_niche[replacement_niche_sz]
    else:
        xs = gas_price_blocks_by_niche[niche]
        fs = gas_price_by_niche[niche]

    assert len(fs) > 0, f'No pts for niche {niche}'

    interpolated = np.interp(blocks, xs, fs)

    assert len(interpolated) == len(blocks)
    for block, pt in zip(blocks, interpolated):
        gas_price_oracle[niche][block] = pt

print('Priced gas price')

# fill gas price into a new table
curr.execute(
    '''
    CREATE TEMP TABLE IF NOT EXISTS tmp_gas_prices_by_niche (
        niche TEXT NOT NULL,
        block_number INTEGER NOT NULL,
        gas_price NUMERIC(78, 0) NOT NULL
    );

    CREATE INDEX IF NOT EXISTS tmp_idx_gas_prices_block ON tmp_gas_prices_by_niche (block_number);
    '''
)

del niche
to_insert = [
    (niche, block_number, gas_price)
    for niche in gas_price_oracle
    for block_number, gas_price in gas_price_oracle[niche].items()
]
psycopg2.extras.execute_batch(
    curr,
    '''
    INSERT INTO tmp_gas_prices_by_niche (niche, block_number, gas_price) VALUES (%s, %s, %s)
    ''',
    to_insert
)

curr.execute(
    '''
    CREATE TEMP TABLE tmp_relay_results_after_fee AS
    SELECT
        tcarr.id result_id,
        campaign_id,
        candidate_arbitrage_id,
        real_profit_before_fee - gas_used * gp.gas_price as profit_after_fee,
        gp.niche,
        gp.gas_price,
        ca.block_number
    FROM (SELECT * FROM top_candidate_arbitrage_campaigns WHERE removed is not True) tcac
    JOIN top_candidate_arbitrage_relay_results tcarr ON tcarr.campaign_id = tcac.id
    JOIN candidate_arbitrages ca on ca.id = tcarr.candidate_arbitrage_id
    LEFT JOIN tmp_gas_prices_by_niche gp ON gp.niche = tcac.niche AND gp.block_number = ca.block_number
    '''
)

curr.execute(
    'SELECT count(*) FROM tmp_relay_results_after_fee WHERE profit_after_fee IS NULL'
)

print(f'Found {curr.fetchone()[0]} broken records')

# find the largest arbitrage per campaign
curr.execute(
    '''
    CREATE TEMP TABLE largest_arbs_in_campaign AS
    WITH largest_arb_by_campaign AS
    (
        SELECT campaign_id, MAX(profit_after_fee) max_profit
        FROM tmp_relay_results_after_fee
        GROUP BY campaign_id
    )
    SELECT trr.candidate_arbitrage_id, trr.campaign_id, block_number, profit_after_fee
    FROM tmp_relay_results_after_fee trr
    JOIN largest_arb_by_campaign lac ON lac.campaign_id = trr.campaign_id AND lac.max_profit = trr.profit_after_fee
    '''
)

if True:
    # dump large arbitrage profits
    curr.execute(
        '''
        SELECT lca.candidate_arbitrage_id, block_number, profit_after_fee
        FROM tmp_relay_results_after_fee trr
        JOIN lca_dedup lca ON lca.candidate_arbitrage_id = trr.candidate_arbitrage_id 
        '''
    )
    with open('tmp_lca_results.csv', mode='w') as fout:
        for id_, bn, p in curr:
            fout.write(f'{id_},{bn},{p}\n')

# # compute some stats
# curr.execute(
#     '''
#     SELECT
#         MAX(profit_after_fee),
#         PERCENTILE_DISC(0.5) WITHIN GROUP ()
#     '''
# )

curr.execute(
    '''
    SELECT block_number, MAX(profit_after_fee)
    FROM largest_arbs_in_campaign
    GROUP BY block_number
    ORDER BY block_number ASC
    '''
)

block_numbers = []
profit_after_fees = []

with open('large_profits_per_block.csv', mode='w') as fout:
    for bn, p in curr:
        bn = int(bn)
        p = p / (10 ** 18)
        block_numbers.append(bn)
        profit_after_fees.append(p)
        fout.write(f'{bn},{p}\n')

local_conn = sqlite3.connect('./tmp.db')
local_curr = local_conn.cursor()
local_curr.execute(
    '''
    CREATE TABLE IF NOT EXISTS block_timestamps (
        block_number INTEGER NOT NULL PRIMARY KEY,
        timestamp    INTEGER NOT NULL
    );
    '''
)


# build block timestamp caches if needex
local_curr.execute('SELECT COUNT(*) FROM block_timestamps')
(n_timestamps,) = local_curr.fetchone()

if n_timestamps == 0:
    print('filling timestamps')
    curr.execute('SELECT MIN(start_block), MAX(end_block) FROM block_samples')
    min_block, max_block = curr.fetchone()
    print(f'Sampling timestamps betwen {min_block} and {max_block}')
    w3 = web3.Web3(web3.WebsocketProvider('ws://10.10.111.111:8546',
            websocket_timeout=60 * 5,
            websocket_kwargs={
                'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
            },
    ))
    blocks_to_query = np.linspace(min_block, max_block, 10_000, endpoint=True, dtype=int)
    for i, block_number in enumerate(blocks_to_query):
        if i % 500 == 0:
            print(f'{i / len(blocks_to_query) * 100:.2f}%')
        ts = w3.eth.get_block(int(block_number))['timestamp']
        local_curr.execute(
            '''
            INSERT INTO block_timestamps (block_number, timestamp)
            VALUES (?, ?)
            ''',
            (int(block_number), int(ts))
        )
    local_conn.commit()

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

plt.scatter(timestamps, profit_after_fees, s=1, color='black')
plt.xlabel('Timestamp')
plt.ylabel('ETH profit')
plt.title('Attainable large ETH profit\nmedian gas price oracle')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#
# Plot the length of opportunity
#
curr.execute(
    '''
    SELECT trr.block_number, trr.campaign_id, trr.profit_after_fee, lac.profit_after_fee, lac.block_number
    FROM tmp_relay_results_after_fee trr
    JOIN largest_arbs_in_campaign lac ON lac.campaign_id = trr.campaign_id
    WHERE trr.block_number >= lac.block_number
    ORDER BY trr.block_number, trr.campaign_id
    '''
)

campaign_durations = {}
campaign_bounds = {}
campaign_max_profit = {}
campaign_died = None
for block_number, campaign_id, profit_after_fees, max_profit_after_fee, max_profit_block in curr:
    campaign_max_profit[campaign_id] = int(max_profit_after_fee)

    if campaign_id == campaign_died:
        continue

    profit_after_fees    = int(profit_after_fees)
    max_profit_after_fee = int(max_profit_after_fee)

    pct_diff_from_max = (profit_after_fees - max_profit_after_fee) / max_profit_after_fee * 100
    if pct_diff_from_max < -50:
        # campaign just died
        campaign_durations[campaign_id] = block_number - max_profit_block
        campaign_bounds[campaign_id] = (max_profit_block, block_number)
        campaign_died = campaign_id
    else:
        campaign_durations[campaign_id] = block_number - max_profit_block + 1

# dump campaign durations

with open('lc_campaign_durations.csv', mode='w') as fout:
    for id_, d in campaign_durations.items():
        fout.write(f'{id_},{d}\n')
with open('lc_campaign_bounds.csv', mode='w') as fout:
    for id_, (start, end) in campaign_bounds.items():
        fout.write(f'{id_},{start},{end},{campaign_max_profit[campaign_id]}\n')

max_profits = []
durations = []

assert set(campaign_max_profit.keys()) == set(campaign_durations.keys())

for campaign_id, max_profit in campaign_max_profit.items():
    duration = campaign_durations[campaign_id]
    max_profits.append(max_profit / (10 ** 18))
    durations.append(duration)

plt.scatter(max_profits, durations, s=1, color='black')
plt.xlabel('ETH profit')
plt.ylabel('Duration (blocks)')
plt.tight_layout()
plt.show()

