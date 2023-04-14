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

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams['hatch.linewidth'] = 0.2


# curr.execute('SET TRANSACTION READ ONLY')
curr.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')

gas_pricers = ['25th percentile', 'median', '75th percentile']

if True:
    input('ENTER to overwrite...')
    with open('tmp_campaign_durations_out.csv', mode='w') as fout:
        for gas_pricer in gas_pricers:
            curr.execute(
                '''
                WITH maximizing_blocks AS (
                    SELECT cac.id, min(block_number) block_number_max_profit
                    FROM (SELECT * FROM candidate_arbitrage_campaigns WHERE gas_pricer = %s) cac
                    JOIN candidate_arbitrage_campaign_member cacm ON cacm.candidate_arbitrage_campaign = cac.id
                    JOIN candidate_arbitrages ca ON ca.id = cacm.candidate_arbitrage_id
                    WHERE cac.max_profit_after_fee_wei = cacm.profit_after_fee_wei
                    GROUP BY cac.id
                )
                SELECT cac.id, mb.block_number_max_profit, cac.max_profit_after_fee_wei, cac.block_number_end, cacm.profit_after_fee_wei, ca.block_number
                FROM (SELECT * FROM candidate_arbitrage_campaigns WHERE gas_pricer = %s) cac
                JOIN maximizing_blocks mb ON mb.id = cac.id
                JOIN candidate_arbitrage_campaign_member cacm ON cacm.candidate_arbitrage_campaign = cac.id
                JOIN candidate_arbitrages ca ON ca.id = cacm.candidate_arbitrage_id
                WHERE ca.block_number >= mb.block_number_max_profit
                ORDER BY cac.id, ca.block_number
                ''',
                {
                    'gas_pricer': gas_pricer
                }
            )

            campaign_durations = {}
            campaign_bounds = {}
            campaign_ends = {}
            campaign_max_profit = {}
            campaign_died = None
            last_campaign_id = None
            for campaign_id, max_profit_block, max_profit_after_fee, block_number_end, profit_after_fees, block_number in curr:
                if campaign_id == campaign_died:
                    continue

                campaign_max_profit[campaign_id] = int(max_profit_after_fee)
                campaign_ends[campaign_id] = block_number_end

                # by default, set campaign to span its length
                campaign_bounds[campaign_id] = (max_profit_block, block_number_end)
                campaign_durations[campaign_id] = block_number_end - max_profit_block + 1

                profit_after_fees    = int(profit_after_fees)
                max_profit_after_fee = int(max_profit_after_fee)

                pct_diff_from_max = (profit_after_fees - max_profit_after_fee) / max_profit_after_fee * 100
                if pct_diff_from_max < -50:
                    # campaign just died
                    campaign_durations[campaign_id] = block_number - max_profit_block
                    campaign_bounds[campaign_id] = (max_profit_block, block_number)
                    campaign_died = campaign_id

            for id, dur in campaign_durations.items():
                start, end = campaign_bounds[id]
                max_profit = campaign_max_profit[id]
                fout.write(f'{id},{gas_pricer},{dur},{start},{end},{max_profit}\n')


    exit()

if True:
    for gas_pricer in ['median']:
        
        curr.execute(
            '''
            SELECT max_profit_after_fee_wei
            FROM candidate_arbitrage_campaigns cac
            WHERE gas_pricer = %s
            ''',
            (gas_pricer,)
        )
        profits = [x / (10 ** 18) for (x,) in curr]
        plt.hist(profits, bins=30, label=gas_pricer)
    plt.title('Profit After Fees\nRandom Sample')
    plt.xlabel('ETH Profit')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.savefig('profit_distribution_random_sample_arbs.png', format='png', dpi=300)
    plt.show()
    exit()


curr.execute('SELECT COUNT(*) FROM carr_dedup')
(n_tot,) = curr.fetchone()
print(f'Have {n_tot:,} total samples')


def get_success_rate():
    print('Counting success rate....')
    curr.execute(
        '''
        SELECT
            COUNT(*)
        FROM carr_dedup cd
        JOIN candidate_arbitrage_relay_results carr ON cd.candidate_arbitrage_id = carr.candidate_arbitrage_id
        WHERE carr.shoot_success = True
        ''',
    )
    (n_success,) = curr.fetchone()
    print(f'Have {n_tot:,} total relayed and {n_success:,} success')
    print(f'Success rate {n_success / n_tot * 100:.2f}%')
    print(f'Failure rate {(n_tot - n_success) / n_tot * 100:.2f}%')
    return n_success

if False:
    print('Counting number that are profitable at different gas prices')
    # n_success = get_success_rate()
    print('Counting how many are profitable at different gas-price oracles')

    n_profitable = []
    for gas_pricer in gas_pricers:
        curr.execute(
            '''
            SELECT COUNT(*)
            FROM (
                SELECT min(ca.id)
                FROM (
                    SELECT DISTINCT cacm.candidate_arbitrage_id id
                    FROM (SELECT * FROM candidate_arbitrage_campaigns WHERE gas_pricer = %(gas_pricer)s) cac
                    JOIN candidate_arbitrage_campaign_member cacm ON cacm.candidate_arbitrage_campaign = cac.id
                ) x
                JOIN candidate_arbitrages ca ON ca.id = x.id
                GROUP BY ca.block_number, ca.exchanges, ca.directions
            ) x
            ''',
            {
                'gas_pricer': gas_pricer
            }
        )
        (n,) = curr.fetchone()
        print(f'Found {n:,} profitable transactions for gas-pricer {gas_pricer}')
        n_profitable.append(n)

    exit()

n_success = get_success_rate()

if True:
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
                failure_reason LIKE 'Broken token%' broken_token,
                failure_reason LIKE 'incompatible%' incompatible,
                failure_reason LIKE 'Other%' other,
                (failure_reason LIKE 'Bad exchange%') OR (failure_reason LIKE 'Broken exchange%') bad_exchange,
                failure_reason LIKE 'No arbitrage after fee' no_after_fee,
                (failure_reason LIKE 'token-exchange interf%') OR (failure_reason LIKE 'Token % interferes with%') interference,
                failure_reason LIKE 'Balancer v1: Token % not available %' not_avail,
                *
            FROM (
                SELECT ca.candidate_arbitrage_id, failure_reason
                FROM carr_dedup ca
                JOIN candidate_arbitrage_relay_results carr ON ca.candidate_arbitrage_id = carr.candidate_arbitrage_id
            ) x
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
        print(f'{d} & ${n:,}$ & ${round(n / n_tot * 100, 1):.1f}\\%$ \\\\')
    print('\\midrule')
    print(f'All & ${n_failures:,}$ & ${round(n_failures / n_tot * 100, 1):.1f}\\%$ \\\\')


