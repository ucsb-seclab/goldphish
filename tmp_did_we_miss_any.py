import web3._utils.filters
import itertools
from pricers.balancer import FINALIZE_TOPIC, PUBLIC_SWAP_TOPIC, BalancerPricer
from collections import deque
import collections
import datetime
import random
import psycopg2
import psycopg2.extras
import numpy as np
import tabulate
import web3
import scipy.stats
import sqlite3
from backtest.top_of_block.relay import load_pricer_for

from backtest.utils import connect_db
from find_circuit.find import PricingCircuit, detect_arbitrages_bisection
from find_circuit.monitor import meets_thresholds
from utils import WETH_ADDRESS, connect_web3, get_block_timestamp

db = connect_db()
w3 = connect_web3()
print('connected to postgresql')
db.autocommit = False

curr = db.cursor()

curr.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')

# def setup_weth_arb_tables(curr: psycopg2.extensions.cursor):
#     """
#     Build temp table 'tmp_weth_arbs' of WETH-profiting arbitrages
#     """
#     weth_token_id = 57

#     curr.execute(
#         '''
#         CREATE TEMP TABLE tmp_weth_arbs (
#             id INTEGER NOT NULL,
#             block_number INTEGER NOT NULL,
#             revenue NUMERIC(78, 0) NOT NULL,
#             coinbase_xfer NUMERIC(78, 0),
#             fee NUMERIC(78, 0) NOT NULL,
#             net_profit NUMERIC(78, 0),
#             txn_hash BYTEA
#         );

#         INSERT INTO tmp_weth_arbs (id, block_number, revenue, coinbase_xfer, fee, txn_hash)
#         SELECT sa.id, sa.block_number, sac.profit_amount, sa.coinbase_xfer, sa.gas_used * sa.gas_price, txn_hash
#         FROM sample_arbitrages_no_fp sa
#         JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
#         WHERE sac.profit_token = %s;
#         ''',
#         (weth_token_id,)
#     )
#     n_weth = curr.rowcount
#     print(f'have {n_weth:,} arbitrages with WETH profit')

#     curr.execute(
#         '''
#         UPDATE tmp_weth_arbs
#         SET net_profit = (revenue - fee) - coinbase_xfer
#         WHERE coinbase_xfer IS NOT NULL
#         '''
#     )
#     n_nonnull_coinbase = curr.rowcount
#     print(f'Have coinbase transfer info for {n_nonnull_coinbase:,} arbitrages; {n_nonnull_coinbase / n_weth * 100:.2f}%')
#     print()


# setup_weth_arb_tables(curr)

# curr.execute(
#     '''
#     SELECT id, block_number, revenue, txn_hash
#         FROM (
#         SELECT
#             b.id,
#             b.block_number,
#             b.txn_hash,
#             b.revenue,
#             bool_or(is_uniswap_v2) has_uniswap_v2,
#             bool_or(is_uniswap_v3) has_uniswap_v3,
#             bool_or(is_sushiswap) has_sushiswap,
#             bool_or(is_shibaswap) has_shibaswap,
#             bool_or(is_balancer_v1) has_balancer_v1,
#             bool_or(is_balancer_v2) has_balancer_v2,
#             bool_and(is_known) all_known,
#             count(*) n_exchanges
#         FROM (
#             SELECT *, is_uniswap_v2 or is_uniswap_v3 or is_sushiswap or is_shibaswap or is_balancer_v1 or is_balancer_v2 is_known
#             FROM (
#                 SELECT
#                     sa.id,
#                     sa.block_number,
#                     sa.txn_hash,
#                     sa.revenue,
#                     EXISTS(SELECT 1 FROM uniswap_v2_exchanges e WHERE e.address = sae.address) is_uniswap_v2,
#                     EXISTS(SELECT 1 FROM uniswap_v3_exchanges e WHERE e.address = sae.address) is_uniswap_v3,
#                     EXISTS(SELECT 1 FROM sushiv2_swap_exchanges e WHERE e.address = sae.address) is_sushiswap,
#                     EXISTS(SELECT 1 FROM shibaswap_exchanges e WHERE e.address = sae.address) is_shibaswap,
#                     EXISTS(SELECT 1 FROM balancer_exchanges e WHERE e.address = sae.address) is_balancer_v1,
#                     sae.address = '\\xBA12222222228d8Ba445958a75a0704d566BF2C8'::bytea is_balancer_v2
#                 FROM (SELECT * FROM tmp_weth_arbs WHERE revenue >= %s) sa
#                 JOIN sample_arbitrage_backrun_detections bd ON bd.sample_arbitrage_id = sa.id
#                 JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
#                 JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
#                 JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
#                 JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
#                 WHERE bd.rerun_exactly = true
#             ) a
#         ) b
#         GROUP BY b.id, b.block_number, b.txn_hash, b.revenue
#     ) c
#     WHERE all_known = true AND n_exchanges <= 3
#     ORDER BY block_number ASC
#     ''',
#     (10 ** 18,)
# )


rows = []
with open('missed.csv') as fin:
# with open('/data/robert/tmp_thing.csv') as fin:
    for line in fin:
        if line.strip() == '':
            break
        # id_, block_number, revenue, txn_hash = line.strip().split(',')
        id_, txn_hash, block_number, revenue = line.strip().split(',')
        rows.append((
            int(id_),
            int(block_number),
            int(float(revenue) * (10 ** 18)),
            # int(revenue),
            bytes.fromhex(txn_hash),
        ))

print(f'investigating.... {len(rows):,}')

balancer = bytes.fromhex('0xBA12222222228d8Ba445958a75a0704d566BF2C8'[2:])

curr2 = db.cursor()
n_balancer = 0
n_not_met_threshold = 0
n_found = 0
n_not_found = 0
missed = list()
for i, (id_, block_number, revenue, txn_hash) in enumerate(rows):
    if txn_hash.hex() != '4da87d8ee426068bbeadd7aff81b9254229c13c306eddc3de46fc7ccd5cbfc6d':
        continue

    if i % 100 == 0 and (n_found + n_not_found) > 0:
        print(f'Found {n_found / (n_found + n_not_found) * 100:.2f}%')
        print(f'had {n_balancer:,} balancer -- {n_balancer / (n_balancer + n_not_found + n_found):.2f}%')
        print(f'had {n_not_met_threshold:,} not meet threshold -- {n_not_met_threshold / (n_not_found + n_found):.2f}%')

    curr2.execute(
        '''
        SELECT sace.id, sae.address
        FROM (SELECT * FROM sample_arbitrage_cycles WHERE sample_arbitrage_id = %s) sac
        JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
        JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
        JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
        ORDER BY sace.id ASC
        ''',
        (id_,)
    )

    exchanges = curr2.fetchall()
    if len(set(x for (x, _) in exchanges)) != len(exchanges):
        print(f'Split?????? {txn_hash.hex()}')
        continue

    exchanges = [x.tobytes() for (_, x) in exchanges]

    if balancer in exchanges:
        n_balancer += 1
        continue

    curr2.execute(
        '''
        SELECT
        FROM top_candidate_arbitrage_campaigns
        WHERE exchanges @> %(exchanges)s AND %(exchanges)s <@ exchanges AND start_block <= %(block_number)s and %(block_number)s - 1 <= end_block
        LIMIT 1
        ''',
        {
            'exchanges': exchanges,
            'block_number': block_number,
        }
    )
    if curr2.rowcount == 0:
        n_not_found += 1
        missed.append((id_, txn_hash, block_number, revenue))

        if False:
            print('checking for matching candidate...')
            curr2.execute(
                '''
                SELECT id, profit_no_fee, block_number
                FROM candidate_arbitrages
                WHERE exchanges = %(exchanges)s AND block_number < %(block_number)s AND %(block_number)s - 500 <= block_number
                ORDER BY block_number DESC
                LIMIT 1
                ''',
                {
                    'exchanges': exchanges,
                    'block_number': block_number,
                }
            )
            if curr2.rowcount > 0:
                (ca_id, profit_no_fee, old_block_no) = curr2.fetchone()
                print(f'Had {profit_no_fee / (10 ** 18):.2f} ETH profit detected in the past.... ({block_number - old_block_no:,} blocks ago) {txn_hash.hex()}')
                if profit_no_fee >= (10 ** 18):
                    curr2.execute(
                        '''
                        SELECT failure_reason
                        FROM top_candidate_arbitrage_relay_results tcarr
                        WHERE candidate_arbitrage_id = %s
                        ''',
                        (ca_id,)
                    )
                    if curr2.rowcount == 0:
                        print(f'Why no relay of {ca_id} ???')
                    else:
                        (failure_reason,) = curr2.fetchone()
                        if failure_reason is None:
                            curr.execute(
                                '''
                                SELECT start_block, end_block
                                FROM top_candidate_arbitrage_relay_results tcarr
                                JOIN top_candidate_arbitrage_campaigns tcac on tcac.id = tcarr.campaign_id
                                WHERE tcarr.candidate_arbitrage_id = %s
                                ''',
                                (ca_id,)
                            )
                            old_start, old_end = curr.fetchone()
                            print(f'bounds: {old_start:,} -> {old_end:,}')
                            print(f'Why no campaign with candidate {ca_id} pairing {txn_hash.hex()} at {block_number:,}')
                        print(f'Failure reason: {failure_reason}')
            else:
                print('Did not see a record in the recent past')

        assert txn_hash.hex() == '4da87d8ee426068bbeadd7aff81b9254229c13c306eddc3de46fc7ccd5cbfc6d'
        exchanges = ['0xa478c2975ab1ea89e8196811f51a7b7ade33eb11', '0x2e41132dab88a9bad80740a1392d322bf023d494', '0x59a19d8c652fa0284f44113d0ff9aba70bd46fb4']
        directions = [
            ('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', '0x6B175474E89094C44Da98b954EedeAC495271d0F'),
            ('0x6B175474E89094C44Da98b954EedeAC495271d0F', '0xba100000625a3754423978a60c9317c58a424e3D'),
            ('0xba100000625a3754423978a60c9317c58a424e3D', '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2')
        ]
        pricers = []
        for exc in exchanges:
            p = load_pricer_for(w3, curr2, w3.toChecksumAddress(exc))
            if not meets_thresholds(p, block_number - 1):
                print('not meet threshold!!!')
                n_not_met_threshold += 1
            pricers.append(p)

        pc = PricingCircuit(
            pricers, directions
        )
        ts = get_block_timestamp(w3, block_number)
        det = detect_arbitrages_bisection(pc, block_identifier=block_number - 1, try_all_directions=False, timestamp=ts)
        if len(det) == 0:
            print(f'No detection')
        else:
            (fa,) = det
            print(f'Detected profit {fa.profit / (10 ** 18):,}')

        print('block', block_number)

        for p in pricers:
            if isinstance(p, BalancerPricer):
                print(f'Finalized? {p.address} {p.get_finalized(block_number - 500)}')

                curr.execute('SELECT origin_block FROM balancer_exchanges WHERE address = %s', (bytes.fromhex(p.address[2:]),))
                (origin,) = curr.fetchone()
                print(f'{p.address} origin {origin:,}')

                f: web3._utils.filters.Filter = w3.eth.filter({
                    'address': p.address,
                    'fromBlock': origin - 1,
                    'toBlock': origin + 10000,
                })
                logs = f.get_all_entries()
                print(f'Have {len(logs):,} logs')
                for log in logs:
                    if log['topics'][0] == PUBLIC_SWAP_TOPIC:
                        print(f'Public swap {log["blockNumber"]:,}')
                    if log['topics'][0] == FINALIZE_TOPIC:
                        print(f'Finalize {log["blockNumber"]:,}')
                print()
        exit()

        pricers = []
        for exc in exchanges:
            p = load_pricer_for(w3, curr2, w3.toChecksumAddress(exc))
            if not meets_thresholds(p, block_number - 1):
                print('not meet threshold!!!')
                n_not_met_threshold += 1
            pricers.append(p)

        if not any(isinstance(p, BalancerPricer) for p in pricers):
            print('not balancer')
            continue

        # if len(pricers) == 2:
        #     other_token = set(pricers[0].get_tokens(block_number - 1)).intersection(pricers[1].get_tokens(block_number - 1)).difference([WETH_ADDRESS])
        #     if len(other_token) > 1:
        #         print('idk other token....')
        #     else:
        #         ot = next(iter(other_token))
        #         pc = PricingCircuit(
        #             pricers, [(WETH_ADDRESS, ot), (ot, WETH_ADDRESS)]
        #         )
        #         ts = get_block_timestamp(w3, block_number)
        #         det = detect_arbitrages_bisection(pc, block_identifier=block_number - 1, try_all_directions=False, timestamp=ts)
        #         if len(det) == 0:
        #             print(f'No detection')
        #         else:
        #             (fa,) = det
        #             print(f'Detected profit {fa.profit / (10 ** 18):,}')


        # print(f'Did not find anything at all in this cycle set for {txn_hash.hex()} -- made revenue {revenue / (10 ** 18):.5f} ETH')
        # for exc in exchanges:
        #     print('    ' + web3.Web3.toChecksumAddress(exc))
    else:
        n_found += 1
        # print('found')
        # (maybe_id, my_revenue, my_block) = curr2.fetchone()
        # revenue_diff = my_revenue - revenue
        # block_diff = block_number - my_block
        # print(f'{block_number}: most recent ID {maybe_id} -- we make {revenue_diff / (10 ** 18):,.5f} more {block_diff:,} blocks earlier')

print(f'Found {n_found / (n_found + n_not_found) * 100:.2f}%')
print(f'n_found', n_found)
print(f'n_found', n_not_found)



# print(f'Random sample of 20 missed transactions (from {n_not_found})')
# for id_, txn_hash, block_number, revenue in random.sample(list(missed), 20):
#     print(f'{id_},{txn_hash.hex()},{block_number},{revenue / (10 ** 18)}')

