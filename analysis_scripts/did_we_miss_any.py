from collections import deque
import collections
import datetime
import psycopg2
import psycopg2.extras
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import scipy.stats
import sqlite3

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


setup_weth_arb_tables(curr)

curr.execute(
    '''
    SELECT id, block_number, revenue, txn_hash
        FROM (
        SELECT
            b.id,
            b.block_number,
            b.txn_hash,
            b.revenue,
            bool_or(is_uniswap_v2) has_uniswap_v2,
            bool_or(is_uniswap_v3) has_uniswap_v3,
            bool_or(is_sushiswap) has_sushiswap,
            bool_or(is_shibaswap) has_shibaswap,
            bool_or(is_balancer_v1) has_balancer_v1,
            bool_or(is_balancer_v2) has_balancer_v2,
            bool_and(is_known) all_known,
            count(*) n_exchanges
        FROM (
            SELECT *, is_uniswap_v2 or is_uniswap_v3 or is_sushiswap or is_shibaswap or is_balancer_v1 or is_balancer_v2 is_known
            FROM (
                SELECT
                    sa.id,
                    sa.block_number,
                    sa.txn_hash,
                    sa.revenue,
                    EXISTS(SELECT 1 FROM uniswap_v2_exchanges e WHERE e.address = sae.address) is_uniswap_v2,
                    EXISTS(SELECT 1 FROM uniswap_v3_exchanges e WHERE e.address = sae.address) is_uniswap_v3,
                    EXISTS(SELECT 1 FROM sushiv2_swap_exchanges e WHERE e.address = sae.address) is_sushiswap,
                    EXISTS(SELECT 1 FROM shibaswap_exchanges e WHERE e.address = sae.address) is_shibaswap,
                    EXISTS(SELECT 1 FROM balancer_exchanges e WHERE e.address = sae.address) is_balancer_v1,
                    sae.address = '\\xBA12222222228d8Ba445958a75a0704d566BF2C8'::bytea is_balancer_v2
                FROM (SELECT * FROM tmp_weth_arbs WHERE revenue >= %s) sa
                JOIN sample_arbitrage_backrun_detections bd ON bd.sample_arbitrage_id = sa.id
                JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
                JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
                JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
                JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
                WHERE bd.rerun_exactly = true
            ) a
        ) b
        GROUP BY b.id, b.block_number, b.txn_hash, b.revenue
    ) c
    WHERE all_known = true AND n_exchanges <= 3
    ORDER BY block_number ASC
    ''',
    (10 ** 18,)
)

print(f'investigating.... {curr.rowcount:,}')

balancer = bytes.fromhex('0xBA12222222228d8Ba445958a75a0704d566BF2C8'[2:])

curr2 = db.cursor()
n_balancer = 0
n_found = 0
n_not_found = 0
for i, (id_, block_number, revenue, txn_hash) in enumerate(curr):
    if i % 100 == 0 and (n_found + n_not_found) > 0:
        print(f'Found {n_found / (n_found + n_not_found) * 100:.2f}%')
        print(f'had {n_balancer:,} balancer -- {n_balancer / (n_balancer + n_not_found + n_found):.2f}%')

    txn_hash = txn_hash.tobytes()

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
        WHERE exchanges = %(exchanges)s AND start_block <= %(block_number)s and end_block <= %(block_number)s
        LIMIT 1
        ''',
        {
            'exchanges': exchanges,
            'block_number': block_number,
        }
    )
    if curr2.rowcount == 0:
        n_not_found += 1
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

