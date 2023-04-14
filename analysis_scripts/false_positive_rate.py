"""
Checks the false-positive rate of arbitrage labels
"""

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

local_db = sqlite3.connect('tmp_fp_rate.sqlite3')
local_curr = local_db.cursor()

local_curr.execute('''SELECT 1 FROM sqlite_schema WHERE type='table' AND name = 'fp_random_samples' ''')
N_SAMPLES = 100


if len(local_curr.fetchall()) < 1:
    # need to generate sample

    # a stable method of selecting 100 transactions
    print('Selecting 100 transactions at random')

    r = random.Random(100)
    curr.execute(
        '''
        SELECT MIN(id), MAX(id) FROM sample_arbitrages_no_fp WHERE block_number < 15628035
        '''
    )
    assert curr.rowcount == 1
    min_id, max_id = curr.fetchone()

    sampled_ids = set()
    while len(sampled_ids) < N_SAMPLES:
        n_needed = N_SAMPLES - len(sampled_ids)
        assert n_needed > 0
        print(f'Randomly sampling {n_needed:,} IDs')
        new_sample = list(set([r.randint(min_id, max_id) for _ in range(n_needed)]))
        # see how many do not exist
        curr.execute(
            '''
            SELECT id FROM sample_arbitrages_no_fp WHERE id = ANY(%s) AND block_number < 15628035
            ''',
            (new_sample,)
        )
        existing_id_new_sample = set(x for (x,) in curr)
        sampled_ids.update(existing_id_new_sample)

    # lastly, ensure that the transaction ids are all unique for our sample
    # (they should be!!!!!!!)

    curr.execute(
        '''
        SELECT COUNT(DISTINCT txn_hash) FROM sample_arbitrages_no_fp WHERE id = ANY(%s)
        ''',
        (list(sampled_ids),)
    )
    (n_distinct_hash,) = next(curr)
    assert n_distinct_hash == N_SAMPLES

    local_curr.execute(
        '''
        CREATE TABLE fp_random_samples (
            sample_id INTEGER NOT NULL
        );
        '''
    )
    local_curr.executemany(
        '''
        INSERT INTO fp_random_samples (sample_id) VALUES (?)
        ''',
        [(x,) for x in sorted(sampled_ids)]
    )
    local_db.commit()
else:
    print('fetching sample ids from sqlite db')
    local_curr.execute('SELECT sample_id FROM fp_random_samples')
    sampled_ids = set(int(x) for (x,) in local_curr)
    assert len(sampled_ids) == N_SAMPLES


psycopg2.extras.execute_values(
    curr,
    '''
    CREATE TEMP TABLE tmp_fp_random_samples AS
    (SELECT * FROM (VALUES %s) t(sample_id))
    ''',
    [(x,) for x in sorted(sampled_ids)]
)

# count how many are 'simple' (ie only one cycle)
curr.execute(
    '''
    SELECT COUNT(*) 
    FROM sample_arbitrages_no_fp sa WHERE EXISTS(SELECT FROM tmp_fp_random_samples WHERE sample_id = id) AND n_cycles > 1
    '''
)
(n_more_than_one_cycle,) = curr.fetchone()
print(f'Have {n_more_than_one_cycle:,} samples with more than one cycle ({n_more_than_one_cycle / N_SAMPLES * 100:.2f}%)')

# how many are uniswap (sushiswap) only?
curr.execute(
    '''
    CREATE TEMP TABLE tmp_fp_sample_exchange_summary AS
    SELECT
        b.id,
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
                sa.gas_price,
                sa.coinbase_xfer,
                sa.txn_hash,
                EXISTS(SELECT 1 FROM uniswap_v2_exchanges e WHERE e.address = sae.address) is_uniswap_v2,
                EXISTS(SELECT 1 FROM uniswap_v3_exchanges e WHERE e.address = sae.address) is_uniswap_v3,
                EXISTS(SELECT 1 FROM sushiv2_swap_exchanges e WHERE e.address = sae.address) is_sushiswap,
                EXISTS(SELECT 1 FROM shibaswap_exchanges e WHERE e.address = sae.address) is_shibaswap,
                EXISTS(SELECT 1 FROM balancer_exchanges e WHERE e.address = sae.address) is_balancer_v1,
                sae.address = '\\xBA12222222228d8Ba445958a75a0704d566BF2C8'::bytea is_balancer_v2
            FROM (SELECT * FROM sample_arbitrages_no_fp where EXISTS(SELECT FROM tmp_fp_random_samples WHERE sample_id = id)) sa
            JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
            JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
            JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
            JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
        ) a
    ) b
    GROUP BY b.id
    '''
)
assert curr.rowcount == N_SAMPLES - n_more_than_one_cycle, f'expect {N_SAMPLES - n_more_than_one_cycle} rows but got {curr.rowcount}'

# how many are all known exchanges?
curr.execute(
    'SELECT SUM(CASE WHEN all_known THEN 1 ELSE 0 END) FROM tmp_fp_sample_exchange_summary'
)
(n_all_known,) = curr.fetchone()
print(f'Have {n_all_known} arbitrages that use only known exchanges ({n_all_known / N_SAMPLES*100:.2f}%)')

# how many only use uniswap v2/v3/sushiswap/shibaswap
curr.execute(
    'SELECT SUM(CASE WHEN NOT has_balancer_v1 AND NOT has_balancer_v2 THEN 1 ELSE 0 END) FROM tmp_fp_sample_exchange_summary WHERE all_known'
)
(n_all_known,) = curr.fetchone()
print(f'Have {n_all_known} arbitrages that use easy exchanges ({n_all_known / N_SAMPLES*100:.2f}%)')

# generate a CSV file so I can import it into google drive


token_decimals: typing.Dict[str, int] = {}


w3 = web3.Web3(web3.WebsocketProvider('ws://10.10.111.111:8546'))
assert w3.isConnected()
print(f'Connected to web3 chain_id={w3.eth.chain_id}')

simple_erc20_just_decimals = [
        {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [
            {
                "name": "",
                "type": "uint8"
            }
        ],
        "payable": True,
        "stateMutability": "view",
        "type": "function"
    },
]

with open('tmp_false_positive_out.csv', mode='w') as fout:
    for i, id_ in enumerate(sorted(sampled_ids)):
        print(f'{i}/{N_SAMPLES}')
        curr.execute(
            '''
            SELECT txn_hash, n_cycles
            FROM sample_arbitrages_no_fp
            WHERE id = %s
            ''',
            (id_,)
        )
        (txn_hash, n_cycles) = next(curr)
        txn_hash = txn_hash.tobytes()

        if n_cycles == 1:
            curr.execute(
                '''
                SELECT profit_amount, t.address, t.symbol
                FROM sample_arbitrage_cycles_no_fp sac
                JOIN tokens t ON profit_token = t.id
                WHERE sample_arbitrage_id = %s
                ''',
                (id_,)
            )
            assert curr.rowcount == 1
            profit_amt, profit_token, profit_token_sym = next(curr)
            profit_token = web3.Web3.toChecksumAddress((profit_token.tobytes()))

            if profit_token not in token_decimals:
                print(f'Querying decimals for {profit_token}')
                contract: web3.contract.Contract = w3.eth.contract(address=profit_token, abi=simple_erc20_just_decimals)
                d = contract.functions.decimals().call(block_identifier='latest')
                token_decimals[profit_token] = d
                print(f'Found decimals={d} for {profit_token}')

            decimals = token_decimals[profit_token]

            fout.write(f'{i+1},{id_},https://etherscan.io/tx/0x{bytes(txn_hash).hex()},{n_cycles},{profit_token},{profit_token_sym},{profit_amt / (10 ** decimals)}\n')
        else:
            fout.write(f'{i+1},{id_},https://etherscan.io/tx/0x{bytes(txn_hash).hex()},{n_cycles}\n')
