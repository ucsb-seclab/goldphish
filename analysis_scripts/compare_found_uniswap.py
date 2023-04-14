"""
Compare the ones found by uniswap pricers and what actually happened
"""

import time
import psycopg2
import psycopg2.extensions

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
setup_only_uniswap_tables(curr)

#
# setup only weth and only-uniswap tables
#
curr.execute(
    '''
    CREATE TABLE tmp_only_uniswap_only_weth (
        id INTEGER NOT NULL PRIMARY KEY
    );

    INSERT INTO tmp_only_uniswap_only_weth
    SELECT id
    FROM tmp_weth_arbs tow
    WHERE EXISTS(SELECT 1 FROM only_uniswap_arbitrages oua WHERE tow.id = oua.id);
    '''
)
n_only_uniswap_only_weth = curr.rowcount
print(f'[*] Have {n_only_uniswap_only_weth:,} that only use uniswap and WETH')


#
# find arbitrages that use at least 1 uniswap v3
#

start = time.time()
curr.execute(
    '''
    CREATE TEMP TABLE tmp_uniswap_v3_exchange_ids AS
    SELECT sace.id
    FROM uniswap_v3_exchanges uv3
    JOIN sample_arbitrage_exchanges sae ON sae.address = uv3.address;

    CREATE TEMP TABLE uniswap_v3_cycle_items (
        id INTEGER NOT NULL PRIMARY KEY
    );

    INSERT INTO uniswap_v3_cycle_items (id)
    SELECT sacei.id
    FROM sample_arbitrage_cycle_exchange_items sacei
    WHERE sacei.exchange_id IN (SELECT id FROM tmp_uniswap_v3_exchange_ids);
    '''
)
elapsed = time.time() - start
print(f'[*] found {curr.rowcount} cycle exchage items that used uniswap v3 (took {elapsed:.2f} seconds)')
start = time.time()
curr.execute(
    '''
    CREATE TEMP TABLE tmp_only_uniswap_only_weth_uses_v3 (
        id INTEGER NOT NULL PRIMARY KEY
    );

    INSERT INTO tmp_only_uniswap_only_weth_uses_v3 (id)
    SELECT distinct sac.sample_arbitrage_id
    FROM sample_arbitrage_cycles sac
    JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
    JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
    WHERE EXISTS(SELECT 1 FROM uniswap_v3_cycle_items ci WHERE ci.id = sacei.id);
    '''
)
elapsed = time.time() - start
n_only_uniswap_only_weth_uses_v3 = curr.rowcount
print(f'Found {n_only_uniswap_only_weth_uses_v3:,} arbitrages that use only uniswap, use v3, and take weth profit (took {elapsed:.2f} seconds)')


#
# Do backrun detection -- see which ones are profitable only when played behind prior
# transactions
#
#
# ----- BACKRUN DETECTION STRATEGY -----
# I do this because we don't have a perfect way to reorder transactions in a block.
# Since ganache requires mining a block at the fork point.
# TODO hack ganache to make it not require this?
#
# * limit to only uniswap v2, v3 scraped txns
# * for each scraped arbitrage
#   * re-run it on the top of block using our model
#   * see if it remains profitable (before fees) -- 
#

#
# Separate fees of top-of-block transactions vs
#


#
# Backward-match 
#
