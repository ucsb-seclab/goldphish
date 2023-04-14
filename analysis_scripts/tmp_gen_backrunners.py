import collections
import datetime
import math
import sqlite3
import psycopg2
import common
import tabulate
import os
import numpy as np
import matplotlib.pyplot as plt


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

common.setup_weth_arb_tables(curr)

curr.execute(
    '''
    CREATE TEMP TABLE tmp_backrunners (
        sample_arbitrage_id INTEGER NOT NULL,
        txn_hash BYTEA NOT NULL,
        block_number INTEGER NOT NULL
    );

    CREATE TEMP TABLE tmp_not_backrunners (
        sample_arbitrage_id INTEGER NOT NULL,
        txn_hash BYTEA NOT NULL,
        block_number INTEGER NOT NULL
    );

    INSERT INTO tmp_not_backrunners (sample_arbitrage_id, txn_hash, block_number)
    SELECT sample_arbitrage_id, sa.txn_hash, sa.block_number
    FROM sample_arbitrage_backrun_detections sabd
    JOIN sample_arbitrages_no_fp sa ON sa.id = sabd.sample_arbitrage_id
    WHERE sabd.rerun_exactly = true AND sa.n_cycles = 1;

    INSERT INTO tmp_backrunners (sample_arbitrage_id, txn_hash, block_number)
    SELECT sample_arbitrage_id, sa.txn_hash, sa.block_number
    FROM sample_arbitrage_backrun_detections sabd
    JOIN sample_arbitrages_no_fp sa ON sa.id = sabd.sample_arbitrage_id
    WHERE (sabd.rerun_reverted = true OR sabd.rerun_no_arbitrage = true) AND sa.n_cycles = 1;
    '''
)


with open('tmp_marks_backrunners_profit.csv', mode='w') as fout:
    curr.execute(
        '''
        SELECT bs.end_block, SUM(net_profit)
        FROM tmp_weth_arbs twa
        JOIN tmp_backrunners tb ON tb.sample_arbitrage_id = twa.id
        JOIN block_samples bs ON bs.start_block <= tb.block_number AND tb.block_number <= bs.end_block
        GROUP BY bs.end_block
        ORDER BY bs.end_block ASC 
        '''
    )

    for bs, p in curr:
        fout.write(f'b,{bs},{int(p)}\n')

    curr.execute(
        '''
        SELECT bs.end_block, SUM(net_profit)
        FROM tmp_weth_arbs twa
        JOIN tmp_not_backrunners tb ON tb.sample_arbitrage_id = twa.id
        JOIN block_samples bs ON bs.start_block <= tb.block_number AND tb.block_number <= bs.end_block
        GROUP BY bs.end_block
        ORDER BY bs.end_block ASC 
        '''
    )

    for bs, p in curr:
        fout.write(f'nb,{bs},{int(p)}\n')

    curr.execute(
        '''
        SELECT bs.end_block, SUM(net_profit)
        FROM tmp_weth_arbs twa
        JOIN block_samples bs ON bs.start_block <= twa.block_number AND twa.block_number <= bs.end_block
        GROUP BY bs.end_block
        ORDER BY bs.end_block ASC 
        '''
    )
    for bs, p in curr:
        fout.write(f'all,{bs},{int(p)}\n')
