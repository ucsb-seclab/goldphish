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

curr.execute(
    '''
    SELECT avg(ss.num_model_queries / r.num_model_queries::float), avg(ss.wall_seconds / r.wall_seconds::float), count(*)
    FROM (select * from seek_candidates_profile_results_one_penny where is_baseline = false) ss
    JOIN seek_candidates_profile_results r ON r.is_baseline = True and r.block_number = ss.block_number
    where r.num_model_queries > 0
    '''
)
(avg, wall, n_mat) = curr.fetchone()
print(n_mat, avg, wall)
