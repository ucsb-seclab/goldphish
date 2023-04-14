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


curr.execute('SELECT COUNT(*) FROM top_candidate_arbitrage_campaigns')
(n_campaigns,) = curr.fetchone()


common.setup_weth_arb_tables(curr)

curr.execute(
    '''
    CREATE TEMP TABLE weth_arbs_no_br
    '''
)
