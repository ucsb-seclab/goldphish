from collections import deque
import itertools
import time
import psycopg2
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import scipy.stats


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

curr.execute(
    '''
    
    '''
)

