import collections
import datetime
import subprocess
import argparse
import psycopg2.extras
import time
from backtest.utils import connect_db

db = connect_db()
curr = db.cursor()

parser = argparse.ArgumentParser()
parser.add_argument('ID', type=int)

args = parser.parse_args()

curr.execute('SELECT start_block, end_block FROM block_samples WHERE priority = %s', (args.ID,))
start_block, end_block = curr.fetchone()
print(f'Querying for id={args.ID} from {start_block} to {end_block}')


curr.execute(
    '''
    INSERT INTO interference_edges (i1, i2)
    SELECT c1.id, c2.id
    FROM (SELECT * FROM candidate_arbitrage_campaigns WHERE gas_pricer = 'median' AND %s <= block_number_start and block_number_end <= %s) c1
    JOIN (SELECT * FROM candidate_arbitrage_campaigns WHERE gas_pricer = 'median') c2 ON
        int4range(c1.block_number_start, c1.block_number_end + 1) &&
        int4range(c2.block_number_start, c2.block_number_end + 1)
        AND
        c1.id < c2.id
        AND
        c1.exchanges && c2.exchanges
    WHERE c1.gas_pricer = 'median' AND
          c2.gas_pricer = 'median'
    ''',
    (start_block, end_block)
)

print('done')
db.commit()

