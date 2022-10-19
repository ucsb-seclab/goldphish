import datetime
import subprocess
import argparse
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
    INSERT INTO carr_dedup (candidate_arbitrage_id)
    SELECT min(ca.id)
    FROM candidate_arbitrages ca
    WHERE %s <= ca.block_number AND ca.block_number <= %s
    GROUP BY ca.block_number, ca.exchanges, ca.directions
    ''',
    (start_block, end_block)
)
db.commit()
