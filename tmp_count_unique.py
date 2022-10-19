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
    INSERT INTO unique_count_in_block_samples (priority, n_unique)
    SELECT %s, count(*)
    FROM (
        SELECT DISTINCT exchanges, directions, ca.block_number
        FROM candidate_arbitrages ca
        WHERE %s <= block_number AND block_number <= %s
    ) x
    ''',
    (args.ID, start_block, end_block)
)
assert curr.rowcount == 1
db.commit()
