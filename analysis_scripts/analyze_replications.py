"""
Analysis for the replicating arbitrages scraped from the chain
"""

import psycopg2
import psycopg2.extensions
import numpy as np
import tabulate

from common import gen_false_positives, label_zerox_exchanges

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

# gen_false_positives(curr)

# show how much we've processed

# print('counting arbitrages')
curr.execute(
    '''
    SELECT COUNT(*)
    FROM sample_arbitrages sa
    JOIN sample_arbitrage_cycles sac ON sa.id = sac.sample_arbitrage_id
    '''
)
(n_samples,) = curr.fetchone()
print(f'Have {n_samples:,} arbitrages with one cycle')

# print('counting replicated arbitrages')
curr.execute(
    '''
    SELECT count(*)
    FROM sample_arbitrage_replications
    WHERE verification_started = true AND verification_finished = true AND supported = true
    '''
)

(n_run,) = curr.fetchone()

print(f'Processed {n_run:,} sample arbitrages through replicator ({n_run / n_samples * 100:.2f}%)')

curr.execute(
    '''
    SELECT count(*)
    FROM sample_arbitrage_replications
    WHERE verification_started = true AND verification_finished = true AND supported = true AND replicated = false
    '''
)

(n_not_replicated,) = curr.fetchone()
print(f'Have {n_not_replicated:,} that failed replication ({n_not_replicated / n_run * 100:.2f}%)')
print()

curr.execute(
    '''
    SELECT sar.sample_arbitrage_id, (sar.our_profit - sac.profit_amount) profit_diff
    FROM sample_arbitrage_replications sar
    JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sar.sample_arbitrage_id
    WHERE verification_started = true AND verification_finished = true AND supported = true AND replicated = true
    '''
)
assert curr.rowcount == n_run - n_not_replicated, f'expected {curr.rowcount} == {n_run - n_not_replicated}'

sa_ids = []
sa_diffs = []

for sa_id, diff in curr:
    sa_ids.append(sa_id)
    sa_diffs.append(int(diff))

assert len(sa_ids) == n_run - n_not_replicated


percentile_marks = [1, 5, 25, 50, 75, 95, 99]
percentiles = np.percentile(sa_diffs, percentile_marks)

tab = []
for mark, p in zip(percentile_marks, percentiles):
    tab.append((f'{mark:02d}%', f'{p / (10 ** 18):.8f} ETH'))

print(tabulate.tabulate(tab, headers=('Percentile', 'ETH difference *')))

print('* greater = we found more')
