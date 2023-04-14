import collections
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np
import collections
import datetime
import math
import sqlite3
import psycopg2
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

curr.execute('SELECT id FROM top_candidate_arbitrage_campaigns WHERE end_block >= 14500000')
allowed_ids = set(x for (x,) in curr)

durations = []
with open('lc_campaign_durations.csv') as fin:
    for line in fin:
        id_, dur = line.strip().split(',')
        id_ = int(id_)
        if id_ not in allowed_ids:
            continue
        dur = int(dur)
        durations.append(dur)

print(f'Have {len(durations):,} durations')
mean_dur = np.mean(durations)
print(f'Mean duration: {mean_dur}')
print(f'Median duration: {np.percentile(durations, 50)}')

counts_by_duration = collections.defaultdict(lambda: 0)
for d in durations:
    counts_by_duration[d] = counts_by_duration[d] + 1

distinct_durations = []
dur_counts = []

for d in range(min(durations), 20): # max(durations) + 1):
    distinct_durations.append(d)
    dur_counts.append(counts_by_duration[d])

# for d, c in counts_by_duration.items():
#     if d < 1_000:
#         distinct_durations.append(d)
#         dur_counts.append(c)

def func(x, a, c):
    return a*np.exp(-c*(x)) + 1

popt, pcov = scipy.optimize.curve_fit(func, distinct_durations, dur_counts)
print(popt)

space = np.linspace(min(distinct_durations), max(distinct_durations), 100)
def fit(x):
    return func(x, *popt)

line = [fit(x) for x in space]

plt.plot(space, line)
plt.scatter(distinct_durations, dur_counts)
plt.yscale('log')
plt.show()
