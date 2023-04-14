import os
import numpy as np
from matplotlib import pyplot as plt

fname = '/home/robert/Downloads/token_liquidities.csv'

token_liquidities = []
with open(fname) as fin:
    for line in fin:
        addr, usd_locked = line.strip().split(',')
        locked = np.float128(usd_locked)
        token_liquidities.append(locked)

total_liquidity = np.sum(token_liquidities)

print(f'${total_liquidity:,} total liquidity')

csum = np.cumsum(token_liquidities)

percentile_marks = [50, 90, 95, 99, 99.9, 99.99]

cum_percent = csum / total_liquidity

for i, perc in enumerate(cum_percent):
    if len(percentile_marks) == 0:
        break

    next_mark = percentile_marks[0]
    if perc * 100 > next_mark:
        del percentile_marks[0]
        print(f'top {i-1:,} tokens have {next_mark}% of the liquidity')
