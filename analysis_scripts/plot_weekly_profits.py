import datetime
from dateutil import tz
import os
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np

txn_hashses = []
block_num_shot = []
gas_theirs = []
gas_mine = []
profit_amts = []
gas_prices = []

block_ts_nums = []
block_ts_timestamps = []

with open('tmp/block_timestamps.txt') as fin:
    for line in fin:
        block_num, timestamp = line.strip().split(',')
        block_num = int(block_num)
        timestamp = float(timestamp)

        block_ts_nums.append(block_num)
        block_ts_timestamps.append(timestamp)


with open('tmp/reshoot_log.txt') as fin:
    for line in fin:
        block_num,tx_hash,gt,gm,gp,pa = line.strip().split(',')

        block_num = int(block_num)
        gt = int(gt)
        gm = int(gm)
        gp = int(gp)
        pa = int(pa)

        txn_hashses.append(tx_hash)
        block_num_shot.append(block_num)
        gas_theirs.append(gt)
        gas_mine.append(gm)
        gas_prices.append(gp)
        profit_amts.append(pa)


# start each week on monday (sorry americans)

first_week_cal = datetime.datetime.fromtimestamp(block_ts_timestamps[0], tz.tzutc()).isocalendar()
first_week_start = datetime.datetime.fromisocalendar(first_week_cal[0], first_week_cal[1], 1).replace(tzinfo=tz.tzutc())
last_week_cal = datetime.datetime.fromtimestamp(block_ts_timestamps[-1], tz.tzutc()).isocalendar()
last_week_start = datetime.datetime.fromisocalendar(last_week_cal[0], last_week_cal[1], 1).replace(tzinfo=tz.tzutc())


print('first week', first_week_start)
print('last week', last_week_start)


n_weeks = (last_week_start - first_week_start).days // 7 + 1

print(f'Total of {n_weeks:,} weeks')


week_profits_ether = np.zeros(n_weeks, dtype=float)
week_count_arbs = np.zeros(n_weeks, dtype=int)
block_num_ts_interp = np.interp(block_num_shot, block_ts_nums, block_ts_timestamps)

our_net_profits = []
seen_tstamps = set()
n_dupes = 0
for block_ts, txn_hash, gt, gp, pa in zip(block_num_ts_interp, txn_hashses, gas_mine, gas_prices, profit_amts):
    # find week number of this shot
    if txn_hash in seen_tstamps:
        print('SAW THIS ALREADY!!!!!')
        n_dupes += 1
        continue
    seen_tstamps.add(txn_hash)
    dt = datetime.datetime.fromtimestamp(block_ts, tz.tzutc())
    week = (dt - first_week_start).days // 7

    wei_fee = gt * gp
    net = pa - wei_fee
    week_profits_ether[week] += net / (10 ** 18)
    week_count_arbs[week] += 1

print('number of dupes', n_dupes)
print('last dt', dt)

xs = [
    (first_week_start + datetime.timedelta(days=i * 7)) for i in range(n_weeks)
]
plt.gca().get_xaxis().set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.bar(xs, week_profits_ether, width = 6)
plt.gcf().autofmt_xdate()
plt.title('Ether profit per week, reshoot')
plt.xlabel('Week')
plt.ylabel('Profit (ether)')
plt.show()

xs = [
    (first_week_start + datetime.timedelta(days=i * 7)) for i in range(n_weeks)
]
plt.gca().get_xaxis().set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.bar(xs, week_count_arbs, width = 6)
plt.gcf().autofmt_xdate()
plt.title('Arbitrages per week, reshoot')
plt.xlabel('Week')
plt.ylabel('Arbitrages (count)')
plt.show()
