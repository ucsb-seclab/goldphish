import datetime
import matplotlib.pyplot as plt
import numpy as np
import sqlite3

# COLORS https://colorkit.co/palette/1abc9c-16a085-2ecc71-27ae60-3498db-2980b9-9b59b6-8e44ad-34495e-2c3e50-f1c40f-f39c12-e67e22-d35400-e74c3c-c0392b-ecf0f1-bdc3c7-95a5a6-7f8c8d/

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams['hatch.linewidth'] = 0.2

interp_timestamps = []
buf_ys1 = []
buf_ys2 = []
buf_ys3 = []

with open('tmp_backrun.csv') as fin:
    for line in fin:
        ts, ys2, ys3 = line.strip().split(',')
        interp_timestamps.append(float(ts))
        buf_ys1.append(0)
        buf_ys2.append(float(ys2))
        buf_ys3.append(float(ys3))

bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]

plt.fill_between([min(bin_rights_datetimes), max(bin_rights_datetimes)], [0], [100], color='#bbbbbb', label='Unknown')


plt.fill_between(bin_rights_datetimes, buf_ys1, buf_ys3, label='Back-running', hatch='//')
plt.fill_between(bin_rights_datetimes, buf_ys1, buf_ys2, label='Not back-running', hatch='\\\\')
plt.ylabel('Count (%)')
plt.xlabel('Date')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
plt.xticks(rotation=75)

handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,0]
plt.margins(x=0, y=0)

plt.title('Back-running Behavior Over Time')
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower left')#, ncol=3, bbox_to_anchor=(0.5, 0.94))
plt.tight_layout()
plt.savefig('backrunners.png', format='png', dpi=300)
plt.show()

exit()

#
# Produce stacked graph
#

local_conn = sqlite3.connect('./tmp.db')
local_curr = local_conn.cursor()

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

local_curr.execute('SELECT block_number, timestamp FROM block_timestamps ORDER BY block_number ASC')

ts_block_numbers = []
ts_timestamps = []
for bn, t in local_curr:
    ts_block_numbers.append(bn)
    ts_timestamps.append(t)


profits_all = {}
profits_br = {}
profits_nbr = {}

with open('tmp_marks_backrunners_profit.csv') as fin:
    for line in fin:
        cat, bn, profit = line.strip().split(',')
        bn = int(bn)
        profit = int(profit)

        if cat == 'b':
            profits_br[bn] = profit
        elif cat == 'nb':
            profits_nbr[bn] = profit
        elif cat == 'all':
            profits_all[bn] = profit

all_blocks = sorted(set(profits_all.keys()))

interp_timestamps = np.interp(all_blocks, ts_block_numbers, ts_timestamps)

bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]

buf_ys1 = []
buf_ys2 = []
buf_ys3 = []
for block in all_blocks:
    buf_ys1.append(0)
    buf_ys2.append(profits_nbr.get(block, 0) / profits_all.get(block, 1) * 100)
    buf_ys3.append((profits_nbr.get(block, 0) + profits_br.get(block, 0)) / profits_all.get(block, 1) * 100)

ax1.fill_between([min(bin_rights_datetimes), max(bin_rights_datetimes)], [0], [100], color='#bbbbbb', label='Other')


ax1.fill_between(bin_rights_datetimes, buf_ys1, buf_ys3, label='Back-running arbitrages', hatch='//')
ax1.fill_between(bin_rights_datetimes, buf_ys1, buf_ys2, label='Non-back-running arbitrages', hatch='\\\\')
# ax1.set_title('Back-running Behavior Over Time\nBy Profit')
ax1.set_ylabel('Share of Profit (%)')
# ax1.xlabel('Date')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
# ax1.xaxis.set_tick_params(labelrotation=75)

handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,0]


# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower left')
ax1.margins(x=0, y=0)
# plt.tight_layout()
# plt.savefig('backrunners_profits.png', format='png', dpi=300)
# plt.show()
# exit()


interp_timestamps = []
buf_ys1 = []
buf_ys2 = []
buf_ys3 = []

with open('tmp_backrun.csv') as fin:
    for line in fin:
        ts, ys2, ys3 = line.strip().split(',')
        interp_timestamps.append(float(ts))
        buf_ys1.append(0)
        buf_ys2.append(float(ys2))
        buf_ys3.append(float(ys3))

bin_rights_datetimes = [datetime.datetime.fromtimestamp(t)for t in interp_timestamps]

ax2.fill_between([min(bin_rights_datetimes), max(bin_rights_datetimes)], [0], [100], color='#bbbbbb', label='Unknown')


ax2.fill_between(bin_rights_datetimes, buf_ys1, buf_ys3, label='Back-running', hatch='//')
ax2.fill_between(bin_rights_datetimes, buf_ys1, buf_ys2, label='Not back-running', hatch='\\\\')
ax2.set_ylabel('Count (%)')
ax2.set_xlabel('Date')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
plt.xticks(rotation=75)

handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,0]
ax2.margins(x=0, y=0)

fig.suptitle('Back-running Behavior Over Time')
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right')#, ncol=3, bbox_to_anchor=(0.5, 0.94))
fig.tight_layout()
# plt.tight_layout()
plt.savefig('backrunners.png', format='png', dpi=300)
plt.show()

