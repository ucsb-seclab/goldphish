from matplotlib import pyplot as plt

dat = []
with open('/home/robert/Downloads/out.csv') as fin:
    for line in fin:
        block,_,dollars = line.strip().split(',')
        dat.append((int(block), float(dollars)))

xs = [x for x, _ in dat]
ys = [y for _, y in dat]

plt.plot(xs, ys)
plt.ylabel('$ Arbitrage opportunity')
plt.xlabel('Block number')
plt.title('WBTC-USDC arbitrage opportunities\nUniswap v2/v3')
plt.show()

xs_dollarvalue = []
ys_duration = []

(prev_block_num, prev_dollars) = dat[0]
for x, y in dat[1:]:
    if not (prev_dollars * 0.99 < y < prev_dollars * 1.01):
        # reset
        xs_dollarvalue.append(prev_dollars)
        ys_duration.append(x - prev_block_num)
        prev_block_num, prev_dollars = x, y

plt.scatter(xs_dollarvalue, ys_duration, s=2)
plt.ylabel('Duration (blocks)')
plt.xlabel('$ of arbitrage')
plt.title('WBTC-USDC arbitrage opportunity durations\nUniswap v2/v3')
plt.show()

