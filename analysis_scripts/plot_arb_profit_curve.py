import matplotlib.pyplot as plt
import scipy.optimize

xs = []
ys1 = []
ys2 = []

with open('/home/robert/Downloads/plot_points.csv') as fin:
    for line in fin:
        if line.startswith('#'):
            continue
        _,x,_,y1,_,y2 = line.strip().split(',')
        x = float(x)
        y1 = float(y1)
        y2 = float(y2)
        xs.append(x)
        ys1.append(y1)
        ys2.append(y2)
        if x >= 175_000:
            break

def quad_func(x, a, b, c):
    return a * (x ** 2) + b * x + c


opt = scipy.optimize.curve_fit(quad_func, xdata=xs, ydata=ys1, p0=[-1, 10, 100])

optimized = lambda x: quad_func(x, *(opt[0]))

ys_fit = [optimized(x) for x in xs]

plt.plot(xs, ys1)
# plt.plot(xs, ys_fit)
plt.title('Profit curve\nblock 13,601,583 USDC-HEX arbitrage')
plt.xlabel('$ In')
plt.ylabel('$ Profit')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.axhline(y=679.60, color='green', linewidth=0.5)
# plt.plot(xs, ys2)
plt.show()
