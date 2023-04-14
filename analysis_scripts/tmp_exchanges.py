import datetime
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman Cyr"
plt.rcParams["font.size"] = 12

categories = []
weights = []
with open('tmp_exchanges.csv') as fin:
    for line in fin:
        cat, w = line.strip().split(',')
        w = int(w)
        categories.append(cat)
        weights.append(w)

others = sum(weights[11:])
categories = categories[:11]
weights = weights[:11]
categories.append('Known Other')
weights.append(others)

tup = sorted(zip(categories, weights), key=lambda x: x[1], reverse=True)
categories = [x[0] for x in tup]
weights = [x[1] for x in tup]

colors = ['gray' if c == 'Unknown' else 'C0' for c in categories]

plt.bar(categories, weights, color=colors)

plt.xticks(rotation=75)

plt.title('Use Count of DEX Applications')
plt.xlabel('DEX Application')
plt.ylabel('Count of Uses in Arbitrage')
plt.tight_layout()
plt.savefig('dexes.png', format='png', dpi=300)
plt.show()
