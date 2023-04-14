"""
Attempts to determine which exchanges were used
"""

import collections
import itertools
from statistics import median
from matplotlib.colors import Colormap
import psycopg2
import tabulate
import web3
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from common import gen_false_positives, label_zerox_exchanges, setup_weth_arb_tables

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

parser = argparse.ArgumentParser()
parser.add_argument('--no-sanity', action='store_true', dest='no_sanity')

args = parser.parse_args()


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

label_zerox_exchanges(curr)
gen_false_positives(curr)

# basic facts
curr.execute('SELECT COUNT(*) FROM sample_arbitrage_exchanges_no_fp')
(n_exchanges,) = curr.fetchone()
print(f'Have {n_exchanges:,} exchanges')


curr.execute('SELECT COUNT(*) FROM sample_arbitrage_cycle_exchange_items_no_fp')
(raw_cnt,) = curr.fetchone()

if not args.no_sanity:
    #
    # Sanity check -- ensure every sample_arbitrage_cycle_exchange_items
    # event counts only once

    curr.execute(
        '''
        SELECT COUNT(*)
        FROM sample_arbitrages_no_fp sa
        JOIN sample_arbitrage_cycles_no_fp sac ON sa.id = sac.sample_arbitrage_id
        JOIN sample_arbitrage_cycle_exchanges_no_fp sace ON sac.id = sace.cycle_id
        JOIN sample_arbitrage_cycle_exchange_items_no_fp sacei ON sace.id = sacei.cycle_exchange_id
        '''
    )
    (join_cnt,) = curr.fetchone()
    assert raw_cnt == join_cnt, f'expected {raw_cnt} == {join_cnt}'

    print('sanity check completed.')
    print()


n_exchange_uses = raw_cnt
print(f'Have {n_exchange_uses:,} exchange usages')


#
# Create a temp table to store each exchange's weight
#
curr.execute(
    '''
    CREATE TEMP TABLE tmp_exchange_weights (
        id       INTEGER NOT NULL,
        address  BYTEA NOT NULL,
        weight   INTEGER NOT NULL,
        category TEXT DEFAULT 'Unknown'
    );

    INSERT INTO tmp_exchange_weights
    SELECT sae.id, sae.address, COUNT(*)
    FROM sample_arbitrage_cycle_exchange_items_no_fp sacei
    JOIN sample_arbitrage_exchanges_no_fp sae ON sacei.exchange_id = sae.id
    GROUP BY sae.id, sae.address;
    '''
)
assert curr.rowcount == n_exchanges

#
# Show most weighty exchanges
#
curr.execute(
    '''
    SELECT address, weight
    FROM tmp_exchange_weights
    ORDER BY weight DESC
    LIMIT 5
    '''
)

tab = []
for baddr, weight in curr:
    tab.append((web3.Web3.toChecksumAddress(baddr.tobytes()), weight, f'{weight / n_exchange_uses * 100:.1f}%'))

print(tabulate.tabulate(tab, headers=['Exchange', 'Usages', 'Percent']))
print()


#
# Label all uniswap exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Uniswap v2'
    WHERE EXISTS(SELECT 1 FROM uniswap_v2_exchanges uv2 WHERE tew.address = uv2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

# router
curr.execute(
    '''
    UPDATE tmp_exchange_weights SET category = 'Uniswap v2'
    WHERE encode(address, 'hex') = '7a250d5630b4cf539739df2c5dacb4c659f2488d'
    '''
)
assert curr.rowcount == 1


curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Uniswap v3'
    WHERE EXISTS(SELECT 1 FROM uniswap_v3_exchanges uv3 WHERE tew.address = uv3.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges


#
# Label kyberswap
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Kyberswap'
    WHERE EXISTS(SELECT 1 FROM kyberswap_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label Dodo
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'dodo'
    WHERE EXISTS(SELECT 1 FROM dodo_exchanges e WHERE tew.address = e.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label 1inch
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = '1inch'
    WHERE EXISTS(SELECT 1 FROM oneinch_exchanges e WHERE tew.address = e.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label DeFi Plaza
curr.execute(
    '''
    UPDATE tmp_exchange_weights SET category = 'defi plaza'
    WHERE encode(address, 'hex') = 'e68c1d72340aeefe5be76eda63ae2f4bc7514110'
    '''
)
assert curr.rowcount == 1


#
# Label Balancer
curr.execute(
    '''
    UPDATE tmp_exchange_weights SET category = 'Balancer v2'
    WHERE encode(address, 'hex') = 'ba12222222228d8ba445958a75a0704d566bf2c8'
    '''
)
assert curr.rowcount == 1

curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Balancer v1'
    WHERE EXISTS(SELECT 1 FROM balancer_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label sushiswap exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Sushi Swap'
    WHERE EXISTS(SELECT 1 FROM sushiv2_swap_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges


#
# Label shibaswap exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Shiba Swap'
    WHERE EXISTS(SELECT 1 FROM shibaswap_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges


#
# Label crypto.com exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'crypto.com'
    WHERE EXISTS(SELECT 1 FROM cryptodotcom_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges


#
# Label curve.fi exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'curve.fi'
    WHERE EXISTS(SELECT 1 FROM curvefi_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label indexed.finance exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'indexed.finance'
    WHERE EXISTS(SELECT 1 FROM indexed_finance_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label PowerPool exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'PowerPool'
    WHERE EXISTS(SELECT 1 FROM powerpool_index_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label Orion exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Orion V2'
    WHERE EXISTS(SELECT 1 FROM orion_v2_exchanges s2 WHERE tew.address = s2.address)
    '''
)
# assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label Convergence exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Convergence'
    WHERE EXISTS(SELECT 1 FROM convergence_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label SakeSwap exchanges
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'SakeSwap'
    WHERE EXISTS(SELECT 1 FROM sakeswap_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label Cream Finance
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Cream'
    WHERE EXISTS(SELECT 1 FROM cream_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label Bitberry
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Bitberry'
    WHERE EXISTS(SELECT 1 FROM bitberry_exchanges s2 WHERE tew.address = s2.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label Bancor v2
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Bancor V2'
    WHERE EXISTS(SELECT 1 FROM bancor_v2_exchanges b WHERE tew.address = b.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label WSwap
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'WSwap'
    WHERE EXISTS(SELECT 1 FROM wswap_exchanges b WHERE tew.address = b.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label Equalizer
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Equalizer'
    WHERE EXISTS(SELECT 1 FROM equalizer_exchanges b WHERE tew.address = b.address)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Label zerox exchanges
    # WHERE EXISTS(SELECT 1 FROM tmp_zerox_exchanges tzx WHERE tew.id = tzx.exchange_id)
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = '0x'
    WHERE EXISTS(SELECT 1 FROM tmp_exchanges_with_zerox tzx WHERE tew.id = tzx.exchange_id)
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

# a mapping from uniswap v2 clone factory to name
# this is used to label associated exchanges
clone_factory_map = {
    '0x8a93B6865C4492fF17252219B87eA6920848EdC0': 'SwipeSwap',
}
for address, name in clone_factory_map.items():
    curr.execute(
        '''
        UPDATE tmp_exchange_weights tew SET category = %(category)s
        WHERE EXISTS(
            SELECT 1
            FROM uniswap_v2_clone_exchanges e
            JOIN uniswap_v2_clone_factories f ON e.factory_id = f.id
            WHERE f.address = %(factory_addr)s AND tew.address = e.address
          ) AND tew.category = 'Unknown'
        ''',
        {
            'category': name,
            'factory_addr': bytes.fromhex(address[2:]),
        }
    )



#
# Label uniswap v2 clones that haven't been labeled yet
curr.execute(
    '''
    UPDATE tmp_exchange_weights tew SET category = 'Uniswap V2 Clones'
    WHERE EXISTS(SELECT 1 FROM uniswap_v2_clone_exchanges e WHERE tew.address = e.address) AND
      tew.category = 'Unknown'
    '''
)
assert curr.rowcount > 0
assert curr.rowcount < n_exchanges

#
# Print most-used uniswap v2 clone families
curr.execute(
    '''
    SELECT address, sum_weight
    FROM (
        SELECT f.address, SUM(tew.weight) sum_weight
        FROM uniswap_v2_clone_exchanges e
        JOIN uniswap_v2_clone_factories f ON e.factory_id = f.id
        JOIN tmp_exchange_weights tew ON tew.address = e.address
        WHERE tew.category = 'Uniswap V2 Clones'
        GROUP BY f.address
    ) s
    ORDER BY sum_weight DESC
    '''
)
print(f'Have {curr.rowcount:,} uniswap v2 clone families')

tab = []
for addr, weight in curr:
    addr = web3.Web3.toChecksumAddress(addr.tobytes())
    tab.append([addr, weight, f'{weight / n_exchange_uses * 100:.1f}%'])
print('Most used uniswap v2 clone factories')
print(tabulate.tabulate(tab, headers=['Factory', 'Usages', 'Percent']))
print()


#
# Print most used non-labeled exchange
curr.execute(
    '''
    SELECT id, address, weight
    FROM tmp_exchange_weights
    WHERE category = 'Unknown'
    ORDER BY RANDOM() -- weight DESC
    LIMIT 20
    '''
)

tab = []
for id_, baddr, weight in list(curr):
    addr = web3.Web3.toChecksumAddress(baddr.tobytes())

    # find an example arbitrage with this exchange
    curr.execute(
        '''
        SELECT txn_hash
        FROM sample_arbitrages sa
        JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
        JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
        JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
        WHERE sacei.exchange_id = %s AND
              NOT EXISTS(SELECT 1 FROM sample_arbitrage_cycle_exchange_item_is_zerox is_zerox WHERE is_zerox.sample_arbitrage_cycle_exchange_item_id = sacei.id)
        LIMIT 1
        ''',
        (id_,)
    )
    (txn_hash,) = curr.fetchone()

    tab.append((addr, weight, f'{weight / n_exchange_uses * 100:.1f}%', f'0x' + txn_hash.tobytes().hex()))

print('Most used unlabeled exchanges')
print(tabulate.tabulate(tab, headers=['Exchange', 'Usages', 'Percent', 'Example transaction']))
print()


# Count number of unknown exchange-usages
curr.execute("SELECT SUM(weight) FROM tmp_exchange_weights WHERE category = 'Unknown'")
(n_unexplained_usages,) = curr.fetchone()
curr.execute("SELECT count(*) FROM tmp_exchange_weights WHERE category = 'Unknown'")
(n_unexplained_exchanges,) = curr.fetchone()

print(f'Have {n_unexplained_usages:,} unexplained exchange-usages ({n_unexplained_usages / n_exchange_uses * 100:.4f}%)')
print(f'Have {n_unexplained_exchanges:,} unexplained exchanges ({n_unexplained_exchanges / n_exchanges * 100:.2f}%)')
print()

#
# Display exchange-usage breakdown
curr.execute(
    '''
    SELECT category, sum_weight
    FROM (
        SELECT category, SUM(weight) sum_weight
        FROM tmp_exchange_weights
        GROUP BY category
    ) a
    ORDER BY sum_weight desc
    '''
)

categories = []
weights = []
n_others = 0
for i, (cat, w) in enumerate(curr):
    if i >= 10000:
        n_others += w
    else:
        categories.append(cat)
        weights.append(w)

if n_others > 0:
    categories.append('Known Other')
    weights.append(n_others)

tup = sorted(zip(categories, weights), key=lambda x: x[1], reverse=True)
categories = [x[0] for x in tup]
weights = [x[1] for x in tup]

sum_weights = sum(weights)

for c, w in tup:
    print(f'{c} & ${w:,}$ & {w / sum_weights * 100:.1f}\\%\\')

tab = [(c, w, f'{w / n_exchange_uses * 100:.3f}%') for c, w in zip(categories, weights)]
print('Exchange use categories')
print(tabulate.tabulate(tab, headers=['Exchange Category', 'Count arbitrage uses', 'Percent of uses']))
print()

colors = ['gray' if c == 'Unknown' else 'C0' for c in categories]

with open('tmp_exchanges.csv', mode='w') as fout:
    for cat, w in zip(categories, weights):
        fout.write(f'{cat},{w}\n')

plt.bar(categories, weights, color=colors)

plt.xticks(rotation=90)

plt.title('Usage Count of DEX Applications')
plt.xlabel('DEX Application')
plt.ylabel('Count of Uses in Arbitrage')
plt.tight_layout()
plt.savefig('dexes.png', format='png', dpi=300)
plt.show()


exit()

# find out which 2-exchange arbitrages had the best yields

print('computing WETH arbitrages...')
setup_weth_arb_tables(curr)

curr.execute(
    '''
    SELECT COUNT(*)
    FROM sample_arbitrages_no_fp
    '''
)
(n_arbs_no_fp,) = curr.fetchone()

curr.execute(
    '''
    SELECT COUNT(*)
    FROM tmp_weth_arbs twa
    JOIN sample_arbitrages_no_fp sa ON sa.id = twa.id
    '''
)
(n_weth_arbs_no_fp,) = curr.fetchone()

print(f'Have {n_weth_arbs_no_fp:,} arbitrages that had WETH profit ({n_weth_arbs_no_fp / n_arbs_no_fp * 100:.2f}%)')

curr.execute(
    '''
    SELECT twa.id, twa.revenue, twa.net_profit, array_agg(tw.category)
    FROM tmp_weth_arbs twa
    JOIN sample_arbitrages_no_fp sa ON sa.id = twa.id
    JOIN sample_arbitrage_cycles_no_fp sac ON sa.id = sac.sample_arbitrage_id
    JOIN sample_arbitrage_cycle_exchanges_no_fp sace ON sac.id = sace.cycle_id
    JOIN sample_arbitrage_cycle_exchange_items_no_fp sacei ON sace.id = sacei.cycle_exchange_id
    JOIN tmp_exchange_weights tw ON tw.id = sacei.exchange_id
    GROUP BY twa.id, twa.revenue, twa.net_profit
    ''',
)
assert curr.rowcount == n_weth_arbs_no_fp, f'expected {curr.rowcount} == {n_weth_arbs_no_fp}'

two_cycle_categories = [tuple(sorted(x)) for x in itertools.combinations_with_replacement(categories, 2)]
yields = collections.defaultdict(lambda: [])

for _, revenue, net_profit, cat in curr:
    if len(cat) != 2:
        continue
    if revenue <= 0:
        continue

    yield_ = int(net_profit) / int(revenue) * 100
    cat = tuple(sorted(cat))

    assert cat in two_cycle_categories

    yields[cat].append(yield_)

n_arbs_can_compute_yield = sum(len(x) for x in yields.values())
print(f'Have {n_arbs_can_compute_yield:,} length 2 arbitrages with yield computed ({n_arbs_can_compute_yield / n_weth_arbs_no_fp * 100:.2f}% of WETH arbs)')

matrix = np.zeros((len(categories), len(categories)))
matrix_touched = np.zeros(matrix.shape, dtype=int)

sorted_categories = sorted(categories)

min_median_yield = 100
max_median_yield = 0
for (i1, c1), (i2, c2) in itertools.combinations_with_replacement(list(enumerate(categories)), 2):
    if c1 > c2:
        ((i1, c1), (i2, c2)) = ((i2, c2), (i1, c1))

    if (c1, c2) in yields:
        median_yield = np.median(yields[(c1, c2)])
        if median_yield < -50:
            continue
        min_median_yield = min(min_median_yield, median_yield)
        max_median_yield = max(max_median_yield, median_yield)
        matrix[i1,i2] = median_yield
        matrix[i2,i1] = median_yield
        matrix_touched[i1, i2] = 1
        matrix_touched[i2, i1] = 1

# put -1_000's where we haven't touched a cell
matrix_with_invalids = np.multiply(matrix, matrix_touched) - (1_000 * (1 - matrix_touched))

cmap: Colormap = plt.cm.get_cmap('plasma').copy()
cmap.set_under('gray')

plt.imshow(matrix_with_invalids, cmap=cmap, vmin=min_median_yield, vmax=max_median_yield)
plt.title('2-exchange arbitrage\nmedian percentage yield by niche')
plt.colorbar()
tick_marks = np.arange(len(categories))
plt.xticks(tick_marks, categories, rotation=90)
plt.yticks(tick_marks, categories)
plt.tight_layout()
plt.show()
