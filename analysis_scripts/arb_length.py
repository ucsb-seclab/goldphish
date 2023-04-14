import psycopg2
from matplotlib import pyplot as plt


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


blocks_opens = []
profit_no_fees = []

curr.execute(
    '''
    SELECT
        (end_block_exclusive - start_block),
        max_profit_wei_no_fee
    FROM candidate_arbitrage_opportunities
    WHERE end_block_exclusive IS NOT NULL
    '''
)

for blocks, profits in curr:
    blocks_opens.append(int(blocks))
    profit_no_fees.append(int(profits))


max_open_range = max(blocks_opens)
min_open_range = min(blocks_opens)

assert len(blocks_opens) == len(profit_no_fees)

print(f'have info for {len(blocks_opens)} arbitrages')
print(f'max time arbitrage was open: {max_open_range:,} blocks')
print(f'min time arbitrage was open: {min_open_range:,} blocks')


profit_no_fees_eth = [x / (10 ** 18) for x in profit_no_fees]


plt.scatter(profit_no_fees_eth, blocks_opens, s=2)
plt.xlabel('max profit')
plt.ylabel('blocks open')
plt.show()
