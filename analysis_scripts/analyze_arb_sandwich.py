import collections
import datetime
import math
import sqlite3
import psycopg2
import tabulate
import os
import numpy as np
import matplotlib.pyplot as plt
import web3

from common import gen_false_positives, setup_backrun_arb_tables, setup_weth_arb_tables

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

# commented out because false positives were already generated
# gen_false_positives(curr)
# exit()

if True:
    # arbitrage tables 
    setup_weth_arb_tables(curr)

    curr.execute(
        '''
        SELECT
            id, net_profit, txn_hash
        FROM tmp_weth_arbs twa
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld
            WHERE jld.sample_arbitrage_id = twa.id
        )
        ORDER BY twa.net_profit DESC
        LIMIT 20
        '''
    )
    tab = []
    for id_, net_profit, txn_hash in curr:
        tab.append((id_, net_profit / 10 ** 18, txn_hash.hex()))
    print(tabulate.tabulate(tab, headers=['ID', 'Profit (ETH)', 'hash']))
    exit()

if True:
    setup_weth_arb_tables(curr)
    setup_backrun_arb_tables(curr)

    curr.execute(
        '''
        CREATE TEMP VIEW tmp_yields AS
        SELECT *, net_profit::float / revenue * 100 as yield
        FROM tmp_weth_arbs
        WHERE revenue > 0;
        '''
    )

    tab = []

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_weth_arbs twa
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
        )
        '''
    )
    row1a = curr.fetchone()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
        FROM tmp_yields tp
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = tp.id
        )
        '''
    )
    row1b = curr.fetchone()
    print(f'\\textbf{{All}} & ', end = '')
    for p in row1a:
        print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    for p in row1b:
        print(f' ${round(p, 2)}$ &', end = '')
    print()

    tab.append(('All', *tuple(x / (10 ** 18) for x in row1a), *row1b))

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_weth_arbs twa
        WHERE EXISTS(
            SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = twa.txn_hash
        )
        AND NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
        )
        '''
    )
    row1a = curr.fetchone()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
        FROM tmp_yields tp
        WHERE EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = tp.txn_hash)
        AND NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = tp.id
        )
        '''
    )
    row1b = curr.fetchone()
    print(f'\\flashbots & ', end = '')
    for p in row1a:
        print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    for p in row1b:
        print(f' ${round(p, 2)}$ &', end = '')
    print()
    tab.append(('Flashbots', *tuple(x / (10 ** 18) for x in row1a), *row1b))

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_weth_arbs twa
        WHERE NOT EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = twa.txn_hash)
        AND NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
        )
        '''
    )
    row1a = curr.fetchone()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
        FROM tmp_yields tp
        WHERE NOT EXISTS(SELECT FROM flashbots_transactions ft WHERE ft.transaction_hash = tp.txn_hash)
        AND NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = tp.id
        )
        '''
    )
    row1b = curr.fetchone()
    print(f'Not \\flashbots & ', end = '')
    for p in row1a:
        print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    for p in row1b:
        print(f' ${round(p, 2)}$ &', end = '')
    print()
    tab.append(('Not-flashbots', *tuple(x / (10 ** 18) for x in row1a), *row1b))

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_weth_arbs twa
        JOIN tmp_backrunners tb ON twa.id = tb.sample_arbitrage_id
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
        )
        '''
    )
    row1a = curr.fetchone()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
        FROM tmp_yields tp
        JOIN tmp_backrunners tb ON tp.id = tb.sample_arbitrage_id
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = tp.id
        )
        '''
    )
    row1b = curr.fetchone()
    print(f'Back-running & ', end = '')
    for p in row1a:
        print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    for p in row1b:
        print(f' ${round(p, 2)}$ &', end = '')
    print()
    tab.append(('Back-running', *tuple(x / (10 ** 18) for x in row1a), *row1b))

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY revenue),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY net_profit),
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY fee),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY fee)
        FROM tmp_weth_arbs twa
        JOIN tmp_not_backrunners tb ON twa.id = tb.sample_arbitrage_id
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
        )
        '''
    )
    row1a = curr.fetchone()

    curr.execute(
        '''
        SELECT
            PERCENTILE_DISC(0.25) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.5) WITHIN GROUP(ORDER BY yield),
            PERCENTILE_DISC(0.75) WITHIN GROUP(ORDER BY yield)
        FROM tmp_yields tp
        JOIN tmp_not_backrunners tb ON tp.id = tb.sample_arbitrage_id
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = tp.id
        )
        '''
    )
    row1b = curr.fetchone()
    print(f'Not back-running & ', end = '')
    for p in row1a:
        print(f' ${round(p / (10 ** 18), 3)}$ &', end = '')
    for p in row1b:
        print(f' ${round(p, 2)}$ &', end = '')
    print()
    tab.append(('Not back-running', *tuple(x / (10 ** 18) for x in row1a), *row1b))


    print(tabulate.tabulate(tab, headers=['Percentile', 'Backrunner profit (ETH)', 'non-backrunner profit (ETH)']))
    print()
    exit()

if False:
    setup_weth_arb_tables(curr)


    # get top arbs
    curr.execute(
        '''
        SELECT txn_hash, net_profit
        FROM tmp_weth_arbs twa
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
        )
        ORDER BY twa.net_profit DESC
        LIMIT 10
        '''
    )
    tab = []
    for txn_hash, p in curr:
        tab.append((txn_hash.tobytes().hex(), p / (10 ** 18)))
    print('Most profitable arbitrages')
    print(tabulate.tabulate(tab, headers=['Txn Hash', 'Profit'], floatfmt=('.4f')))
    print()

    # sum profit made
    # sum profit after fees in WETH and then USD
    curr.execute(
        '''
        SELECT
            SUM(net_profit),
            SUM(net_profit / power(10, 18) * eth_price_usd)
        FROM tmp_weth_arbs twa
        JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
        WHERE NOT EXISTS(
            SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
        )
        ''',
    )
    tot_eth_profit, tot_usd_profit = curr.fetchone()
    print(f'Total ETH profit: {tot_eth_profit / (10 ** 18):,.2f} ETH')
    print(f'Total USD profit: {tot_usd_profit:,.2f} USD')
    print()

    # top relayers
    curr.execute(
        '''
        SELECT *
        FROM (
            SELECT
                SUM(net_profit) sum_profit_eth,
                SUM(net_profit / power(10, 18) * eth_price_usd) sum_profit_usd,
                sa.shooter
            FROM tmp_weth_arbs twa
            JOIN sample_arbitrages sa ON sa.id = twa.id
            JOIN eth_price_blocks ep ON twa.block_number = ep.block_number
            where net_profit is not null
                and NOT EXISTS(
                    SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
                )
            GROUP BY sa.shooter
        ) x
        ORDER BY sum_profit_eth DESC
        LIMIT 10
        '''
    )
    tab = []
    # for npe, npu, addr in curr:
    for npe, npu, addr in curr:
        addr = web3.Web3.toChecksumAddress(addr.tobytes())
        tab.append((
            addr,
            f'{npe / (10 ** 18):,.0f}',
            f'{npe / tot_eth_profit * 100:.2f}%',
            f'{npu:.0f}',
            f'{npu / tot_usd_profit * 100:,.2f}%',
        ))

    print()
    print()
    print(tabulate.tabulate(tab, headers=['Shooter contract', 'Profit (ETH)', 'ETH Profit (% of total)', 'Profit (USD)', '(%)']))



curr.execute(
    '''
    CREATE TEMP TABLE tmp_sandwich_arbs AS
    SELECT
        x.*,
        revenue - fees weth_profit,
        (revenue - fees) / power(10, 18) * eth_price_usd usd_profit
    FROM (
        SELECT
            rear_arbitrage_gain + front_arbitrage_loss revenue,
            rear_arbitrage_gas_price::numeric * rear_arbitrage_gas_used::numeric + rear_arbitrage_coinbase_xfer + front_arbitrage_gas_price::numeric * front_arbitrage_gas_used::numeric fees,
            jld.sample_arbitrage_id,
            relayer,
            front_arbitrage_txn_hash,
            rear_arbitrage_txn_hash,
            block_number
        FROM arb_sandwich_detections jld
        JOIN sample_arbitrage_cycles_no_fp sac ON sac.sample_arbitrage_id = jld.sample_arbitrage_id
        WHERE sac.profit_token = 57
    ) x
    JOIN eth_price_blocks ep ON ep.block_number = x.block_number
    '''
)

#
# sum sandwich profit
curr.execute(
    '''
    SELECT sum(weth_profit) sum_weth_profit, sum(usd_profit) sum_usd_profit
    FROM tmp_sandwich_arbs
    '''
)
sum_weth_profit, sum_usd_profit = curr.fetchone()
print(f'Sandwich profit, WETH: {sum_weth_profit / (10 ** 18):,f}')
print(f'Sandwich profit, USD: {sum_usd_profit:,f}')
print()


#
# sum profitable sandwichers
print('WARNING: this needs to be adjusted for gas fees in the front-running transaction')
print()
curr.execute(
    '''
    SELECT relayer, sum_weth_profit, sum_usd_profit
    FROM (
        SELECT relayer, sum(weth_profit) sum_weth_profit, sum(usd_profit) sum_usd_profit
        FROM tmp_sandwich_arbs
        GROUP BY relayer
    ) x
    ORDER BY sum_weth_profit desc
    LIMIT 10
    '''
)
tab = []
for relayer, weth_profit, usd_profit in curr:
    tab.append((
        web3.Web3.toChecksumAddress(relayer.tobytes()),
        f'{weth_profit / (10 ** 18):,.3f}',
        f'{usd_profit:,.2f}'
    ))
print(tabulate.tabulate(tab, headers=['Relayer', 'WETH profit', 'USD profit']))
print()

#
# largest sandwich profit
curr.execute(
    '''
    SELECT 
        front_arbitrage_txn_hash,
        rear_arbitrage_txn_hash,
        weth_profit
    FROM tmp_sandwich_arbs
    ORDER BY weth_profit
    DESC
    LIMIT 20
    '''
)
tab = []
for tx1, tx2, p in curr:
    tab.append((
        '0x' + tx1.tobytes().hex(),
        '0x' + tx2.tobytes().hex(),
        f'{p / (10 ** 18):.2f}',
    ))
print(tabulate.tabulate(tab, headers=['front txn', 'rear txn', 'profit WETH']))
print()

#
# smallest jit-run profit
#
curr.execute(
    '''
    SELECT 
        front_arbitrage_txn_hash,
        rear_arbitrage_txn_hash,
        weth_profit
    FROM tmp_sandwich_arbs
    ORDER BY weth_profit
    ASC
    LIMIT 20
    '''
)
tab = []
for tx1, tx2, p in curr:
    tab.append((
        '0x' + tx1.tobytes().hex(),
        '0x' + tx2.tobytes().hex(),
        f'{p / (10 ** 18):.2f}',
    ))
print('smallest sandwich profit')
print(tabulate.tabulate(tab, headers=['front txn', 'rear txn', 'profit WETH']))


if False:
    # dump per-block aggregate data for plotting
    setup_weth_arb_tables(curr)

    curr.execute(
        '''
        CREATE TEMP TABLE tmp_backrunners (
            sample_arbitrage_id INTEGER NOT NULL,
            txn_hash BYTEA NOT NULL
        );

        CREATE TEMP TABLE tmp_not_backrunners (
            sample_arbitrage_id INTEGER NOT NULL,
            txn_hash BYTEA NOT NULL
        );

        INSERT INTO tmp_not_backrunners (sample_arbitrage_id, txn_hash)
        SELECT sample_arbitrage_id, sa.txn_hash
        FROM sample_arbitrage_backrun_detections sabd
        JOIN sample_arbitrages_no_fp sa ON sa.id = sabd.sample_arbitrage_id
        WHERE sabd.rerun_exactly = true AND sa.n_cycles = 1 and block_number < 15628035;

        INSERT INTO tmp_backrunners (sample_arbitrage_id, txn_hash)
        SELECT sample_arbitrage_id, sa.txn_hash
        FROM sample_arbitrage_backrun_detections sabd
        JOIN sample_arbitrages_no_fp sa ON sa.id = sabd.sample_arbitrage_id
        WHERE (sabd.rerun_reverted = true OR sabd.rerun_no_arbitrage = true) AND sa.n_cycles = 1 and block_number < 15628035;
        '''
    )

    with open('tmp_marks_sandwich.csv', mode='w') as fout:
        curr.execute(
            '''
            SELECT bs.end_block, SUM(net_profit)
            FROM tmp_weth_arbs twa
            JOIN tmp_backrunners tb ON tb.sample_arbitrage_id = twa.id
            JOIN block_samples bs ON bs.start_block <= twa.block_number AND twa.block_number <= bs.end_block
            WHERE NOT EXISTS(
                SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
            )
            GROUP BY bs.end_block
            ORDER BY bs.end_block ASC 
            '''
        )
        print(f'Got {curr.rowcount}')

        for bs, p in curr:
            fout.write(f'b,{bs},{int(p)}\n')

        curr.execute(
            '''
            SELECT bs.end_block, SUM(net_profit)
            FROM tmp_weth_arbs twa
            JOIN tmp_not_backrunners tb ON tb.sample_arbitrage_id = twa.id
            JOIN block_samples bs ON bs.start_block <= twa.block_number AND twa.block_number <= bs.end_block
            WHERE NOT EXISTS(
                SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
            )
            GROUP BY bs.end_block
            ORDER BY bs.end_block ASC 
            '''
        )

        for bs, p in curr:
            fout.write(f'nb,{bs},{int(p)}\n')

        curr.execute(
            '''
            SELECT bs.end_block, SUM(net_profit)
            FROM tmp_weth_arbs twa
            JOIN block_samples bs ON bs.start_block <= twa.block_number AND twa.block_number <= bs.end_block
            WHERE NOT EXISTS(
                SELECT FROM arb_sandwich_detections jld WHERE jld.sample_arbitrage_id = twa.id
            ) AND NOT EXISTS(
                SELECT FROM tmp_backrunners tb WHERE tb.sample_arbitrage_id = twa.id
            ) AND NOT EXISTS (
                SELECT FROM tmp_not_backrunners tb WHERE tb.sample_arbitrage_id = twa.id
            )
            GROUP BY bs.end_block
            ORDER BY bs.end_block ASC 
            '''
        )

        for bs, p in curr:
            fout.write(f'o,{bs},{int(p)}\n')

        curr.execute(
            '''
            SELECT bs.end_block, SUM(weth_profit)
            FROM tmp_sandwich_arbs tsa
            JOIN block_samples bs ON bs.start_block <= tsa.block_number AND tsa.block_number <= bs.end_block
            GROUP BY bs.end_block
            ORDER BY bs.end_block ASC
            '''
        )

        for bs, p in curr:
            fout.write(f's,{bs},{int(p)}\n')

