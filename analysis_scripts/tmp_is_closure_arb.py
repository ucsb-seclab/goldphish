import collections
import datetime
import math
import sqlite3
import psycopg2
import common
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


curr.execute('SELECT COUNT(*) FROM top_candidate_arbitrage_campaigns')
(n_campaigns,) = curr.fetchone()


curr.execute(
    '''
    select count(*) cnt, sum(case when terminating_transaction is not null then 1 else 0 end) cnt_found
    from top_candidate_arbitrage_campaign_terminations
    '''
)
tot_checked, tot_found_termination = curr.fetchone()

print(f'Checked {tot_checked:,} of {n_campaigns:,} campaigns ({tot_checked / n_campaigns * 100:.1f}%)')
print(f'Found {tot_found_termination:,} terminations for {tot_checked:,} campaigns ({tot_found_termination / tot_checked * 100:.2f}%)')

remove_backrun = True
if remove_backrun:
    curr.execute(
        '''
        CREATE TEMP TABLE arbs_that_terminate AS
        SELECT sa.id sample_arbitrage_id, tcac.campaign_id, sa.block_number
        FROM (SELECT * FROM top_candidate_arbitrage_campaign_terminations WHERE terminating_transaction is not null) tcac
        JOIN sample_arbitrages_no_fp sa
        ON sa.txn_hash = tcac.terminating_transaction
        '''
    )
    n_are_arbs = curr.rowcount
    print(f'Found {n_are_arbs:,} arbitrages among {tot_found_termination:,} terminations ({n_are_arbs / tot_found_termination * 100:.2f}%)')
    curr.execute(
        '''
        DELETE FROM arbs_that_terminate att
        WHERE EXISTS (
            SELECT
            FROM sample_arbitrage_backrun_detections bd
            WHERE bd.sample_arbitrage_id = att.sample_arbitrage_id AND (rerun_reverted = True or rerun_no_arbitrage = True)
        )
        '''
    )
    n_backrun = curr.rowcount
    print(f'Removed {n_backrun:,} that were back-running ({n_backrun / n_are_arbs * 100:.2f}%)')
    n_are_arbs -= n_backrun

else:
    curr.execute(
        '''
        CREATE TEMP TABLE arbs_that_terminate AS
        SELECT sa.id sample_arbitrage_id, tcac.campaign_id, sa.block_number
        FROM (SELECT * FROM top_candidate_arbitrage_campaign_terminations WHERE terminating_transaction is not null) tcac
        JOIN sample_arbitrages_no_fp sa
        ON sa.txn_hash = tcac.terminating_transaction
        '''
    )
    n_are_arbs = curr.rowcount
    print(f'Found {n_are_arbs:,} arbitrages among {tot_found_termination:,} terminations ({n_are_arbs / tot_found_termination * 100:.2f}%)')

common.setup_weth_arb_tables(curr)

curr.execute(
    '''
    CREATE TEMP TABLE tmp_last_profit_before_terminate AS
    SELECT tcarr.campaign_id, max(ca.block_number) block_number
    FROM arbs_that_terminate att
    JOIN top_candidate_arbitrage_relay_results tcarr on tcarr.campaign_id = att.campaign_id
    join candidate_arbitrages ca on ca.id = tcarr.candidate_arbitrage_id and ca.block_number < att.block_number
    group by tcarr.campaign_id
    '''
)


curr.execute(
    '''
    CREATE TEMP TABLE tmp_last_profits AS
    SELECT tcarr.campaign_id, tcarr.real_profit_before_fee
    FROM tmp_last_profit_before_terminate lp
    JOIN top_candidate_arbitrage_relay_results tcarr on tcarr.campaign_id = lp.campaign_id
    join candidate_arbitrages ca on ca.id = tcarr.candidate_arbitrage_id and ca.block_number = lp.block_number
    '''
)

curr.execute(
    '''
    SELECT tmp.real_profit_before_fee, twa.revenue
    FROM arbs_that_terminate att
    JOIN tmp_last_profits tmp ON att.campaign_id = tmp.campaign_id
    LEFT JOIN tmp_weth_arbs twa ON twa.id = att.sample_arbitrage_id
    '''
)

mine = []
theirs = []
n_not_weth = 0
for mbf, rev in curr:
    if rev is None:
        n_not_weth += 1
        continue
    mine.append(int(mbf) / (10 ** 18))
    theirs.append(int(rev) / (10 ** 18))

print(f'Have {n_not_weth:,} arbitrages that were not WETH {n_not_weth / n_are_arbs * 100:.2f}%')

diffs = np.array(mine) - np.array(theirs)

marks = [5, 10, 25, 50, 75, 90, 95]
tab = zip(marks, np.percentile(diffs, marks))
print(tabulate.tabulate(tab, headers=['Percentile', 'Difference (me - them) (ETH)']))


plt.hist(diffs, bins=20)
plt.show()
