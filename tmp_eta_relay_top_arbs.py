import datetime
import time
from backtest.utils import connect_db

db = connect_db()
curr = db.cursor()

curr.execute(
    '''
    SELECT COUNT(*) FROM large_candidate_arbitrages
    ''',
)

(n_arbitrages,) = curr.fetchone()

print(f'Have {n_arbitrages:,} large arbitrages in total to relay')

curr.execute(
    '''
    CREATE TEMP TABLE tmp_count_by_block AS
    SELECT block_number, count(*)::integer as cnt
    FROM large_candidate_arbitrages
    GROUP BY block_number;

    CREATE INDEX tmp_cbb_bn_idx ON tmp_count_by_block (block_number);
    '''
)
# print(f'Have {curr.rowcount:,} unique blocks to process')

last_marks = []
last_ts = []

while True:
    time.sleep(5)
    db.commit()
    curr.execute(
        '''
        SELECT SUM(cnt)::integer
        FROM (SELECT * FROM top_candidate_arbitrage_reservations WHERE claimed_on IS NOT NULL) tcar
        JOIN tmp_count_by_block tcb ON tcar.start_block <= tcb.block_number AND tcb.block_number <= LEAST(tcar.end_block, tcar.progress)
        '''
    )
    (n_processed,) = curr.fetchone()
    last_marks.append(n_processed)
    last_ts.append(time.time())
    last_marks = last_marks[-500:]
    last_ts = last_ts[-500:]

    if len(last_marks) >= 2:
        period_processed = last_marks[-1] - last_marks[0]
        elapsed = last_ts[-1] - last_ts[0]
        nps = period_processed / elapsed

        if nps == 0:
            print('n_processed', n_processed)
            continue

        remain = n_arbitrages - n_processed
        eta_to_target = remain / nps
        eta_to_target_td = datetime.timedelta(seconds=eta_to_target)
        print(f'{n_processed:,} of {n_arbitrages:,} top arbs relayed ({n_processed / n_arbitrages * 100:.2f}%) ETA {eta_to_target_td}')

