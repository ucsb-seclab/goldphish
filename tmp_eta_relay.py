import datetime
import time
from backtest.utils import connect_db

db = connect_db()
curr = db.cursor()

TARGET_PRIORITY = 29

curr.execute(
    '''
    SELECT COUNT(*)
    FROM candidate_arbitrage_reshoot_blocks
    WHERE priority <= %s
    ''',
    (TARGET_PRIORITY,)
)

(n_blocks_to_target,) = curr.fetchone()

print(f'Have {n_blocks_to_target:,} blocks in total to relay to complete to priority {TARGET_PRIORITY}')

last_marks = []
last_ts = []

while True:
    time.sleep(5)
    db.rollback()
    curr.execute('select count(*) from candidate_arbitrage_reshoot_blocks where completed_on is not null')
    (n_blocks,) = curr.fetchone()
    last_marks.append(n_blocks)
    last_ts.append(time.time())
    last_marks = last_marks[-100:]
    last_ts = last_ts[-100:]

    if len(last_marks) >= 2:
        curr.execute(
            '''
            SELECT priority
            FROM candidate_arbitrage_reshoot_blocks
            WHERE completed_on IS NULL
            ORDER BY priority ASC
            LIMIT 1
            '''
        )
        (priority_in_progress,) = curr.fetchone()

        curr.execute(
            '''
            SELECT COUNT(*), SUM(CASE WHEN completed_on IS NULL THEN 0 ELSE 1 END)
            FROM candidate_arbitrage_reshoot_blocks
            WHERE priority = %s
            ''',
            (priority_in_progress,)
        )

        n_in_priority, n_completed = curr.fetchone()

        period_blocks = last_marks[-1] - last_marks[0]
        elapsed = last_ts[-1] - last_ts[0]
        nps = period_blocks / elapsed

        if nps == 0:
            print('n_blocks', n_blocks)
            continue

        remain_in_priority = n_in_priority - n_completed
        eta_this_priority = remain_in_priority / nps
        eta_this_priority_td = datetime.timedelta(seconds=eta_this_priority)
        print(f'{n_completed:,} of {n_in_priority:,} blocks done in priority {priority_in_progress} ({n_completed / n_in_priority * 100:.2f}%) ETA {eta_this_priority_td}')

        remain = n_blocks_to_target - n_blocks
        eta_to_target = remain / nps
        eta_to_target_td = datetime.timedelta(seconds=eta_to_target)
        print(f'{n_blocks:,} of {n_blocks_to_target:,} blocks done to target {TARGET_PRIORITY} ({n_blocks / n_blocks_to_target * 100:.2f}%) ETA {eta_to_target_td}')

