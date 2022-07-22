import datetime
import time
from backtest.utils import connect_db

db = connect_db()
curr = db.cursor()

curr.execute(
    '''
    SELECT SUM(block_number_end - block_number_start + 1)
    FROM candidate_arbitrage_reservations
    WHERE block_number_start >= 13000000
    '''
)
(total_blocks,) = curr.fetchone()

print(f'Have {total_blocks:,} total blocks to process')

last_marks = []
last_ts = []

while True:
    time.sleep(5)
    db.rollback()
    curr.execute('select sum(progress - block_number_start + 1) from candidate_arbitrage_reservations where progress is not null and claimed_on is not null')
    (n_blocks,) = curr.fetchone()
    last_marks.append(n_blocks)
    last_ts.append(time.time())
    last_marks = last_marks[-100:]
    last_ts = last_ts[-100:]

    if len(last_marks) >= 2:
        period_blocks = last_marks[-1] - last_marks[0]
        elapsed = last_ts[-1] - last_ts[0]
        nps = period_blocks / elapsed

        if nps == 0:
            print('n_blocks', n_blocks)
            continue

        remain = total_blocks - n_blocks
        eta = remain / nps
        eta_td = datetime.timedelta(seconds=eta)
        print(f'{n_blocks:,} of {total_blocks:,} blocks done ({n_blocks / total_blocks * 100:.2f}%) ETA {eta_td}')

