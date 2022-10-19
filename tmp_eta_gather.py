import datetime
import time
from backtest.utils import connect_db

db = connect_db()
curr = db.cursor()

curr.execute(
    '''
    SELECT COUNT(*)
    FROM gather_sample_arbitrages_reservations
    '''
)
(n_ress,) = curr.fetchone()

print(f'Have {n_ress:,} total reservations to process')

last_marks = []
last_ts = []

while True:
    time.sleep(5)
    db.rollback()
    curr.execute('select count(*) from gather_sample_arbitrages_reservations where finished_on is not null')
    (n_ress_completed,) = curr.fetchone()
    last_marks.append(n_ress_completed)
    last_ts.append(time.time())
    last_marks = last_marks[-400:]
    last_ts = last_ts[-400:]

    if len(last_marks) >= 2:
        period_ress = last_marks[-1] - last_marks[0]
        elapsed = last_ts[-1] - last_ts[0]
        nps = period_ress / elapsed

        if nps == 0:
            print('n_blocks', period_ress)
            continue

        remain = n_ress - n_ress_completed
        eta = remain / nps
        eta_td = datetime.timedelta(seconds=eta)
        print(f'{n_ress_completed:,} of {n_ress:,} reservations done ({n_ress_completed / n_ress * 100:.2f}%) ETA {eta_td}')

