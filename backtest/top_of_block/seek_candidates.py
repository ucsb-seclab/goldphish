import argparse
import collections
import signal
import itertools
import logging
import os
import sys
import time
import typing
import backoff
import numpy as np
import web3
import web3.types
import web3._utils.filters
import psycopg2
import psycopg2.extensions
import tempfile

from backtest.top_of_block.common import load_pool
from backtest.top_of_block.constants import MIN_PROFIT_PREFILTER
from backtest.utils import connect_db
import pricers
import find_circuit
from pricers.pricer_pool import PricerPool
from utils import get_block_timestamp
import utils.profiling


l = logging.getLogger(__name__)

LOG_BATCH_SIZE = 100

DEBUG = False

TMP_REMOVE_ME_FOR_FIXUP_ONLY = True

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'seek-candidates'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (run before mass scan)')
    parser.add_argument('--fixup-queue', action='store_true', help='Fix the queue in the event that a worker had a spurious shutdown')

    return parser_name, seek_candidates

def seek_candidates(w3: web3.Web3, args: argparse.Namespace):
    l.info('Starting candidate searching')
    db = connect_db()
    curr = db.cursor()
    time.sleep(4)

    if args.setup_db and args.fixup_queue:
        print('Cannot have both --setup-db and --fixup-queue', file=sys.stderr)
        exit(1)

    if args.fixup_queue:
        fixup_reservations(curr)
        db.commit()
        return

    if args.setup_db:
        setup_db(curr)
        fill_queue(w3, curr)
        db.commit()
        return

    if args.worker_name is None:
        print('Must supply worker_name', file=sys.stderr)
        exit(1)

    storage_dir = os.path.join(os.getenv('STORAGE_DIR', '/mnt/goldphish'), 'tmp')
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)

    cancel_requested = False
    def set_cancel_requested(_, __):
        nonlocal cancel_requested
        l.info('Cancellation requested')
        cancel_requested = True

    signal.signal(signal.SIGHUP, set_cancel_requested)

    batch_size_blocks = 100 # batch size for getting logs
    with tempfile.TemporaryDirectory(dir=storage_dir) as tmpdir:
        while not cancel_requested:
            l.debug(f'getting new reservation')
            maybe_rez = get_reservation(curr, args.worker_name)
            if maybe_rez is None:
                # we're at the end
                break

            reservation_id, reservation_start, reservation_end = maybe_rez

            # occasionaly database will disconnect while loading the pool
            # (dunno why) -- if that happens, just back off a bit, reconnect,
            # and try again
            def reconnect_db(_):
                nonlocal db
                nonlocal curr
                db = connect_db()
                curr = db.cursor()

            @backoff.on_exception(
                backoff.expo,
                psycopg2.OperationalError,
                max_time = 10 * 60,
                factor = 4,
                on_backoff = reconnect_db,
            )
            def get_pricer_with_retry() -> PricerPool:
                return load_pool(w3, curr, tmpdir)

            pricer = get_pricer_with_retry()
            pricer.warm(reservation_start)

            curr_block = reservation_start
            while curr_block <= reservation_end:
                this_end_block = min(curr_block + batch_size_blocks - 1, reservation_end)
                for block_number, logs in get_relevant_logs(w3, pricer, curr_block, this_end_block):

                    if cancel_requested:
                        l.debug('shutting down main loop')
                        break

                    update = pricer.observe_block(block_number, logs)
                    utils.profiling.maybe_log()
                    while True:
                        try:
                            process_candidates(w3, pricer, block_number, update, curr)
                            if not DEBUG:
                                with utils.profiling.profile('db.update'):
                                    curr.execute(
                                        'UPDATE candidate_arbitrage_reservations SET progress = %s, updated_on = now()::timestamp where id=%s',
                                        (block_number, reservation_id),
                                    )
                                with utils.profiling.profile('db.commit'):
                                    db.commit()
                            break
                        except Exception as e:
                            db.rollback()
                            if 'execution aborted (timeout = 5s)' in str(e):
                                l.exception('Encountered timeout, trying again in a little bit')
                                time.sleep(30)
                            else:
                                raise e

                if cancel_requested:
                    break

                curr_block = this_end_block + 1

            # mark reservation as completed
            if not DEBUG:
                if not cancel_requested:
                    assert this_end_block == reservation_end
                    l.debug(f'Completed reservation id={reservation_id:,}')
                    curr.execute(
                        'UPDATE candidate_arbitrage_reservations SET completed_on = NOW()::timestamp WHERE id = %s',
                        (reservation_id,)
                    )
                    db.commit()
                else:
                    # cancellation was requested
                    curr.execute(
                        '''
                        UPDATE candidate_arbitrage_reservations SET block_number_end = progress, completed_on = NOW()::timestamp WHERE id = %s
                        RETURNING progress
                        ''',
                        (reservation_id,),
                    )
                    assert curr.rowcount == 1
                    (end_inclusive,) = curr.fetchone()

                    if end_inclusive < reservation_end:
                        l.info('Splitting off unifinished reservation into new one')
                        curr.execute(
                            '''
                            INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end, priority)
                            SELECT %s, %s, priority
                            FROM candidate_arbitrage_reservations WHERE id = %s
                            RETURNING id
                            ''',
                            (end_inclusive + 1, reservation_end, reservation_id,)
                        )
                        assert curr.rowcount == 1
                        (new_id,) = curr.fetchone()
                        l.debug(f'Created new reservation id={new_id:,} {end_inclusive + 1:,} -> {reservation_end:,}')
                    db.commit()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_arbitrages (
            id              SERIAL PRIMARY KEY NOT NULL,
            block_number    INTEGER NOT NULL,
            exchanges       bytea[] NOT NULL,
            directions      bytea[] NOT NULL,
            amount_in       NUMERIC(78, 0) NOT NULL,
            profit_no_fee   NUMERIC(78, 0) NOT NULL,
            verify_started  TIMESTAMP WITHOUT TIME ZONE,
            verify_finished TIMESTAMP WITHOUT TIME ZONE,
            verify_run      BOOLEAN DEFAULT FALSE
        );

        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrages_block_number ON candidate_arbitrages (block_number);
        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrages_verify_run ON candidate_arbitrages (verify_run);

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_reservations (
            id                 SERIAL PRIMARY KEY NOT NULL,
            block_number_start INTEGER NOT NULL,
            block_number_end   INTEGER NOT NULL,
            progress           INTEGER,
            worker             TEXT,
            updated_on         TIMESTAMP WITHOUT TIME ZONE,
            claimed_on         TIMESTAMP WITHOUT TIME ZONE,
            completed_on       TIMESTAMP WITHOUT TIME ZONE,
            priority           INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_blocks_to_verify (
            block_number INTEGER PRIMARY KEY NOT NULL,
            max_profit_no_fee NUMERIC(78, 0) NOT NULL,
            verify_started  TIMESTAMP WITHOUT TIME ZONE,
            verify_finished TIMESTAMP WITHOUT TIME ZONE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_blocks_to_verify_block_number ON candidate_arbitrage_blocks_to_verify(block_number);
        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrage_blocks_to_verify_max_profit_no_fee ON candidate_arbitrage_blocks_to_verify (max_profit_no_fee);
        """
    )

    curr.execute('SELECT count(*) FROM candidate_arbitrage_reservations')


def fill_queue(w3: web3.Web3, curr: psycopg2.extensions.cursor):
    curr.execute('SELECT COUNT(*) FROM candidate_arbitrage_reservations')
    (n_queued,) = curr.fetchone()

    if n_queued > 0:
        l.debug('not filling queue')
        return

    # fill queue from sample priorities, splitting 1 into 3
    curr.execute('SELECT start_block, end_block, priority from block_samples order by start_block asc')
    for start_block, end_block, priority in list(curr):
        divide_1 = start_block + (end_block - start_block) // 3
        divide_2 = start_block + (end_block - start_block) * 2 // 3

        curr.execute(
            '''
            INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end, priority)
            VALUES (%s, %s, %s)
            ''',
            (start_block, divide_1, priority),
        )
        curr.execute(
            '''
            INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end, priority)
            VALUES (%s, %s, %s)
            ''',
            (divide_1 + 1, divide_2, priority),
        )
        curr.execute(
            '''
            INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end, priority)
            VALUES (%s, %s, %s)
            ''',
            (divide_2 + 1, end_block, priority),
        )


    curr.execute('SELECT COUNT(*) FROM candidate_arbitrage_reservations')
    (n_inserted,) = curr.fetchone()

    l.info(f'Added {n_inserted} reservations')


def scan_start(curr: psycopg2.extensions.cursor) -> int:
    curr.execute(
        '''
        SELECT LEAST(
            (SELECT MIN(origin_block) FROM uniswap_v2_exchanges),
            (SELECT MIN(origin_block) FROM uniswap_v3_exchanges),
            (SELECT MIN(origin_block) FROM sushiv2_swap_exchanges),
            (SELECT MIN(origin_block) FROM balancer_exchanges),
            (SELECT MIN(origin_block) FROM balancer_v2_exchanges)
        )
        '''
    )
    (ret,) = curr.fetchone()
    return ret


def fixup_reservations(curr: psycopg2.extensions.cursor):
    answer = input('WARNING: this is destructive. ENSURE ALL WORKERS ARE OFF. To continue type YES: ')
    if answer.strip().upper() != 'YES':
        print('Quitting, bye.')
        curr.connection.rollback() # unnecessary but do it anyway
        return

    if time.time() < 1659889982.4060774 + 80 * 60:
        l.warning('Breaking down reservations!!')

        min_res_size = LOG_BATCH_SIZE // 2

        curr.execute(
            '''
            SELECT id, block_number_start, block_number_end, priority
            FROM candidate_arbitrage_reservations
            WHERE completed_on IS NULL and claimed_on IS NULL and (block_number_end - block_number_start + 1) > %s AND priority < 100
            ''',
            (min_res_size * 2,),
        )

        n_closed = curr.rowcount
        l.debug(f'halving {n_closed:,} reservations')
        input('continue?')
        for id_, block_number_start, block_number_end, priority in list(curr):
            midpoint = (block_number_end + block_number_start) // 2
            assert block_number_start < midpoint < block_number_end
            # break in half
            curr.execute(
                '''
                INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end, priority)
                VALUES (%s, %s, %s)
                ''',
                (block_number_start, midpoint, priority),
            )
            curr.execute(
                '''
                INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end, priority)
                VALUES (%s, %s, %s)
                ''',
                (midpoint + 1, block_number_end, priority),
            )
            curr.execute(
                '''
                DELETE FROM candidate_arbitrage_reservations WHERE id = %s
                ''',
                (id_,)
            )

        return


    # count in-progress arbitrages
    curr.execute(
        '''
        SELECT COUNT(*)
        FROM candidate_arbitrage_reservations
        WHERE claimed_on IS NOT NULL AND completed_on IS NULL
        '''
    )
    (n_in_progress,) = curr.fetchone()
    l.debug(f'Have {n_in_progress:,} in-progress reservations')

    # sanity check
    curr.execute('SELECT COUNT(*) FROM candidate_arbitrage_reservations WHERE progress < block_number_start OR progress > block_number_end')
    (n_broken,) = curr.fetchone()
    assert n_broken == 0

    # force completion on anything that should be done but isn't marked yet
    curr.execute(
        '''
        UPDATE candidate_arbitrage_reservations
        SET completed_on = now()::timestamp
        WHERE progress = block_number_end AND claimed_on IS NOT NULL AND completed_on IS NULL
        RETURNING id
        '''
    )
    n_force_closed = curr.rowcount
    l.debug(f'Forced {n_force_closed:,} reservations to completion')

    # split off anything that's partially in-progress
    curr.execute(
        '''
        INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end, priority)
        SELECT progress + 1, block_number_end, priority
        FROM candidate_arbitrage_reservations car
        WHERE car.claimed_on IS NOT NULL AND
              car.completed_on IS NULL AND
              car.progress IS NOT NULL AND
              car.progress < car.block_number_end
        RETURNING block_number_end - block_number_start + 1
        '''
    )
    diffs = [x for (x,) in curr]
    n_new_reservations = len(diffs)
    l.debug(f'Created {n_new_reservations} new reservations')
    if len(diffs) > 0:
        l.debug(f'Median block count in new reservations {int(np.median(diffs))}')

    # force-end in-progress reservations
    curr.execute(
        '''
        UPDATE candidate_arbitrage_reservations car
        SET block_number_end = progress, completed_on = now()::timestamp
        WHERE car.claimed_on IS NOT NULL AND
              car.completed_on IS NULL AND
              car.progress IS NOT NULL AND
              car.progress < car.block_number_end
        '''
    )
    n_closed = curr.rowcount
    assert n_closed == n_new_reservations
    l.debug(f'Forcibly closed {n_closed} in-progress arbitrage search reservations')

    # force re-open reservations with no progress
    curr.execute(
        '''
        UPDATE candidate_arbitrage_reservations car
        SET claimed_on = null
        WHERE
            car.claimed_on IS NOT NULL AND
            car.completed_on IS NULL AND
            car.progress IS NULL
        '''
    )
    n_force_opened = curr.rowcount
    l.debug(f'Force re-opened {n_force_opened} arbitrage search reservations')

    assert n_force_opened + n_new_reservations + n_force_closed == n_in_progress, f'expected {n_new_reservations + n_force_closed} == {n_in_progress}'


def get_reservation(curr: psycopg2.extensions.cursor, worker_name: str) -> typing.Optional[typing.Tuple[int, int, int]]:
    curr.execute('BEGIN TRANSACTION')

    curr.execute(
        '''
        SELECT id, block_number_start, block_number_end
        FROM candidate_arbitrage_reservations
        WHERE claimed_on IS NULL AND completed_on IS NULL
        ORDER BY priority ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        '''
    )
    if curr.rowcount < 1:
        l.info('Finished queue')
        return

    id_, start, end = curr.fetchone()
    curr.execute(
        '''
        UPDATE candidate_arbitrage_reservations car
        SET claimed_on = NOW()::timestamp, worker = %s
        WHERE id = %s
        ''',
        (worker_name, id_),
    )
    assert curr.rowcount == 1

    assert start <= end

    l.info(f'Processing reservation id={id_:,} from={start:,} to end={end:,} ({end - start:,} blocks)')

    if not DEBUG:
        curr.connection.commit()

    return id_, start, end


def get_relevant_logs(
        w3: web3.Web3,
        pool: PricerPool,
        batch_start_block: int,
        batch_end_block: int
    ) -> typing.Iterator[typing.Tuple[int, typing.List[web3.types.LogReceipt]]]:

    assert batch_start_block <= batch_end_block

    l.debug(f'start get logs from {batch_start_block:,} to {batch_end_block:,}')

    with utils.profiling.profile('get_logs'):

        f: web3._utils.filters.Filter = w3.eth.filter({
            'fromBlock': batch_start_block,
            'toBlock': batch_end_block,
        })

        logs = f.get_all_entries()

        l.debug(f'got {len(logs):,} logs this batch')

    important_addresses = pool.monitored_addresses()
    logs = list(x for x in logs if x['address'] in important_addresses)

    gather = collections.defaultdict(lambda: [])
    for log in logs:
        gather[log['blockNumber']].append(log)

    for i in range(batch_start_block, batch_end_block + 1):
        yield (i, gather[i])


def process_candidates(
        w3: web3.Web3,
        pool: pricers.PricerPool,
        block_number: int,
        updated_exchanges: typing.Dict[typing.Tuple[str, str], typing.List[str]],
        curr: psycopg2.extensions.cursor
    ):
    l.debug(f'{len(updated_exchanges)} exchanges updated in block {block_number:,}')

    next_block_ts = get_block_timestamp(w3, block_number + 1)

    n_ignored = 0
    n_found = 0
    max_profit_no_fee = -1
    for p in find_circuit.profitable_circuits(updated_exchanges, pool, block_number, timestamp=next_block_ts, only_weth_pivot=True):
        if p.profit < MIN_PROFIT_PREFILTER:
            n_ignored += 1
            continue

        if False:
            # some debugging
            exchange_outs = {}
            amount = p.amount_in
            for exc, (token_in, token_out) in zip(p.circuit, p.directions):
                amount_out = exc.token_out_for_exact_in(token_in, token_out, amount, block_identifier=block_number, timestamp=next_block_ts)
                amount_out = pricers.token_transfer.out_from_transfer(token_out, amount_out)
                l.debug(f'{exc.address} out={amount_out}')
                amount = amount_out
                exchange_outs[exc.address] = amount_out
            if amount != p.amount_in + p.profit:
                l.warning(f'did not match up with profit! computed amount_out={amount_out} expected {p.amount_in + p.profit}')

            # run it again using fresh pricers
            amount = p.amount_in
            for exc, (token_in, token_out) in zip(p.circuit, p.directions):
                pricer = exc.copy_without_cache()
                amount_out = pricer.token_out_for_exact_in(token_in, token_out, amount, block_identifier=block_number, timestamp=next_block_ts)
                amount_out = pricers.token_transfer.out_from_transfer(token_out, amount_out)
                amount = amount_out
                l.debug(f'fresh {exc.address} out={amount_out}')
                if exchange_outs[exc.address] != amount_out:
                    l.debug(f'Amount_out {exc.address} changed from {exchange_outs[exc.address]} to {amount_out}')

            if amount != p.amount_in + p.profit:
                l.debug('This did not match after fresh pricer!!!!!!!! weird!!!!')
                l.debug('----------------------------')
                l.debug(f'Found arbitrage')
                l.debug(f'block_number ..... {block_number}')
                l.debug(f'amount_in ........ {p.amount_in}')
                l.debug(f'expected profit .. {p.profit}')
                for p, dxn in zip(p.circuit, p.directions):
                    l.debug(f'    address={p.address} direction={dxn}')
                l.debug('----------------------------')
                raise Exception('what is this')


        # this is potentially profitable, log as a candidate

        # ONLY IF IT DOESNT EXIST ALREADY
        if TMP_REMOVE_ME_FOR_FIXUP_ONLY:
            dxns = [bytes.fromhex(t[2:]) for t, _ in p.directions]
            curr.execute(
                '''
                SELECT EXISTS(
                    SELECT 1
                    FROM candidate_arbitrages
                    WHERE block_number = %s AND exchanges = %s AND directions = %s
                );
                ''',
                (block_number, [bytes.fromhex(x.address[2:]) for x in p.circuit], dxns,)
            )
            (exists_already,) = curr.fetchone()
            if exists_already:
                l.warning(f'already exists, ignoring...')
                continue

        with utils.profiling.profile('db.insert'):
            curr.execute(
                """
                INSERT INTO candidate_arbitrages (block_number, exchanges, directions, amount_in, profit_no_fee)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING (id)
                """,
                (block_number, [bytes.fromhex(x.address[2:]) for x in p.circuit], dxns, p.amount_in, p.profit)
            )
        # (inserted_id,) = curr.fetchone()
        # l.debug(f'inserted candidate id={inserted_id}')
        n_found += 1
        
        max_profit_no_fee = max(max_profit_no_fee, p.profit)

    if n_ignored > 0:
        l.debug(f'Ignored {n_ignored} arbitrages due to not meeting profit threshold in block {block_number:,}')
    if n_found > 0:
        l.debug(f'Found {n_found} candidate arbitrages in block {block_number:,}')

