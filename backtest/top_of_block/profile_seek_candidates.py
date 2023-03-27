"""
Performance-profile seek_candidates
"""

import argparse
import collections
import math
from multiprocessing.sharedctypes import Value
import signal
import itertools
import random
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
from find_circuit.find import find_upper_bound
from find_circuit.find import DEFAULT_FEE_TRANSFER_CALCULATOR, FeeTransferCalculator, FoundArbitrage, PricingCircuit
import pricers
import find_circuit
import find_circuit.monitor
from pricers.balancer import TooLittleInput
from pricers.base import NotEnoughLiquidityException
from pricers.pricer_pool import PricerPool
from utils import WETH_ADDRESS, get_block_timestamp
import utils.profiling

from .seek_candidates import get_relevant_logs


l = logging.getLogger(__name__)

LOG_BATCH_SIZE = 100

DEBUG = False


def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'profile-seek-candidates'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (run before mass scan)')

    return parser_name, profile_seek_candidates

def profile_seek_candidates(w3: web3.Web3, args: argparse.Namespace):
    l.info('Starting profile of candidate-seek')

    db = connect_db()
    curr = db.cursor()
    time.sleep(4)

    if args.setup_db:
        setup_db(curr)
        l.info('Setup db')
        return

    do_profile(w3, curr)

def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS seek_candidates_profile_strategy_reservations_one_penny (
            id SERIAL NOT NULL PRIMARY KEY,
            start_block INTEGER NOT NULL,
            end_block INTEGER NOT NULL,
            baseline_updated_on         TIMESTAMP WITHOUT TIME ZONE,
            baseline_claimed_on         TIMESTAMP WITHOUT TIME ZONE,
            baseline_completed_on       TIMESTAMP WITHOUT TIME ZONE,

            linear_updated_on         TIMESTAMP WITHOUT TIME ZONE,
            linear_claimed_on         TIMESTAMP WITHOUT TIME ZONE,
            linear_completed_on       TIMESTAMP WITHOUT TIME ZONE
        );

        CREATE TABLE IF NOT EXISTS seek_candidates_profile_results_one_penny (
            id SERIAL NOT NULL PRIMARY KEY,
            block_number INTEGER NOT NULL,
            is_baseline BOOLEAN NOT NULL,
            num_model_queries BIGINT NOT NULL,
            wall_seconds DOUBLE PRECISION NOT NULL CHECK (wall_seconds > 0.0)
        );

        CREATE INDEX IF NOT EXISTS idx_scpr_ib ON seek_candidates_profile_results_one_penny (is_baseline);
        '''
    )
    curr.connection.commit()

    curr.execute('SELECT COUNT(*) FROM seek_candidates_profile_strategy_reservations_one_penny')
    (n_ress,) = curr.fetchone()

    if n_ress > 0:
        l.info('Not filling reservations')
        return

    # select reservations
    r = random.Random(0xDEADBEEF)
    
    # pick 100 sections of 100 blocks within range
    curr.execute('SELECT MIN(start_block), MAX(end_block) FROM block_samples')
    start_block, end_block = curr.fetchone()

    starts = sorted([r.randint(start_block, end_block) for _ in range(100)])

    # ensure no starts are within 100 of each other (unlikely!)
    assert not any(abs(x) <= 100 for x in np.diff(sorted(starts)))

    # insert
    for start in starts:
        curr.execute(
            '''
            INSERT INTO seek_candidates_profile_strategy_reservations_one_penny (
                start_block, end_block
            ) VALUES (%s, %s)
            ''',
            (start, start + 99)
        )
    
    if not DEBUG:
        curr.connection.commit()


def do_profile(w3: web3.Web3, curr: psycopg2.extensions.cursor):

    while True:
        res = get_reservation(curr)
        if res is None:
            break

        res_id, is_baseline, start_block, end_block = res

        try:
            do_profile_range(
                w3,
                curr,
                res_id,
                is_baseline,
                start_block,
                end_block,
            )
        except:
            l.fatal(f'Exception while processing res={res_id} is_baseline={is_baseline}')
            raise

        if is_baseline:
            field = 'baseline_completed_on'
        else:
            field = 'linear_completed_on'

        curr.execute(
            f'''
            UPDATE seek_candidates_profile_strategy_reservations_one_penny
            SET
                {field} = now()::timestamp
            WHERE id = %s
            ''',
            (res_id,)
        )
        if not DEBUG:
            curr.connection.commit()


def get_reservation(curr: psycopg2.extensions.cursor):

    # l.critical('DEBUG IS ON')
    # l.critical('DEBUG IS ON')
    # l.critical('DEBUG IS ON')
    # l.critical('DEBUG IS ON')
    # l.critical('DEBUG IS ON')
    # return 5, False, 9929904, 9930003

    curr.execute(
        '''
        SELECT
            id,
            start_block,
            end_block,
            baseline_claimed_on IS NOT NULL,
            linear_claimed_on IS NOT NULL
        FROM seek_candidates_profile_strategy_reservations_one_penny
        WHERE baseline_claimed_on IS NULL OR linear_claimed_on IS NULL
        ORDER BY id ASC
        FOR UPDATE SKIP LOCKED
        LIMIT 1
        '''
    )

    if curr.rowcount == 0:
        l.info('Done')
        return None
    
    id_, start_block, end_block, baseline_started, linear_started = curr.fetchone()


    if not (baseline_started or linear_started):
        is_baseline = id_ % 2 == 0
    elif baseline_started:
        assert not linear_started
        is_baseline = False
    else:
        assert not baseline_started
        is_baseline = True

    if is_baseline:
        field = 'baseline_claimed_on'
    else:
        field = 'linear_claimed_on'
    
    curr.execute(
        f'''
        UPDATE seek_candidates_profile_strategy_reservations_one_penny
        SET
            {field} = now()::timestamp
        WHERE id = %s
        ''',
        (id_,)
    )
    assert curr.rowcount == 1

    if not DEBUG:
        curr.connection.commit()

    return id_, is_baseline, start_block, end_block

def do_profile_range(
    w3: web3.Web3,
    curr: psycopg2.extensions.cursor,
    res_id: int,
    is_baseline: bool,
    start_block: int,
    end_block: int,
):
    l.info(f'Working on reservation id={res_id} baseline={is_baseline}')
    assert (end_block - start_block) <= 100, f'size too big'

    storage_dir = os.path.join(os.getenv('STORAGE_DIR', '/mnt/goldphish'), 'tmp')

    with tempfile.TemporaryDirectory(dir=storage_dir) as tmpdir:
        pricer = load_pool(w3, curr, tmpdir)
        pricer.warm(start_block)

        for block_number, logs in get_relevant_logs(w3, pricer, start_block, end_block):
            update = pricer.observe_block(block_number, logs)
            process_candidates(w3, pricer, block_number, update, curr, res_id, is_baseline)

    l.debug('Done reservation')

def process_candidates(
        w3: web3.Web3,
        pool: pricers.PricerPool,
        block_number: int,
        updated_exchanges: typing.Dict[typing.Tuple[str, str], typing.List[str]],
        curr: psycopg2.extensions.cursor,
        res_id: int,
        is_baseline: bool,
    ):
    l.debug(f'{len(updated_exchanges)} exchanges updated in block {block_number:,}')

    next_block_ts = get_block_timestamp(w3, block_number + 1)

    n_ignored = 0
    n_found = 0

    old_n_queries = find_circuit.find.count_model_queries

    t_start = time.time()
    kwargs = {}
    if not is_baseline:
        kwargs['detection_func'] = detect_arbitrages_linear
    for p in find_circuit.profitable_circuits(
            updated_exchanges,
            pool,
            block_number,
            timestamp=next_block_ts,
            only_weth_pivot=True,
            **kwargs
        ):
        if p.profit < MIN_PROFIT_PREFILTER:
            n_ignored += 1
            continue

        n_found += 1

    elapsed = time.time() - t_start
    n_queries = find_circuit.find.count_model_queries - old_n_queries

    curr.execute(
        '''
        INSERT INTO seek_candidates_profile_results_one_penny (
            block_number, is_baseline, num_model_queries, wall_seconds
        ) VALUES (%s, %s, %s, %s)
        ''',
        (
            block_number,
            is_baseline,
            n_queries,
            elapsed,
        )
    )

    if is_baseline:
        field = 'baseline_updated_on'
    else:
        field = 'linear_updated_on'

    curr.execute(
        f'''
        UPDATE seek_candidates_profile_strategy_reservations_one_penny
        SET
            {field} = now()::timestamp
        WHERE id = %s
        ''',
        (res_id,)
    )
    if not DEBUG:
        curr.connection.commit()






















# small amount of WETH used by the linear searcher (in wei, about 1 cent)
SMIDGE = (10 ** 18) // 100_000

def detect_arbitrages_linear(
        pc: PricingCircuit,
        block_identifier: int,
        timestamp: typing.Optional[int] = None,
        only_weth_pivot = False,
        try_all_directions = True,
        fee_transfer_calculator: FeeTransferCalculator = DEFAULT_FEE_TRANSFER_CALCULATOR,
    ) -> typing.List[FoundArbitrage]:
    ret = []

    t_start = time.time()

    # for each rotation
    for _ in range(len(pc._circuit) if try_all_directions else 1):
        def run_exc(i):
            amt_in = math.ceil(i)
            price_ratio = pc.sample_new_price_ratio(amt_in, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
            return price_ratio - 1

        # try each direction
        for _ in range(2 if try_all_directions else 1):

            if not (only_weth_pivot and pc.pivot_token != WETH_ADDRESS):
                # quickly try pushing 100 tokens -- if unprofitable, fail

                try:
                    for quick_test_amount_in_zeros in range(5, 25): # start quick test at about 10^-10 dollars (July '22)
                        quick_test_amount_in = 10 ** quick_test_amount_in_zeros
                        try:
                            quick_test_pr = pc.sample_new_price_ratio(quick_test_amount_in, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                            break
                        except TooLittleInput:
                            # try the next largest amount
                            continue
                    else:
                        # exhausted quick_test_amount_in options -- probably there's no way to pump enough liquidity
                        # to this exchange just yet
                        continue
                except NotEnoughLiquidityException:
                    # not profitable most likely
                    continue

                if quick_test_pr > 1:
                    # this may be profitable

                    # search for crossing-point where liquidity does not run out
                    lower_bound = quick_test_amount_in
                    upper_bound = (100_000 * (10 ** 18)) # a shit-ton of ether

                    upper_bound = find_upper_bound(pc, lower_bound, upper_bound, block_identifier, fee_transfer_calculator, timestamp=timestamp)

                    out_lower_bound = pc.sample(lower_bound, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                    out_upper_bound = pc.sample(upper_bound, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)

                    if out_upper_bound < 100:
                        # haven't managed to get anything out with the most money we can pump through, abandon
                        continue

                    if out_lower_bound < 100:
                        # search for crossing-point where (some) positive tokens come out of lower bound
                        lower_bound_search_upper = upper_bound
                        while lower_bound < lower_bound_search_upper - 1000:
                            midpoint = (lower_bound + lower_bound_search_upper) // 2
                            midpoint_out = pc.sample(midpoint, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                            if midpoint_out < 100:
                                lower_bound = midpoint
                                out_lower_bound = midpoint_out
                            else:
                                lower_bound_search_upper = midpoint
                        lower_bound = lower_bound_search_upper

                    assert lower_bound <= upper_bound, f'expect {lower_bound} <= {upper_bound}'

                    # NOTE: it may be the case here that out_lower_bound - lower_bound < 0
                    # i.e, the lower bound is not profitable. This can occur if there is significant
                    # input required to get the first units of output produced -- those are essentially a flat
                    # fee which pushes the pricing "parabola" downward

                    mp_lower_bound = pc.sample_new_price_ratio(lower_bound, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                    mp_upper_bound = pc.sample_new_price_ratio(upper_bound, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)

                    if mp_lower_bound < 1:
                        amount_in = lower_bound
                    elif mp_upper_bound > 1:
                        amount_in = upper_bound
                    else:
                        if mp_lower_bound <= mp_upper_bound:
                            # about to fail, dump info
                            for p, (t_in, t_out) in zip(pc.circuit, pc.directions):
                                l.critical(type(p).__name__, p.address, t_in, t_out)
                            with open('/mnt/goldphish/pts.txt', mode='w') as fout:
                                for amt_in in np.linspace(lower_bound, upper_bound, 200):
                                    amt_in = int(np.ceil(amt_in))
                                    profit = pc.sample(amt_in, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator) - amt_in
                                    price = pc.sample_new_price_ratio(amt_in, block_identifier, timestamp=timestamp, debug=True, fee_transfer_calculator=fee_transfer_calculator)
                                    fout.write(f'{amt_in},{profit},{price}\n')
                            l.critical(f'lower_bound {lower_bound}')
                            l.critical(f'upper_bound {upper_bound}')
                            l.critical(f'mp_lower_bound {mp_lower_bound}')
                            l.critical(f'mp_upper_bound {mp_upper_bound}')
                        assert mp_lower_bound > mp_upper_bound
                        assert 1 < mp_lower_bound

                        # gradually increase to see where profit starts falling

                        last_profit = out_lower_bound
                        amount_in = lower_bound
                        while True:
                            next_in = amount_in + SMIDGE
                            if next_in > upper_bound:
                                l.warning('Exceeded upper bound (unusual)!!!!!')
                                break

                            try:
                                profit = pc.sample(next_in, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator) - next_in
                            except ValueError:
                                l.exception('what is this')
                                break

                            if profit < last_profit:
                                break
                            amount_in = next_in
                            last_profit = profit

                    expected_profit = pc.sample(amount_in, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator) - amount_in

                    # quickly reduce input amount (optimizes for rounding)
                    input_reduction = 0
                    first_token_in, first_token_out = pc.directions[0]
                    first_out_normal, _ = pc.circuit[0].token_out_for_exact_in(first_token_in, first_token_out, amount_in, block_identifier=block_identifier)

                    for i in range(0, 21):
                        attempting_reduction = 10 ** i
                        if attempting_reduction >= amount_in:
                            break

                        try:
                            out_reduced, _ = pc.circuit[0].token_out_for_exact_in(first_token_in, first_token_out, amount_in - attempting_reduction, block_identifier=block_identifier)
                        except NotEnoughLiquidityException:
                            l.critical(f'Ran out of liquidity while sampling {amount_in - attempting_reduction} on {pc.circuit[0].address}')
                            raise

                        if first_out_normal == out_reduced:
                            input_reduction = attempting_reduction
                        else:
                            break

                    amount_in -= input_reduction
                    expected_profit += input_reduction

                    if expected_profit > 0:
                        to_add = FoundArbitrage(
                            amount_in   = amount_in,
                            directions  = pc.directions,
                            circuit     = pc.circuit,
                            pivot_token = pc.pivot_token,
                            profit      = expected_profit,
                        )
                        ret.append(to_add)
            pc.flip()
        pc.rotate()

    return ret
