"""
find_arb_termination.py

Finds out when each arbitrage opportunity closed, and why.
"""
import asyncio
import collections
import datetime
import os
import asyncpg
import itertools
import tempfile
import time
from backtest.top_of_block.relay import AutoAdaptShootSuccess, InferredTokenTransferFeeCalculator, auto_adapt_attempt_shoot_candidate, load_pricer_for, open_ganache
from backtest.utils import connect_db
from backtest.top_of_block.common import load_pool
from backtest.top_of_block.seek_candidates import get_relevant_logs
import argparse
import psycopg2.extensions
import typing
import numpy as np
import logging
from find_circuit.find import PricingCircuit, detect_arbitrages_bisection

from pricers.pricer_pool import PricerPool
from utils import ProgressReporter, connect_web3, get_block_timestamp
import web3
import web3.types
import web3.contract
import web3._utils.filters


l = logging.getLogger(__name__)

class CandidateArbitrageCampaign(typing.NamedTuple):
    arbs: typing.List[typing.Tuple[int, int]]
    niche: str
    gas_pricer: int
    block_number_start: int
    block_number_end: int
    max_profit_after_fee_wei: int
    min_profit_after_fee_wei: int


def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'do-arb-duration'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (before run)')

    parser.add_argument('--fill-modified', action='store_true')
    parser.add_argument('--id', type=int, help='worker id, required for processing', default=0)
    parser.add_argument('--n-workers', type=int, help='number of workers', default=1)

    return parser_name, do_fill_duration


def do_fill_duration(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        db.commit()
        return


    if args.fill_modified:
        assert args.id >= 0 and args.n_workers > 0 and args.id < args.n_workers
        fill_modified_exchanges(w3,args, db, curr)
        return

    # curr.execute(
    #     '''
    #     CREATE TEMP TABLE tmp_bad_cac AS
    #     SELECT cac.id FROM candidate_arbitrage_campaigns cac
    #     JOIN block_samples bs ON bs.start_block <= cac.block_number_end AND cac.block_number_end <= bs.end_block
    #     WHERE priority >= 16;


    #     DELETE FROM candidate_arbitrage_campaigns cac
    #     WHERE id IN (SELECT tb.id FROM tmp_bad_cac tb)
    #     '''
    # )
    # l.info(f'deleted {curr.rowcount} rows')
    # input('continue?')
    # db.commit()

    # return

    curr.execute(f'SELECT MIN(start_block), MAX(end_block) FROM block_samples')
    min_block, max_block = curr.fetchone()

    gas_oracle_pts = get_gas_oracle(curr, min_block, max_block)
    origin_blocks = get_exchange_origin_blocks(curr)

    # go by reservation order
    curr.execute(
        '''
        SELECT start_block, end_block, priority
        FROM block_samples
        WHERE priority = %s
        ''',
        (args.id,)
    )

    for start_block, end_block, priority in list(curr):
        process_sample(w3, curr, min_block, max_block, start_block, end_block, priority, gas_oracle_pts, origin_blocks, args.id)

def get_exchange_origin_blocks(curr: psycopg2.extensions.cursor) -> typing.Dict[bytes, int]:
    l.debug(f'getting exchange origin blocks')
    curr.execute(
        '''
        SELECT address, origin_block FROM uniswap_v2_exchanges
        UNION SELECT address, origin_block FROM uniswap_v3_exchanges
        UNION SELECT address, origin_block FROM sushiv2_swap_exchanges
        UNION SELECT address, origin_block FROM shibaswap_exchanges
        UNION SELECT address, origin_block FROM balancer_exchanges
        UNION SELECT address, origin_block FROM balancer_v2_exchanges
        '''
    )
    ret = {a.tobytes(): b for a, b in curr}
    l.debug(f'loaded {len(ret):,} origin blocks')
    return ret


def get_gas_oracle(curr: psycopg2.extensions.cursor, min_block: int, max_block: int) -> typing.Dict[
        str,
        typing.Tuple[typing.List[int], typing.Tuple[typing.List[int], typing.List[int], typing.List[int], typing.List[int], typing.List[int]]]]:
    l.debug(f'interpolating from {min_block:,} to {max_block:,}')

    curr.execute('SELECT DISTINCT niche FROM naive_gas_price_estimate')
    all_niches = set(x for (x,) in curr)
    l.debug(f'Have {len(all_niches):,} niches')

    ret = {}

    for i, niche in enumerate(sorted(all_niches)):
        l.info(f'Processing niche {i + 1} of {len(all_niches)}')

        curr.execute(
            '''
            SELECT block_number, gas_price_min, gas_price_25th, gas_price_median, gas_price_75th, gas_price_max
            FROM naive_gas_price_estimate
            WHERE niche = %s
            ORDER BY block_number ASC
            ''',
            (niche,)
        )

        block_numbers = []
        pts = ([], [], [], [], [])

        for row in curr:
            block_numbers.append(row[0])
            for i, pt in enumerate(row[1:]):
                pts[i].append(int(pt))
        
        for i in range(len(pts)):
            assert len(block_numbers) == len(pts[i])

        ret[niche] = (block_numbers, pts)

    return ret

def process_sample(
        w3: web3.Web3,
        curr2: psycopg2.extensions.cursor,
        min_block: int,
        max_block: int,
        start_block: int,
        end_block: int,
        priority: int,
        gas_oracle_pts: typing.Dict[
            str,
            typing.Tuple[typing.List[int], typing.Tuple[typing.List[int], typing.List[int], typing.List[int], typing.List[int], typing.List[int]]]
        ],
        origin_blocks: typing.Dict[bytes, int],
        worker_id: int
    ):
    l.info(f'Processing priority={priority}')

    does_not_need_lookback = set()
    running_pre_existing = set()
    # keeps a running record of ongoing arbitrages by niche
    running_arbitrages: typing.Dict[typing.Any, CandidateArbitrageCampaign] = {}

    exchanges_updated_since_start = set()
    
    def flush_arbitrage(exchanges, directions, niche, gas_price_group):

        maybe_running_arb = running_arbitrages.get((exchanges, directions, niche, gas_price_group), None)
        if maybe_running_arb is None:
            return
        running_arb = maybe_running_arb

        sz_gas_price_group = {
            0: 'minimum',
            1: '25th percentile',
            2: 'median',
            3: '75th percentile',
            4: 'maximum',
        }[gas_price_group]


        curr2.execute(
            '''
            INSERT INTO candidate_arbitrage_campaigns
            (niche, gas_pricer, block_number_start, block_number_end, max_profit_after_fee_wei, min_profit_after_fee_wei)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            ''',
            (niche, sz_gas_price_group, running_arb.block_number_start, running_arb.block_number_end, running_arb.max_profit_after_fee_wei, running_arb.min_profit_after_fee_wei)
        )
        assert curr2.rowcount == 1
        (id_,) = curr2.fetchone()
        for arb_id, profit in running_arb.arbs:
            curr2.execute(
                '''
                INSERT INTO candidate_arbitrage_campaign_member (candidate_arbitrage_id, candidate_arbitrage_campaign, profit_after_fee_wei)
                VALUES (%s, %s, %s)
                ''',
                (arb_id, id_, profit),
            )
            assert curr2.rowcount == 1

        del running_arbitrages[(exchanges, directions, niche, gas_price_group)]
        return

    def flush_all_arbitrages(exchanges, directions):
        for (e, d, n, i) in list(running_arbitrages.keys()):
            if e == exchanges and d == directions:
                flush_arbitrage(e, d, n, i)

    def push_running_arbitrage(exchanges, directions, niche, gas_price_group, block_number: int, profit: int, arb_id: int):
        k = (exchanges, directions, niche, gas_price_group)
        if k not in running_arbitrages:
            running_arbitrages[k] = CandidateArbitrageCampaign(
                arbs = [(arb_id, profit)],
                niche = niche,
                gas_pricer = gas_price_group,
                block_number_start = block_number,
                block_number_end = block_number,
                max_profit_after_fee_wei = profit,
                min_profit_after_fee_wei = profit,
            )
        else:
            existing: CandidateArbitrageCampaign = running_arbitrages[k]
            existing.arbs.append((arb_id, profit))
            existing = existing._replace(block_number_end = block_number)
            existing = existing._replace(max_profit_after_fee_wei = max(profit, existing.max_profit_after_fee_wei))
            existing = existing._replace(min_profit_after_fee_wei = min(profit, existing.min_profit_after_fee_wei))
            running_arbitrages[k] = existing

    relay_cache = {}

    def requires_lookback(exchanges, directions, niche, gas_price_group) -> bool:
        k_minor = (exchanges, directions)
        k = (exchanges, directions, niche, gas_price_group)
        if k_minor in does_not_need_lookback or k in does_not_need_lookback:
            return False

        if len(exchanges_updated_since_start.intersection(exchanges)) > 0:
            # this updated since we started the scan, if it isn't already in the pre existing
            # arbitrage set then it doesn't need lookback, its definitely fresh
            does_not_need_lookback.add(k_minor)
            return False

        # find the most recent prior block where this was updated
        curr2.execute(
            '''
            SELECT block_number
            FROM exchanges_updated_in_block
            WHERE block_number < %s AND exchange_address = ANY (%s)
            ORDER BY block_number DESC
            LIMIT 1
            ''',
            (start_block, list(exchanges))
        )
        if curr2.rowcount < 1:
            # if not updated in prior block then this is fresh
            does_not_need_lookback.add(k_minor)
            return False

        assert curr2.rowcount == 1
        (most_recent_prior_updated_block,) = curr2.fetchone()

        # if this is prior to origin of any exchanges, then no arbitrage here
        for e in exchanges:
            if origin_blocks[e] > most_recent_prior_updated_block:
                does_not_need_lookback.add(k_minor)
                return False

        # detect arbitrages here
        circuit = []
        for exc in exchanges:
            exc = w3.toChecksumAddress(exc)
            pricer = load_pricer_for(w3, curr2, exc)
            circuit.append(pricer)
        directions = [web3.Web3.toChecksumAddress(x) for x in directions]
        directions = list(zip(directions, directions[1:] + [directions[0]]))

        pc = PricingCircuit(circuit, directions)

        prior_timestamp = get_block_timestamp(w3, most_recent_prior_updated_block + 1)
        maybe_arb = detect_arbitrages_bisection(
            pc,
            most_recent_prior_updated_block,
            prior_timestamp,
            try_all_directions = False,
        )
        if len(maybe_arb) == 0:
            # no prior arbitrage
            l.debug(f'no prior arbitrage')
            does_not_need_lookback.add(k_minor)
            return False

        (arb,) = maybe_arb

        # immediately dismiss prior arbitrage if pre-shoot profit is already below any profitability
        if niche_sz.endswith('|2|balv2|') or niche_sz.endswith('|3|balv2|'):
            # special case -- we cannot find these in our scrape so we have no oracle

            # just use uniswap v3 instead
            replacement_niche_sz = niche_sz.replace('|balv2|', '|uv3|')
            blocks = gas_oracle_pts[replacement_niche_sz][0]
            pts = gas_oracle_pts[replacement_niche_sz][1][gas_price_group]
        else:
            blocks = gas_oracle_pts[niche_sz][0]
            pts = gas_oracle_pts[niche_sz][1][gas_price_group]

        most_generous_gas_price = int(np.interp(most_recent_prior_updated_block, blocks, pts))
        generous_gas_usage = 100_000
        if arb.profit < most_generous_gas_price * generous_gas_usage:
            l.debug(f'Does not meet generous params')
            does_not_need_lookback.add(k_minor)
            return False

        maybe_cached_result = relay_cache.get((exchanges, tuple(directions)), None)
        if maybe_cached_result is not None:
            result = maybe_cached_result
        else:
            # attempt to relay the arbitrage
            with tempfile.TemporaryDirectory(dir='/mnt/goldphish/tmp') as tmpdir:
                proc, w3_ganache, acct, shooter_address = open_ganache(most_recent_prior_updated_block, tmpdir, worker_id)

                result = auto_adapt_attempt_shoot_candidate(
                    w3_ganache,
                    curr2,
                    acct,
                    shooter_address,
                    arb,
                    InferredTokenTransferFeeCalculator(),
                    prior_timestamp,
                    must_recompute = False,
                )

                proc.kill()
                proc.wait()
            relay_cache[(exchanges, tuple(directions))] = result

        if not isinstance(result, AutoAdaptShootSuccess):
            does_not_need_lookback.add(k_minor)
            return False
        
        prior_profit_after_generous_fee = result.profit_no_fee - result.gas * most_generous_gas_price
        if prior_profit_after_generous_fee < 0:
            does_not_need_lookback.add(k_minor)
            return False

        # this was pre-existing, compare the exact prior predicted gas price
        blocks = gas_oracle_pts[niche][0]
        pts = gas_oracle_pts[niche][1][gas_price_group]
        prior_gas_price = int(np.interp(most_recent_prior_updated_block, blocks, pts))
        prior_profit = result.profit_no_fee - result.gas * prior_gas_price
        if prior_profit < 0:
            does_not_need_lookback.add(k)
            return False

        # this was pre-existing at the desired gas oracle price
        l.debug(f'this was pre-existing')
        return True

    t_start = time.time()
    t_last_update = time.time()

    for block_number in range(start_block, end_block + 1):

        if t_last_update + 20 < time.time():
            processed = block_number - start_block
            if processed > 0:
                elapsed = time.time() - t_start
                nps = processed / elapsed
                remain = end_block - block_number + 1
                tot = end_block - start_block + 1
                eta_sec = remain / nps
                eta = datetime.timedelta(seconds = eta_sec)
                l.info(f'Processed {processed:,} of {tot:,} blocks ({processed / tot * 100:.2f}%) ETA {eta}')

            t_last_update = time.time()

        gas_price_cache = {}
        still_running_pre_existing_this_block = set()

        curr2.execute(
            '''
            SELECT
                id,
                exchanges,
                directions,
                shoot_success,
                gas_used,
                real_profit_before_fee,
                EXISTS(SELECT 1 FROM uniswap_v2_exchanges WHERE address = ANY (exchanges)) has_uniswap_v2,
                EXISTS(SELECT 1 FROM uniswap_v3_exchanges WHERE address = ANY (exchanges)) has_uniswap_v3,
                EXISTS(SELECT 1 FROM sushiv2_swap_exchanges WHERE address = ANY (exchanges)) has_sushiswap,
                EXISTS(SELECT 1 FROM shibaswap_exchanges WHERE address = ANY (exchanges)) has_shibaswap,
                EXISTS(SELECT 1 FROM balancer_exchanges WHERE address = ANY (exchanges)) has_balancer_v1,
                EXISTS(SELECT 1 FROM balancer_v2_exchanges WHERE address = ANY (exchanges)) has_balancer_v2
            FROM candidate_arbitrages ca
            JOIN candidate_arbitrage_relay_results carr ON carr.candidate_arbitrage_id = ca.id
            where block_number = %s
            ORDER BY ca.id ASC
            ''',
            (block_number,),
        )


        for \
            candidate_id, \
            exchanges, \
            directions, \
            shoot_success, \
            gas_used, \
            real_profit_before_fee, \
            has_uniswap_v2, \
            has_uniswap_v3, \
            has_sushiswap, \
            has_shibaswap, \
            has_balancer_v1, \
            has_balancer_v2 \
            in list(curr2):

            exchanges = tuple(x.tobytes() for x in exchanges)
            directions = tuple(d.tobytes() for d in directions)

            k_minor = (exchanges, directions)

            if not shoot_success:
                flush_all_arbitrages(k_minor)
                does_not_need_lookback.add(k_minor)
                for (e, d, n, i) in list(running_pre_existing):
                    if e == exchanges and d == directions:
                        running_pre_existing.remove((e, d, n, i))
                continue

            for is_flashbots in [True, False]:
                # compute the niche
                niche_sz = ''


                if is_flashbots:
                    niche_sz += 'fb|'
                else:
                    niche_sz += 'nfb|'
                
                niche_sz += str(len(exchanges)) + '|'
                
                if has_uniswap_v2:
                    niche_sz += 'uv2|'
                if has_uniswap_v3:
                    niche_sz += 'uv3|'
                if has_sushiswap:
                    niche_sz += 'sushi|'
                if has_shibaswap:
                    niche_sz += 'shiba|'
                if has_balancer_v1:
                    niche_sz += 'balv1|'
                if has_balancer_v2:
                    niche_sz += 'balv2|'

                # for each gas price in this niche
                for gas_price_group in range(5): # one for min, 25pct, median, 75pct, max
                    k = (exchanges, directions, niche_sz, gas_price_group)

                    maybe_gas_price = gas_price_cache.get((niche_sz, gas_price_group), None)
                    if maybe_gas_price is not None:
                        gas_price = maybe_gas_price
                    else:
                        if niche_sz.endswith('|2|balv2|') or niche_sz.endswith('|3|balv2|'):
                            # special case -- we cannot find these in our scrape so we have no oracle

                            # just use uniswap v3 instead
                            replacement_niche_sz = niche_sz.replace('|balv2|', '|uv3|')
                            blocks = gas_oracle_pts[replacement_niche_sz][0]
                            pts = gas_oracle_pts[replacement_niche_sz][1][gas_price_group]
                        else:
                            blocks = gas_oracle_pts[niche_sz][0]
                            pts = gas_oracle_pts[niche_sz][1][gas_price_group]
                        assert len(blocks) == len(pts)
                        gas_price = int(np.interp(block_number, blocks, pts))
                        gas_price_cache[(niche_sz, gas_price_group)] = gas_price

                    real_profit_after_fee = real_profit_before_fee - gas_price * gas_used

                    if real_profit_after_fee < 0:
                        if k in running_pre_existing:
                            running_pre_existing.remove(k)
                        does_not_need_lookback.add(k)
                        flush_arbitrage(exchanges, directions, niche_sz, gas_price_group)
                        continue

                    if k in running_pre_existing:
                        l.debug(f'This was pre-existing and is still ongoing')
                        still_running_pre_existing_this_block.add(k)
                        continue

                    rl = requires_lookback(exchanges, directions, niche_sz, gas_price_group)
                    if rl == False:
                        push_running_arbitrage(
                            exchanges, directions, niche_sz, gas_price_group, block_number, real_profit_after_fee, candidate_id,
                        )
                    else:
                        running_pre_existing.add(k)
                        still_running_pre_existing_this_block.add(k)

        # clear arbitrages that fell off profitability this block
        all_tracked_exchanges = set()
        for exchanges, _, _, _ in running_arbitrages.keys():
            all_tracked_exchanges.update(exchanges)

        l.debug(f'Have {len(all_tracked_exchanges):,} exchanges involved in running arbitrages in block {block_number}')
        curr2.execute(
            '''
            SELECT exchange_address
            FROM exchanges_updated_in_block
            WHERE block_number = %s AND exchange_address = ANY (%s)
            ''',
            (block_number, list(all_tracked_exchanges))
        )
        l.debug(f'Of all exchanges, {curr2.rowcount} updated this block')

        updated_exchanges = set(x.tobytes() for (x,) in curr2)
        exchanges_updated_since_start.update(updated_exchanges)

        # flush running arbitrages that state-updated this block but didnt show profit
        for (exchanges, d, n, i), arb in list(running_arbitrages.items()):
            if len(updated_exchanges.intersection(exchanges)) > 0 and arb.block_number_end != block_number:
                flush_arbitrage(exchanges, d, n, i)

        # unmark mark pre-existing arbitrages that didn't show a profitable arbitrage in this block
        n_pre_existing_closed = 0
        for k in list(running_pre_existing):
            exchanges, _, _, _ = k
            if len(updated_exchanges.intersection(exchanges)) > 0 and k not in still_running_pre_existing_this_block:
                # this is no longer running
                running_pre_existing.remove(k)
                does_not_need_lookback.add(k)
                n_pre_existing_closed += 1
        l.debug(f'Had {n_pre_existing_closed} pre-existing arbitrages closed this block')
        l.debug(f'At block {block_number} have {len(running_arbitrages)} running arbitrage-gas pricer combinations')

        # increment end block for everything else
        for k, r in list(running_arbitrages.items()):
            running_arbitrages[k] = r._replace(block_number_end=block_number)

        curr2.connection.commit()



def fill_modified_exchanges(w3: web3.Web3, args: argparse.Namespace, db: psycopg2.extensions.connection, curr: psycopg2.extensions.cursor):
    curr.execute('SELECT MIN(start_block), MAX(end_block) FROM block_samples')
    min_block, max_block = curr.fetchone()

    our_slice_start = min_block + (max_block - min_block) * args.id // args.n_workers
    our_slice_end = (min_block + (max_block - min_block) * (args.id + 1) // args.n_workers) - 1

    BATCH_SIZE = 200

    with tempfile.TemporaryDirectory(dir='/mnt/goldphish/tmp') as tmpdir:
        pool = load_pool(w3, curr, tmpdir)
        pool.warm(our_slice_start - 1)

        t_start = time.time()
        t_last_update = time.time()

        for i in itertools.count():
            batch_start = our_slice_start + BATCH_SIZE * i
            batch_end = min(our_slice_end, our_slice_start + BATCH_SIZE * (i + 1) - 1)

            if batch_start > batch_end:
                break

            if time.time() > t_last_update + 60:
                t_last_update = time.time()

                blocks_processed = batch_start - our_slice_start
                elapsed = time.time() - t_start
                nps = blocks_processed / elapsed
                remaining = our_slice_end - batch_start
                eta_s = remaining / nps
                eta = datetime.timedelta(seconds = eta_s)

                l.info(f'Processed {blocks_processed:,} blocks ({blocks_processed / (our_slice_end - our_slice_start)*100:.2f}%) ETA={eta}')


            logs = get_relevant_logs(w3, pool, batch_start, batch_end)

            for block_number, block_logs in logs:
                update = pool.observe_block(block_number, block_logs)
                updated_exchanges = set().union(*update.values())
                # l.debug(f'block {block_number:,} has {len(updated_exchanges):,} updated exchanges')

                for updated in sorted(updated_exchanges):
                    assert len(updated) == 42

                    curr.execute(
                        '''
                        INSERT INTO exchanges_updated_in_block (block_number, exchange_address)
                        VALUES (%s, %s)
                        ''',
                        (
                            block_number,
                            bytes.fromhex(updated[2:]),
                        )
                    )

    db.commit()
    l.info('Done')


def setup_db(
        curr: psycopg2.extensions.cursor,
    ):
    l.debug(f'setting up database')
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS exchanges_updated_in_block (
            block_number      INTEGER NOT NULL,
            exchange_address  BYTEA NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_exchanges_updated_in_block_block_number ON exchanges_updated_in_block (block_number);
        CREATE INDEX IF NOT EXISTS idx_exchanges_updated_in_block_exchange_address ON exchanges_updated_in_block USING HASH (exchange_address);

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_campaigns (
            id                       SERIAL NOT NULL PRIMARY KEY,
            niche                    TEXT NOT NULL,
            gas_pricer               TEXT NOT NULL,
            block_number_start       INTEGER NOT NULL,
            block_number_end         INTEGER NOT NULL,
            max_profit_after_fee_wei NUMERIC(78, 0) NOT NULL,
            min_profit_after_fee_wei NUMERIC(78, 0) NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrage_campaigns_niche ON candidate_arbitrage_campaigns (niche);
        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrage_campaigns_bns ON candidate_arbitrage_campaigns (block_number_start);
        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrage_campaigns_bne ON candidate_arbitrage_campaigns (block_number_end);

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_campaign_member (
            candidate_arbitrage_id       INTEGER NOT NULL REFERENCES candidate_arbitrages (id) ON DELETE CASCADE,
            candidate_arbitrage_campaign INTEGER NOT NULL REFERENCES candidate_arbitrage_campaigns (id) ON DELETE CASCADE,
            profit_after_fee_wei         NUMERIC(78, 0) NOT NULL
        );
        '''
    )

# def get_resume_point(curr: psycopg2.extensions.cursor, default: int) -> int:
#     curr.execute('SELECT MAX(processed_up_to) FROM arb_termination_progress')
#     (resume_point,) = curr.fetchone()
#     if resume_point is not None:
#         l.info(f'resuming from block {resume_point:,}')
#         return resume_point
#     else:
#         l.info(f'using default start block {default:,}')
#         return default

