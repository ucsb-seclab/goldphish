"""
find_arb_termination.py

Finds out when each arbitrage opportunity closed, and why.
"""
import collections
import datetime
import itertools
import tempfile
import time
from backtest.top_of_block.relay import open_ganache
from backtest.utils import connect_db
from backtest.top_of_block.common import load_pool
from backtest.top_of_block.seek_candidates import get_relevant_logs
import argparse
import psycopg2.extensions
import typing
import numpy as np
import logging

from pricers.pricer_pool import PricerPool
from utils import ProgressReporter, connect_web3
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

    assert args.id >= 0 and args.n_workers > 0 and args.id < args.n_workers

    if args.fill_modified:
        fill_modified_exchanges(w3,args, db, curr)
        return

    curr.execute('SELECT MIN(block_number), MAX(block_number) FROM sample_arbitrages')
    min_block, max_block = curr.fetchone()

    gas_oracle_pts = get_gas_oracle(curr, min_block, max_block)

    # go by reservation order
    curr.execute(
        '''
        SELECT start_block, end_block, priority
        FROM block_samples
        ORDER BY priority ASC
        '''
    )

    curr2 = db.cursor()

    for start_block, end_block, priority in list(curr):
        process_sample(w3, curr, min_block, max_block, start_block, end_block, priority, gas_oracle_pts, args.id)

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
        
        ret[niche] = (block_numbers, pts)

    return ret

def process_sample(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        min_block: int,
        max_block: int,
        start_block: int,
        end_block: int,
        priority: int,
        gas_oracle_pts: typing.Dict[
            str,
            typing.Tuple[typing.List[int], typing.Tuple[typing.List[int], typing.List[int], typing.List[int], typing.List[int], typing.List[int]]]
        ],
        worker_id: int
    ):
    l.info(f'Processing priority={priority}')

    does_not_need_lookback = set()
    # keeps a running record of ongoing arbitrages by niche
    running_arbitrages = collections.defaultdict(lambda : {})

    def flush_arbitrage(*args):
        sz_args = ', '.join(str(x) for x in args)
        l.debug(F'flushing {sz_args}')
        return

    def flush_all_arbitrages(*args):
        sz_args = ', '.join(str(x) for x in args)
        l.debug(F'flushing {sz_args}')
        return

    def requires_lookback(exchanges, directions, niche, gas_price_group) -> bool:
        k_minor = (exchanges, directions)
        k = (exchanges, directions, niche, gas_price_group)
        if k_minor in does_not_need_lookback or k in does_not_need_lookback:
            return False

        # find the most recent prior block where this was updated
        curr.execute(
            '''
            SELECT block_number
            FROM exchanges_updated_in_block
            WHERE block_number < %s AND exchange_address = ANY (%s)
            ORDER BY block_number DESC
            LIMIT 1
            ''',
            (start_block, list(exchanges))
        )
        if curr.rowcount < 1:
            # if not updated in prior block then this is fresh
            does_not_need_lookback.add(k_minor)
            return False

        assert curr.rowcount == 1
        (most_recent_prior_updated_block,) = curr.fetchone()
        l.debug(f'Circuit {exchanges} {directions} most recently updated on {most_recent_prior_updated_block:,}')


        # find if we have a candidate in that block
        curr.execute(
            '''
            SELECT ca.id, carr.shoot_success, carr.gas_used, carr.real_profit_before_fee
            FROM candidate_arbitrages ca
            LEFT JOIN candidate_arbitrage_relay_results carr
                ON ca.id = carr.candidate_arbitrage_id
            WHERE ca.block_number = %s AND
                  ca.exchanges = %s AND
                  ca.directions = %s
            ''',
            (
                most_recent_prior_updated_block,
                list(exchanges),
                list(directions),
            )
        )
        if curr.rowcount >= 2:
            l.critical(f'too many rows!!!')
            for old_arb_id, relay_success, gas_used, real_profit_before_fee in curr:
                l.debug(f'{old_arb_id} {relay_success} {gas_used} {real_profit_before_fee}')
                exit(0)

        assert curr.rowcount <= 1, f'Expected at most 1 row but got {curr.rowcount} for {most_recent_prior_updated_block} {exchanges} {directions}'

        if curr.rowcount < 1:
            # no arbitrage in prior block
            does_not_need_lookback.add(k_minor)
            l.debug(f'No arbitrage in prior block')
            return False

        old_arb_id, relay_success, gas_used, real_profit_before_fee = curr.fetchone()
        if relay_success is None:
            l.warning(f'we need to relay this candidate')
            with tempfile.TemporaryDirectory(dir='/mnt/goldphish/tmp') as tmpdir:
                proc, w3_ganache, acct, shooter_address = open_ganache(block_number, tmpdir, worker_id)

                proc.kill()
                proc.wait()

        if not relay_success:
            # prior arbitrage relay was not success
            does_not_need_lookback.add(k_minor)
            l.debug(f'Arbitrage in prior block failed relay')
            return False
        
        # we need to compute gas price oracle value for this shot
        blocks = gas_oracle_pts[niche][0]
        pts = gas_oracle_pts[niche][gas_price_group]
        gas_price = int(np.interp(most_recent_prior_updated_block, blocks, pts))
        l.debug(f'Gas oracle {gas_price_group} = {gas_price / 10 ** 18:.2f} gwei')

        # 

    def lookback(exchanges, directions) -> bool:
        return False

    for block_number in range(start_block, end_block + 1):
        curr.execute(
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
            in curr:

            exchanges = tuple(e.tobytes() for e in exchanges)
            directions = tuple(t.tobytes() for t in directions)

            k = (exchanges, directions)

            if not shoot_success:
                flush_all_arbitrages(k)
                does_not_need_lookback.add(k)
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
                if has_balancer_v1:
                    niche_sz += 'balv2|'

                # get gas price for this niche
                for i in range(5): # one for min, 25pct, median, 75pct, max
                    rl = requires_lookback(exchanges, directions, niche_sz, i)
                    if rl != False:
                        exit(0)



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

