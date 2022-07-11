"""
find_arb_termination.py

Finds out when each arbitrage opportunity closed, and why.
"""
from backtest.utils import connect_db
from backtest.top_of_block.common import load_pool
from backtest.top_of_block.seek_candidates import get_relevant_logs
import psycopg2.extensions
import typing
import logging

from pricers.pricer_pool import PricerPool
from utils import ProgressReporter
import web3
import web3.types
import web3.contract


l = logging.getLogger(__name__)


class ArbitrageOpportunity(typing.NamedTuple):
    id: int
    start_block: int
    end_block: typing.Optional[int]
    max_profit_wei: int
    min_profit_wei: int
    circuit: typing.List[str]
    directions: typing.List[bool]


def do_find_arb_termination(w3: web3.Web3):
    l.info('Starting diagnosis of failures')
    db = connect_db()
    curr = db.cursor()
    setup_db(curr)

    start_block = 12_369_621 # get_resume_point(curr, 12_369_621)
    end_block = 14_324_572

    progress_reporter = ProgressReporter(
        l, end_block, start_block
    )

    open_arbitrages: typing.List[ArbitrageOpportunity] = []
    batch_size_blocks = 200 # batch size for getting logs
    curr_block = start_block
    pricer = load_pool(w3)
    while curr_block <= end_block:
        this_end_block = min(curr_block + batch_size_blocks - 1, end_block)
        seen_open_arbs = set(oa.id for oa in open_arbitrages)
        for block_number, logs in get_relevant_logs(w3, curr_block, this_end_block):
            open_arbitrages = process_block(w3, curr, block_number, open_arbitrages, logs, pricer)
            seen_open_arbs = seen_open_arbs.union(x.id for x in open_arbitrages)

        l.debug(f'saw {len(seen_open_arbs):,} open arbitrages after processing blocks {curr_block:,} to {this_end_block:,}')

        progress_reporter.observe(batch_size_blocks)
        curr_block = this_end_block + 1
        db.commit()


def process_block(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        block_number: int,
        open_arbitrages: typing.List[ArbitrageOpportunity],
        logs: typing.List[web3.types.TxReceipt],
        pricer: PricerPool,
    ):
    ret: typing.List[ArbitrageOpportunity] = []

    all_arbs: typing.Dict[typing.Any, ArbitrageOpportunity] = {}

    for oa in open_arbitrages:
        excs = tuple(oa.circuit)
        dirs = tuple(oa.directions)
        assert (excs, dirs) not in all_arbs
        all_arbs[(excs, dirs)] = oa

    # figure out which open arbitrages had an exchange touched
    modified_exchanges = pricer.observe_block(logs)
    touched_arbs = set()
    for excs, dirs in all_arbs.keys():
        if len(modified_exchanges.intersection(excs)) > 0:
            touched_arbs.add((excs, dirs))

    # records which arbitrages we saw in the database this block, so that we can look
    # for _absence_ of a record, which indicates the opportunity closed
    arbs_in_db_this_block = set()

    # add in the arbitrages we have from records
    curr.execute(
        '''
        SELECT exchanges, directions, net_profit
        FROM verified_arbitrages va
        JOIN candidate_arbitrages ca ON va.candidate_id = ca.id
        WHERE ca.block_number = %s AND va.net_profit > 0
        ''',
        (block_number,)
    )
    all_recs = curr.fetchall()
    for excs, dxns, profit_no_fee in all_recs:
        excs = [w3.toChecksumAddress(exc.tobytes()) for exc in excs]
        profit_no_fee = int(profit_no_fee)
        assert all(isinstance(d, bool) for d in dxns)
        
        key = (tuple(excs), tuple(dxns))
        arbs_in_db_this_block.add(key)

        if key not in all_arbs:
            # we need a new record for this one
            curr.execute(
                '''
                INSERT INTO candidate_arbitrage_opportunities (
                        start_block,
                        max_profit_wei_no_fee,
                        min_profit_wei_no_fee,
                        exchanges,
                        directions
                    )
                    VALUES (
                        %s, %s, %s, %s, %s
                    )
                    RETURNING id
                ''',
                (
                    block_number,
                    profit_no_fee,
                    profit_no_fee,
                    [bytes.fromhex(x[2:]) for x in excs],
                    dxns
                )
            )
            (id_,) = curr.fetchone()
            ret.append(ArbitrageOpportunity(
                id = id_,
                start_block = block_number,
                end_block = None,
                max_profit_wei = profit_no_fee,
                min_profit_wei = profit_no_fee,
                circuit = excs,
                directions = dxns,
            ))
        else:
            ao = all_arbs[key]
            ao: ArbitrageOpportunity = ao._replace(max_profit_wei = max(ao.max_profit_wei, profit_no_fee))
            ao: ArbitrageOpportunity = ao._replace(min_profit_wei = max(ao.min_profit_wei, profit_no_fee))
            ret.append(ao)

    # figure out which arbitrages dropped off
    for missing_arb in touched_arbs.difference(arbs_in_db_this_block):
        ao = all_arbs[missing_arb]
        curr.execute(
            '''
            UPDATE candidate_arbitrage_opportunities
            SET
                end_block_exclusive = %s,
                max_profit_wei_no_fee = %s,
                min_profit_wei_no_fee = %s
            WHERE id = %s
            ''',
            (block_number, ao.max_profit_wei, ao.min_profit_wei, ao.id),
        )
    return ret


def setup_db(
        curr: psycopg2.extensions.cursor,
    ):
    l.debug(f'setting up database')
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS candidate_arbitrage_opportunities (
            id              SERIAL NOT NULL,
            start_block     INTEGER NOT NULL,
            end_block_exclusive INTEGER,
            max_profit_wei_no_fee NUMERIC(78, 0) NOT NULL,
            min_profit_wei_no_fee NUMERIC(78, 0) NOT NULL,
            exchanges       bytea[] NOT NULL,
            directions      boolean[] NOT NULL
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

