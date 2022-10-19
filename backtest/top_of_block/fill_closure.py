import collections
import datetime
import itertools
import random
import time
import web3
import web3._utils.filters
import web3.contract
import web3.types
import psycopg2.extensions
import psycopg2.extras
import logging
import argparse
import typing
from backtest.top_of_block.relay import InferredTokenTransferFeeCalculator, load_pricer_for

from backtest.utils import connect_db
from find_circuit.find import PricingCircuit, detect_arbitrages_bisection
from pricers.base import BaseExchangePricer, NotEnoughLiquidityException
from utils import BALANCER_VAULT_ADDRESS, get_block_timestamp

l = logging.getLogger(__name__)

DEBUG = False

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'fill-closure'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (before run)')
    parser.add_argument('--reset-db', action='store_true', help='Reset the database (before run)')
    parser.add_argument('--fixup-db', action='store_true', help='Reset the database (before run)')

    parser.add_argument('--id', type=int, default=0)

    parser.add_argument('--top-arbs', action='store_true', help='Do the run for top arbitrages')

    return parser_name, fill_closure


def fill_closure(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()


    if args.fixup_db:
        split_reservations(curr)
        db.commit()
        return

    if args.setup_db:
        setup_db(curr)
        db.commit()
        l.info('Setup db')
        return

    if args.worker_name is None:
        print('Must supply worker_name', file=sys.stderr)
        exit(1)

    l.info(f'Filling closure...')
    while True:
        db.commit()
        curr.execute(
            '''
            SELECT id, start_block, end_block
            FROM top_candidate_closure_reservations
            WHERE claimed_on IS NULL
            ORDER BY id ASC
            FOR UPDATE SKIP LOCKED
            ''',
        )
        if curr.rowcount == 0:
            print('Done')
            break

        (id_, start_block, end_block) = curr.fetchone()
        curr.execute('UPDATE top_candidate_closure_reservations SET claimed_on = now()::timestamp, worker = %s WHERE id = %s', (args.worker_name, id_))
        db.commit()

        process_reservation(curr, w3, start_block, end_block, id_)
        curr.execute('UPDATE top_candidate_closure_reservations SET completed_on = now()::timestamp WHERE id = %s', (id_,))
        db.commit()



def process_reservation(curr: psycopg2.extensions.cursor, w3: web3.Web3, start_block: int, end_block: int, reservation_id: int):
    curr.execute(
        '''
        SELECT id, start_block
        FROM top_candidate_arbitrage_campaigns tcac
        WHERE NOT EXISTS(SELECT FROM top_candidate_arbitrage_campaign_terminations WHERE tcac.id = campaign_id) AND
            %s <= tcac.start_block AND tcac.start_block <= %s
        ORDER BY start_block ASC
        ''',
        (start_block, end_block)
    )
    n_to_check = curr.rowcount
    l.info(f'Have {n_to_check:,} top arbitrage campaigns to check for terminating transaction')

    t_start = time.time()
    t_last_update = t_start
    for i, (id_, start_block) in enumerate(curr.fetchall()):
        if time.time() > t_last_update + 10:
            # do an update
            t_last_update = time.time()
            elapsed = t_last_update - t_start
            nps = i / elapsed
            remain = n_to_check - i
            eta_s = remain / nps
            eta = datetime.timedelta(seconds=eta_s)
            l.info(f'{i:,} of {n_to_check:,} ({i / n_to_check * 100:.2f}%) ETA {eta}')
            if not DEBUG:
                curr.execute('UPDATE top_candidate_closure_reservations SET progress = %s WHERE id = %s', (start_block, reservation_id,))
                curr.connection.commit()
        check_campaign_termination(w3, curr, id_)

    if not DEBUG:
        curr.execute('UPDATE top_candidate_closure_reservations SET progress = end_block WHERE id = %s', (reservation_id,))
        curr.connection.commit()


def check_campaign_termination(w3: web3.Web3, curr: psycopg2.extensions.cursor, id_: int):
    #
    # Gather the members of this campaign
    #
    curr.execute(
        '''
        SELECT exchanges, directions, start_block, end_block
        FROM top_candidate_arbitrage_campaigns
        WHERE id = %s
        ''',
        (id_,)
    )
    assert curr.rowcount == 1
    bexchanges, bdirections, start_block, end_block = curr.fetchone()
    assert start_block <= end_block
    exchanges = [w3.toChecksumAddress(e.tobytes()) for e in bexchanges]
    directions = [w3.toChecksumAddress(t.tobytes()) for t in bdirections]
    directions = list(zip(directions, directions[1:] + [directions[0]]))

    l.debug(f'investigating campaign id={id_} starts at {start_block:,} and ends at {end_block:,} (total {end_block - start_block:,} blocks)')

    curr.execute(
        '''
        SELECT tcarr.id, tcarr.shoot_success, ca.block_number, tcarr.gas_used, tcarr.real_profit_before_fee
        FROM top_candidate_arbitrage_relay_results tcarr
        JOIN candidate_arbitrages ca ON tcarr.candidate_arbitrage_id = ca.id
        WHERE campaign_id = %s
        ORDER BY ca.block_number ASC
        ''',
        (id_,)
    )
    assert curr.rowcount > 0
    l.debug(f'Have {curr.rowcount} members')

    block_with_profit: typing.List[typing.Tuple[int, int]] = []
    result_ids = []

    for result_id, relay_success, block_number, gas_used, real_profit_before_fee in curr:
        assert relay_success
        result_ids.append(result_id)
        l.debug(f'arbitrage member: {block_number:,} {real_profit_before_fee / (10 ** 18):.3f} ETH')
        real_profit_before_fee = int(real_profit_before_fee)

        block_with_profit.append((block_number, real_profit_before_fee))

    max_profit_block, max_profit = max(block_with_profit, key=lambda x: x[1])
    idx_max_profit = block_with_profit.index((max_profit_block, max_profit))
    
    l.debug(f'Max profit was on block {max_profit_block:,}: {max_profit / (10 ** 18):.3f} ETH')

    terminal_threshold = min(10 ** 18, max_profit // 2)
    l.debug(f'Terminal threshold: {terminal_threshold / (10 ** 18):.3f} ETH')

    # see at which block it drops below 50% or 1 ETH, whichever is lowerst
    idx_last_in_campaign = idx_max_profit
    for i, (block_number, profit) in enumerate(block_with_profit[idx_max_profit+1:]):
        if profit <= terminal_threshold:
            terminal_block = block_number
            break
        else:
            idx_last_in_campaign = i + idx_max_profit + 1
    else:
        terminal_block = end_block + 1
            

    l.debug(f'Effectively terminates at block {terminal_block:,}')

    #
    # we need to find all transactions that had logs emitted in the terminal block
    #
    f: web3._utils.filters.Filter = w3.eth.filter({
        'fromBlock': terminal_block,
        'toBlock': terminal_block,
    })
    logs = f.get_all_entries()
    l.debug(f'Have {len(logs):,} logs in block {terminal_block:,}')

    # filter to relevant logs
    relevant_logs: typing.List[web3.types.LogReceipt] = []
    for log in logs:
        if log['address'] in exchanges:
            relevant_logs.append(log)
        elif log['address'] == BALANCER_VAULT_ADDRESS:
            relevant_logs.append(log)
    
    relevant_txns: typing.List[bytes] = set(x['transactionHash'] for x in relevant_logs)

    l.debug(f'Have {len(relevant_logs):,} relevant logs and {len(relevant_txns)} relevant transactions')
    for txn in sorted(relevant_txns):
        l.debug(f'Relevant transaction: https://etherscan.io/tx/{txn.hex()}')

    # recreate the token fee conditions
    result_id_last_in_campaign = result_ids[idx_last_in_campaign]
    curr.execute(
        '''
        SELECT t1.address, itft.fee, itft.round_down, itft.from_address, itft.to_address
        FROM top_candidate_arbitrage_relay_results_used_fees tcarruf
        JOIN inferred_token_fee_on_transfer itft ON itft.id = tcarruf.fee_used
        JOIN tokens t1 ON itft.token_id = t1.id
        WHERE tcarruf.top_candidate_arbitrage_relay_result_id = %s
        ''',
        (result_id_last_in_campaign,)
    )
    fee_calculator = InferredTokenTransferFeeCalculator()
    for taddr, fee, round_down, from_address, to_address in curr:
        token_address = w3.toChecksumAddress(taddr.tobytes())
        from_address = w3.toChecksumAddress(from_address.tobytes())
        to_address = w3.toChecksumAddress(to_address.tobytes())
        fee_calculator.propose(token_address, from_address, to_address, fee, round_down)

    # go through each transaction and see if we can find where the arbitrage goes away

    # detect arbitrages at the start
    pricers: typing.List[BaseExchangePricer] = []
    for exchange in exchanges:
        pricer = load_pricer_for(w3, curr, exchange)
        pricers.append(pricer)

    pc = PricingCircuit(
        pricers,
        directions.copy()
    )

    timestamp_to_use = get_block_timestamp(w3, terminal_block)

    # # was there a candidate in that terminal block? TODO remove
    # curr.execute(
    #     'SELECT COUNT(*) FROM candidate_arbitrages WHERE exchanges = %s AND directions = %s AND block_number = %s',
    #     (
    #         bexchanges,
    #         bdirections,
    #         terminal_block,
    #     )
    # )
    # print(f'Have {curr.fetchone()[0]} candidates in {terminal_block:,}')

    # maybe_fa = detect_arbitrages_bisection(
    #     pc.copy(),
    #     terminal_block + 2,
    #     try_all_directions=False,
    #     fee_transfer_calculator = fee_calculator
    # )
    # print(f'maybe_fa profit: {maybe_fa[0].profit / (10 ** 18):.4f} ETH')
    # exit()

    # gather logs by transaction index, in order of transaction occurrence in the block,
    # and apply them one by one so that we can see where it disappeared
    logs_by_idx: typing.Dict[int, typing.List[web3.types.LogReceipt]] = collections.defaultdict(lambda: [])
    for log in relevant_logs:
        logs_by_idx[log['transactionIndex']].append(log)
    
    found_threshold_txn = None

    for txn_idx, logs in sorted(logs_by_idx.items(), key=lambda x: x[0]):
        txn_hash = logs[0]['transactionHash']
        l.debug(f'Applying transaction index #{txn_idx}: {txn_hash.hex()}')

        for pricer in pricers:
            try:
                pricer.observe_block(logs, force_load = True)
            except Exception as e:
                if 'cannot force load on GULP' in str(e):
                    l.critical('Cannot force load on gulp, giving up....')
                    break
                else:
                    raise


        maybe_fa = detect_arbitrages_bisection(
            pc.copy(),
            terminal_block - 1,
            timestamp = timestamp_to_use,
            try_all_directions = False,
            fee_transfer_calculator = fee_calculator
        )

        if len(maybe_fa) == 0:
            l.debug(f'No arbitrage possible after that transaction')
            found_threshold_txn = txn_hash
            break
        
        new_fa = maybe_fa[0]
        pct_diff = (new_fa.profit - max_profit) / max_profit * 100
        l.debug(f'New profit before fee: {new_fa.profit / (10 ** 18):.8f} ETH ({pct_diff:.3f}%)')

        if new_fa.profit < terminal_threshold:
            found_threshold_txn = txn_hash
            break

    if found_threshold_txn is None:
        l.critical(f'Could not find threshold transaction for campaign id={id_:,}')
        for pricer in pricers:
            l.critical(str(pricer))
    curr.execute(
        '''
        INSERT INTO top_candidate_arbitrage_campaign_terminations (campaign_id, terminating_transaction)
        VALUES (%s, %s)
        ''',
        (
            id_,
            bytes(found_threshold_txn) if found_threshold_txn is not None else None
        )
    )

    if found_threshold_txn:
        l.debug(f'Threshold transaction: {found_threshold_txn.hex()}')

def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS top_candidate_closure_reservations (
            id INTEGER NOT NULL PRIMARY KEY,
            start_block  INTEGER NOT NULL,
            end_block    INTEGER NOT NULL,
            worker       TEXT,
            progress     INTEGER,
            claimed_on   TIMESTAMP WITHOUT TIME ZONE,
            heartbeat    TIMESTAMP WITHOUT TIME ZONE,
            completed_on TIMESTAMP WITHOUT TIME ZONE
        );

        CREATE TABLE IF NOT EXISTS top_candidate_arbitrage_campaign_terminations (
            campaign_id INTEGER PRIMARY KEY NOT NULL REFERENCES top_candidate_arbitrage_campaigns (id) ON DELETE CASCADE,
            terminating_transaction BYTEA
        );
        '''
    )

    curr.execute('SELECT COUNT(*) FROM top_candidate_closure_reservations')
    (n_res,) = curr.fetchone()
    if n_res == 0:
        l.info('Filling reservations')
        curr.execute(
            '''
            INSERT INTO top_candidate_closure_reservations (id, start_block, end_block)
            SELECT priority, start_block, end_block
            FROM block_samples
            '''
        )
        l.info(f'Filled {curr.rowcount:,} reservations')

def split_reservations(curr: psycopg2.extensions.cursor):
    curr.execute('SELECT MAX(id) FROM top_candidate_closure_reservations')
    (max_id,) = curr.fetchone()
    curr.execute(
        '''
        SELECT id, start_block, end_block
        FROM top_candidate_closure_reservations
        WHERE claimed_on is null
        '''
    )
    l.info(f'Splitting up {curr.rowcount} reservations')
    next_id = max_id + 1
    for id_, start_block, end_block in curr.fetchall():
        curr.execute(
            '''
            DELETE FROM top_candidate_closure_reservations WHERE id = %s
            ''',
            (id_,)
        )
        midpoint = (start_block + end_block) // 2
        curr.execute(
            '''
            INSERT INTO top_candidate_closure_reservations (id, start_block, end_block)
            VALUES (%s, %s, %s)
            ''',
            (next_id, start_block, midpoint)
        )
        curr.execute(
            '''
            INSERT INTO top_candidate_closure_reservations (id, start_block, end_block)
            VALUES (%s, %s, %s)
            ''',
            (next_id + 1, midpoint + 1, end_block)
        )
        next_id += 2
    input('ENTER to continue')
