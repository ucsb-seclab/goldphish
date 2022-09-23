import collections
import itertools
import random
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
from pricers.base import BaseExchangePricer
from utils import BALANCER_VAULT_ADDRESS, get_block_timestamp

l = logging.getLogger(__name__)


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

    # pick a random campaign
    l.info(f'Picking random campaign...')
    curr.execute('SELECT MIN(id), MAX(id) FROM top_candidate_arbitrage_campaigns')
    min_id, max_id = curr.fetchone()

    l.debug(f'IDs range from {min_id:,} to {max_id,}')

    r = random.Random(10)
    while True:
        while True:
            trying_id = r.randint(min_id, max_id)
            if trying_id != 57_253:
                continue
            curr.execute(
                'SELECT EXISTS(SELECT 1 FROM top_candidate_arbitrage_campaigns WHERE id = %s)',
                (trying_id,)
            )
            (exists_,) = curr.fetchone()
            if exists_:
                break

        l.info(f'Checking campaign id={trying_id:,}')

        check_campaign_termination(w3, curr, trying_id)


def check_campaign_termination(w3: web3.Web3, curr: psycopg2.extensions.cursor, id_: int):
    #
    # Gather the members of this campaign
    #
    curr.execute(
        '''
        SELECT exchanges, directions
        FROM top_candidate_arbitrage_campaigns
        WHERE id = %s
        ''',
        (id_,)
    )
    assert curr.rowcount == 1
    bexchanges, bdirections = curr.fetchone()
    exchanges = [w3.toChecksumAddress(e.tobytes()) for e in bexchanges]
    directions = [w3.toChecksumAddress(t.tobytes()) for t in bdirections]
    directions = list(zip(directions, directions[1:] + [directions[0]]))

    l.debug(f'investigating campaign id={id_}')

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

    # see at which block it drops below 50%
    idx_last_in_campaign = idx_max_profit
    for i, (block_number, profit) in enumerate(block_with_profit[idx_max_profit+1:]):
        pct_diff = (profit - max_profit) / (max_profit) * 100.0
        if pct_diff <= -50:
            terminal_block = block_number
            break
        else:
            idx_last_in_campaign = i + idx_max_profit + 1
    else:
        # terminates at next update, find that
        max_member_block = max(bn for bn, _ in block_with_profit)
        l.debug(f'Terminates at next update')
        curr.execute(
            '''
            SELECT block_number
            FROM exchanges_updated_in_block
            WHERE block_number > %(block_number)s AND exchange_address = ANY(%(exchanges)s)
            ORDER BY block_number ASC
            LIMIT 1
            ''',
            {
                'block_number': max_member_block,
                'exchanges': bexchanges,
            }
        )
        assert curr.rowcount == 1
        (maybe_terminal_block,) = curr.fetchone()

        # now we need to see if the campaign ended due to a failed candidate arbitrage,
        # indicating that something in the circuit broke
        curr.execute(
            '''
            SELECT id
            FROM candidate_arbitrages
            WHERE exchanges = %s AND directions = %s AND %s <= block_number AND block_number <= %s
            ORDER BY block_number ASC
            ''',
            (bexchanges, bdirections, max_member_block + 1, maybe_terminal_block)
        )
        l.debug(f'Had {curr.rowcount} interceding cadidates')
        if curr.rowcount == 0:
            terminal_block = maybe_terminal_block
        else:
            pass            
            

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

    terminal_block += 2
    timestamp_to_use = get_block_timestamp(w3, terminal_block - 1)

    maybe_fa = detect_arbitrages_bisection(
        pc.copy(),
        terminal_block - 1,
        timestamp = timestamp_to_use,
        try_all_directions = False,
        fee_transfer_calculator = fee_calculator
    )
    print('new profit', maybe_fa[0].profit / (10 ** 18), f'at {terminal_block:,}')
    exit()

    timestamp_to_use = get_block_timestamp(w3, terminal_block - 1)

    maybe_fa = detect_arbitrages_bisection(
        pc.copy(),
        terminal_block - 1,
        timestamp = timestamp_to_use,
        try_all_directions = False,
        fee_transfer_calculator = fee_calculator
    )

    assert len(maybe_fa) > 0

    (fa,) = maybe_fa

    pct_diff = (fa.profit - max_profit) / max_profit * 100
    
    if abs(pct_diff) > 10:
        l.critical(f'Unexpected change in profit: {pct_diff:.3f}%')
        raise Exception('unexpected change in profit')

    l.debug(f'Percent diff from max just before: {pct_diff:.3f}%')

    # gather logs by transaction index, in order of transaction occurrence in the block,
    # and apply them one by one so that we can see where it disappeared
    logs_by_idx: typing.Dict[int, typing.List[web3.types.LogReceipt]] = collections.defaultdict(lambda: [])
    for log in relevant_logs:
        logs_by_idx[log['transactionIndex']].append(log)
    

    for d in directions:
        print(d)

    found_threshold_txn = None

    for txn_idx, logs in sorted(logs_by_idx.items(), key=lambda x: x[0]):
        txn_hash = logs[0]['transactionHash']
        l.debug(f'Applying transaction index #{txn_idx}: {txn_hash.hex()}')

        for pricer, (t1, t2) in zip(pricers, directions):
            t1_before = pricer._balance_cache.get(t1, None)
            t2_before = pricer._balance_cache.get(t2, None)
            pricer.observe_block(logs, force_load = True)
            t1_after = pricer._balance_cache.get(t1, None)
            t2_after = pricer._balance_cache.get(t2, None)
            print('t1', t1)
            print('t2', t2)
            print(f't1_before == t1_after????', t1_before, t1_after, t1_before == t1_after)
            print(f't2_before == t2_after????', t2_before, t2_after, t2_before == t2_after)

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

        if pct_diff <= -50:
            found_threshold_txn = txn_hash
            break

    assert found_threshold_txn is not None

    l.debug(f'Threshold transaction: {found_threshold_txn.hex()}')
