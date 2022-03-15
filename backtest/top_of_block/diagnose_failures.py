import psycopg2
import psycopg2.extensions
import hashlib
import logging
import os
import time
import typing
import web3
from backtest.utils import parse_logs_for_net_profit

import pricers
import find_circuit
from backtest.top_of_block.common import TraceMode, connect_db, load_exchanges, load_naughty_tokens, shoot

from backtest.top_of_block.constants import FNAME_VERIFY_RESULT, NAUGHTY_TOKEN_FNAME
from utils import WETH_ADDRESS, decode_trace_calls, get_abi, pretty_print_trace

l = logging.getLogger(__name__)

MSG_FAIL_USED_NAUGHTY_TOKEN = 'uses naughty token'

def do_diagnose(w3: web3.Web3):
    l.info('Starting diagnosis of failures')
    db = connect_db()
    curr = db.cursor()
    setup_db(curr)

    naughty_tokens = load_naughty_tokens(curr)
    l.debug(f'Already know about {len(naughty_tokens)} naughty tokens')

    for block_number, failed_arbitrage_id, fa in get_failed_arbitrages(w3, curr):
        # if we already know that this contains a naughty token, just skip
        if len(fa.tokens.intersection(naughty_tokens)) > 0:
            l.debug('This guy uses a known naughty token.')
            curr.execute(
                """
                UPDATE failed_arbitrages SET diagnosis = %s WHERE id = %s
                """,
                (MSG_FAIL_USED_NAUGHTY_TOKEN, failed_arbitrage_id,)
            )
            assert curr.rowcount == 1
            curr.connection.commit()
            continue

        maybe_naughty_token, failure_reason = diagnose_single(w3, fa, block_number)

        curr.execute(
            """
            UPDATE failed_arbitrages SET diagnosis = %s WHERE id = %s
            """,
            (failure_reason, failed_arbitrage_id,)
        )
        assert curr.rowcount == 1

        if maybe_naughty_token is not None:
            new_naughty_token, reason = maybe_naughty_token
            curr.execute(
                """
                INSERT INTO naughty_tokens (address, reason, diagnosed_from) VALUES (%s, %s, %s)
                """,
                (bytes.fromhex(new_naughty_token[2:]), reason, failed_arbitrage_id)
            )
            assert curr.rowcount == 1
            naughty_tokens.add(new_naughty_token)
            l.info(f'Found new naughty token: {new_naughty_token} | {reason}')
        curr.connection.commit()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        """
        CREATE TABLE IF NOT EXISTS naughty_tokens (
            id             SERIAL PRIMARY KEY NOT NULL,
            address        BYTEA NOT NULL,
            reason         TEXT NOT NULL,
            diagnosed_from INTEGER NOT NULL REFERENCES failed_arbitrages (id) ON DELETE SET NULL
        );
        """
    )
    curr.connection.commit()


def get_failed_arbitrages(w3: web3.Web3, curr: psycopg2.extensions.cursor) -> typing.Iterator[typing.Tuple[int, int, find_circuit.FoundArbitrage]]:
    """
    Continuously polls for candidate arbitrages in a sleep-loop.
    """
    uniswap_v2_exchanges, uniswap_v3_exchanges = load_exchanges()

    while True:
        curr.execute(
            """
            SELECT id, candidate_id
            FROM failed_arbitrages
            WHERE diagnosis IS NULL
            ORDER BY id ASC
            LIMIT 1
            """
        )
        maybe_id = curr.fetchall()
        if len(maybe_id) == 0:
            l.debug('Nothing to do, sleeping for a bit...')
            curr.connection.commit()
            time.sleep(30)
            continue

        id_, candidate_id = maybe_id[0]

        curr.execute(
            """
            SELECT ca.block_number, ca.exchanges, ca.directions, ca.amount_in, ca.profit_no_fee
            FROM candidate_arbitrages ca
            WHERE id = %s
            """,
            (candidate_id,)
        )
        block_number, exchanges, directions, amount_in, profit = curr.fetchone()

        assert len(exchanges) == len(directions)

        # reconstruct found arbitrage
        amount_in = int(amount_in)
        profit = int(profit)
        circuit: typing.List[pricers.BaseExchangePricer] = []
        for exc in exchanges:
            address = web3.Web3.toChecksumAddress(exc.tobytes())
            
            if address in uniswap_v2_exchanges:
                token0, token1 = uniswap_v2_exchanges[address]
                pricer = pricers.UniswapV2Pricer(w3, address, token0, token1)
            else:
                assert address in uniswap_v3_exchanges
                token0, token1, fee = uniswap_v3_exchanges[address]
                pricer = pricers.UniswapV3Pricer(w3, address, token0, token1, fee)
            circuit.append(pricer)

        if directions[0] == True:
            assert circuit[0].token0 == WETH_ADDRESS
        else:
            assert directions[0] == False
            assert circuit[0].token1 == WETH_ADDRESS

        fa = find_circuit.FoundArbitrage(
            amount_in = amount_in,
            circuit = circuit,
            directions = directions,
            pivot_token = WETH_ADDRESS,
            profit = profit
        )

        l.debug(f'Diagnosing failed arbitrage id={id_:,} candidate_id={candidate_id} block_number={block_number:,} expected_profit={w3.fromWei(profit, "ether"):.6f} ETH')

        yield block_number, id_, fa


erc20 = web3.Web3().eth.contract(address = b'\x00' * 20, abi=get_abi('erc20.abi.json'))
ERC20_BALANCEOF_SELECTOR = erc20.functions.balanceOf(web3.Web3.toChecksumAddress(b'\x00' * 20)).selector[2:]
ERC20_TRANSFER_SELECTOR = erc20.functions.transfer(web3.Web3.toChecksumAddress(b'\x00' * 20), 10).selector[2:]

def diagnose_single(w3: web3.Web3, fa: find_circuit.FoundArbitrage, block_number: int) -> typing.Tuple[typing.Optional[typing.Tuple[str, str]], str]:
    """
    Diagnoses the given re-shoot failure.
    Returns the new naughty token, reason for naughtiness, and the reason for failure
    """

    # recording prior balances also helps with failure diagnosis below
    known_balances = {}
    for exc in fa.circuit:
        for token in [exc.token0, exc.token1]:
            token_contract = w3.eth.contract(
                address = token,
                abi = get_abi('erc20.abi.json'),
            )
            bal = token_contract.functions.balanceOf(exc.address).call(block_identifier=block_number)
            l.debug(f'balance | token={token} balance={bal:,}')
            known_balances[token, exc.address] = bal

    # records the expected amount of transfer out of each exchange
    expected_amounts_out: typing.Dict[typing.Tuple[str, str], int] = {}
    amount = fa.amount_in
    for exc, dxn in zip(fa.circuit, fa.directions):
        if dxn == True:
            token_out = exc.token1
            amount_out = exc.exact_token0_to_token1(amount, block_number)
        else:
            assert dxn == False
            token_out = exc.token0
            amount_out = exc.exact_token1_to_token0(amount, block_number)
        expected_amounts_out[exc.address, token_out] = amount_out
        amount = amount_out

    shooter_address, receipt, (trace, txn) = shoot(w3, fa, block_number, do_trace = TraceMode.ALWAYS)

    exchanges: typing.Set[str] = set(x.address for x in fa.circuit)

    decoded = decode_trace_calls(trace, txn, receipt)
    print('----------------------trace---------------------------')
    pretty_print_trace(decoded, txn, receipt)
    print('------------------------------------------------------')

    if receipt['status'] == 1:
        expected_profit = fa.profit
        movements = parse_logs_for_net_profit(receipt['logs'])
        actual_profit = movements[WETH_ADDRESS][shooter_address]
        assert actual_profit == expected_profit, f'expected {actual_profit:,} == {expected_profit:,}'
        raise NotImplementedError('hmm')
    else:
        # if we saw a revert() in the first call to a token's transfer() or any balanceOf(), it is bugged
        called_transfer = set()
        # known_balances already holds a dict (token, address) -> int as seen returned by balanceOf

        stack = [(0, decoded)]
        while len(stack) > 0:
            depth, item = stack.pop()
            if 'CALL' in item['type']:
                method_sel = item['args'][:4].hex()
                callee = item['callee']
                if method_sel == ERC20_BALANCEOF_SELECTOR:
                    # record the balance
                    if item['actions'][-1]['type'] == 'REVERT':
                        return ((callee, 'reverted in balanceOf'), MSG_FAIL_USED_NAUGHTY_TOKEN)
                    if item['actions'][-1]['type'] == 'RETURN':
                        balance_of_addr = w3.toChecksumAddress(item['args'][12 + 4 : 32 + 4])
                        got_balance = int.from_bytes(item['actions'][-1]['data'][:32], byteorder='big', signed=False)

                        # ensure that it returns what we expect, if we are expecting a particular return value
                        if (callee, balance_of_addr) in known_balances:
                            if got_balance != known_balances[callee, balance_of_addr]:
                                return ((callee, 'unexpected balanceOf after transfer'), MSG_FAIL_USED_NAUGHTY_TOKEN)
                        known_balances[callee,balance_of_addr] = got_balance

                elif method_sel == ERC20_TRANSFER_SELECTOR:
                    # if the first transfer() reverts, this is a broken token
                    if callee not in called_transfer:
                        if len(item['actions']) > 0 and item['actions'][-1]['type'] == 'REVERT':
                            return ((callee, 'transfer reverts'), MSG_FAIL_USED_NAUGHTY_TOKEN)
                        called_transfer.add(callee)

                    # if transfer() emits a Transfer event in the wrong amount, the semantics are different
                    (_, args) = erc20.decode_function_input(item['args'])
                    recipient = args['_to']
                    value = args['_value']
                    for action in item['actions']:
                        if action['type'] == 'TRANSFER' and action['to'] == recipient and action['value'] != value:
                            return ((callee, 'weird transfer event'), MSG_FAIL_USED_NAUGHTY_TOKEN)

                    # if we knew the balance of the sender, record the expected change
                    if (callee, item['from']) in known_balances:
                        known_balances[callee, item['from']] -= value

                    # if we knew the balance of the recipient, record the expected change
                    if (callee, recipient) in known_balances:
                        known_balances[callee, recipient] += value

                    # if this is sent from an exchange, ensure we are sending the expected amount
                    if item['from'] in exchanges:
                        assert (item['from'], callee) in expected_amounts_out
                        expected_amount = expected_amounts_out[item['from'], callee]
                        if expected_amount != value:
                            exchange_addr = item['from']
                            raise Exception(f'amount out from {exchange_addr} was not as expected: wanted {expected_amount:,} but got {value:,}')

                for sub_action in reversed(item['actions']):
                    stack.append((depth + 1, sub_action))
            if item['type'] == 'root':
                for sub_action in reversed(item['actions']):
                    stack.append((depth + 1, sub_action))
    raise Exception('himom')
