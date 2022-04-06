import itertools
import psycopg2
import psycopg2.extensions
import logging
import time
import typing
import web3
import web3._utils.filters
import web3.types
import web3.contract
from backtest.top_of_block.verify import check_candidates
from backtest.utils import ERC20_TRANSFER_TOPIC, parse_logs_for_net_profit

import pricers
import find_circuit
from backtest.top_of_block.common import TraceMode, WrappedFoundArbitrage, connect_db, load_exchanges, load_naughty_tokens, shoot
from backtest.top_of_block.constants import UNISWAP_V2_SYNC_TOPIC, univ2

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

        diagnosis = diagnose_single(w3, curr, fa, block_number)

        if isinstance(diagnosis, DiagnosisNaughtyToken):
            curr.execute(
                """
                UPDATE failed_arbitrages SET diagnosis = %s WHERE id = %s
                """,
                (MSG_FAIL_USED_NAUGHTY_TOKEN, failed_arbitrage_id,)
            )
            assert curr.rowcount == 1

            curr.execute(
                """
                INSERT INTO naughty_tokens (address, reason, diagnosed_from) VALUES (%s, %s, %s)
                """,
                (bytes.fromhex(diagnosis.address[2:]), diagnosis.root_cause, failed_arbitrage_id)
            )
            assert curr.rowcount == 1
            naughty_tokens.add(diagnosis.address)
            l.info(f'Found new naughty token: {diagnosis.address} | {diagnosis.root_cause}')
        else:
            assert isinstance(diagnosis, DiagnosisNotEncodable)
            curr.execute(
                """
                UPDATE failed_arbitrages SET diagnosis = %s WHERE id = %s
                """,
                ('not encodable', failed_arbitrage_id,)
            )
            assert curr.rowcount == 1

        curr.connection.commit()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        """
        CREATE TABLE IF NOT EXISTS naughty_tokens (
            id             SERIAL PRIMARY KEY NOT NULL,
            address        BYTEA NOT NULL,
            reason         TEXT NOT NULL,
            diagnosed_from INTEGER NOT NULL REFERENCES failed_arbitrages (id) ON DELETE CASCADE
        );
        """
    )
    curr.connection.commit()


def get_failed_arbitrages(w3: web3.Web3, curr: psycopg2.extensions.cursor) -> typing.Iterator[typing.Tuple[int, int, WrappedFoundArbitrage]]:
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

        fa = WrappedFoundArbitrage(
                find_circuit.FoundArbitrage(
                amount_in = amount_in,
                circuit = circuit,
                directions = directions,
                pivot_token = WETH_ADDRESS,
                profit = profit
            ),
            candidate_id
        )

        l.debug(f'Diagnosing failed arbitrage id={id_:,} candidate_id={candidate_id} block_number={block_number:,} expected_profit={w3.fromWei(profit, "ether"):.6f} ETH')

        yield block_number, id_, fa


erc20: web3.contract.Contract = web3.Web3().eth.contract(address = b'\x00' * 20, abi=get_abi('erc20.abi.json'))
ERC20_BALANCEOF_SELECTOR = erc20.functions.balanceOf(web3.Web3.toChecksumAddress(b'\x00' * 20)).selector[2:]
ERC20_TRANSFER_SELECTOR = erc20.functions.transfer(web3.Web3.toChecksumAddress(b'\x00' * 20), 10).selector[2:]


class DiagnosisNaughtyToken(typing.NamedTuple):
    address: str
    root_cause: str


class DiagnosisNotEncodable(typing.NamedTuple):
    pass


def diagnose_single(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        fa: WrappedFoundArbitrage,
        block_number: int
    ) -> typing.Union[DiagnosisNaughtyToken, DiagnosisNotEncodable]:
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
        amount = pricers.out_from_transfer(token_out, amount_out)

    # if this doesn't match the expected profit, figure out why
    computed_profit = amount - fa.amount_in
    if computed_profit != fa.profit:
        for pricer in fa.circuit:
            l.debug(str(pricer) + ' ' + pricer.address)
        l.info(f'Computed ({computed_profit}) and expected profit ({fa.profit}) mismatch')
        maybe_naughty_token = diagnose_output_mismatch(w3, curr, fa, amount, block_number)
        if maybe_naughty_token is not None:
            return DiagnosisNaughtyToken(maybe_naughty_token, 'unexpected balance change with no logs')
        raise Exception('what')
        return

    shooter_address, (shoot_result,) = shoot(w3, [fa], block_number, gaslimit=300_000, do_trace = TraceMode.ALWAYS)

    if shoot_result.encodable == False:
        return DiagnosisNotEncodable()

    exchanges: typing.Set[str] = set(x.address for x in fa.circuit)

    decoded = decode_trace_calls(shoot_result.trace, shoot_result.tx_params, shoot_result.receipt)
    print('----------------------trace---------------------------')
    pretty_print_trace(decoded, shoot_result.tx_params, shoot_result.receipt)
    print('------------------------------------------------------')

    if shoot_result.receipt['status'] == 1:
        expected_profit = fa.profit
        movements = parse_logs_for_net_profit(shoot_result.receipt['logs'])
        actual_profit = movements[WETH_ADDRESS][shooter_address]
        assert actual_profit == expected_profit, f'expected {actual_profit:,} == {expected_profit:,}'
        raise NotImplementedError('no failure here, unexpected!')
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
                        return DiagnosisNaughtyToken(callee, 'reverted in balanceOf')
                    if item['actions'][-1]['type'] == 'RETURN':
                        balance_of_addr = w3.toChecksumAddress(item['args'][12 + 4 : 32 + 4])
                        got_balance = int.from_bytes(item['actions'][-1]['data'][:32], byteorder='big', signed=False)

                        # ensure that it returns what we expect, if we are expecting a particular return value
                        if (callee, balance_of_addr) in known_balances:
                            if got_balance != known_balances[callee, balance_of_addr]:
                                return DiagnosisNaughtyToken(callee, 'unexpected balanceOf after transfer')
                        known_balances[callee,balance_of_addr] = got_balance

                elif method_sel == ERC20_TRANSFER_SELECTOR:
                    # if the first transfer() reverts, this is a broken token
                    if callee not in called_transfer:
                        if len(item['actions']) > 0 and item['actions'][-1]['type'] == 'REVERT':
                            return DiagnosisNaughtyToken(callee, 'transfer reverts')
                        called_transfer.add(callee)

                    # if transfer() emits a Transfer event in the wrong amount, the semantics are different
                    (_, args) = erc20.decode_function_input(item['args'])
                    recipient = args['_to']
                    value = args['_value']
                    for action in item['actions']:
                        if action['type'] == 'TRANSFER' and action['to'] == recipient and action['value'] != value:
                            return DiagnosisNaughtyToken(callee, 'weird transfer event')

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


def diagnose_output_mismatch(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        fa: find_circuit.FoundArbitrage,
        amount: int,
        block_number: int
    ) -> typing.Optional[str]:
    """
    Diagnose the reason for amout_out mismatch -- if because of a weird token that changes balances with no logs,
    returns the token address.
    """
    all_uniswap_v2s: typing.List[pricers.UniswapV2Pricer] = []
    for p in fa.circuit:
        if isinstance(p, pricers.UniswapV2Pricer):
            all_uniswap_v2s.append(p)
    
    if len(all_uniswap_v2s) == 0:
        raise NotImplementedError('not sure about how to handle this yet')

    last_syncs: typing.List[typing.Tuple[pricers.UniswapV2Pricer, typing.Any, typing.Set[str]]] = []
    # check for balance mismatches
    for p in all_uniswap_v2s:
        last_sync = get_last_uniswap_v2_sync(w3, p.address, block_number)
        sync = p.contract.events.Sync().processLog(last_sync)

        l.debug(f'last sync from {p.address} was on {last_sync["blockNumber"]}')
        l.debug(f'last sync args: {sync["args"]}')

        # see if this matches what we think is the case on this block
        bal0_real = p.token0_contract.functions.balanceOf(p.address).call(block_identifier=block_number)
        bal1_real = p.token1_contract.functions.balanceOf(p.address).call(block_identifier=block_number)

        brokens = set()
        if bal0_real != sync['args']['reserve0']:
            l.warning(f'Balance mismatch: {p.address} token={p.token0} sync said {sync["args"]["reserve0"]} but balance said {bal0_real}')
            brokens.add(p.token0)
        if bal1_real != sync['args']['reserve1']:
            brokens.add(p.token1)
            l.warning(f'Balance mismatch: {p.address} token={p.token1} sync said {sync["args"]["reserve1"]} but balance said {bal1_real}')

        last_syncs.append((p, sync, brokens, (bal0_real, bal1_real)))


    # go over each broken thing and see if a Transfer() happened after Sync()
    for p, sync, broken_tokens, _ in last_syncs:
        if len(broken_tokens) == 0:
            # nothing here to diagnose
            continue

        for token_addr in broken_tokens:
            contract: web3.contract.Contract = w3.eth.contract(
                address = token_addr,
                abi = get_abi('erc20.abi.json'),
            )
            f = contract.events.Transfer().build_filter()
            f.fromBlock = sync['blockNumber']
            f.toBlock = block_number
            f.args.to.match_single(p.address)
            logs = f.deploy(w3).get_all_entries()

            for log in logs:
                if log['blockNumber'] > sync['blockNumber'] or (sync['blockNumber'] == log['blockNumber'] and sync['logIndex'] < log['logIndex']):
                    l.warning(f'Found transfer into exchange after Sync')
                    l.warning(f'Sync {sync}')
                    l.warning(f'Log {log}')
                    raise Exception('Transfer after Sync()')
            
            # mysterious balance change
            return token_addr

    # last resort -- see if the balanceOf changes at all?
    for p, sync, _, (bal0, bal1) in last_syncs:
        for i in itertools.count(1):
            this_block = block_number - i
            if this_block < sync['blockNumber']:
                break
            if i % 10 == 0:
                l.debug(f'seeking balance change in block {this_block:,}')
            new_bal0 = p.token0_contract.functions.balanceOf(p.address).call(block_identifier=this_block)
            new_bal1 = p.token1_contract.functions.balanceOf(p.address).call(block_identifier=this_block)

            brokens = set()
            if new_bal0 != bal0:
                l.critical(f'bal0 changed: {bal0} -> {new_bal0} block={this_block:,}')
                brokens.add(p.token0)
            if new_bal1 != bal1:
                l.critical(f'bal1 changed: {bal1} -> {new_bal1} block={this_block:,}')
                brokens.add(p.token1)

            for token_addr in brokens:
                f: web3._utils.filters.Filter = w3.eth.filter({
                    'address': token_addr,
                    'fromBlock': this_block + 1,
                    'toBlock': this_block + 1,
                })
                for log in f.get_all_entries():
                    if log['topics'][0] == ERC20_TRANSFER_TOPIC:
                        txn = erc20.events.Transfer().processLog(log)
                        assert txn['args']['to'] != p.address
                        assert txn['args']['from'] != p.address
                    else:
                        raise Exception('dont understand this log')
                raise Exception(f'seems like this token is broken but not sure')

    # at this point we have no clue about why/how the balance could have changed
    raise Exception('unknown mismatch cause')


def get_last_uniswap_v2_sync(w3: web3.Web3, address: str, on_or_before: int) -> web3.types.LogReceipt:
    # get the last Sync() event emitted from this address
    batch_size_blocks = 200
    for i in itertools.count(0):
        start_block = on_or_before - (i + 1) * batch_size_blocks + 1
        end_block   = on_or_before - i * batch_size_blocks
        assert start_block > 10_000_000
        l.debug(f'Searching for recent Sync() of {address} from {start_block} to {end_block}')
        f: web3._utils.filters.Filter = w3.eth.filter({
            'address': address,
            'topics': ['0x' + UNISWAP_V2_SYNC_TOPIC.hex()],
            'fromBlock': start_block,
            'toBlock': end_block,
        })
        logs = f.get_all_entries()
        if len(logs) > 0:
            return logs[-1]


def diagnose_mismatched_profit(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        fa: WrappedFoundArbitrage,
        block_number: int
    ):
    raise NotImplementedError('this was never finished')
    pricer = find_circuit.find.PricingCircuit(fa.circuit, fa.directions)
    remeasured_output = pricer.sample(fa.amount_in, block_number)
    # there's a small bug sometimes where the profit gets fucked with (+/- 1 wei), see if that applies here
    remeasured_profit = remeasured_output - fa.amount_in
    diff = remeasured_profit - fa.profit
    if diff == 0:
        l.debug(f'saw no diff, so not diagnosing mismatched profit')
        return False

    l.warning(f'Saw profit diff of {diff}')

    amount_in_to_try = fa.amount_in + diff
    amount_out = pricer.sample(amount_in_to_try, block_number)
    amount_out_default = pricer.sample(fa.amount_in, block_number)

    if amount_out != amount_out_default:
        raise NotImplementedError('not sure how to handle this either')
    else:
        l.info(f'same amount_out after subtracting {diff} from amount_in')

    # this is the weird situation where everything is correct, we just forgot to adjust profit
    # when we saved a few wei due to rounding
    l.info('correcting prior circuit profit-finding error')

    fa.fa = fa.fa._replace(profit = remeasured_profit)

    curr.execute('SELECT EXISTS(SELECT 1 FROM verified_arbitrages WHERE candidate_id = %s)', (fa.id,))
    (does_exist,) = curr.fetchone()
    assert does_exist == False, 'should not have a verified arbitrage of this ID (inconsistent)'

    curr.execute('SELECT id FROM failed_arbitrages WHERE candidate_id = %s', (fa.id,))
    assert curr.rowcount == 1, f'should have 1 failed aritrage of this candidate_id, but have {curr.rowcount}'
    (failed_arbitrage_id,) = curr.fetchone()

    check_candidates(w3, curr, block_number, [fa])

    curr.execute('SELECT EXISTS(SELECT 1 FROM verified_arbitrages WHERE candidate_id = %s)', (fa.id,))
    (does_exist,) = curr.fetchone()
    assert does_exist == True, 'should have a verified arbitrage of this ID now'

    curr.execute('DELETE FROM failed_arbitrages WHERE id = %s', (failed_arbitrage_id,))
    assert curr.rowcount == 1
    curr.execute('UPDATE candidate_arbitrages SET profit_no_fee = %s WHERE id = %s', (remeasured_profit, fa.id))
    assert curr.rowcount == 1

    raise Exception('himom')
