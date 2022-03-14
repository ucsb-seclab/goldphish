import argparse
import collections
import io
import itertools
import os
import typing
import networkx as nx
import sys
import web3
import web3.types
import web3.exceptions
import web3.contract
import web3._utils.events
import web3._utils.filters
import time
import logging
import gzip
import pricers
import find_circuit
import shooter
import hashlib
import logging.handlers
import enum
import sys
from eth_utils import event_abi_to_log_topic

from pricers.uniswap_v2 import UniswapV2Pricer
from ..utils import funded_deployer, get_ganache_fork, mine_block, parse_logs_for_net_profit
from utils import TETHER_ADDRESS, UNI_ADDRESS, USDC_ADDRESS, WBTC_ADDRESS, WETH_ADDRESS, ProgressReporter, get_abi, decode_trace_calls, get_block_logs, pretty_print_trace, setup_logging

l = logging.getLogger(__name__)

THRESHOLDS = {
    WETH_ADDRESS: web3.Web3.toWei('0.01', 'ether'),
    USDC_ADDRESS: 10 * (10 ** 6),
    TETHER_ADDRESS: 10 * (10 ** 6),
    UNI_ADDRESS: web3.Web3.toWei('0.25', 'ether'), # UNI also uses 18 decimals and had a max price of 40, so this is about $400, optimistically
    WBTC_ADDRESS: 1 * (10 ** 8) // 10_000, # like $1-4 ish?
}

IMPORTANT_TOPICS = []

# Set up the important log topics we'll need to listen
univ2: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('uniswap_v2/IUniswapV2Pair.json')['abi'],
)
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Mint().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Burn().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Swap().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Sync().abi))

univ3: web3.contract.Contract = web3.Web3().eth.contract(
    address=None,
    abi=get_abi('uniswap_v3/IUniswapV3Pool.json')['abi'],
)
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Mint().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Burn().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Initialize().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.SetFeeProtocol().abi)) # I'm not sure this is ever used?
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.CollectProtocol().abi)) # I'm not sure this is ever used?
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Swap().abi))

IMPORTANT_TOPICS_HEX = ['0x' + x.hex() for x in IMPORTANT_TOPICS]

univ2_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ2_excs.csv.gz')
assert os.path.isfile(univ2_fname), f'should have file {univ2_fname}'
univ3_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ3_excs.csv.gz')
assert os.path.isfile(univ3_fname)


FNAME_EXCHANGES_WITH_BALANCES = '/mnt/goldphish/tmp/exchanges_prefilter.csv'
FNAME_CANDIDATE_CSV = '/mnt/goldphish/tmp/top_block_candidate_arbitrages.csv'
FNAME_VERIFY_RESULT = '/mnt/goldphish/tmp/verify_out.csv'


class TraceMode(enum.Enum):
    NEVER   = 0
    ALWAYS  = 1
    ON_FAIL = 2


def load_exchange_balances(w3: web3.Web3):
    if os.path.exists(FNAME_EXCHANGES_WITH_BALANCES):
        l.debug(f'already prefetched balances (\'{FNAME_EXCHANGES_WITH_BALANCES}\' exists), no need to redo')
        return

    tip = w3.eth.get_block('latest')
    block_number = tip['number'] - 10
    l.debug(f'Prefiltering based off balances in block {block_number:,}')

    with open(FNAME_EXCHANGES_WITH_BALANCES + '.tmp', mode='w') as fout:
        with gzip.open(univ2_fname, mode='rt') as fin:
            for i, line in enumerate(fin):
                if i % 100 == 0:
                    l.debug(f'Processed {i:,} uniswap v2 exchanges for prefilter')
                    fout.flush()
                address, origin_block, token0, token1 = line.strip().split(',')
                address = w3.toChecksumAddress(address)
                origin_block = int(origin_block)
                token0 = w3.toChecksumAddress(token0)
                token1 = w3.toChecksumAddress(token1)

                # record liquidity balance of both token0 and token1
                try:
                    bal_token0 = w3.eth.contract(
                        address=token0,
                        abi=get_abi('erc20.abi.json'),
                    ).functions.balanceOf(address).call()
                    bal_token1 = w3.eth.contract(
                        address=token1,
                        abi=get_abi('erc20.abi.json'),
                    ).functions.balanceOf(address).call()
                    fout.write(f'2,{address},{origin_block},{token0},{token1},{bal_token0},{bal_token1}\n')
                except:
                    l.exception('could not get balance, ignoring')


        l.debug('loaded all uniswap v2 exchanges')

        with gzip.open(univ3_fname, mode='rt') as fin:
            for i, line in enumerate(fin):
                if i % 100 == 0:
                    l.debug(f'Processed {i:,} uniswap v3 exchanges for prefilter')
                    fout.flush()
                address, origin_block, token0, token1, fee = line.strip().split(',')
                address = w3.toChecksumAddress(address)
                origin_block = int(origin_block)
                token0 = w3.toChecksumAddress(token0)
                token1 = w3.toChecksumAddress(token1)
                fee = int(fee)

                # record liquidity balance of both token0 and token1
                try:
                    bal_token0 = w3.eth.contract(
                        address=token0,
                        abi=get_abi('erc20.abi.json'),
                    ).functions.balanceOf(address).call()
                    bal_token1 = w3.eth.contract(
                        address=token1,
                        abi=get_abi('erc20.abi.json'),
                    ).functions.balanceOf(address).call()
                    fout.write(f'3,{address},{origin_block},{token0},{token1},{fee},{bal_token0},{bal_token1}\n')
                except:
                    l.exception('could not get balance, ignoring')

        l.debug('finished load of exchange graph')
    os.rename(FNAME_EXCHANGES_WITH_BALANCES + '.tmp', FNAME_EXCHANGES_WITH_BALANCES)


def load_pool(w3: web3.Web3) -> pricers.PricerPool:
    l.debug('starting load of exchange graph')
    t_start = time.time()
    univ2_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ2_excs.csv.gz')
    assert os.path.isfile(univ2_fname), f'should have file {univ2_fname}'
    univ3_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ3_excs.csv.gz')
    assert os.path.isfile(univ3_fname)

    ret = pricers.PricerPool(w3)

    n_ignored = 0
    with open(FNAME_EXCHANGES_WITH_BALANCES) as fin:
        for line in fin:
            if line.startswith('2'):
                _, address, origin_block, token0, token1, bal0, bal1 = line.strip().split(',')

                origin_block = int(origin_block)
                bal0 = int(bal0)
                bal1 = int(bal1)
                if token0 in THRESHOLDS:
                    if bal0 < THRESHOLDS[token0]:
                        n_ignored += 1
                        continue
                if token1 in THRESHOLDS:
                    if bal1 < THRESHOLDS[token1]:
                        n_ignored += 1
                        continue
                ret.add_uniswap_v2(address, token0, token1, origin_block)
            else:
                assert line.startswith('3')

                _, address, origin_block, token0, token1, fee, bal0, bal1 = line.strip().split(',')
                fee = int(fee)

                origin_block = int(origin_block)
                bal0 = int(bal0)
                bal1 = int(bal1)
                if token0 in THRESHOLDS:
                    if bal0 < THRESHOLDS[token0]:
                        n_ignored += 1
                        continue
                if token1 in THRESHOLDS:
                    if bal1 < THRESHOLDS[token1]:
                        n_ignored += 1
                        continue
                ret.add_uniswap_v3(address, token0, token1, fee, origin_block)

    l.debug(f'Kept {ret.exchange_count:,} and ignored {n_ignored:,} exchanges below threshold ({n_ignored / (n_ignored + ret.exchange_count) * 100:.2f}%)')

    t_end = time.time()
    l.debug(f'Took {t_end - t_start:.2f} seconds to load into pricing pool')
    return ret


def get_relevant_logs(w3: web3.Web3, start_block: int, end_block: int) -> typing.Iterator[typing.Tuple[int, typing.List[web3.types.LogReceipt]]]:
    assert start_block <= end_block
    f: web3._utils.filters.Filter = w3.eth.filter({
        'topics': [IMPORTANT_TOPICS_HEX],
        'fromBlock': start_block,
        'toBlock': end_block,
    })
    logs = f.get_all_entries()
    gather = collections.defaultdict(lambda: [])
    for log in logs:
        gather[log['blockNumber']].append(log)

    for i in range(start_block, end_block + 1):
        yield (i, gather[i])


# profit must be enough to pay for 100k gas @ 10 gwei (both overly optimistic)
MIN_PROFIT_PREFILTER = (130_000) * (10 * (10 ** 9))

def process_candidates(w3: web3.Web3, pool: pricers.PricerPool, block_number: int, updated_exchanges: typing.Set[str], fout: io.TextIOWrapper, fout_errors: io.TextIOWrapper):
    # update pricer
    l.debug(f'{len(updated_exchanges)} exchanges updated in block {block_number:,}')

    # buffer results here so if a web3 exception occurs, we can safely raise the exception
    # without writing partial-block results to the output file
    to_write = []

    n_ignored = 0
    n_found = 0
    for p in find_circuit.profitable_circuits(updated_exchanges, pool, block_number, only_weth_pivot=True):
        if p.profit < MIN_PROFIT_PREFILTER:
            n_ignored += 1
            continue

        # this is potentially profitable, log as a candidate
        exchanges = [f'{x.address}|{isinstance(x, UniswapV2Pricer)}|{dxn}' for x, dxn in zip(p.circuit, p.directions)]
        exchanges_str = '#'.join(exchanges)
        to_write.append(f'{block_number},{p.amount_in},{exchanges_str},{p.profit},{w3.fromWei(p.profit, "ether")}\n')
        n_found += 1

    if len(to_write) > 0:
        fout.write(''.join(to_write))
        fout.flush()

    if n_ignored > 0:
        l.debug(f'Ignored {n_ignored} arbitrages due to not meeting profit threshold in block {block_number:,}')
    if n_found > 0:
        l.debug(f'Found {n_found} candidate arbitrages in block {block_number:,}')


def load_exchanges() -> typing.Tuple[typing.Dict[str, typing.Tuple[str, str]], typing.Dict[str, typing.Tuple[str, str, int]]]:
    uniswap_v2_exchanges: typing.Dict[str, typing.Tuple[str, str]] = {}
    uniswap_v3_exchanges: typing.Dict[str, typing.Tuple[str, str, int]] = {}

    start_load = time.time()
    l.debug('Loading exchanges')
    with gzip.open(univ2_fname, mode='rt') as fin:
        for line in fin:
            address, origin_block, token0, token1 = line.strip().split(',')
            address = web3.Web3.toChecksumAddress(address)
            token0 = web3.Web3.toChecksumAddress(token0)
            token1 = web3.Web3.toChecksumAddress(token1)
            uniswap_v2_exchanges[address] = (token0, token1)

    l.debug('loaded all uniswap v2 exchanges')

    with gzip.open(univ3_fname, mode='rt') as fin:
        for line in fin:
            address, origin_block, token0, token1, fee = line.strip().split(',')
            address = web3.Web3.toChecksumAddress(address)
            origin_block = int(origin_block)
            token0 = web3.Web3.toChecksumAddress(token0)
            token1 = web3.Web3.toChecksumAddress(token1)
            fee = int(fee)
            uniswap_v3_exchanges[address] = (token0, token1, fee)

    end = time.time()
    l.debug(f'finished loading exchanges, took {end - start_load:.2f} seconds')
    l.debug(f'Have {len(uniswap_v2_exchanges):,} uniswap v2 exchanges and {len(uniswap_v3_exchanges):,} uniswap v3 exchanges')

    return (uniswap_v2_exchanges, uniswap_v3_exchanges)


def do_verify(w3: web3.Web3):
    l.info('Starting verification of profitability')

    uniswap_v2_exchanges, uniswap_v3_exchanges = load_exchanges()

    with open(FNAME_VERIFY_RESULT, mode='w') as fout:
        with open(FNAME_CANDIDATE_CSV) as fin:
            while True:
                line = fin.readline()
                if line == '':
                    # no line yet
                    time.sleep(0.5)
                    continue

                # reconstruct the found_arbitrage
                block_number, amount_in, exchange_str, profit, _ = line.strip().split(',')
                block_number = int(block_number)
                amount_in = int(amount_in)
                profit = int(profit)

                circuit = []
                directions = []
                for exchange in exchange_str.split('#'):
                    address, is_uniswap_v2, zero_for_one = exchange.split('|')
                    is_uniswap_v2 = {'True': True, 'False': False}[is_uniswap_v2]
                    zero_for_one = {'True': True, 'False': False}[zero_for_one]

                    if is_uniswap_v2:
                        token0, token1 = uniswap_v2_exchanges[address]
                        pricer = pricers.UniswapV2Pricer(w3, address, token0, token1)
                    else:
                        token0, token1, fee = uniswap_v3_exchanges[address]
                        pricer = pricers.UniswapV3Pricer(w3, address, token0, token1, fee)

                    if len(circuit) == 0:
                        if zero_for_one:
                            pivot = token0
                        else:
                            pivot = token1
                        assert pivot == WETH_ADDRESS

                    circuit.append(pricer)
                    directions.append(zero_for_one)

                fa = find_circuit.FoundArbitrage(amount_in, circuit, directions, pivot_token=WETH_ADDRESS, profit = profit)

                # time to reshoot
                maybe_gas = check_reshoot(w3, fa, block_number)
                if maybe_gas is not None:
                    for i in itertools.count(1):
                        block = w3.eth.get_block(block_number + i)
                        if len(block['transactions']) > 0:
                            gasprice = w3.eth.get_transaction_receipt(block['transactions'][-1])['effectiveGasPrice']
                            break
                    total_fee = gasprice * maybe_gas
                    net_profit = fa.profit - total_fee
                    profit_sz = {True: 'Profitable', False: 'NotProfitable'}[net_profit > 0]
                    fout.write(f'{line.strip()}!{gasprice}/{block["number"]}!{maybe_gas}!{net_profit}!{net_profit / (10 ** 18)}!{profit_sz}\n')
                    fout.flush()
                else:
                    sha = hashlib.sha1(line.strip().encode('ascii')).hexdigest()
                    l.warning(f'verification failed: {sha}')
                    if sha == '807a96816f3809809035b6e37ef3fc07377159aa':
                        raise Exception('aha!')
                    if '12369879' in line.strip() and '0x886072A44BDd944495eFF38AcE8cE75C1EacDAF6' in line:
                        import pdb; pdb.set_trace()
                    fout.write(f'FAILED:{line.strip()}\n')
                    fout.flush()


def check_reshoot(w3: web3.Web3, fa: find_circuit.FoundArbitrage, block_number: int) -> typing.Optional[int]:
    """
    Attempts re-shoot and returns gas usage on success. On failure, returns None.
    """
    shooter_address, receipt, maybe_trace = shoot(w3, fa, block_number, do_trace = TraceMode.ON_FAIL)

    if receipt['status'] != 1:
        trace, txn = maybe_trace
        print('----------------------trace---------------------------')
        decoded = decode_trace_calls(trace, txn, receipt)
        pretty_print_trace(decoded, txn, receipt)
        print('------------------------------------------------------')
        return None

    movements = parse_logs_for_net_profit(receipt['logs'])
    if movements[WETH_ADDRESS][shooter_address] == fa.profit:
        return receipt['gasUsed']
    return None


NAUGHTY_TOKEN_FNAME = '/mnt/goldphish/tmp/naughty_tokens.txt'
def do_diagnose(w3: web3.Web3):
    naughty_tokens = set()
    if os.path.isfile(NAUGHTY_TOKEN_FNAME):
        with open(NAUGHTY_TOKEN_FNAME) as fin:
            for line in fin:
                address, reason = line.strip().split(',')
                assert w3.isChecksumAddress(address)
                naughty_tokens.add(address)

    l.debug(f'Already know about {len(naughty_tokens)} naughty tokens')

    uniswap_v2_exchanges, uniswap_v3_exchanges = load_exchanges()

    with open(FNAME_VERIFY_RESULT) as fin, \
         open(NAUGHTY_TOKEN_FNAME, mode='a') as fout_naughty, \
         open('/mnt/goldphish/tmp/diagnose_out.txt', mode='w') as fout_log:

        while True:
            line = fin.readline()
            if line == '':
                l.debug('got to end of file, sleeping a bit')
                time.sleep(20)
                continue

            if not line.startswith('FAILED:'):
                # the shoot worked, no need to diagnose
                continue

            line = line[len('FAILED:'):]

            # parse it out
            block_number, amount_in, exchange_str, profit, _ = line.strip().split(',')
            block_number = int(block_number)
            amount_in = int(amount_in)
            profit = int(profit)

            circuit = []
            directions = []
            all_tokens = set()
            for exchange in exchange_str.split('#'):
                address, is_uniswap_v2, zero_for_one = exchange.split('|')
                is_uniswap_v2 = {'True': True, 'False': False}[is_uniswap_v2]
                zero_for_one = {'True': True, 'False': False}[zero_for_one]

                if is_uniswap_v2:
                    token0, token1 = uniswap_v2_exchanges[address]
                    pricer = pricers.UniswapV2Pricer(w3, address, token0, token1)
                else:
                    token0, token1, fee = uniswap_v3_exchanges[address]
                    pricer = pricers.UniswapV3Pricer(w3, address, token0, token1, fee)

                all_tokens.add(token0)
                all_tokens.add(token1)
                circuit.append(pricer)
                directions.append(zero_for_one)

            # if we already know that this contains a naughty token, just skip
            if len(all_tokens.intersection(naughty_tokens)) > 0:
                l.debug('This guy uses a known naughty token.')
                fout_log.write(f'KNOWN_NAUGHTY:{line}:{all_tokens.intersection(naughty_tokens)}\n')
                continue

            fa = find_circuit.FoundArbitrage(amount_in, circuit, directions, pivot_token=WETH_ADDRESS, profit = profit)

            try:
                maybe_naughty_token, failure_reason = diagnose_single(w3, fa, block_number)
            except:
                sha = hashlib.sha1(line.encode('ascii')).hexdigest()
                l.critical(f'failed to diagnose {sha}: {line}')
                raise

            assert ':' not in failure_reason
            fout_log.write(f'{failure_reason}:{line}\n')
            if maybe_naughty_token is not None:
                new_naughty_token, reason = maybe_naughty_token
                fout_naughty.write(f'{new_naughty_token},{reason}\n')
                fout_naughty.flush()
                naughty_tokens.add(new_naughty_token)
                l.info(f'Found new naughty token: {new_naughty_token} | {reason}')


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
                        return ((callee, 'reverted in balanceOf'), 'naughty token - balanceOf reverted')
                    if item['actions'][-1]['type'] == 'RETURN':
                        balance_of_addr = w3.toChecksumAddress(item['args'][12 + 4 : 32 + 4])
                        got_balance = int.from_bytes(item['actions'][-1]['data'][:32], byteorder='big', signed=False)

                        # ensure that it returns what we expect, if we are expecting a particular return value
                        if (callee, balance_of_addr) in known_balances:
                            if got_balance != known_balances[callee, balance_of_addr]:
                                return ((callee, 'unexpected balanceOf after transfer'), 'naughty token - balanceOf after transfer was unusual')
                        known_balances[callee,balance_of_addr] = got_balance

                elif method_sel == ERC20_TRANSFER_SELECTOR:
                    # if the first transfer() reverts, this is a broken token
                    if callee not in called_transfer:
                        if len(item['actions']) > 0 and item['actions'][-1]['type'] == 'REVERT':
                            return ((callee, 'transfer reverts'), 'naughty token - transfer reverts')
                        called_transfer.add(callee)

                    # if transfer() emits a Transfer event in the wrong amount, the semantics are different
                    (_, args) = erc20.decode_function_input(item['args'])
                    recipient = args['_to']
                    value = args['_value']
                    for action in item['actions']:
                        if action['type'] == 'TRANSFER' and action['to'] == recipient and action['value'] != value:
                            return ((callee, 'weird transfer event'), 'naughty token - transfer emitted weird event')

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


def shoot(w3: web3.Web3, fa: find_circuit.FoundArbitrage, block_number: int, do_trace: TraceMode = TraceMode.NEVER) -> typing.Tuple[str, web3.types.TxReceipt, typing.Optional[typing.Any]]:
    account = funded_deployer()
    for ganache in get_ganache_fork(w3, block_number):

        for pricer in fa.circuit:
            pricer.set_web3(ganache)

        # deploy shooter
        shooter_address = shooter.deploy.deploy_shooter(ganache, account, max_priority = 2, max_fee_total = w3.toWei(1, 'ether'))

        # wrap some ether
        weth: web3.contract.Contract = ganache.eth.contract(
            address='0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            abi=get_abi('weth9/WETH9.json')['abi'],
        )
        wrap = weth.functions.deposit().buildTransaction({'value': 100, 'from': account.address})
        wrap_hash = ganache.eth.send_transaction(wrap)
        wrap_receipt = ganache.eth.wait_for_transaction_receipt(wrap_hash)
        assert wrap_receipt['status'] == 1

        # transfer to shooter
        xfer = weth.functions.transfer(shooter_address, 100).buildTransaction({'from': account.address})
        xfer_hash = ganache.eth.send_transaction(xfer)
        xfer_receipt = ganache.eth.wait_for_transaction_receipt(xfer_hash)
        assert xfer_receipt['status'] == 1

        shot = shooter.composer.construct_from_found_arbitrage(fa, 0, ganache.eth.get_block('latest')['number'] + 1)

        l.debug(f'About to shoot {shot.hex()}')

        txn: web3.types.TxParams = {
            'chainId': ganache.eth.chain_id,
            'from': account.address,
            'to': shooter_address,
            'value': 0,
            'nonce': ganache.eth.get_transaction_count(account.address),
            'data': shot,
            'gas': 10_000_000, # huge gas to avoid any out of gas error
            'maxPriorityFeePerGas': 2,
            'maxFeePerGas': 1000 * (10 ** 9),
        }
        signed_txn = ganache.eth.account.sign_transaction(txn, account.key)
        txn_hash = ganache.eth.send_raw_transaction(signed_txn.rawTransaction)

        mine_block(ganache)

        receipt = ganache.eth.get_transaction_receipt(txn_hash)

        should_trace = (do_trace == TraceMode.ALWAYS) or (receipt['status'] != 1 and do_trace == TraceMode.ON_FAIL)

        if should_trace:
            trace = ganache.provider.make_request('debug_traceTransaction', [receipt['transactionHash'].hex()])
            return shooter_address, receipt, (trace['result']['structLogs'], txn)

        return shooter_address, receipt, None



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['verify', 'diagnose'], help='verification mode', default=None)
    parser.add_argument('--verify', action='store_true', help='verification mode', default=False)

    args = parser.parse_args()

    if args.mode == 'verify':
        setup_logging('top_block_verify', suppress=['shooter.deploy'])
    elif args.mode == 'diagnose':
        setup_logging('top_block_diagnose', suppress=['shooter.deploy'])
    else:
        setup_logging('top_block_candidates', suppress=['shooter.deploy'])

    l.info('Booting up...')

    #
    # Connect to web3
    #

    w3 = web3.Web3(web3.WebsocketProvider(
        'ws://172.17.0.1:8546',
        websocket_timeout=60 * 5,
        websocket_kwargs={
            'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
        },
    ))
    if not w3.isConnected():
        l.error(f'Could not connect to web3')
        exit(1)

    l.debug(f'Connected to web3, chainId={w3.eth.chain_id}')


    if args.mode == 'verify':
        l.info(f'Verifying candidate arbitrages')
        do_verify(w3)
    elif args.mode == 'diagnose':
        l.info(f'Diagnosing failures')
        do_diagnose(w3)
    else:
        l.info('Starting candidate searching')
        time.sleep(4)

        try:
            load_exchange_balances(w3)
            pricer = load_pool(w3)

            #
            # Load all candidate profitable arbitrages
            with open(FNAME_CANDIDATE_CSV, mode='w') as fout, open('/mnt/goldphish/tmp/reshoot_log_top_block_errors.csv', mode='w') as fout_errors:
                start_block = 12_369_621
                end_block = w3.eth.get_block('latest')['number']
                progress_reporter = ProgressReporter(l, end_block, start_block)
                batch_size_blocks = 200
                curr_block = start_block
                while curr_block < end_block:
                    this_end_block = min(curr_block + batch_size_blocks - 1, end_block)
                    n_logs = 0
                    for block_number, logs in get_relevant_logs(w3, curr_block, this_end_block):
                        updated_exchanges = pricer.observe_block(logs)
                        while True:
                            try:
                                process_candidates(w3, pricer, block_number, updated_exchanges, fout, fout_errors)
                                break
                            except Exception as e:
                                if 'execution aborted (timeout = 5s)' in str(e):
                                    l.exception('Encountered timeout, trying again in a little bit')
                                    time.sleep(30)
                                else:
                                    raise e
                        progress_reporter.observe(n_items=1)
                        n_logs += len(logs)
                    l.debug(f'Found {n_logs} logs this batch')
                    curr_block = this_end_block + 1

        except Exception as e:
            l.exception('top-level exception')
            raise e

if __name__ == '__main__':
    main()

