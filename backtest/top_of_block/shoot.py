import collections
import datetime
import decimal
import itertools
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import numpy as np
import web3
import web3.types
import web3.contract
import web3._utils.filters
import typing
import logging
import argparse
import psycopg2.extensions
import threading
import queue
from backtest.top_of_block.constants import MIN_PROFIT_PREFILTER
from find_circuit.find import DEFAULT_FEE_TRANSFER_CALCULATOR, BuiltinFeeTransferCalculator, FeeTransferCalculator, FoundArbitrage, PricingCircuit, detect_arbitrages_bisection
import pricers

from backtest.utils import connect_db, erc20
from eth_account import Account
from eth_account.signers.local import LocalAccount
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.liquidity_bootstrapping_pool import BalancerV2LiquidityBootstrappingPoolPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.base import BaseExchangePricer
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer
from shooter.composer import construct_arbitrage
from shooter.encoder import serialize

from utils import BALANCER_VAULT_ADDRESS, WETH_ADDRESS, decode_trace_calls, get_abi, pretty_print_trace

l = logging.getLogger(__name__)

DEBUG = False

SHOOTER_ARTIFACT_PATH = pathlib.Path(__file__).parent.parent.parent / 'artifacts' / 'contracts' / 'shooter.sol' / 'Shooter.json'

assert os.path.isfile(SHOOTER_ARTIFACT_PATH)

with open(SHOOTER_ARTIFACT_PATH) as fin:
    SHOOTER_ARTIFACT = json.load(fin)

generic_shooter: web3.contract.Contract = web3.Web3().eth.contract(address=web3.Web3.toChecksumAddress(b'\x00'*20), abi=SHOOTER_ARTIFACT['abi'])

DO_APPROVE_SELECTOR = bytes.fromhex(
    generic_shooter.functions.doApprove(
        web3.Web3.toChecksumAddress(b'\x00'*20),
        web3.Web3.toChecksumAddress(b'\x00'*20),
    ).selector[2:]
)

QUEUE_FINISHED = 0xDEADBEEF

BANNED_TOKENS = frozenset((
    '0xD46bA6D942050d489DBd938a2C909A5d5039A161', # Ampleforth -- rebasing token, fucks up with DEX
))


def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'do-shoot'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (before run)')

    return parser_name, connect


def connect(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        fill_queue(curr)
        db.commit()
        return

    if args.worker_name is None:
        print('Must supply worker_name', file=sys.stderr)
        exit(1)

    l.info('Starting arbitrage campaign connection')

    time.sleep(4)

    fee_calculator = InferredTokenTransferFeeCalculator()

    while True:
        maybe_rez = get_reservation(curr, args.worker_name)

        if maybe_rez is None:
            break
        
        reservation_id, block_number = maybe_rez
        process_reservation(w3, curr, reservation_id, block_number, fee_calculator)


class ConnectedArbitrageCampaign(typing.NamedTuple):
    exchanges: typing.List[str]
    directions: typing.List[typing.Tuple[str, str]]

    def __hash__(self) -> int:
        return hash((tuple(self.exchanges), tuple(self.directions)))


def process_reservation(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        reservation_id: int,
        block_number: int,
        fee_calculator: 'InferredTokenTransferFeeCalculator',
    ):
    fee_calculator.sync(curr, block_number)

    timestamp_to_use = w3.eth.get_block(block_number + 1)['timestamp']

    candidates = get_candidates_in_block(curr, block_number)

    l.debug(f'Have {len(candidates):,} arbitrages to test')

    # for keeping progress
    t_start = time.time()
    percent_marks = [len(candidates) * x // 100 for x in range(20, 100, 20)]

    # some stats
    n_banned_skipped = 0
    n_no_arb_on_fee = 0

    # cache of pricers at current block
    pricer_cache: typing.Dict[str, BaseExchangePricer] = {}

    new_banned_tokens    = set()
    new_banned_exchanges = set()

    banned_tokens     = load_broken_tokens_at(curr, block_number).union(BANNED_TOKENS)
    banned_exchanges  = set()

    results_success: typing.Dict[int, AutoAdaptShootSuccess] = {}
    results_failure: typing.Dict[int, str]                   = {}

    proc, w3_ganache, acct, shooter_address = open_ganache(block_number)

    for i, candidate in enumerate(candidates):
        l.debug(f'reshooting id={candidate.id_}')
        # report progress
        if i > 0 and i in percent_marks:
            elapsed = time.time() - t_start
            nps = i / elapsed
            remain = len(candidates) - i
            eta_seconds = remain / nps
            eta = datetime.timedelta(seconds=eta_seconds)

            l.debug(f'Working, {i:,}/{len(candidates):,} ({round(i / len(candidates) * 100)}%) complete, ETA {eta}')

        # find if we should skip bc of a banned exchange
        these_banned_exchanges = set(candidate.exchanges).intersection(banned_exchanges)
        if len(these_banned_exchanges) > 0:
            # this is a banned arbitrage, ignore it
            n_banned_skipped += 1
            sz_banned_exchanges = ','.join(sorted(these_banned_exchanges))
            results_failure[candidate.id_] = f'Banned exchange/s: {sz_banned_exchanges}'
            l.debug(f'Skipping arbitrage id={candidate.id_} because it uses banned tokens {sz_banned_exchanges}')
            continue

        # find if we should skip bc of a banned token
        all_tokens = set(x for x, _ in candidate.directions)

        these_banned_tokens = all_tokens.intersection(banned_tokens)
        if len(these_banned_tokens) > 0:
            # this is a banned arbitrage, ignore it
            n_banned_skipped += 1
            sz_banned_tokens = ','.join(sorted(these_banned_tokens))
            results_failure[candidate.id_] = f'Banned token/s: {sz_banned_tokens}'
            l.debug(f'Skipping arbitrage id={candidate.id_} because it uses banned tokens {sz_banned_tokens}')
            continue

        # get pricers
        circuit = []
        for x in candidate.exchanges:
            if x in pricer_cache:
                circuit.append(pricer_cache[x])
            else:
                pricer = load_pricer_for(w3, curr, x)
                assert pricer is not None
                pricer_cache[x] = pricer
                circuit.append(pricer)

        pc = PricingCircuit(
            circuit,
            candidate.directions
        )

        fa = FoundArbitrage(
            circuit = circuit,
            directions = candidate.directions,
            pivot_token = WETH_ADDRESS,
            amount_in = candidate.amount_in,
            profit = candidate.profit_before_fee,
        )

        maybe_fa = detect_arbitrages_bisection(
            pc,
            block_number,
            timestamp = timestamp_to_use,
            try_all_directions = False,
            fee_transfer_calculator = fee_calculator
        )

        fee_calculator.infer_relays_and_aliases(fa, shooter_address)
        
        has_fees = any(fee_calculator.has_fee(t) for t in all_tokens)
        if len(maybe_fa) == 0:
            # no arbitrages found, is it because we had fees?
            if has_fees:
                results_failure[candidate.id_] = 'No arbitrage after applying inferred fees'
                n_no_arb_on_fee += 1
            else:
                l.critical(f'Could not replicate arbitrage!!!!!')
                l.critical(f'id = {candidate.id_}')
                l.critical(f'block_number = {block_number}')
                l.critical(f'found_arbitrage = {fa}')
                raise Exception('could not replicate!!')

        (fa,) = maybe_fa

        if not has_fees:
            if fa.profit != candidate.profit_before_fee:
                l.critical(f'profit changed!!! old: {candidate.profit_before_fee / 10 ** 18:.5f} vs new {fa.profit / 10 ** 18:.5f} ETH')
            else:
                l.debug('passed; same profit as expected')


        result = auto_adapt_attempt_shoot_candidate(
            w3_ganache,
            curr,
            acct,
            shooter_address,
            fa,
            fee_calculator,
        )
        if isinstance(result, DiagnosisBrokenToken):
            banned_tokens.add(result.token_address)
            new_banned_tokens.add(result.token_address)
            n_banned_skipped += 1
        elif isinstance(result, DiagnosisBadExchange):
            banned_exchanges.add(result.exchange)
            new_banned_exchanges.add(result.exchange)
            n_banned_skipped += 1
        elif isinstance(result, DiagnosisNoArbitrageOnFeeApplied):
            n_no_arb_on_fee += 1
        elif isinstance(result, AutoAdaptShootSuccess):
            results_success[candidate.id_] = result
        else:
            raise Exception(f'Not sure what this is: {result}')

    l.debug(f'Skipped {n_banned_skipped:,} arbitrages due to banned token or exchange use')
    l.debug(f'Had {n_no_arb_on_fee:,} arbitrages that diasappeared on applying fee')
    l.debug(f'Had {len(results_success):,} successful arbitrages in {block_number}')

    proc.kill()
    proc.wait()

    return


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_arbitrage_reshoot_blocks (
            id                 SERIAL PRIMARY KEY NOT NULL,
            block_number       INTEGER NOT NULL,
            worker             TEXT,
            claimed_on         TIMESTAMP WITHOUT TIME ZONE,
            completed_on       TIMESTAMP WITHOUT TIME ZONE            
        );

        CREATE TABLE IF NOT EXISTS inferred_token_fee_on_transfer (
            id SERIAL             PRIMARY KEY NOT NULL,
            token_id              INTEGER NOT NULL REFERENCES tokens (id) ON DELETE CASCADE,
            fee                   NUMERIC(20, 20),
            round_down            BOOLEAN,
            from_address          BYTEA,
            to_address            BYTEA,
            block_number_inferred INTEGER NOT NULL,
            updated_on            TIMESTAMP WITHOUT TIME ZONE
        );

        CREATE INDEX IF NOT EXISTS idx_inferred_token_fee_on_transfer_updated_on ON inferred_token_fee_on_transfer (updated_on);
        CREATE INDEX IF NOT EXISTS idx_inferred_token_fee_on_transfer_token_id ON inferred_token_fee_on_transfer (token_id);

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_shoot_results (
            candidate_arbitrage_id INTEGER NOT NULL REFERENCES candidate_arbitrages (id) ON DELETE CASCADE,
            shoot_success          BOOLEAN NOT NULL,
            failure_reason         TEXT,
            gas_used               INTEGER CHECK ((shoot_success = true and gas_used is not null) OR (shoot_success = false and gas_used is null)),
            had_fee_on_xfer_token  BOOLEAN,
            real_profit_before_fee NUMERIC(78, 0) NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_arbitrage_shoot_results_candidate_arbitrage_id ON candidate_arbitrage_shoot_results (candidate_arbitrage_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_arbitrage_shoot_results_shoot_success ON candidate_arbitrage_shoot_results (shoot_success);

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_shoot_results_used_fees (
            candidate_arbitrage_id INTEGER NOT NULL REFERENCES candidate_arbitrages (id) ON DELETE CASCADE,
            fee_used               INTEGER NOT NULL REFERENCES inferred_token_fee_on_transfer (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS broken_tokens (
            token_id               INTEGER NOT NULL PRIMARY KEY REFERENCES tokens (id) ON DELETE CASCADE,
            broken_at_block_number INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_broken_tokens_block_number ON broken_tokens (broken_at_block_number);

        CREATE TABLE IF NOT EXISTS banned_exchanges (
            address                BYTEA NOT NULL,
            broken_at_block_number INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_banned_exchanges ON banned_exchanges (broken_at_block_number);
        """
    )
    pass

def fill_queue(curr: psycopg2.extensions.cursor):
    curr.execute('SELECT COUNT(*) FROM candidate_arbitrage_reshoot_blocks;')
    (n_queued,) = curr.fetchone()

    if n_queued > 0:
        l.debug('not filling queue')
        return

    curr.execute(
        '''
        INSERT INTO candidate_arbitrage_reshoot_blocks (block_number)
        SELECT DISTINCT block_number
        FROM candidate_arbitrages
        '''
    )
    assert curr.rowcount > 0
    l.info(f'filled queue with {curr.rowcount:,} blocks')


def get_reservation(curr: psycopg2.extensions.cursor, worker_name: str) -> typing.Optional[typing.Tuple[int, int]]:
    curr.execute('BEGIN TRANSACTION')

    curr.execute(
        '''
        SELECT id, block_number
        FROM candidate_arbitrage_reshoot_blocks
        WHERE id = 43 -- claimed_on IS NULL AND completed_on IS NULL AND block_number >= 13100000
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        '''
    )
    if curr.rowcount < 1:
        l.info('Finished queue')
        return

    id_, block_number = curr.fetchone()
    curr.execute(
        '''
        UPDATE candidate_arbitrage_reshoot_blocks
        SET claimed_on = NOW()::timestamp, worker = %s
        WHERE id = %s
        ''',
        (worker_name, id_),
    )
    assert curr.rowcount == 1

    if not DEBUG:
        curr.connection.commit()

    l.info(f'Processing reservation id={id_:,} block_number={block_number})')

    return id_, block_number


_port = 0
def open_ganache(block_number: int) -> typing.Tuple[subprocess.Popen, web3.Web3, LocalAccount, str]:
    global _port
    acct: LocalAccount = Account.from_key(bytes.fromhex('f96003b86ed95cb86eae15653bf4b0bc88691506141a1a9ae23afd383415c268'))

    bin_loc = '/opt/ganache-fork/src/packages/ganache/dist/node/cli.js'
    cwd_loc = '/opt/ganache-fork/'

    my_pid = os.getpid()
    my_slice = my_pid % 1000 # I assume this is enough that none of our workers will have duplicate slices
    n_per_slice = 10 # number of ports allocated per slice before they must be reused
    slice_start = 10000 + my_slice * n_per_slice
    ganache_port = slice_start + _port
    _port = (_port + 1) % n_per_slice

    web3_host = os.getenv('WEB3_HOST', 'ws://172.17.0.1:8546')
    p = subprocess.Popen(
        [
            'node',
            bin_loc,
            '--fork.url', web3_host,
            '--fork.blockNumber', str(block_number),
            '--server.port', str(ganache_port),
            '--chain.chainId', '1',
            # '--chain.hardfork', 'arrowGlacier',
            '--wallet.accounts', f'{acct.key.hex()},{web3.Web3.toWei(1000, "ether")}',
        ],
        stdout=subprocess.DEVNULL,
        cwd=cwd_loc,
    )

    l.debug(f'spawned ganache on PID={p.pid} port={ganache_port}')

    w3 = web3.Web3(web3.WebsocketProvider(
            f'ws://localhost:{ganache_port}',
            websocket_timeout=60 * 5,
            websocket_kwargs={
                'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
            },
        )
    )

    def patch_make_batch_request(requests: typing.Tuple[str, typing.Any]):
        ret = []
        for method, args in requests:
            ret.append(w3.provider.make_request(method, args))
        return ret

    w3.provider.make_request_batch = patch_make_batch_request

    while not w3.isConnected():
        time.sleep(0.01)

    for _ in range(10):
        if w3.eth.get_balance(acct.address) == web3.Web3.toWei(1000, 'ether'):
            break
        l.debug('Backing off on balance check...')
        time.sleep(0.01)

    #
    # deploy the shooter
    #
    shooter = w3.eth.contract(
        bytecode = SHOOTER_ARTIFACT['bytecode'],
        abi = SHOOTER_ARTIFACT['abi'],
    )

    constructor_txn = shooter.constructor().buildTransaction({'from': acct.address})
    txn_hash = w3.eth.send_transaction(constructor_txn)
    receipt = w3.eth.wait_for_transaction_receipt(txn_hash)

    shooter_addr = receipt['contractAddress']
    l.info(f'deployed shooter to {shooter_addr} with admin key {acct.address}')

    shooter = w3.eth.contract(
        address = shooter_addr,
        abi = SHOOTER_ARTIFACT['abi'],
    )

    #
    # fund the shooter with some wrapped ether
    #
    amt_to_send = w3.toWei(100, 'ether')
    weth: web3.contract.Contract = w3.eth.contract(
        address=WETH_ADDRESS,
        abi=get_abi('weth9/WETH9.json')['abi'],
    )
    wrap = weth.functions.deposit().buildTransaction({'value': amt_to_send, 'from': acct.address})
    wrap_hash = w3.eth.send_transaction(wrap)
    wrap_receipt = w3.eth.wait_for_transaction_receipt(wrap_hash)
    assert wrap_receipt['status'] == 1

    # transfer to shooter
    xfer = weth.functions.transfer(shooter_addr, amt_to_send).buildTransaction({'from': acct.address})
    xfer_hash = w3.eth.send_transaction(xfer)
    xfer_receipt = w3.eth.wait_for_transaction_receipt(xfer_hash)
    assert xfer_receipt['status'] == 1

    l.info(f'Transferred {amt_to_send / (10 ** 18):.2f} ETH to shooter')

    return p, w3, acct, shooter_addr


class CandidateArbitrage(typing.NamedTuple):
    id_: int
    exchanges: typing.List[str]
    directions: typing.List[typing.Tuple[str, str]]
    amount_in: int
    profit_before_fee: int
    block_number: int


def get_candidates_in_block(
        curr: psycopg2.extensions.cursor,
        block_number: int
    ) -> typing.List[
        CandidateArbitrage
    ]:
    curr.execute(
        '''
        SELECT id, exchanges, directions, amount_in, profit_no_fee
        FROM candidate_arbitrages
        WHERE block_number = %s
        ''',
        (block_number,)
    )

    ret = []
    for id_, bexchanges, bdirections, amount_in, profit in curr:
        exchanges = [web3.Web3.toChecksumAddress(x.tobytes()) for x in bexchanges]
        directions = [web3.Web3.toChecksumAddress(x.tobytes()) for x in bdirections]
        directions = list(zip(directions, directions[1:] + [directions[0]]))
        
        assert len(exchanges) == len(directions)

        ret.append(CandidateArbitrage(
            id_ = id_,
            exchanges = exchanges,
            directions = directions,
            amount_in = int(amount_in),
            profit_before_fee = int(profit),
            block_number=block_number
        ))
    
    return ret

class AutoAdaptShootSuccess(typing.NamedTuple):
    amount_in: int
    profit_no_fee: int
    gas: int

class DiagnosisBrokenToken(typing.NamedTuple):
    token_address: str
    reason: str

class DiagnosisFeeOnTransfer(typing.NamedTuple):
    from_address: str
    to_address: str
    token_address: str
    fee: decimal.Decimal
    round_down: bool

class DiagnosisNoArbitrageOnFeeApplied(typing.NamedTuple):
    pass

class DiagnosisBadExchange(typing.NamedTuple):
    exchange: str
    reason: str

def auto_adapt_attempt_shoot_candidate(
        w3_ganache: web3.Web3,
        curr: psycopg2.extensions.cursor,
        account: LocalAccount,
        shooter_address: str,
        fa: FoundArbitrage,
        fee_transfer_calculator: 'InferredTokenTransferFeeCalculator',
        must_recompute = False
    ) -> typing.Union[AutoAdaptShootSuccess, DiagnosisBrokenToken]:
    """
    Attempt the arbitrage by all means possible, reverting and re-shooting
    when a tax (fee) on transfer is detected.
    """
    # holds the list of inferred fees, so if we keep getting it wrong we can
    # start to make some reasonable conclusions
    inferred_fees: typing.Dict[typing.Any, typing.List[decimal.Decimal]] = collections.defaultdict(lambda: [])

    while True:
        result = w3_ganache.provider.make_request('evm_snapshot', [])
        snapshot_id = int(result['result'][2:], base=16)

        if must_recompute:
            l.debug(f'recomputing arbitrage (fresh pricers)')
            pc = PricingCircuit(
                [load_pricer_for(w3_ganache, curr, p.address) for p in fa.circuit], fa.directions
            )
            maybe_fa = detect_arbitrages_bisection(
                pc,
                'latest',
                try_all_directions=False,
                fee_transfer_calculator=fee_transfer_calculator
            )
            assert len(maybe_fa) <= 1
            if len(maybe_fa) == 0:
                l.debug(f'No arbitrage on re-shoot')
                return DiagnosisNoArbitrageOnFeeApplied()
            l.debug(f'Expect to make {fa.profit / (10 ** 18):.5f} ETH on re-shoot')
            fa = maybe_fa[0]

        try:
            receipt = attempt_shoot_candidate(
                w3_ganache,
                account,
                shooter_address,
                fa,
                fee_transfer_calculator
            )

            if receipt['status'] != 1:
                diagnosis = diagnose_failure(w3_ganache, account, shooter_address, fa, fee_transfer_calculator, receipt)
                if isinstance(diagnosis, DiagnosisBrokenToken):
                    return diagnosis
                elif isinstance(diagnosis, DiagnosisBadExchange):
                    return diagnosis
                elif isinstance(diagnosis, DiagnosisFeeOnTransfer):
                    diagnosis: DiagnosisFeeOnTransfer
                    k = (diagnosis.token_address, diagnosis.from_address, diagnosis.to_address)
                    all_inferences = inferred_fees[k]

                    if len(all_inferences) > 0:
                        median_fee = np.median(all_inferences + [diagnosis.fee])
                        min_fee = min(diagnosis.fee, np.min(all_inferences))
                        max_fee = max(diagnosis.fee, np.max(all_inferences))

                        l.debug(f'n_failures {len(all_inferences) + 1}')
                        l.debug(f'min_fee    {min_fee}')
                        l.debug(f'median_fee {median_fee}')
                        l.debug(f'max_fee    {max_fee}')

                        # adjust fee down gradually if we keep failing repeatedly
                        using_fee = min_fee - decimal.Decimal('0.00001')
                        l.debug(f'using fee {using_fee}')
                    else:
                        all_inferences.append(diagnosis.fee)
                        using_fee = diagnosis.fee

                    # plug-in fee and attempt to shoot again
                    fee_transfer_calculator.propose(
                        diagnosis.token_address,
                        diagnosis.from_address,
                        diagnosis.to_address,
                        using_fee,
                        diagnosis.round_down
                    )
                must_recompute = True
                l.debug(f'Diagnosed, re-shooting...')
            else:
                # shoot success
                weth: web3.contract.Contract = w3_ganache.eth.contract(
                    address=WETH_ADDRESS,
                    abi=get_abi('erc20.abi.json'),
                )
                shooter_balance = weth.functions.balanceOf(shooter_address).call()
                l.debug(f'shooter_balance = {shooter_balance / (10**18):.5f} ETH')
                profit = shooter_balance - 100 * (10 ** 18)
                if profit < 0:
                    l.critical('------------------------')
                    l.critical(f'Profit was negative! {profit / (10 ** 18):.5f} ETH')
                    l.critical(f'amount_in ........ {fa.amount_in}')
                    l.critical(f'expected profit .. {fa.profit} wei')
                    l.critical('------------------------')
                    diagnose_failure(w3_ganache, account, shooter_address, fa, fee_transfer_calculator, receipt)
                    raise Exception('what')
                return AutoAdaptShootSuccess(
                    fa.amount_in,
                    profit_no_fee = profit,
                    gas = receipt['gasUsed']
                )
        finally:
            result = w3_ganache.provider.make_request('evm_revert', [snapshot_id])
            assert result['result'] == True, 'snapshot revert should be success'


def attempt_shoot_candidate(
        w3_ganache: web3.Web3,
        account: LocalAccount,
        shooter_address: str,
        fa: FoundArbitrage,
        fee_transfer_calculator: FeeTransferCalculator,
    ) -> web3.types.TxReceipt:
    encoded, approvals_required = construct_arbitrage(
        fa,
        shooter_address,
        'latest',
        fee_transfer_calculator
    )

    for addr, token in approvals_required:
        token_extended = bytes.fromhex(token[2:]).rjust(32, b'\x00')
        addr_extended = bytes.fromhex(addr[2:]).rjust(32, b'\x00')
        payload = DO_APPROVE_SELECTOR + token_extended + addr_extended
        txn = {
            'from': account.address,
            'to': shooter_address,
            'data': payload,
            'chainId': 1,
            'gas': 1_000_000,
            'nonce': w3_ganache.eth.get_transaction_count(account.address),
            'gasPrice': 500 * (10 ** 9)
        }
        signed = w3_ganache.eth.account.sign_transaction(txn, account.key)

        txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
        receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)

        assert receipt['status'] == 1, f'failed to shoot approval for token {token} address {addr}'


    payload = serialize(encoded)

    # selector 0 for arbitrage
    payload = b'\x00' * 4 + payload

    txn = {
        'from': account.address,
        'to': shooter_address,
        'data': payload,
        'chainId': 1,
        'gas': 1_000_000,
        'nonce': w3_ganache.eth.get_transaction_count(account.address),
        'gasPrice': 500 * (10 ** 9)
    }
    signed = w3_ganache.eth.account.sign_transaction(txn, account.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)

    return receipt


def diagnose_failure(
        w3_ganache: web3.Web3,
        account: LocalAccount,
        shooter_address: str,
        fa: FoundArbitrage,
        fee_transfer_calculator: FeeTransferCalculator,
        receipt: web3.types.TxReceipt,
    ) -> DiagnosisBrokenToken:
    l.debug('diagnosing shoot failure...')
    l.debug(f'gas usage: {receipt["gasUsed"]:,}')

    # gather /expected/ token transfer calls out of exchanges
    expected_transfers: typing.Dict[typing.Tuple[str, str], int] = {}
    last_token = fa.pivot_token
    curr_amt = fa.amount_in
    for i, (p, (t_in, t_out)) in enumerate(zip(fa.circuit, fa.directions)):
        assert last_token == t_in
        curr_amt, _ = p.token_out_for_exact_in(t_in, t_out, curr_amt, receipt['blockNumber'] - 1, timestamp=None)
        last_token = t_out

        if i + 1 < len(fa.circuit):
            next_exchange_addr = fa.circuit[i + 1].address
        else:
            next_exchange_addr = None

        if isinstance(p, (BalancerV2WeightedPoolPricer, BalancerV2LiquidityBootstrappingPoolPricer)):
            sender = BALANCER_VAULT_ADDRESS
        else:
            sender = p.address

        amt_transferred = curr_amt

        if i + 1 < len(fa.circuit):
            if isinstance(fa.circuit[i + 1], (BalancerV2WeightedPoolPricer, BalancerV2LiquidityBootstrappingPoolPricer)):
                next_exchange_addr = BALANCER_VAULT_ADDRESS
            else:
                next_exchange_addr = fa.circuit[i + 1].address
        else:
            next_exchange_addr = shooter_address

        # If the next exchange is Balancer V1 or V2, or if the current exchange is Balancer V1, then we need to account for two fees:
        # (1) to self, (2) to next exchange
        if isinstance(p, BalancerPricer) or \
            (i + 1 < len(fa.circuit) and \
                isinstance(fa.circuit[i + 1], (BalancerPricer, BalancerV2WeightedPoolPricer, BalancerV2LiquidityBootstrappingPoolPricer))
            ):
            # transfer #1
            curr_amt = fee_transfer_calculator.out_from_transfer(t_out, sender, shooter_address, curr_amt)
            expected_transfers[(sender, t_out)] = (amt_transferred, curr_amt)

            # transfer #2 only happens if this token is not the end of the circuit (ie, there is no more exchanges)
            if i + 1 < len(fa.circuit):
                amt_transferred_2 = curr_amt
                curr_amt = fee_transfer_calculator.out_from_transfer(t_out, shooter_address, next_exchange_addr, curr_amt)
                expected_transfers[(shooter_address, t_out)] = (amt_transferred_2, curr_amt)
        else:
            curr_amt = fee_transfer_calculator.out_from_transfer(t_out, sender, next_exchange_addr, curr_amt)
            expected_transfers[(sender, t_out)] = (amt_transferred, curr_amt)


        assert curr_amt >= 0, 'negative token balance is not possible'
    l.debug(f'Expect {(curr_amt - fa.amount_in) / (10 ** 18):.5f} ETH profit')
    assert last_token == fa.pivot_token

    prior_balances: typing.Dict[typing.Tuple[str, str], int] = {}
    # gather token balances of exchanges to ensure transfers /should/ succeed
    for address, token in sorted(expected_transfers.keys()):
        this_erc20: web3.contract.Contract = w3_ganache.eth.contract(address=token, abi=get_abi('erc20.abi.json'))
        bal = this_erc20.functions.balanceOf(address).call(block_identifier=receipt['blockNumber'] - 1)
        prior_balances[(address, token)] = bal

    for (address, token), (amount, _) in expected_transfers.items():
        # ignore shooter, which is merely relaying and has no start balance
        if address == shooter_address:
            continue

        prior_bal = prior_balances[(address, token)]
        if amount > prior_bal:
            # We thought we could pull more token out than existed in the DeX at the start of execution.
            # This probably means that we have a 'rebasing' token or something, which means your balance
            # can go _down_ without ever interacting with the token.
            for p in fa.circuit:
                if p.address == address and isinstance(p, UniswapV2Pricer):
                    bal = p.get_value_locked(token_address=token, block_identifier = receipt['blockNumber'] - 1)
                    l.debug(f'Uniswap V2 exchange {p.address} reports token={token} balance of {bal} but actual balance is {prior_bal} !!')
                    return DiagnosisBrokenToken(token_address=token, reason='Unexpected balance decrease')

        assert amount <= prior_bal, f'Owner {address} needs to transfer out {amount} more than available ({prior_bal}) for {fa}'
        l.debug(f'Expect {address} to transfer {amount} of {token} out (prior balance {prior_bal})')

    all_tokens: typing.Set[str] = set(x for x, _ in fa.directions)
    all_uniswap_v2: typing.Set[str] = set(x.address for x in fa.circuit if isinstance(x, UniswapV2Pricer))

    txn = w3_ganache.eth.get_transaction(receipt['transactionHash'])

    with tempfile.NamedTemporaryFile() as f:
        fname = f.name
        got = w3_ganache.provider.make_request(
            'debug_traceTransactionToFile',
            [
                receipt['transactionHash'].hex(),
                {'disableStorage': True, 'file_name': fname},
            ]
        )
        assert got['result'] == 'OK', f'expected OK result but got {got}'

        stat = os.stat(fname)
        l.debug(f'File size was {stat.st_size // 1024 // 1024} MB')

        trace = {'result': json.load(f)}

    if 'result' not in trace:
        print(trace)

    with open('/mnt/goldphish/trace.txt', mode='w') as fout:
        for sl in trace['result']['structLogs']:
            fout.write(str(sl) + '\n')

    decoded = decode_trace_calls(trace['result']['structLogs'], txn, receipt)

    #
    # DFS through call tree to see if source of revert was a token
    #

    pretty_print_trace(decoded, txn, receipt)

    # Also note transfer recipients for phase 2 below, just to save on traversals
    xfer_recipients: typing.Set[typing.Tuple[str, str]] = set()

    q = [decoded]
    while len(q) > 0:
        item = q.pop()
        assert w3_ganache.isChecksumAddress(item['callee'])

        if item['callee'] in all_uniswap_v2 and \
            len(item['actions']) > 0 and \
            item['actions'][-1]['type'] == 'REVERT' and \
            item['actions'][-1]['message'].strip() == b'UniswapV2: OVERFLOW':

            # overflowed
            l.debug(f'Exchange {item["callee"]} overflowed, marking as bad...')
            return DiagnosisBadExchange(
                exchange = item['callee'],
                reason = 'Uniswap V2: OVERFLOW'
            )

        if item['callee'] in all_tokens and item['callee'] != WETH_ADDRESS:
            token: str = item['callee']
            try:
                f, decoded_func_input = erc20.decode_function_input(item['args'])
            except ValueError as e:
                if 'Could not find any function with matching selector' in str(e):
                    f = None
                    decoded_func_input = None
                else:
                    raise

            # we don't care if this isn't a transfer
            if f is not None and f.fn_name in ['transfer', 'transferFrom']:
                if f.fn_name == 'transfer':
                    from_addr = item['from']
                elif f.fn_name == 'transferFrom':
                    from_addr = w3_ganache.toChecksumAddress(decoded_func_input['_from'])

                # ensure that the amount transferred is as expected
                maybe_xfer = expected_transfers.get((from_addr, token), None)
                assert maybe_xfer is not None, f'Unexpected transfer from {from_addr} of {token}'
                (expected_amount, _) = maybe_xfer

                # sometimes can send more, than expected that's ok
                assert decoded_func_input['_value'] >= expected_amount, f'expected {from_addr} to send {expected_amount} of {token} but got {decoded_func_input["_value"]}'

                did_revert = (len(item['actions']) > 0) and (item['actions'][-1]['type'] == 'REVERT')

                # Still may have indicated non-success -- if this is a call to transfer or transferFrom, we need to check
                # the return boolean value
                if not did_revert and len(item['actions']) > 0 and item['actions'][-1]['type'] == 'RETURN':
                    returned = item['actions'][-1]['data']
                    if len(returned) > 0 and all(x == 0 for x in returned[:32]):
                        # returned False from transfer (i.e., reverted)
                        l.debug(f'Token {token} returned False from transfer() call')
                        did_revert = True

                if did_revert:
                    # see if this is a transfer call
                    l.debug(f'Broken token: {token}')
                    return DiagnosisBrokenToken(token_address=token, reason=f'token reverts on transfer')
                else:
                    # did not revert, but we still want to note the transfer recipients
                    recipient = w3_ganache.toChecksumAddress(decoded_func_input['_to'])
                    xfer_recipients.add((recipient, token))

        
        # add all outbound calls but do NOT descend into a token's internals
        if item['callee'] not in all_tokens:
            for sub_item in reversed(item['actions']):
                if 'CALL' in sub_item['type']:
                    q.append(sub_item)


    # Add record of prior balances for all transfer recipients
    for address, token in xfer_recipients:
        if (address, token) not in prior_balances:
            # must query
            this_erc20: web3.contract.Contract = w3_ganache.eth.contract(address=token, abi=get_abi('erc20.abi.json'))
            bal = this_erc20.functions.balanceOf(address).call(block_identifier=receipt['blockNumber'] - 1)
            prior_balances[(address, token)] = bal

    for (address, token), val in prior_balances.items():
        for p in fa.circuit:
            if isinstance(p, UniswapV2Pricer) and p.address == address:
                tvl = p.get_value_locked(token, receipt['blockNumber'] - 1)
                if tvl != val:
                    l.critical(f'uniswap v2 {address} reserves-balance mismatch - balanceOf={val} vs reserves={tvl}')
                if val < tvl:
                    # Uniswap thinks that it has more in reserves than in reality, we can't handle this type of token
                    l.debug(f'Marking token {token} as broken')
                    return DiagnosisBrokenToken(token_address=token, reason='Unexpected balance decrease')

    #
    # DFS through call tree to see if we have unexpected fee-on-transfer
    #

    did_receive_transfer_from: typing.Dict[typing.Tuple[str, str], str] = {}
    expected_amount_recieved: typing.Dict[typing.Tuple[str, str], int] = {}
    transfer_amount_attempted: typing.Dict[typing.Tuple[str, str], int] = {}

    q = [decoded]
    while len(q) > 0:
        item = q.pop()

        if 'callee' not in item:
            print(item)

        if item['callee'] in all_tokens and item['callee'] != WETH_ADDRESS:
            token = item['callee']
            try:
                f, decoded_func_input = erc20.decode_function_input(item['args'])
            except ValueError as e:
                if 'Could not find any function with matching selector' in str(e):
                    f = None
                    decoded_func_input = None
                else:
                    raise

            if f is not None:
                if f.fn_name in ['transfer', 'transferFrom']:
                    if f.fn_name == 'transfer':
                        from_addr = item['from']
                    elif f.fn_name == 'transferFrom':
                        from_addr = w3_ganache.toChecksumAddress(decoded_func_input['_from'])

                    recipient = decoded_func_input['_to']

                    expected_amount_recieved[(recipient, token)] = fee_transfer_calculator.out_from_transfer(
                        token,
                        from_addr,
                        recipient,
                        decoded_func_input['_value']
                    )
                    transfer_amount_attempted[(recipient, token)] = decoded_func_input['_value']

                    # This is a transfer -- just note that the transfer occurred
                    assert (recipient, token) not in did_receive_transfer_from
                    did_receive_transfer_from[(recipient, token)] = from_addr

                elif f.fn_name == 'balanceOf':
                    assert len(item['actions']) > 0
                    assert item['actions'][-1]['type'] == 'RETURN', f'expected last item to be return but got {item["actions"][-1]["type"]}'

                    #
                    # This may be what we want to check in this stage, see if this is /before/ or /after/ the transfer
                    owner = w3_ganache.toChecksumAddress(decoded_func_input['_owner'])
                    if (owner, token) in did_receive_transfer_from:

                        # we want to check the balance here!
                        prior_balance = prior_balances[(owner, token)]

                        new_balance = int.from_bytes(item["actions"][-1]['data'][:32], byteorder='big', signed=False)

                        # see if this matches what we expect
                        expected_balance = prior_balances[(owner, token)] + expected_amount_recieved[(owner, token)]
                        assert new_balance <= (expected_balance * 110 // 100), f'Balance went much higher than expected after transfer for token {token} actual_balance={new_balance} expected_balance={expected_balance}'

                        l.debug(f'evaluating owner {owner} balance of {token}')
                        l.debug(f'new_balance      ' + str(new_balance))
                        l.debug(f'expected_balance ' + str(expected_balance))

                        if new_balance < expected_balance:
                            l.debug(f'saw unexpected balance {new_balance} ({hex(new_balance)}) on owner {owner} for token {token}')
                            # there was probably a fee -- let's figure it out
                            real_amount_received = new_balance - prior_balance
                            dtransfer_amount = decimal.Decimal(transfer_amount_attempted[(owner, token)])
                            fee_rate_before_round = decimal.Decimal(real_amount_received) / dtransfer_amount
                            fee_rate = round(fee_rate_before_round, 5)

                            if fee_rate > 2:
                                raise Exception('fee_rate too big!!!')

                            # attempt to see if that fee_rate would have worked -- and if so, which way to round
                            amt_out = dtransfer_amount * fee_rate
                            amt_out_floor = math.floor(amt_out)
                            amt_out_ceil = math.ceil(amt_out)
                            if amt_out_floor == real_amount_received:
                                round_down = True
                            elif amt_out_ceil == real_amount_received:
                                round_down = False
                            else:
                                # neither of these worked, just guess we need to round down
                                l.warning(f'Token {token} did not have clean inferred fee_rate ({fee_rate_before_round}) -- assuming rounding down')
                                round_down = True

                            l.debug(f'Computed fee_rate={fee_rate} (round_down={round_down}) on token {token} from {did_receive_transfer_from[(owner, token)]} to {owner}')
                            return DiagnosisFeeOnTransfer(
                                from_address = did_receive_transfer_from[(owner, token)],
                                to_address = owner,
                                token_address = token,
                                fee = fee_rate,
                                round_down = round_down
                            )
            else:
                raise NotImplementedError(f'Not sure how to handle {f.fn_name} function call')
        else:
            # add all outbound calls, but DO NOT dive into a tokens' internal calls
            for sub_item in reversed(item['actions']):
                if 'CALL' in sub_item['type']:
                    q.append(sub_item)


    raise Exception('himom')


def load_broken_tokens_at(curr: psycopg2.extensions.cursor, block_number: int):
    """
    Load tokens that were banned before block_number
    """
    curr.execute(
        '''
        SELECT address
        FROM broken_tokens
        JOIN tokens ON tokens.id = broken_tokens.token_id
        WHERE broken_at_block_number <= %s
        ''',
        (block_number,)
    )
    return set(web3.Web3.toChecksumAddress(x.tobytes()) for (x,) in curr)


def load_pricer_for(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        exchange: str,
    ) -> typing.Optional[pricers.BaseExchangePricer]:
    bexchange = bytes.fromhex(exchange[2:])

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM uniswap_v2_exchanges uv2
        JOIN tokens t0 ON uv2.token0_id = t0.id
        JOIN tokens t1 ON uv2.token1_id = t1.id
        WHERE uv2.address = %s
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM sushiv2_swap_exchanges sv2
        JOIN tokens t0 ON sv2.token0_id = t0.id
        JOIN tokens t1 ON sv2.token1_id = t1.id
        WHERE sv2.address = %s
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM shibaswap_exchanges ss
        JOIN tokens t0 ON ss.token0_id = t0.id
        JOIN tokens t1 ON ss.token1_id = t1.id
        WHERE ss.address = %s
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address, originalfee
        FROM uniswap_v3_exchanges uv3
        JOIN tokens t0 ON uv3.token0_id = t0.id
        JOIN tokens t1 ON uv3.token1_id = t1.id
        WHERE uv3.address = %s            
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1
        token0, token1, fee = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV3Pricer(w3, exchange, token0, token1, fee)
        return p

    curr.execute(
        '''
        SELECT EXISTS(SELECT 1 FROM balancer_exchanges WHERE address = %s)
        ''',
        (bexchange,)
    )
    (is_balancerv1,) = curr.fetchone()
    if is_balancerv1:
        p = BalancerPricer(w3, exchange)
        return p

    curr.execute(
        '''
        SELECT pool_id, pool_type
        FROM balancer_v2_exchanges
        WHERE address = %s
        ''',
        (bexchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1
        pool_id, pool_type = curr.fetchone()
        pool_id = pool_id.tobytes()

        vault = w3.eth.contract(
            address = BALANCER_VAULT_ADDRESS,
            abi = get_abi('balancer_v2/Vault.json'),
        )

        if pool_type in ['WeightedPool', 'WeightedPool2Tokens']:
            return BalancerV2WeightedPoolPricer(w3, vault, exchange, pool_id)
        elif pool_type in ['LiquidityBootstrappingPool', 'NoProtocolFeeLiquidityBootstrappingPool']:
            return BalancerV2LiquidityBootstrappingPoolPricer(w3, vault, exchange, pool_id)

    l.error(f'Could not find exchange for address {exchange}')
    return None

class TokenFee(typing.NamedTuple):
    token: str
    from_address: str
    to_address: str
    fee: decimal.Decimal
    round_down: bool
    block_number: int
    updated_on: datetime.datetime

class InferredTokenTransferFeeCalculator(BuiltinFeeTransferCalculator):

    last_updated: datetime.datetime
    proposed_mover_to_token_fees: typing.Dict[typing.Tuple[str, str, str], TokenFee]
    mover_to_token_fees: typing.Dict[typing.Tuple[str, str, str], TokenFee]
    tokens_with_fee: typing.Set[str]
    proposed_tokens_with_fee: typing.Set[str]
    alias_map: typing.Dict[str, str]
    relayed_by_shooter: typing.Dict[typing.Tuple[str, str], str]

    updated: typing.List[TokenFee]


    def __init__(self) -> None:
        super().__init__()

        self.proposed_mover_to_token_fees = {}
        self.mover_to_token_fees = {}
        self.updated = []
        self.last_updated = datetime.datetime(year=1990, month=1, day=1, tzinfo=datetime.timezone.utc)
        self.tokens_with_fee = set()
        self.proposed_tokens_with_fee = set()
        self.alias_map = {}
        self.relayed_by_shooter = {}

    def clear_proposals(self):
        self.proposed_mover_to_token_fees.clear()
    
    def alias(self, address: str, as_: str):
        """
        Alias a given address as a `as_` -- computing all fees to/from it as if it goes to/from `as_`
        (useful only for balancer v2 really, since it uses a centralized vault)
        """
        self.alias_map[address] = as_

    def mark_requires_relay(self, from_: str, to: str, relayer: str):
        """
        Any transfers from `from_` to `to` will subtract a fee equivalent to being sent to `checkpointer`
        and then relayed back out
        """
        self.relayed_by_shooter[(from_, to)] = relayer

    def infer_relays_and_aliases(self, fa: FoundArbitrage, shooter_address: str):
        # first, add balacer vault aliases to records
        for p in fa.circuit:
            if isinstance(p, (BalancerV2LiquidityBootstrappingPoolPricer, BalancerV2WeightedPoolPricer)):
                self.alias(p.address, BALANCER_VAULT_ADDRESS)
        
        for p1, p2 in zip(fa.circuit, fa.circuit[1:]):
            # anything going _into_ Balancer (V1 or V2) must be relayed
            if isinstance(p2, (BalancerPricer, BalancerV2LiquidityBootstrappingPoolPricer, BalancerV2WeightedPoolPricer)):
                self.mark_requires_relay(p1.address, p2.address, shooter_address)

            # anything coming _out_ of Balancer V1 must be relayed
            if isinstance(p1, BalancerPricer):
                self.mark_requires_relay(p1.address, p2.address, shooter_address)

    def propose(self, token_address: str, from_address: str, to_address: str, fee: decimal.Decimal, round_down: bool):
        t = TokenFee(
            token = token_address,
            from_address = from_address,
            to_address = to_address,
            fee = fee,
            round_down = round_down,
            block_number = None,
            updated_on = datetime.datetime.utcnow(),
        )
        self.proposed_tokens_with_fee.add(token_address)
        self.proposed_mover_to_token_fees[(token_address, from_address, to_address)] = t

    def sync(self, curr: psycopg2.extensions.cursor, suggested_block_number: int):
        """
        Synchronize state of inferred transfer fee.
        """

        # fresh, full-pull
        curr.execute(
            '''
            SELECT t.address, fee, round_down, from_address, to_address, block_number_inferred, updated_on
            FROM inferred_token_fee_on_transfer inf
            JOIN tokens t ON t.id = inf.token_id
            WHERE updated_on > %s
            ''',
            (self.last_updated,)
        )
        for baddr, fee, round_down, bfrom, bto, block_number, updated_on in curr:
            t = TokenFee(
                token        = web3.Web3.toChecksumAddress(baddr.tobytes()),
                from_address = web3.Web3.toChecksumAddress(bfrom.tobytes()),
                to_address   = web3.Web3.toChecksumAddress(bto.tobytes()),
                block_number = block_number,
                updated_on   = updated_on,
                fee          = fee,
                round_down   = round_down,
            )

            k = (t.token, t.from_address, t.to_address)
            existing_val = self.mover_to_token_fees.get(k, None)
            if existing_val is not None:
                # conflict resolution
                should_update = abs(suggested_block_number - existing_val.block_number) > abs(suggested_block_number - t.block_number)
                if should_update:
                    self.mover_to_token_fees[k] = t

            self.tokens_with_fee.add(t.token)

        l.debug(f'Loaded {len(self.mover_to_token_fees):,} token fee records')

        # dump updated tokens
        for t in self.updated:
            curr.execute(
                '''
                INSERT INTO inferred_token_fee_on_transfer (token_id, fee, round_down, from_address, to_address, block_number_inferred, updated_on)
                SELECT t.id, %(fee)s, %(round_down)s %(from_addr)s, %(to_addr)s, %(block_number)s, now()::timestamp
                FROM tokens WHERE address = %(token_addr)s
                ''',
                {
                    'fee':          t.fee,
                    'round_down':   t.round_down,
                    'from_addr':    bytes.fromhex(t.from_address[2:]),
                    'to_addr':      bytes.fromhex(t.to_address[2:]),
                    'block_number': t.block_number,
                    'token_addr':   bytes.fromhex(t.token[2:]),
                }
            )
            assert curr.rowcount == 1
        
        l.debug(f'Inserted {len(self.updated)} inferred token fee records')
        self.updated.clear()

    def out_from_transfer(self, token: str, from_: str, to_: str, amount: int) -> int:
        maybe_relayer = self.relayed_by_shooter.get((from_, to_), None)

        from_ = self.alias_map.get(from_, from_)
        to_   = self.alias_map.get(to_, to_)

        if maybe_relayer:
            intermediate_out = self._out_from_transfer(token, from_, maybe_relayer, amount)
            return self._out_from_transfer(token, maybe_relayer, to_, intermediate_out)
        else:
            return self._out_from_transfer(token, from_, to_, amount)

    def _out_from_transfer(self, token: str, from_: str, to_: str, amount: int) -> int:
        k = (token, from_, to_)
        maybe_t = self.proposed_mover_to_token_fees.get(k, None)
        if maybe_t is not None:
            if maybe_t.round_down:
                return int(amount * maybe_t.fee)
            else:
                return math.ceil(amount * maybe_t.fee)

        maybe_t = self.mover_to_token_fees.get((token, from_, to_), None)
        if  maybe_t is not None:
            if maybe_t.round_down:
                return int(amount * maybe_t.fee)
            else:
                return math.ceil(amount * maybe_t.fee)

        return super().out_from_transfer(token, from_, to_, amount)

    def has_fee(self, token: str) -> bool:
        """
        Returns True if this token has fees, either committed or proposed
        """
        return token in self.tokens_with_fee or token in self.proposed_tokens_with_fee

