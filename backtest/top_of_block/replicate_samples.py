"""
Attempts to re-find arbitrages that actually occurred on the blockchain.
"""

import argparse
import subprocess
import decimal
import json
import os
import pathlib
import random
import time
import typing
import web3
import web3.contract
import web3.types
import web3._utils.filters
import psycopg2
import psycopg2.extensions
import logging
import pricers
import pricers.balancer_v2.common

from backtest.utils import connect_db
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.liquidity_bootstrapping_pool import BalancerV2LiquidityBootstrappingPoolPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.base import BaseExchangePricer, NotEnoughLiquidityException
from pricers.token_transfer import out_from_transfer
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer
from find_circuit import PricingCircuit
from find_circuit.find import FoundArbitrage, detect_arbitrages_bisection
from shooter.encoder import BalancerV1Swap, BalancerV2Swap, UniswapV2Swap, UniswapV3Swap, serialize
from utils import BALANCER_VAULT_ADDRESS, WETH_ADDRESS, decode_trace_calls, get_abi, pretty_print_trace
from eth_account import Account
from eth_account.signers.local import LocalAccount

l = logging.getLogger(__name__)

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'replicate'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (run before mass scan)')

    return parser_name, replicate


def replicate(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        return

    l.info('Starting replication')

    time.sleep(4)

    while True:
        candidate = get_candidate(curr)

        if candidate is None:
            # out of work
            break

        try_replicate(w3, curr, candidate)
        db.commit()

        # some jitter for the parallel workers
        if random.choice((True, False)):
            time.sleep(random.expovariate(1 / 0.005))

    l.info('done')


def setup_db(curr: psycopg2.extensions.cursor):
    l.info('setting up database')
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS sample_arbitrage_replications (
            sample_arbitrage_id   INTEGER NOT NULL REFERENCES sample_arbitrages (id) ON DELETE CASCADE,
            verification_started  BOOLEAN NOT NULL DEFAULT FALSE,
            verification_finished BOOLEAN,
            supported             BOOLEAN,
            replicated            BOOLEAN,
            our_profit            NUMERIC(78, 0),
            percent_change        DOUBLE PRECISION
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_sample_arbitrage_replication_id ON sample_arbitrage_replications (sample_arbitrage_id);
        CREATE INDEX IF NOT EXISTS idx_sample_arbitrage_replication_started ON sample_arbitrage_replications (verification_started);
        '''
    )
    curr.execute(
        '''
        INSERT INTO sample_arbitrage_replications (sample_arbitrage_id)
        SELECT sample_arbitrage_id
        FROM sample_arbitrage_cycles sac
        WHERE
            NOT EXISTS(SELECT 1 FROM sample_arbitrage_replications sar WHERE sar.sample_arbitrage_id = sac.sample_arbitrage_id) AND
            EXISTS(SELECT 1 FROM sample_arbitrages sa WHERE sa.id = sac.sample_arbitrage_id)
        '''
    )
    l.info(f'inserted {curr.rowcount:,} replication rows')

    curr.connection.commit()
    l.info('done setting up database')
    pass


def get_candidate(curr: psycopg2.extensions.cursor):
    """
    Get the next candidate ID from the queue
    """
    curr.execute(
        '''
        SELECT sample_arbitrage_id
        FROM sample_arbitrage_replications sar
        WHERE verification_started = false
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        '''
    )
    if curr.rowcount == 0:
        curr.connection.rollback()
        l.info('No more work')
        return None

    assert curr.rowcount == 1

    (id_,) = curr.fetchone()
    curr.execute('UPDATE sample_arbitrage_replications SET verification_started = true WHERE sample_arbitrage_id = %s', (id_,))
    assert curr.rowcount == 1
    # curr.connection.commit()

    l.debug(f'Processing id_={id_}')

    return id_

def try_replicate(w3: web3.Web3, curr: psycopg2.extensions.cursor, candidate: int):
    # if we could not replicate it exactly via backrun-detector, then ignore
    curr.execute(
        '''
        SELECT rerun_exactly FROM sample_arbitrage_backrun_detections
        WHERE sample_arbitrage_id = %s
        ''',
        (candidate,)
    )
    assert curr.rowcount == 1

    (rerun_exactly,) = curr.fetchone()

    if not rerun_exactly:
        l.debug(f'Did not rerun exactly, cannot support replicating {candidate}')
        curr.execute(
            '''
            UPDATE sample_arbitrage_replications
            SET verification_finished = true, supported = false
            WHERE sample_arbitrage_id = %s
            ''',
            (candidate,)
        )
        assert curr.rowcount == 1
        return

    curr.execute(
        '''
        SELECT sac.id, sac.profit_amount, sa.txn_hash, sa.block_number, t.address
        FROM sample_arbitrages sa
        JOIN sample_arbitrage_cycles sac ON sa.id = sac.sample_arbitrage_id
        JOIN tokens t ON t.id = sac.profit_token
        WHERE sa.id = %s
        ''',
        (candidate,)
    )

    assert curr.rowcount == 1, f'Expected rowcount of 1 for {candidate} but got {curr.rowcount}'

    (sac_id, original_profit_amount, txn_hash, block_number, pivot_token) = curr.fetchone()
    original_profit_amount = int(original_profit_amount)
    txn_hash = txn_hash.tobytes()
    pivot_token = web3.Web3.toChecksumAddress(pivot_token.tobytes())

    l.debug('getting directions...')

    # get exchange directions
    curr.execute(
        '''
        SELECT sace.id, tin.address, tout.address
        FROM sample_arbitrage_cycle_exchanges sace
        JOIN tokens tin  ON token_in  = tin.id
        JOIN tokens tout ON token_out = tout.id
        WHERE cycle_id = %s
        order by sace.id asc
        ''',
        (sac_id,)
    )
    assert curr.rowcount > 1, f'Cannot form a cycle with only {curr.rowcount} on {candidate}'

    directions = []
    sace_ids = []
    for sace_id, tin_baddr, tout_baddr in curr:
        tin_address = web3.Web3.toChecksumAddress(tin_baddr.tobytes())
        tout_address = web3.Web3.toChecksumAddress(tout_baddr.tobytes())
        d = (tin_address, tout_address)

        assert d not in directions, f'cannot have duplicate directions on {candidate} (saw duplicate {d})'

        directions.append((tin_address, tout_address))
        sace_ids.append(sace_id)

    # ensure directions point end-to-end
    for (_, t2), (t3, _) in zip(directions, directions[1:] + [directions[0]]):
        assert t2 == t3, f'directions did not align for {candidate}'

    # ensure pivot is somewhere in directions
    assert any((pivot_token in x) for x in directions), f'could not find pivot token in directions for {candidate}'

    l.debug('getting exchanges...')

    # get the exchanges
    curr.execute(
        '''
        SELECT sace.id, sae.address
        FROM sample_arbitrage_cycles sac
        JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
        JOIN sample_arbitrage_cycle_exchange_items sacei ON sace.id = sacei.cycle_exchange_id
        JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
        WHERE sac.sample_arbitrage_id = %s
        ORDER BY sace.id ASC
        ''',
        (candidate,)
    )
    assert curr.rowcount >= len(sace_ids), f'wrong number of rows ({curr.rowcount}) while getting exchange addresses on {candidate}'

    if curr.rowcount > len(sace_ids):
        # one of the exchange items is split across several exchanges
        # we don't support this, so give up
        l.debug(f'Exchange flow was split, cannot support {candidate}, skipping')
        curr.execute(
            '''
            UPDATE sample_arbitrage_replications
            SET verification_finished = true, supported = false
            WHERE sample_arbitrage_id = %s
            ''',
            (candidate,)
        )
        assert curr.rowcount == 1
        return

    circuit: typing.List[BaseExchangePricer] = []
    # attempt to load each of the exchanges
    for expected_id, (actual_id, baddr) in zip(sace_ids, list(curr)):
        baddr = baddr.tobytes()
        assert expected_id == actual_id, f'expected {expected_id} == {actual_id} on {candidate}'

        maybe_pricer = load_pricer_for(w3, curr, baddr, txn_hash)

        if maybe_pricer is not None:
            circuit.append(maybe_pricer)
        else:
            # pricer not found
            l.debug(f'Could not find pricer for {web3.Web3.toChecksumAddress(baddr)} on {candidate}')
            curr.execute(
                '''
                UPDATE sample_arbitrage_replications
                SET verification_finished = true, supported = false
                WHERE sample_arbitrage_id = %s
                ''',
                (candidate,)
            )
            assert curr.rowcount == 1
            return

    assert len(circuit) == len(directions) # this should not be possible to voilate, but just to be sure

    # rotate so pivot is in place
    pc = PricingCircuit(circuit, directions)
    while pc.pivot_token != pivot_token:
        pc.rotate()

    maybe_arbs = detect_arbitrages_bisection(pc, block_number - 1, try_all_directions=False)

    assert len(maybe_arbs) <= 1, f'Got too many arbitrages out from detection on {candidate}'

    if len(maybe_arbs) == 0:
        l.error(f'could not replicate id={candidate} https://etherscan.io/tx/0x{txn_hash.hex()}')
        curr.execute(
            '''
            UPDATE sample_arbitrage_replications
            SET verification_finished = true, supported = true, replicated = false
            WHERE sample_arbitrage_id = %s
            ''',
            (candidate,)
        )
        assert curr.rowcount == 1
        return

    # get our profit and compare it to theirs
    arb = maybe_arbs[0]

    profit_percent_changed = None
    if original_profit_amount > 0:
        original_profit_amount_dec = decimal.Decimal(original_profit_amount)
        diff = decimal.Decimal(arb.profit) - original_profit_amount_dec
        profit_percent_changed = diff / original_profit_amount_dec * 100
        l.debug(f'Profit changed by {profit_percent_changed:.2f}%')

    can_attempt_ganache_replicate = arb.pivot_token == WETH_ADDRESS and len(arb.circuit) <= 3

    if can_attempt_ganache_replicate:
        has_whacky_token = False
        # find out if it uses any bad tokens we don't know about
        all_tokens = set(x for x, _ in directions).union(x for _, x in directions)
        assert len(all_tokens) == len(pc.circuit)
        for t in all_tokens:
            curr.execute(
                '''
                SELECT EXISTS(
                    SELECT 1
                    FROM whacky_tokens wt
                    JOIN tokens t ON wt.token_id = t.id
                    WHERE t.address = %s
                );
                ''',
                (bytes.fromhex(t[2:]),)
            )
            (is_whacky,) = curr.fetchone()
            if is_whacky:
                has_whacky_token = True
                break

        if not has_whacky_token:
            ganache_replicate(arb, block_number)

    curr.execute(
        '''
        UPDATE sample_arbitrage_replications
        SET
            verification_finished = true,
            supported = true,
            replicated = true,
            our_profit = %s,
            percent_change = %s
        WHERE sample_arbitrage_id = %s
        ''',
        (arb.profit, profit_percent_changed, candidate)
    )
    assert curr.rowcount == 1
    return


def load_pricer_for(w3: web3.Web3, curr: psycopg2.extensions.cursor, exchange: bytes, txn_hash: bytes) -> typing.Optional[pricers.BaseExchangePricer]:
    """
    Attempt to get a Pricer for this exchange.

    NOTE: if the 'exchange' is Balancer Vault, then we need to find the pool_id from the txn_hash (TODO fix this)
    """
    assert isinstance(txn_hash, bytes)

    exchange_address = w3.toChecksumAddress(exchange)

    if exchange_address == BALANCER_VAULT_ADDRESS:
        receipt = w3.eth.get_transaction_receipt(txn_hash)

        # find pool_id
        found = False
        for log in receipt['logs']:
            if log['address'] == BALANCER_VAULT_ADDRESS and log['topics'][0] == pricers.balancer_v2.common.SWAP_TOPIC:
                assert found != True, f'Could not identify single Balancer pool in txn_hash {txn_hash.hex()}'
                found = True
                pool_id = log['topics'][1]

        assert found == True, f'Could not find Balancer pool in txn https://etherscan.io/tx/0x{txn_hash.hex()}'

        curr.execute(
            '''
            SELECT address, pool_type
            FROM balancer_v2_exchanges
            WHERE pool_id = %s
            ''',
            (pool_id,)
        )
        assert curr.rowcount == 1, f'Did not find Balancer pool with id {pool_id.hex()}'
        (pool_address, pool_type) = curr.fetchone()
        pool_address = web3.Web3.toChecksumAddress(pool_address.tobytes())

        vault = w3.eth.contract(
            address = BALANCER_VAULT_ADDRESS,
            abi = get_abi('balancer_v2/Vault.json'),
        )

        if pool_type in ['WeightedPool', 'WeightedPool2Tokens']:
            return BalancerV2WeightedPoolPricer(w3, vault, pool_address, pool_id)
        elif pool_type in ['LiquidityBootstrappingPool', 'NoProtocolFeeLiquidityBootstrappingPool']:
            return BalancerV2LiquidityBootstrappingPoolPricer(w3, vault, pool_address, pool_id)
        
        # we don't know about this pool_type
        l.debug(f'Cannot handle pool_type = {pool_type} for balancer pool_id={pool_id.hex()}')
        return None

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM uniswap_v2_exchanges uv2
        JOIN tokens t0 ON uv2.token0_id = t0.id
        JOIN tokens t1 ON uv2.token1_id = t1.id
        WHERE uv2.address = %s
        ''',
        (exchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange_address, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM sushiv2_swap_exchanges sv2
        JOIN tokens t0 ON sv2.token0_id = t0.id
        JOIN tokens t1 ON sv2.token1_id = t1.id
        WHERE sv2.address = %s
        ''',
        (exchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange_address, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM shibaswap_exchanges ss
        JOIN tokens t0 ON ss.token0_id = t0.id
        JOIN tokens t1 ON ss.token1_id = t1.id
        WHERE ss.address = %s
        ''',
        (exchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange_address, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address, originalfee
        FROM uniswap_v3_exchanges uv3
        JOIN tokens t0 ON uv3.token0_id = t0.id
        JOIN tokens t1 ON uv3.token1_id = t1.id
        WHERE uv3.address = %s            
        ''',
        (exchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1
        token0, token1, fee = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV3Pricer(w3, exchange_address, token0, token1, fee)
        return p

    curr.execute(
        '''
        SELECT EXISTS(SELECT 1 FROM balancer_exchanges WHERE address = %s)
        ''',
        (exchange,)
    )
    (is_balancerv1,) = curr.fetchone()
    if is_balancerv1:
        p = BalancerPricer(w3, exchange_address)
        return p

    return None


class ReshootResult(typing.NamedTuple):
    success: bool
    profit: int
    gasUsed: int

def ganache_replicate(fa: FoundArbitrage, block_number) -> ReshootResult:
    proc, w3_ganache, acct, shooter_addr = open_ganache(block_number - 1)

    intermediate_arbitrage, approvals_required = construct_arbitrage(fa, shooter_addr, block_number)
    for step in intermediate_arbitrage:
        print(step)
    payload = serialize(intermediate_arbitrage)

    artifact_path = pathlib.Path(__file__).parent.parent.parent / 'artifacts' / 'contracts' / 'shooter.sol' / 'Shooter.json'

    assert os.path.isfile(artifact_path)

    with open(artifact_path) as fin:
        artifact = json.load(fin)

    shooter = w3_ganache.eth.contract(
        address = shooter_addr,
        abi = artifact['abi'],
    )

    for addr, token in approvals_required:
        shooter.functions.doApprove(token, addr).transact({'from': acct.address})
        l.debug(f'approved {addr} to use token {token}')

    txn = {
        'from': acct.address,
        'to': shooter_addr,
        'data': b'\x00'*4 + payload,
        'chainId': 1,
        'gas': 1_000_000,
        'nonce': w3_ganache.eth.get_transaction_count(acct.address),
        'gasPrice': 500 * (10 ** 9)
    }
    signed = w3_ganache.eth.account.sign_transaction(txn, acct.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)

    weth: web3.contract.Contract = w3_ganache.eth.contract(
        address = WETH_ADDRESS,
        abi = get_abi('erc20.abi.json'),
    )

    new_balance = weth.functions.balanceOf(shooter_addr).call()
    real_profit = new_balance - web3.Web3.toWei(100, 'ether')

    l.debug(f'actual profit: {real_profit} expected profit {fa.profit}')

    tr = w3_ganache.provider.make_request('debug_traceTransaction', [txn_hash.hex()])

    decoded = decode_trace_calls(tr['result']['structLogs'], txn, receipt)
    pretty_print_trace(decoded, txn, receipt)

    proc.kill()
    proc.wait()

    if real_profit != fa.profit:
        raise Exception('what')

    if receipt['status'] == 1:
        return ReshootResult(
            success = True,
            profit  = real_profit,
            gasUsed = receipt['gasUsed'],
        )
    else:
        return ReshootResult(
            success = False,
            profit  = None,
            gasUsed = None
        )


def construct_arbitrage(fa: FoundArbitrage, shooter_addr: str, block_number) -> typing.Tuple[typing.List, typing.List[typing.Tuple[str, str]]]:
    assert fa.pivot_token == WETH_ADDRESS
    ret = []
    approvals_required: typing.List[typing.Tuple[str, str]] = []

    assert len(fa.circuit) >= 2
    assert len(fa.directions) >= 2

    amount_in = fa.amount_in

    for p, (token_in, token_out) in zip(fa.circuit, fa.directions):
        amount_out, _ = p.token_out_for_exact_in(
            token_in,
            token_out,
            amount_in,
            block_number - 1,
        )

        if isinstance(p, UniswapV2Pricer):
            ret.append(UniswapV2Swap(
                amount_in=None,
                amount_out=amount_out,
                exchange=p.address,
                to=None,
                zero_for_one=(bytes.fromhex(token_in[2:]) < bytes.fromhex(token_out[2:]))
            ))
        elif isinstance(p, UniswapV3Pricer):
            ret.append(UniswapV3Swap(
                amount_in=amount_in,
                exchange=p.address,
                to=[],
                zero_for_one=(bytes.fromhex(token_in[2:]) < bytes.fromhex(token_out[2:])),
                leading_exchanges=None,
                must_send_input=False
            ))
        elif isinstance(p, BalancerPricer):
            ret.append(BalancerV1Swap(
                amount_in=amount_in,
                exchange=p.address,
                token_in=token_in,
                token_out=token_out,
                to=None,
                requires_approval=False,
            ))
            approvals_required.append((p.address, token_in))
        elif isinstance(p, (BalancerV2WeightedPoolPricer, BalancerV2LiquidityBootstrappingPoolPricer)):
            ret.append(BalancerV2Swap(
                pool_id=p.pool_id,
                amount_in=amount_in,
                amount_out=amount_out,
                token_in=token_in,
                token_out=token_out,
                to=None
            ))
            approvals_required.append((BALANCER_VAULT_ADDRESS, token_in))

        amount_in = out_from_transfer(token_out, amount_out)

    if isinstance(ret[0], UniswapV2Swap):
        ret[0] = ret[0]._replace(amount_in = fa.amount_in)

    if isinstance(ret[0], UniswapV3Swap):
        ret[0]  = ret[0]._replace(must_send_input = True)

    ret[-1] = ret[-1]._replace(to = shooter_addr)

    for i in range(len(ret) - 1):
        p1 = ret[i]
        p2 = ret[i + 1]

        if isinstance(p2, (BalancerV1Swap, BalancerV2Swap)):
            ret[i] = p1._replace(to = shooter_addr)
        else:
            ret[i] = p1._replace(to = p2.exchange)


    gathered = _recurse_gather_uniswap_v3(ret, [])
    assert isinstance(gathered, list)
    assert len(gathered) > 0
    return gathered, approvals_required

def _recurse_gather_uniswap_v3(l, acc):
    if len(l) == 0:
        return acc

    if isinstance(l[0], UniswapV3Swap):
        uv3 = l[0]._replace(leading_exchanges=acc)
        return _recurse_gather_uniswap_v3(l[1:], [uv3])

    return _recurse_gather_uniswap_v3(l[1:], acc + [l[0]])

_port = 0
def open_ganache(block_number: int) -> typing.Tuple[subprocess.Popen, web3.Web3, LocalAccount, str]:
    global _port
    acct: LocalAccount = Account.from_key(bytes.fromhex('f96003b86ed95cb86eae15653bf4b0bc88691506141a1a9ae23afd383415c268'))

    bin_loc = '/opt/ganache-fork/src/packages/ganache/dist/node/cli.js'
    cwd_loc = '/opt/ganache-fork/'

    my_pid = os.getpid()
    ganache_port = 34451 + (my_pid % 500) + _port
    _port += 1

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
        time.sleep(0.1)

    assert w3.eth.get_balance(acct.address) == web3.Web3.toWei(1000, 'ether')

    #
    # deploy the shooter
    #
    artifact_path = pathlib.Path(__file__).parent.parent.parent / 'artifacts' / 'contracts' / 'shooter.sol' / 'Shooter.json'

    assert os.path.isfile(artifact_path)

    with open(artifact_path) as fin:
        artifact = json.load(fin)

    shooter = w3.eth.contract(
        bytecode = artifact['bytecode'],
        abi = artifact['abi'],
    )

    constructor_txn = shooter.constructor().buildTransaction({'from': acct.address})
    txn_hash = w3.eth.send_transaction(constructor_txn)
    receipt = w3.eth.wait_for_transaction_receipt(txn_hash)

    shooter_addr = receipt['contractAddress']
    l.info(f'deployed shooter to {shooter_addr} with admin key {acct.address}')

    shooter = w3.eth.contract(
        address = shooter_addr,
        abi = artifact['abi'],
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
