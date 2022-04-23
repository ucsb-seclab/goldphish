import os
import random
import psycopg2
import psycopg2.extensions
import enum
import gzip
import time
import logging
import typing
import web3
import web3.contract
import web3.exceptions
import web3.types
import backoff
import find_circuit
import shooter
import pricers

from utils import get_abi
from ..utils import CancellationToken, GanacheContextManager, funded_deployer, mine_block
from .constants import FNAME_EXCHANGES_WITH_BALANCES, THRESHOLDS, univ2_fname, univ3_fname


l = logging.getLogger(__name__)


class TraceMode(enum.Enum):
    NEVER   = 0
    ALWAYS  = 1
    ON_FAIL = 2


def load_naughty_tokens(curr: psycopg2.extensions.cursor) -> typing.Set[str]:
    curr.execute('SELECT address FROM naughty_tokens')
    return set(web3.Web3.toChecksumAddress(a.tobytes()) for (a,) in curr)


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


class ShootResult(typing.NamedTuple):
    arbitrage: find_circuit.FoundArbitrage
    maybe_receipt: typing.Optional[web3.types.TxReceipt]
    maybe_tx_params: typing.Optional[web3.types.TxParams]
    maybe_trace: typing.Optional[typing.Any]
    encodable: bool

    @property
    def tx_params(self) -> web3.types.TxParams:
        assert self.maybe_tx_params is not None
        return self.maybe_tx_params

    @property
    def receipt(self) -> web3.types.TxReceipt:
        assert self.maybe_receipt is not None
        return self.maybe_receipt

    @property
    def trace(self) -> typing.Any:
        assert self.maybe_trace is not None
        return self.maybe_trace


def ganache_deploy_shooter(w3: web3.Web3, destination: str):
    """
    Deploy the shooter on ganache by manually setting the code.
    Assumes the shooter is stateless.
    """
    assert web3.Web3.isChecksumAddress(destination)

    artifact = shooter.deploy.retrieve_shooter()
    assert 'deployedBytecode' in artifact

    bytecode = artifact['deployedBytecode']

    l.debug(f'deploying shooter to {destination}')

    result = w3.provider.make_request('evm_setAccountCode', [destination, bytecode])
    assert result['result'] == True, f'expected to see result=True in {result}'

    return


@backoff.on_exception(
        backoff.expo,
        web3.exceptions.TransactionNotFound,
        max_tries=4,
    )
def get_txn_with_backoff(w3: web3.Web3, txn_hash: bytes) -> web3.types.TxReceipt:
    return w3.eth.get_transaction_receipt(txn_hash)

def shoot(
        w3: web3.Web3,
        fas: typing.List[find_circuit.FoundArbitrage],
        block_number: int,
        do_trace: TraceMode = TraceMode.NEVER,
        gaslimit: typing.Optional[int] = None,
        cancellation_token: typing.Optional[CancellationToken] = None
    ) -> typing.Tuple[str, typing.List[ShootResult]]:
    assert len(fas) > 0
    ret = []
    account = funded_deployer()
    with GanacheContextManager(w3, block_number) as ganache:

        # deploy shooter
        shooter_address = '0x0044a204aaE0000091D100002cBC095180000F90' # we own the key to this address
        ganache_deploy_shooter(ganache, shooter_address)

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

        for fa in fas:
            if cancellation_token is not None and cancellation_token.cancel_requested():
                return

            for pricer in fa.circuit:
                pricer.set_web3(ganache)

            # checkpoint ganache
            result = ganache.provider.make_request('evm_snapshot', [])
            snapshot_id = int(result['result'][2:], base=16)

            try:
                shot = shooter.composer.construct_from_found_arbitrage(fa, 0, ganache.eth.get_block('latest')['number'] + 1)

                l.debug(f'About to shoot {shot.hex()}')

                if gaslimit is None:
                    gaslimit = 10_000_000

                txn: web3.types.TxParams = {
                    'chainId': ganache.eth.chain_id,
                    'from': account.address,
                    'to': shooter_address,
                    'value': 0,
                    'nonce': ganache.eth.get_transaction_count(account.address),
                    'data': shot,
                    'gas': gaslimit,
                    'maxPriorityFeePerGas': 2,
                    'maxFeePerGas': 5000 * (10 ** 9),
                }
                signed_txn = ganache.eth.account.sign_transaction(txn, account.key)
                txn_hash = ganache.eth.send_raw_transaction(signed_txn.rawTransaction)

                mine_block(ganache)
                receipt = get_txn_with_backoff(ganache, txn_hash)

                should_trace = (do_trace == TraceMode.ALWAYS) or (receipt['status'] != 1 and do_trace == TraceMode.ON_FAIL)

                if should_trace:
                    start = time.time()
                    trace = ganache.provider.make_request('debug_traceTransaction', [receipt['transactionHash'].hex()])
                    l.info(f'Spent {time.time() - start:f} seconds tracing')
                    if 'result' not in trace:
                        l.critical(f'Why does this not have result? {trace}')
                    maybe_trace = trace['result']['structLogs']
                else:
                    maybe_trace = None

                ret.append(ShootResult(
                    arbitrage = fa,
                    maybe_receipt = receipt,
                    maybe_tx_params = txn,
                    maybe_trace = maybe_trace,
                    encodable = True
                ))
            except shooter.composer.ConstructionException:
                l.exception('encoding params didnt make sense')
                ret.append(ShootResult(
                    arbitrage = fa,
                    maybe_receipt = None,
                    maybe_tx_params = None,
                    maybe_trace = None,
                    encodable = False
                ))
            except shooter.encoder.ExceedsEncodableParamsException:
                l.warning('exceeds encodable params')
                ret.append(ShootResult(
                    arbitrage = fa,
                    maybe_receipt = None,
                    maybe_tx_params = None,
                    maybe_trace = None,
                    encodable = False
                ))

            # reset snapshot
            result = ganache.provider.make_request('evm_revert', [snapshot_id])
            assert result['result'] == True, 'snapshot revert should be success'
    return shooter_address, ret


class WrappedFoundArbitrage:
    id: int
    fa: find_circuit.FoundArbitrage

    def __init__(self, fa: find_circuit.FoundArbitrage, db_id: int) -> None:
        assert isinstance(db_id, int)
        assert db_id >= 0
        self.id = db_id
        self.fa = fa

    @property
    def amount_in(self) -> int:
        return self.fa.amount_in
    
    @property
    def circuit(self) -> typing.List[pricers.base.BaseExchangePricer]:
        return self.fa.circuit
    
    @property
    def directions(self) -> typing.List[bool]:
        return self.fa.directions
    
    @property
    def pivot_token(self) -> str:
        return self.fa.pivot_token
    
    @property
    def profit(self) -> int:
        return self.fa.profit

    @property
    def tokens(self) -> typing.Set[str]:
        return self.fa.tokens


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
