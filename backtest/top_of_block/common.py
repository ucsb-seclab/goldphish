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
import find_circuit
import shooter
import pricers

from utils import get_abi
from ..utils import GanacheContextManager, funded_deployer, mine_block
from .constants import univ2_fname, univ3_fname


l = logging.getLogger(__name__)


class TraceMode(enum.Enum):
    NEVER   = 0
    ALWAYS  = 1
    ON_FAIL = 2


def connect_db() -> psycopg2.extensions.connection:
    db = psycopg2.connect(
        host = 'ethereum-measurement-pg',
        port = 5432,
        user = 'measure',
        password = 'password',
        database = 'eth_measure_db',
    )
    db.autocommit = False
    l.debug(f'connected to postgresql')
    return db


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


def shoot(w3: web3.Web3, fas: typing.List[find_circuit.FoundArbitrage], block_number: int, do_trace: TraceMode = TraceMode.NEVER, gaslimit: typing.Optional[int] = None) -> typing.Tuple[str, typing.List[ShootResult]]:
    assert len(fas) > 0
    tries_remaining = 4
    shot_arbitrages = set()
    ret = []
    while True:
        try:
            account = funded_deployer()
            with GanacheContextManager(w3, block_number) as ganache:

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

                for fa in fas:
                    if fa in shot_arbitrages:
                        continue

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
                            'maxFeePerGas': 1000 * (10 ** 9),
                        }
                        signed_txn = ganache.eth.account.sign_transaction(txn, account.key)
                        txn_hash = ganache.eth.send_raw_transaction(signed_txn.rawTransaction)

                        mine_block(ganache)

                        receipt = ganache.eth.get_transaction_receipt(txn_hash)

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

                        shot_arbitrages.add(fa)
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
                    assert result['result'] == True
                else:
                    # done
                    break
        except web3.exceptions.TransactionNotFound:
            tries_remaining -= 1
            if tries_remaining <= 0:
                raise
            ts_to_sleep = 60 + random.randint(0, 120)
            l.exception(f'transaction was not found; retrying. sleeping for {ts_to_sleep}sec, tries remaining = {tries_remaining}')
            time.sleep(ts_to_sleep)
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

