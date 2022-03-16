import psycopg2
import psycopg2.extensions
import enum
import gzip
import time
import logging
import typing
import web3
import web3.contract
import web3.types
import find_circuit
import shooter

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


def shoot(w3: web3.Web3, fa: find_circuit.FoundArbitrage, block_number: int, do_trace: TraceMode = TraceMode.NEVER, gaslimit: typing.Optional[int] = None) -> typing.Tuple[str, web3.types.TxReceipt, typing.Optional[typing.Any]]:
    account = funded_deployer()
    with GanacheContextManager(w3, block_number) as ganache:

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
            trace = ganache.provider.make_request('debug_traceTransaction', [receipt['transactionHash'].hex()])
            return shooter_address, receipt, (trace['result']['structLogs'], txn)

        return shooter_address, receipt, None
