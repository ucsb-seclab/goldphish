"""
Temporary file to test the shooter on some basic stuff while it is being constructed
"""

import json
import os
import pathlib
import subprocess
import time
import typing
import logging
import numpy as np

import web3
import web3.contract

from eth_account import Account
from eth_account.signers.local import LocalAccount
from backtest.utils import connect_db
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.uniswap_v3 import UniswapV3Pricer

from utils import BALANCER_VAULT_ADDRESS, WBTC_ADDRESS, WETH_ADDRESS, connect_web3, get_abi, setup_logging, decode_trace_calls, pretty_print_trace

from pricers import UniswapV2Pricer

l = logging.getLogger(__name__)

# spawn ganache connecting to web3 endpoint

_port = 0
def open_ganache(block_number: int) -> typing.Tuple[subprocess.Popen, web3.Web3, LocalAccount, str]:
    global _port
    acct: LocalAccount = Account.from_key(bytes.fromhex('f96003b86ed95cb86eae15653bf4b0bc88691506141a1a9ae23afd383415c268'))

    bin_loc = '/opt/ganache-fork/src/packages/ganache/dist/node/cli.js'
    cwd_loc = '/opt/ganache-fork/'

    my_pid = os.getpid()
    ganache_port = 34451 + (my_pid % 10_000) + _port
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
            '--chain.hardfork', 'arrowGlacier',
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
    artifact_path = pathlib.Path(__file__).parent / 'artifacts' / 'contracts' / 'shooter.sol' / 'Shooter.json'

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


class UniswapV2Swap(typing.NamedTuple):
    amount_in: int
    amount_out: int
    exchange: str
    to: str
    zero_for_one: bool

    def serialize(self) -> bytes:
        builder = []
        builder.append(
            int.to_bytes(self.amount_in, length=32, byteorder='big', signed=False)
        )
        builder.append(
            int.to_bytes(self.amount_out, length=32, byteorder='big', signed=False)
        )
        builder.append(
            bytes.fromhex(self.exchange[2:])
        )
        builder.append(
            bytes.fromhex(self.to[2:])
        )
        builder.append(
            b'\x01' if self.zero_for_one else b'\x00'
        )
        return b''.join(builder)

class UniswapV3Swap(typing.NamedTuple):
    amount_in: int
    exchange: str
    to: str
    zero_for_one: bool
    leading_exchanges: typing.List
    must_send_input: bool

    def serialize(self) -> bytes:
        assert self.amount_in > 0

        if len(self.leading_exchanges) > 0:
            extradata = serialize(self.leading_exchanges)
        else:
            extradata = b''
        extradata = (b'\x01' if self.must_send_input else b'\x00') + extradata

        builder = []
        builder.append(
            int.to_bytes(self.amount_in, length=32, byteorder='big', signed=True)
        )
        builder.append(
            bytes.fromhex(self.exchange[2:])
        )
        builder.append(
            bytes.fromhex(self.to[2:])
        )
        builder.append(
            b'\x01' if self.zero_for_one else b'\x00'
        )
        builder.append(
            int.to_bytes(len(extradata), length=2, byteorder='big', signed=False)
        )
        builder.append(
            extradata
        )
        return b''.join(builder)


class BalancerV1Swap(typing.NamedTuple):
    amount_in: int
    exchange: str
    token_in: str
    token_out: str
    to: str
    requires_approval: bool

    def serialize(self) -> bytes:
        assert self.amount_in > 0
        builder = [
            int.to_bytes(self.amount_in, length=32, byteorder='big', signed=False),
            bytes.fromhex(self.exchange[2:]),
            bytes.fromhex(self.token_in[2:]),
            bytes.fromhex(self.token_out[2:]),
            bytes.fromhex(self.to[2:]),
            b'\x01' if self.requires_approval else b'\x00',
        ]
        return b''.join(builder)


class BalancerV2Swap(typing.NamedTuple):
    pool_id: bytes
    amount_in: int
    amount_out: int
    token_in: str
    token_out: str
    to: str

    def serialize(self) -> bytes:
        assert self.amount_in > 0
        assert self.amount_out > 0
        assert len(self.pool_id) == 32
        builder = [
            self.pool_id,
            int.to_bytes(self.amount_in, length=32, byteorder='big', signed=False),
            int.to_bytes(self.amount_out, length=32, byteorder='big', signed=False),
            bytes.fromhex(self.token_in[2:]),
            bytes.fromhex(self.token_out[2:]),
            bytes.fromhex(self.to[2:]),
        ]

        return b''.join(builder)


def serialize(l: typing.List[typing.Union[UniswapV2Swap, UniswapV3Swap, BalancerV1Swap, BalancerV2Swap]]) -> bytes:
    # assert 2 <= len(l) <= 3
    builder = []
    builder.append(
        int.to_bytes(len(l), length=1, byteorder='big', signed=False),
    )
    serialized = [x.serialize() for x in l]
    offsets = [0] + list(np.cumsum([len(x) for x in serialized])[:-1])
    offsets = [o + (1 + 3 * 3) for o in offsets]
    assert len(offsets) == len(l)

    # pad with 0 offset to 3
    offsets.extend(0 for _ in range(3 - len(offsets)))

    for i, o in enumerate(offsets):
        if i < len(l):
            type_ = {
                UniswapV2Swap:  1,
                UniswapV3Swap:  2,
                BalancerV1Swap: 3,
                BalancerV2Swap: 4,
            }[type(l[i])]
            builder.append(
                int.to_bytes(type_, length=1, byteorder='big', signed=False)
            )
            builder.append(
                int.to_bytes(int(o), length=2, byteorder='big', signed=False)
            )
        else:
            builder.append(
                b'\x00\x00\x00'
            )

    builder.extend(serialized)
    return b''.join(builder)

def main():

    setup_logging()
    db = connect_db()
    curr = db.cursor()

    # w3 = connect_web3()
    # vault: web3.contract.Contract = w3.eth.contract(
    #     address = BALANCER_VAULT_ADDRESS,
    #     abi = get_abi('balancer_v2/Vault.json')
    # )

    # curr.execute(
    #     '''
    #     SELECT pool_id FROM balancer_v2_exchanges;
    #     '''
    # )
    # weth_bals = []
    # pool_to_tokens = {}
    # for (pool_id,) in curr:
    #     pool_id = pool_id.tobytes()
    #     tokens, _, _ = vault.functions.getPoolTokens(pool_id).call()
    #     tokens = set(tokens)
    #     if WETH_ADDRESS not in tokens:
    #         continue

    #     tokens, balances, _ = vault.functions.getPoolTokens(pool_id).call()
    #     pool_to_tokens[pool_id] = set(tokens)
    #     for tok, bal in zip(tokens, balances):
    #         if tok == WETH_ADDRESS:
    #             break
    #     else:
    #         raise Exception('what')
    #     weth_bals.append((bal, pool_id))

    # weth_bals = sorted(weth_bals, key=lambda x: x[0], reverse=True)
    # for balance, pool_id in weth_bals[:20]:
    #     print(f'{pool_id.hex()} {balance / (10 ** 18): 10.2f} ETH')
    #     for t in sorted(pool_to_tokens[pool_id]):
    #         print(f'    {t}')

    # exit()

    # t0 = bytes.fromhex(WBTC_ADDRESS[2:])
    # t1 = bytes.fromhex(WETH_ADDRESS[2:])
    # if t0 >= t1:
    #     t0, t1 = t1, t0

    # curr.execute(
    #     """
    #     SELECT uv2.address
    #     FROM uniswap_v2_exchanges uv2
    #     JOIN tokens t0 ON t0.id = uv2.token0_id
    #     JOIN tokens t1 ON t1.id = uv2.token1_id
    #     WHERE t0.address = %s AND t1.address = %s
    #     """,
    #     (t0, t1)
    # )
    # (baddr,) = curr.fetchone()
    # baddr = baddr.tobytes()
    # print(f'target uniswap address {web3.Web3.toChecksumAddress(baddr)}')

    # curr.execute(
    #     '''
    #     SELECT id, txn_hash, block_number
    #     FROM (
    #         SELECT sa.id, sa.txn_hash, sa.block_number, array_agg(sae.address) arr
    #         FROM sample_arbitrages sa
    #         JOIN sample_arbitrage_cycles sac ON sa.id = sac.sample_arbitrage_id
    #         JOIN sample_arbitrage_cycle_exchanges sace ON sac.id = sace.cycle_id
    #         JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
    #         JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
    #         GROUP BY sa.id, sa.txn_hash, sa.block_number
    #     ) x
    #     WHERE array_length(arr, 1) = 2 AND %s = ANY (arr) AND %s = ANY (arr) AND block_number >= 13000000 AND EXISTS(SELECT 1 FROM sample_arbitrage_backrun_detections sabd WHERE sabd.sample_arbitrage_id = x.id AND sabd.rerun_exactly = true)
    #     LIMIT 10
    #     ''',
    #     (
    #         bytes.fromhex(BALANCER_VAULT_ADDRESS[2:]),
    #         baddr,
    #     ),
    # )
    # for id_, txn_hash, block_number in curr:
    #     print(f'id_ = {id_} block_number={block_number} txn=https://etherscan.io/tx/0x{txn_hash.tobytes().hex()}')

    # exit()

    proc, w3_ganache, acct, shooter_addr = open_ganache(15183206)
    l.info(f'ganache opened, pid={proc.pid} with account {acct.address} funded')

    result = w3_ganache.provider.make_request('evm_snapshot', [])
    snapshot_id = int(result['result'][2:], base=16)


    #
    # attempt to do a swap on just Uniswap v2 WETH -> USDT pair
    #

    # find out what the going rate is for ETH -> USDT
    uv2 = UniswapV2Pricer(w3_ganache, '0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852', WETH_ADDRESS, '0xdAC17F958D2ee523a2206206994597C13D831ec7')
    usdt_out, _ = uv2.token_out_for_exact_in(WETH_ADDRESS, '0xdAC17F958D2ee523a2206206994597C13D831ec7', web3.Web3.toWei(1, 'ether'), block_identifier='latest')
    assert usdt_out > 0
    l.debug(f'expect to get {usdt_out} out from uniswap v2')

    payload = serialize([UniswapV2Swap(
        amount_in=web3.Web3.toWei(1, 'ether'),
        amount_out=usdt_out,
        exchange=uv2.address,
        to=shooter_addr,
        zero_for_one=True,
    )])

    txn = {'from': acct.address, 'to': shooter_addr, 'data': b'\x00'*4 + payload, 'chainId': 1, 'gas': 600_000, 'nonce': w3_ganache.eth.get_transaction_count(acct.address), 'gasPrice': 100 * (10 ** 9)}
    signed = w3_ganache.eth.account.sign_transaction(txn, acct.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)

    # check out balance in the destination token
    usdt_contract: web3.contract.Contract = w3_ganache.eth.contract(
        address='0xdAC17F958D2ee523a2206206994597C13D831ec7',
        abi = get_abi('erc20.abi.json'),
    )
    bal = usdt_contract.functions.balanceOf(shooter_addr).call()
    assert bal == usdt_out

    l.info('passed uniswap v2 single swap')

    result = w3_ganache.provider.make_request('evm_revert', [snapshot_id])
    assert result['result'] == True, 'snapshot revert should be success'
    result = w3_ganache.provider.make_request('evm_snapshot', [])
    snapshot_id = int(result['result'][2:], base=16)


    #
    # attempt to do a swap on Uniswap v3 WETH->USDT
    #

    uv3 = UniswapV3Pricer(
        w3_ganache,
        '0x11b815efB8f581194ae79006d24E0d814B7697F6',
        WETH_ADDRESS,
        '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        500
    )

    # find out what 1 WETH gets you
    usdt_out, _ = uv3.token_out_for_exact_in(WETH_ADDRESS, '0xdAC17F958D2ee523a2206206994597C13D831ec7', web3.Web3.toWei('1', 'ether'), block_identifier='latest')
    assert usdt_out > 0
    l.debug(f'Expect to get {usdt_out} USDT out from uniswap v3')

    payload = serialize([UniswapV3Swap(
        amount_in=web3.Web3.toWei(1, 'ether'),
        exchange=uv3.address,
        to=shooter_addr,
        zero_for_one=True,
        leading_exchanges=[],
        must_send_input=True
    )])

    txn = {'from': acct.address, 'to': shooter_addr, 'data': b'\x00'*4 + payload, 'chainId': 1, 'gas': 600_000, 'nonce': w3_ganache.eth.get_transaction_count(acct.address), 'gasPrice': 100 * (10 ** 9)}
    signed = w3_ganache.eth.account.sign_transaction(txn, acct.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)
    assert receipt['status'] == 1

    # check out balance in the destination token
    usdt_contract: web3.contract.Contract = w3_ganache.eth.contract(
        address='0xdAC17F958D2ee523a2206206994597C13D831ec7',
        abi = get_abi('erc20.abi.json'),
    )
    bal = usdt_contract.functions.balanceOf(shooter_addr).call()
    assert bal == usdt_out

    l.info('passed uniswap v3 single swap')

    result = w3_ganache.provider.make_request('evm_revert', [snapshot_id])
    assert result['result'] == True, 'snapshot revert should be success'
    result = w3_ganache.provider.make_request('evm_snapshot', [])
    snapshot_id = int(result['result'][2:], base=16)

    #
    # Attempt to do a swap on Balancer v1
    #

    balancer = BalancerPricer(w3_ganache, address='0x25af1F2c3772d6F19Aa6615571203757365D29C6')
    amt_out, _ = balancer.token_out_for_exact_in(
        token_in=WETH_ADDRESS,
        token_out=WBTC_ADDRESS,
        token_amount_in=web3.Web3.toWei(1, 'ether'),
        block_identifier='latest',
    )

    payload = serialize([BalancerV1Swap(
        amount_in=web3.Web3.toWei(1, 'ether'),
        exchange=balancer.address,
        token_in=WETH_ADDRESS,
        token_out=WBTC_ADDRESS,
        to=shooter_addr,
        requires_approval=True,
    )])

    txn = {'from': acct.address, 'to': shooter_addr, 'data': b'\x00'*4 + payload, 'chainId': 1, 'gas': 600_000, 'nonce': w3_ganache.eth.get_transaction_count(acct.address), 'gasPrice': 100 * (10 ** 9)}
    signed = w3_ganache.eth.account.sign_transaction(txn, acct.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)
    assert receipt['status'] == 1

    # tr = w3_ganache.provider.make_request('debug_traceTransaction', [txn_hash.hex()])

    # decoded = decode_trace_calls(tr['result']['structLogs'], txn, receipt)
    # pretty_print_trace(decoded, txn, receipt)


    #
    # Attempt to do a swap on Balancer v2
    #

    artifact_path = pathlib.Path(__file__).parent / 'artifacts' / 'contracts' / 'shooter.sol' / 'Shooter.json'

    assert os.path.isfile(artifact_path)

    with open(artifact_path) as fin:
        artifact = json.load(fin)

    shooter: web3.contract.Contract = w3_ganache.eth.contract(
        address = shooter_addr,
        abi = artifact['abi'],
    )

    shooter.functions.doApprove(WETH_ADDRESS, BALANCER_VAULT_ADDRESS).transact({'from': acct.address})

    vault = w3_ganache.eth.contract(
        address = BALANCER_VAULT_ADDRESS,
        abi = get_abi('balancer_v2/Vault.json'),
    )
    balancer_v2 = BalancerV2WeightedPoolPricer(w3_ganache, vault, '0xA6F548DF93de924d73be7D25dC02554c6bD66dB5', bytes.fromhex('a6f548df93de924d73be7d25dc02554c6bd66db500020000000000000000000e'))
    wbtc_out, _ = balancer_v2.token_out_for_exact_in(
        token_in=WETH_ADDRESS,
        token_out=WBTC_ADDRESS,
        token_amount_in=web3.Web3.toWei(1, 'ether'),
        block_identifier='latest'
    )
    print(f'expect to get {wbtc_out} wbtc')

    payload = serialize([
        BalancerV2Swap(
            balancer_v2.pool_id,
            web3.Web3.toWei(1, 'ether'),
            wbtc_out,
            WETH_ADDRESS,
            WBTC_ADDRESS,
            shooter_addr,
        )
    ])
    txn = {'from': acct.address, 'to': shooter_addr, 'data': b'\x00'*4 + payload, 'chainId': 1, 'gas': 600_000, 'nonce': w3_ganache.eth.get_transaction_count(acct.address), 'gasPrice': 100 * (10 ** 9)}
    signed = w3_ganache.eth.account.sign_transaction(txn, acct.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)
    assert receipt['status'] == 1

    # tr = w3_ganache.provider.make_request('debug_traceTransaction', [txn_hash.hex()])

    # decoded = decode_trace_calls(tr['result']['structLogs'], txn, receipt)
    # pretty_print_trace(decoded, txn, receipt)


    #
    # Attempt real uniswap v2 - sushiswap v2 arbitrage
    #

    proc.kill()
    proc.wait(10)

    # replicate https://etherscan.io/tx/0xcf583f20ad922936f4e7389a50a02d420385067ef2a612fbd177370869473fcc

    proc, w3_ganache, acct, shooter_addr = open_ganache(13271519)

    result = w3_ganache.provider.make_request('evm_snapshot', [])
    snapshot_id = int(result['result'][2:], base=16)

    shot = serialize([
        UniswapV2Swap(
            amount_in=76294691978820684801,
            amount_out=207437769544,
            exchange='0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852',
            to='0x06da0fd433C1A5d7a4faa01111c044910A184553',
            zero_for_one=True,
        ),
        UniswapV2Swap(
            amount_in=0,
            amount_out=76647249124042447025,
            exchange='0x06da0fd433C1A5d7a4faa01111c044910A184553',
            to=shooter_addr,
            zero_for_one=False
        ),
    ])

    txn = {
        'from': acct.address,
        'to': shooter_addr,
        'data': b'\x00'*4 + shot,
        'chainId': 1,
        'gas': 600_000,
        'nonce': w3_ganache.eth.get_transaction_count(acct.address),
        'gasPrice': 300 * (10 ** 9)
    }
    signed = w3_ganache.eth.account.sign_transaction(txn, acct.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)
    assert receipt['status'] == 1

    # tr = w3_ganache.provider.make_request('debug_traceTransaction', [txn_hash.hex()])

    # decoded = decode_trace_calls(tr['result']['structLogs'], txn, receipt)
    # pretty_print_trace(decoded, txn, receipt)

    result = w3_ganache.provider.make_request('evm_revert', [snapshot_id])
    assert result['result'] == True, 'snapshot revert should be success'
    result = w3_ganache.provider.make_request('evm_snapshot', [])
    snapshot_id = int(result['result'][2:], base=16)

    #
    # Reproduce a uniswap v2/v3 arbitrage
    #

    # replicate  https://etherscan.io/tx/0xc17553f6cd4e4bb94ed4a581b0f03d047d65262fb14ea666b7d1a86a9d4db93b

    proc, w3_ganache, acct, shooter_addr = open_ganache(13085220)

    shot = serialize([
        UniswapV3Swap(
            amount_in=106198401025,
            exchange='0x11b815efB8f581194ae79006d24E0d814B7697F6',
            to=shooter_addr,
            zero_for_one=False,
            must_send_input=False,
            leading_exchanges=[
                UniswapV2Swap(
                    amount_in=32100000000000000000,
                    amount_out=106198401025,
                    exchange='0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852',
                    to='0x11b815efB8f581194ae79006d24E0d814B7697F6',
                    zero_for_one=True,
                ),
            ]
        )
    ])

    txn = {
        'from': acct.address,
        'to': shooter_addr,
        'data': b'\x00'*4 + shot,
        'chainId': 1,
        'gas': 600_000,
        'nonce': w3_ganache.eth.get_transaction_count(acct.address),
        'gasPrice': 300 * (10 ** 9)
    }
    signed = w3_ganache.eth.account.sign_transaction(txn, acct.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)
    assert receipt['status'] == 1

    # tr = w3_ganache.provider.make_request('debug_traceTransaction', [txn_hash.hex()])

    # decoded = decode_trace_calls(tr['result']['structLogs'], txn, receipt)
    # pretty_print_trace(decoded, txn, receipt)

    proc.kill()
    proc.wait(10)

    #
    # Reproduce a Uniswap v2 / Balancer V1 swap
    #

    # replicate https://etherscan.io/tx/0x3d4ea175e44b15bdadcb8e125a9b8bd30a527a9702e2f1cdec4c74a61827d5ae

    proc, w3_ganache, acct, shooter_addr = open_ganache(13140970)

    artifact_path = pathlib.Path(__file__).parent / 'artifacts' / 'contracts' / 'shooter.sol' / 'Shooter.json'

    assert os.path.isfile(artifact_path)

    with open(artifact_path) as fin:
        artifact = json.load(fin)

    shooter: web3.contract.Contract = w3_ganache.eth.contract(
        address = shooter_addr,
        abi = artifact['abi'],
    )

    shooter.functions.doApprove(WBTC_ADDRESS, '0x25af1F2c3772d6F19Aa6615571203757365D29C6').transact({'from': acct.address})

    payload = serialize([
        UniswapV2Swap(
            amount_in=3344855733020686848,
            amount_out=25745647,
            exchange='0xBb2b8038a1640196FbE3e38816F3e67Cba72D940',
            to=shooter_addr,
            zero_for_one=False,
        ),
        BalancerV1Swap(
            amount_in=25745647,
            token_in=WBTC_ADDRESS,
            token_out=WETH_ADDRESS,
            requires_approval=False,
            to=shooter_addr,
            exchange='0x25af1F2c3772d6F19Aa6615571203757365D29C6',
        )
    ])

    txn = {
        'from': acct.address,
        'to': shooter_addr,
        'data': b'\x00'*4 + payload,
        'chainId': 1,
        'gas': 600_000,
        'nonce': w3_ganache.eth.get_transaction_count(acct.address),
        'gasPrice': 300 * (10 ** 9)
    }
    signed = w3_ganache.eth.account.sign_transaction(txn, acct.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)
    assert receipt['status'] == 1

    # tr = w3_ganache.provider.make_request('debug_traceTransaction', [txn_hash.hex()])

    # decoded = decode_trace_calls(tr['result']['structLogs'], txn, receipt)
    # pretty_print_trace(decoded, txn, receipt)

    proc.kill()
    proc.wait(10)

    #
    # Replicate Uniswap v2 / Balancer v2 arbitrage
    #

    # replicates https://etherscan.io/tx/0x561c691c63565c42fa17ab0a0f9ae8dbdbc23475cebce7670d1a04f1f780101a

    proc, w3_ganache, acct, shooter_addr = open_ganache(13140936)

    artifact_path = pathlib.Path(__file__).parent / 'artifacts' / 'contracts' / 'shooter.sol' / 'Shooter.json'

    assert os.path.isfile(artifact_path)

    with open(artifact_path) as fin:
        artifact = json.load(fin)

    shooter: web3.contract.Contract = w3_ganache.eth.contract(
        address = shooter_addr,
        abi = artifact['abi'],
    )

    shooter.functions.doApprove(WBTC_ADDRESS, BALANCER_VAULT_ADDRESS).transact({'from': acct.address})

    payload = serialize([
        UniswapV2Swap(
            28563953093248548864,
            219174863,
            '0xBb2b8038a1640196FbE3e38816F3e67Cba72D940',
            shooter_addr,
            zero_for_one=False,
        ),
        BalancerV2Swap(
            bytes.fromhex('a6f548df93de924d73be7d25dc02554c6bd66db500020000000000000000000e'),
            219174863,
            28621119951572604960,
            WBTC_ADDRESS,
            WETH_ADDRESS,
            shooter_addr
        )
    ])

    txn = {
        'from': acct.address,
        'to': shooter_addr,
        'data': b'\x00'*4 + payload,
        'chainId': 1,
        'gas': 600_000,
        'nonce': w3_ganache.eth.get_transaction_count(acct.address),
        'gasPrice': 300 * (10 ** 9)
    }
    signed = w3_ganache.eth.account.sign_transaction(txn, acct.key)

    txn_hash = w3_ganache.eth.send_raw_transaction(signed['rawTransaction'])
    receipt = w3_ganache.eth.wait_for_transaction_receipt(txn_hash)
    # assert receipt['status'] == 1

    tr = w3_ganache.provider.make_request('debug_traceTransaction', [txn_hash.hex()])

    decoded = decode_trace_calls(tr['result']['structLogs'], txn, receipt)
    pretty_print_trace(decoded, txn, receipt)


main()
