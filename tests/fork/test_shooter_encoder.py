import collections
import typing
import pytest
import web3
import web3.types
import web3.contract
import random
import web3.constants
from eth_account.signers.local import LocalAccount
from eth_utils import event_abi_to_log_topic

import shooter.encoder

from utils import get_abi

uv3: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('uniswap_v3/IUniswapV3Pool.json')['abi'],
)

erc20: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('erc20.abi.json'),
)

ERC20_TRANSFER_TOPIC = event_abi_to_log_topic(erc20.events.Transfer().abi)

def replay_to_txn(w3: web3.Web3, ganache: web3.Web3, txn: str):
    """
    Replay transactions from the top of the block `txn` appears in.
    """
    print('replaying...')
    receipt = w3.eth.get_transaction_receipt(txn)
    hashes = []
    for i in range(receipt['transactionIndex']):
        to_replay = w3.eth.get_raw_transaction_by_block(receipt['blockHash'], i)
        hashes.append(ganache.eth.send_raw_transaction(to_replay))
    for h in hashes:
        r = ganache.eth.wait_for_transaction_receipt(h)
    print('done replay')


def read_mem(start, read_len, mem):
    b = b''
    for idx in range(start, start + read_len):
        byte = idx % 32
        word = idx // 32
        b += bytes.fromhex(mem[word][byte*2:byte*2+2])
    return b


def random_addr(r: random.Random) -> str:
    b = b''
    for _ in range(20):
        b += bytes([r.randint(0, 255)])
    return web3.Web3.toChecksumAddress(b)


def parse_logs_for_net_profit(logs: typing.List[web3.types.LogReceipt]) -> typing.Dict[str, typing.Dict[str, int]]:
    ret = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for log in logs:
        if len(log['topics']) > 0 and log['topics'][0] == ERC20_TRANSFER_TOPIC:
            xfer = erc20.events.Transfer().processLog(log)
            ret[log['address']][xfer['args']['from']] -= xfer['args']['value']
            ret[log['address']][xfer['args']['to']] += xfer['args']['value']
    return {k: dict(v) for k, v in ret.items()}


@pytest.mark.ganache_block_num(14_180_071 - 1)
@pytest.mark.ganache_mine_mode('TIMED')
def test_shoot_two_exchanges_basic(
        mainnet_chain: web3.Web3,
        ganache_chain: web3.Web3,
        funded_deployer: LocalAccount,
        deployed_shooter: str
    ):
    """
    Based off transaction:
    0x4bbd3ead05d44408bff9ce948501d7e6179f8d7da5a85ad9c9b05274653d2343
    Block number: 14,180,071
    """
    w3 = ganache_chain
    print('latest block.......', w3.eth.get_block('latest')['number'])
    replay_to_txn(mainnet_chain, ganache_chain, '0x4bbd3ead05d44408bff9ce948501d7e6179f8d7da5a85ad9c9b05274653d2343')

    # wrap some ETH
    weth: web3.contract.Contract = w3.eth.contract(
        address='0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        abi=get_abi('weth9/WETH9.json')['abi'],
    )
    wrap = weth.functions.deposit().buildTransaction({'value': 100, 'from': funded_deployer.address})
    wrap_hash = w3.eth.send_transaction(wrap)
    wrap_receipt = w3.eth.wait_for_transaction_receipt(wrap_hash)
    assert wrap_receipt['status'] == 1

    # transfer to shooter
    xfer = weth.functions.transfer(deployed_shooter, 100).buildTransaction({'from': funded_deployer.address})
    xfer_hash = w3.eth.send_transaction(xfer)
    xfer_receipt = w3.eth.wait_for_transaction_receipt(xfer_hash)
    assert xfer_receipt['status'] == 1

    coinbase_xfer_amt = 0 # w3.toWei('0.01', 'ether')
    target_block = w3.eth.get_block('latest')['number'] + 1
    enc = shooter.encoder.encode_basic(
        target_block=target_block,
        amount_in=0xcceead6811c5d300e0,
        coinbase_xfer=coinbase_xfer_amt,
        exchanges=[
            shooter.encoder.UniswapV3Record(
                address='0xFd76bE67FFF3BAC84E3D5444167bbC018f5968b6',
                amount_out=0, # unused
                zero_for_one=False,
                recipient=shooter.encoder.FundsRecipient.SHOOTER
            ),
            shooter.encoder.UniswapV2Record(
                address='0x18Cd890F4e23422DC4aa8C2D6E0Bd3F3bD8873d8',
                zero_for_one=True,
                amount_in_explicit=3154113260025438813,
                amount_out=3780334308014288994528,
                recipient=shooter.encoder.FundsRecipient.MSG_SENDER,
            )
        ]
    )

    print('encoded call....')
    print(enc.hex())
    print()

    assert isinstance(enc, bytes)

    print('balance shooter', w3.fromWei((w3.eth.get_balance(deployed_shooter)), 'ether'))
    txn: web3.types.TxParams = {
        'chainId': w3.eth.chain_id,
        'from': funded_deployer.address,
        'to': deployed_shooter,
        'value': coinbase_xfer_amt,
        'nonce': w3.eth.get_transaction_count(funded_deployer.address),
        'data': enc,
        'gas': 400_000,
        'maxPriorityFeePerGas': 2,
        'maxFeePerGas': 300 * (10 ** 9),
    }
    signed_txn = w3.eth.account.sign_transaction(txn, funded_deployer.key)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(receipt)
    print('target block', target_block)

    trace = w3.provider.make_request('debug_traceTransaction', [tx_hash.hex()])
    with open('/mnt/goldphish/trace.txt', mode='w') as fout:
        for l in trace['result']['structLogs']:
            fout.write(str(l) + '\n')

    print('------------------ self trace --------------------------')

    parsed = decode_trace_calls(trace['result']['structLogs'])
    pretty_print_trace(parsed)

    print('--------------------------------------------------------')

    print('------------------ old trace --------------------------')

    traceOld = mainnet_chain.provider.make_request('debug_traceTransaction', [
        '0x4bbd3ead05d44408bff9ce948501d7e6179f8d7da5a85ad9c9b05274653d2343',
        {'enableMemory': True}
    ])
    parsedOld = decode_trace_calls(traceOld['result']['structLogs'])
    pretty_print_trace(parsedOld)
    with open('/mnt/goldphish/trace_old.txt', mode='w') as fout:
        for l in traceOld['result']['structLogs']:
            fout.write(str(l) + '\n')

    print('--------------------------------------------------------')    

    # first call should be to uniswap v3, decode it and ensure the args are as expected
    uv3_call = parsed['actions'][0]
    assert uv3_call['callee'] == '0xFd76bE67FFF3BAC84E3D5444167bbC018f5968b6'
    (func, dec) = uv3.decode_function_input(uv3_call['args'])
    assert func.abi['name'] == 'swap'

    assert dec['recipient'] == deployed_shooter
    assert dec['zeroForOne'] == False
    assert dec['amountSpecified'] == 0xcceead6811c5d300e0

    print('receipt', receipt)
    assert receipt['status'] == 1

    print('balance shooter', w3.fromWei((w3.eth.get_balance(deployed_shooter)), 'ether'))

    # parse receipt for Transaction logs and ensure that they show we make a profit
    profits = parse_logs_for_net_profit(receipt['logs'])
    profit_token = None
    for moved_token, net_flows in profits.items():
        for addr, net_flow in net_flows.items():
            print(f'{moved_token} {addr} {net_flow}')
            if addr == deployed_shooter:
                assert net_flow > 0
                assert profit_token is None
                profit_token = moved_token

    assert profit_token is not None
    print('profit_token', profit_token)

