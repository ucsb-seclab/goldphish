"""
mainly test-the-tests stuff;  to make sure we're setting up exchanges correctly
"""

import json
import time
import pytest
import web3
import web3.contract
import web3.types
from eth_account.signers.local import LocalAccount
import pricers
import find_circuit.find
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer

from utils import decode_trace_calls, get_abi, pretty_print_trace

def deploy_v3_pool(
        w3: web3.Web3,
        funded_account: LocalAccount,
        deployed_uniswap_v3_factory,
        token_a,
        token_b
    ):
    with open('./abis/uniswap_v3/IUniswapV3Factory.json') as fin:
        artifact = json.load(fin)

    factory: web3.contract.Contract = w3.eth.contract(
        address = deployed_uniswap_v3_factory,
        abi = artifact,
    )

    txn = factory.functions.createPool(token_a, token_b, 3_000).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    (event,) = factory.events.PoolCreated().processReceipt(receipt)

    return event['args']['pool']


def deploy_v2_pool(
        w3: web3.Web3,
        funded_account: LocalAccount,
        deployed_uniswap_v2_factory,
        token_a,
        token_b
    ):
    with open('./abis/uniswap_v2/IUniswapV2Factory.json') as fin:
        artifact = json.load(fin)
    
    factory: web3.contract.Contract = w3.eth.contract(
        address = deployed_uniswap_v2_factory,
        abi = artifact,
    )

    txn = factory.functions.createPair(token_a, token_b).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    (event,) = factory.events.PairCreated().processReceipt(receipt)

    return event['args']['pair']



def authorize_nfp(w3: web3.Web3, funded_account: LocalAccount, nf_position_manager: str, token: str):

    contract = w3.eth.contract(
        address = token,
        abi = get_abi('erc20.abi.json'),
    )

    txn = contract.functions.approve(nf_position_manager, (0x1 << 256) - 1).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1



def mint_token(
        w3: web3.Web3,
        funded_account: LocalAccount,
        amount: int,
        token: str,
        to: str,
    ):
    with open('./contracts/artifacts/ERC20PresetMinterPauser.json') as fin:
        artifact = json.load(fin)

    contract: web3.contract.Contract = w3.eth.contract(
        address = token,
        abi = artifact['abi'],
    )

    txn = contract.functions.mint(to, amount).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1


def wrap_weth(w3: web3.Web3, funded_account: LocalAccount, amount: int):
    weth: web3.contract.Contract = w3.eth.contract(
        address='0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        abi=get_abi('weth9/WETH9.json')['abi'],
    )

    txn = weth.functions.deposit().buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
        'value': amount,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1


def init_uniswap_v3(w3: web3.Web3, pool: str, funded_account: LocalAccount, sqrt_price_x96):
    contract = w3.eth.contract(
        address = pool,
        abi = get_abi('uniswap_v3/IUniswapV3Pool.json')['abi'],
    )

    txn = contract.functions.initialize(sqrt_price_x96).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1


def mint_uniswap_v3(
        w3: web3.Web3,
        funded_account: LocalAccount,
        nf_position_manager: str,
        token0: str,
        token1: str,
        tick_lower: int,
        tick_upper: int,
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int,
        amount1_min: int):

    if bytes.fromhex(token0[2:]) > bytes.fromhex(token1[2:]):
        token0, token1 = token1, token0
        amount0_desired, amount1_desired = amount1_desired, amount0_desired
        amount0_min, amount1_min = amount1_min, amount0_min

    with open('./contracts/artifacts/NonfungiblePositionManager.json') as fin:
        artifact = json.load(fin)
    
    contract = w3.eth.contract(
        address = nf_position_manager,
        abi = artifact['abi'],
    )

    txn = contract.functions.mint(
        (token0,
        token1,
        3_000,
        tick_lower,
        tick_upper,
        amount0_desired,
        amount1_desired,
        amount0_min,
        amount1_min,
        funded_account.address,
        int(time.time() + 60),)
    ).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    if receipt['status'] != 1:
        trace = w3.provider.make_request('debug_traceTransaction', [receipt['transactionHash'].hex()])
        with open('/mnt/goldphish/trace.txt', mode='w') as fout:
            for log in trace['result']['structLogs']:
                fout.write(str(log) + '\n')

        print('------------------our trace---------------------------')
        decoded = decode_trace_calls(trace['result']['structLogs'], txn, receipt)
        pretty_print_trace(decoded, txn, receipt)
        print('------------------------------------------------------')


def mint_uniswap_v2(w3: web3.Web3, funded_account: LocalAccount, pair, token0, token1, amount0, amount1):
    if bytes.fromhex(token0[2:]) < bytes.fromhex(token1[2:]):
        token0, token1 = token1, token0
        amount0, amount1 = amount1, amount0

    # transfer both amounts in
    token0_contract: web3.contract.Contract = w3.eth.contract(
        address = token0,
        abi = get_abi('erc20.abi.json')
    )

    txn = token0_contract.functions.transfer(pair, amount0).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    token1_contract: web3.contract.Contract = w3.eth.contract(
        address = token1,
        abi = get_abi('erc20.abi.json')
    )
    txn = token1_contract.functions.transfer(pair, amount1).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    with open('./abis/uniswap_v2/IUniswapV2Pair.json') as fin:
        artifact = json.load(fin)

    pair_contract: web3.contract.Contract = w3.eth.contract(
        address = pair,
        abi = artifact['abi'],
    )

    txn = pair_contract.functions.mint(funded_account.address).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1


def test_setup_univ2(ganache_chain, funded_account, deployed_uniswap_v2_factory):
    assert deployed_uniswap_v2_factory is not None
    assert web3.Web3.isChecksumAddress(deployed_uniswap_v2_factory)


def test_setup_univ3(ganache_chain, funded_account, deployed_uniswap_v3_factory):
    assert deployed_uniswap_v3_factory is not None
    assert web3.Web3.isChecksumAddress(deployed_uniswap_v3_factory)


def test_setup_weth(ganache_chain, weth):
    assert web3.Web3.isChecksumAddress(weth)


def test_setup_tokens(ganache_chain, token_a, token_b):
    assert web3.Web3.isChecksumAddress(token_a)
    assert web3.Web3.isChecksumAddress(token_b)
    assert token_a != token_b


def test_basic_arbitrage(
        ganache_chain,
        funded_account: LocalAccount,
        deployed_uniswap_v3_factory: str,
        deployed_uniswap_v2_factory: str,
        nf_position_manager: str,
        weth: str,
        token_a: str,
    ):
    w3: web3.Web3 = ganache_chain
    
    # create a v3 weth-token_a pool
    univ3_pool = deploy_v3_pool(ganache_chain, funded_account, deployed_uniswap_v3_factory, weth, token_a)

    print('got pool', univ3_pool)
    print('funded account', funded_account.address)

    # mint some 25:75 WETH liquidity in the pool

    # wrap some weth
    wrap_weth(w3, funded_account, w3.toWei(100, 'ether'))

    mint_token(w3, funded_account, 100 * (10 ** 18), token_a, funded_account.address)

    print('nf_position_manager:', nf_position_manager)
    authorize_nfp(w3, funded_account, nf_position_manager, weth)
    authorize_nfp(w3, funded_account, nf_position_manager, token_a)

    # mint it into uniswap v3
    init_uniswap_v3(w3, univ3_pool, funded_account, 1 * (1 << 96))
    mint_uniswap_v3(
        w3,
        funded_account,
        nf_position_manager,
        weth,
        token_a,
        tick_lower = -887220,
        tick_upper =  887220,
        amount0_desired = w3.toWei(50, 'ether'),
        amount1_desired = w3.toWei(50, 'ether'),
        amount0_min = w3.toWei(50, 'ether'),
        amount1_min = w3.toWei(50, 'ether')
    )

    # create a v2 weth-token_a pool
    univ2_pool = deploy_v2_pool(w3, funded_account, deployed_uniswap_v2_factory, weth, token_a)

    # mint liquidity, v2
    mint_uniswap_v2(w3, funded_account, univ2_pool, weth, token_a, w3.toWei(48, 'ether'), w3.toWei(50, 'ether'))

    token0, token1 = sorted([weth, token_a], key=lambda x: bytes.fromhex(x[2:]))

    # attempt to find arbitrage
    exchanges = [
        UniswapV3Pricer(w3, univ3_pool, token0, token1, 3_000),
        UniswapV2Pricer(w3, univ2_pool, token0, token1)
    ]
    got = find_circuit.find.detect_arbitrages(exchanges, block_identifier=w3.eth.get_block('latest')['number'])
    assert len(got) > 0
    print(got)

