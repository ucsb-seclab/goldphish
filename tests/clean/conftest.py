import json
import time
import typing
import pytest
import web3
import web3.types
import web3.contract
from eth_account import Account
from eth_account.signers.local import LocalAccount
import subprocess
import os


UNISWAP_V2_DEPLOYER = '0x9C33eaCc2F50E39940D3AfaF2c7B8246B681A374'
UNISWAP_V3_DEPLOYER = '0x6C9FC64A53c1b71FB3f9Af64d1ae3A4931A5f4E9'
WETH_DEPLOYER       = '0x4F26FfBe5F04ED43630fdC30A87638d53D0b0876'
WETH                = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'


@pytest.fixture(scope='module')
def funded_account():
    return Account.from_key(b'\x30' * 32)


_next_ganache_port = 6555
@pytest.fixture(scope='function')
def ganache_chain(funded_account: LocalAccount):
    global _next_ganache_port

    p = subprocess.Popen(
        [
            'yarn',
            'ganache-cli',
            '--server.ws',
            '--server.port', str(_next_ganache_port),
            '--wallet.accounts', f'{funded_account.key.hex()},{web3.Web3.toWei(2_000, "ether")}',
            '--wallet.unlockedAccounts', f'{UNISWAP_V2_DEPLOYER}',
            '--wallet.unlockedAccounts', f'{UNISWAP_V3_DEPLOYER}',
            '--wallet.unlockedAccounts', f'{WETH_DEPLOYER}',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    provider = web3.WebsocketProvider(
        f'ws://127.0.0.1:{_next_ganache_port}',
        websocket_timeout=60 * 5,
        websocket_kwargs={
            'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
        },
    )

    w3 = web3.Web3(
        provider
    )

    _next_ganache_port += 1

    print('waiting....')
    while not w3.isConnected():
        time.sleep(0.1)
    print('connected')

    assert w3.isConnected()

    yield w3

    p.kill()
    p.wait()
    subprocess.check_call(['killall', 'node'])


@pytest.fixture()
def deployed_uniswap_v2_factory(ganache_chain, funded_account: LocalAccount):
    w3: web3.Web3 = ganache_chain

    # transfer some funds to the uniswap v2 deployer account
    txn: web3.types.TxParams = {
        'from': funded_account.address,
        'to': UNISWAP_V2_DEPLOYER,
        'value': w3.toWei(2, 'ether'),
        'nonce': w3.eth.get_transaction_count(funded_account.address),
        'chainId': w3.eth.chain_id,
        'gas': 21000,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
    }
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    # use the deployer to deploy the factory
    with open('./contracts/artifacts/UniswapV2Factory.json') as fin:
        artifact = json.load(fin)
    
    factory: web3.contract.Contract = w3.eth.contract(
        abi=artifact['abi'],
        bytecode=artifact['bytecode'],
    )
    txn = factory.constructor(w3.toChecksumAddress(b'\x00' * 20)).buildTransaction({
        'from': UNISWAP_V2_DEPLOYER,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 4_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1
    assert receipt['contractAddress'] == '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f'
    return receipt['contractAddress']


@pytest.fixture()
def deployed_uniswap_v3_factory(ganache_chain, funded_account: LocalAccount):
    w3: web3.Web3 = ganache_chain

    # transfer some funds to the uniswap v3 deployer account
    txn: web3.types.TxParams = {
        'from': funded_account.address,
        'to': UNISWAP_V3_DEPLOYER,
        'value': w3.toWei(2, 'ether'),
        'nonce': w3.eth.get_transaction_count(funded_account.address),
        'chainId': w3.eth.chain_id,
        'gas': 21000,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
    }
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    # use the deployer to deploy the factory
    with open('./contracts/artifacts/UniswapV3Factory.json') as fin:
        artifact = json.load(fin)
    
    factory: web3.contract.Contract = w3.eth.contract(
        abi=artifact['abi'],
        bytecode=artifact['bytecode'],
    )
    txn = factory.constructor().buildTransaction({
        'from': UNISWAP_V3_DEPLOYER,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1
    assert receipt['contractAddress'] == '0x1F98431c8aD98523631AE4a59f267346ea31F984'
    return receipt['contractAddress']

@pytest.fixture()
def weth(ganache_chain, funded_account: LocalAccount):
    w3: web3.Web3 = ganache_chain

    # transfer some funds to the weth9 deployer account
    txn: web3.types.TxParams = {
        'from': funded_account.address,
        'to': WETH_DEPLOYER,
        'value': w3.toWei(2, 'ether'),
        'nonce': w3.eth.get_transaction_count(funded_account.address),
        'chainId': w3.eth.chain_id,
        'gas': 21000,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
    }
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    # fast-forward the nonce
    result = w3.provider.make_request('evm_setAccountNonce', [WETH_DEPLOYER, 446])
    assert result['result'] == True

    with open('./abis/weth9/WETH9.json') as fin:
        artifact = json.load(fin)
    
    contract: web3.contract.Contract = w3.eth.contract(
        abi=artifact['abi'],
        bytecode=artifact['bytecode'],
    )
    txn = contract.constructor().buildTransaction({
        'from': WETH_DEPLOYER,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 2_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    assert receipt['contractAddress'] == WETH
    return receipt['contractAddress']


@pytest.fixture()
def token_a(ganache_chain, funded_account: LocalAccount):
    """
    Deploys an ERC20 token
    """
    w3: web3.Web3 = ganache_chain

    with open('./contracts/artifacts/ERC20PresetMinterPauser.json') as fin:
        artifact = json.load(fin)
    
    contract: web3.contract.Contract = w3.eth.contract(
        abi=artifact['abi'],
        bytecode=artifact['bytecode'],
    )
    txn = contract.constructor('TokenA', 'TA').buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 2_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    return receipt['contractAddress']


@pytest.fixture()
def token_b(ganache_chain, funded_account: LocalAccount):
    """
    Deploys an ERC20 token
    """
    w3: web3.Web3 = ganache_chain

    with open('./contracts/artifacts/ERC20PresetMinterPauser.json') as fin:
        artifact = json.load(fin)
    
    contract: web3.contract.Contract = w3.eth.contract(
        abi=artifact['abi'],
        bytecode=artifact['bytecode'],
    )
    txn = contract.constructor('TokenB', 'TB').buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 2_000_000,
    })
    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    return receipt['contractAddress']


@pytest.fixture()
def nf_position_manager(ganache_chain, funded_account: LocalAccount, deployed_uniswap_v3_factory, weth):
    """
    Deploys the position manager helper contract
    """
    w3: web3.Web3 = ganache_chain

    with open('./contracts/artifacts/NonfungiblePositionManager.json') as fin:
        artifact = json.load(fin)
    
    contract: web3.contract.Contract = w3.eth.contract(
        abi=artifact['abi'],
        bytecode=artifact['bytecode'],
    )
    txn = contract.constructor(deployed_uniswap_v3_factory, weth, w3.toChecksumAddress(b'\xa1' * 20)).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })

    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    return receipt['contractAddress']


@pytest.fixture()
def v3_swap_router(ganache_chain, funded_account: LocalAccount, deployed_uniswap_v3_factory, weth):
    """
    Deploys the position manager helper contract
    """
    w3: web3.Web3 = ganache_chain

    with open('./contracts/artifacts/V3SwapRouter.json') as fin:
        artifact = json.load(fin)
    
    contract: web3.contract.Contract = w3.eth.contract(
        abi=artifact['abi'],
        bytecode=artifact['bytecode'],
    )
    txn = contract.constructor(deployed_uniswap_v3_factory, weth).buildTransaction({
        'from': funded_account.address,
        'maxFeePerGas': 80 * (10 ** 9),
        'maxPriorityFeePerGas': 3 * (10 ** 9),
        'gas': 6_000_000,
    })

    tx_hash = w3.eth.send_transaction(txn)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt['status'] == 1

    return receipt['contractAddress']

