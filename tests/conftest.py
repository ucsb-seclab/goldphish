"""
tests/utils.py

Utilities for testing procedures
"""

import json
import os
import signal
import time
import web3
import pathlib
import pytest
import subprocess
from eth_account import Account
from eth_account.signers.local import LocalAccount

import shooter.deploy

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "ganache_block_num",
    )
    config.addinivalue_line(
        "markers", "ganache_mine_mode",
    )

@pytest.fixture(scope='function')
def funded_deployer():
    ret: LocalAccount = Account.from_key(bytes.fromhex('0x02f690498604807a0970b2c39634e3fe71e13920187c6a4c23b7a3a0b3fb4638'[2:]))
    return ret

@pytest.fixture(scope='module')
def mainnet_chain():
    return web3.Web3(
        web3.WebsocketProvider(
            'ws://172.17.0.1:8546',
            websocket_timeout=60 * 5,
            websocket_kwargs={
                'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
            },
        ),
    )

_next_ganache_port = 5555
@pytest.fixture(scope='function')
def ganache_chain(request, mainnet_chain: web3.Web3, funded_deployer: LocalAccount):
    global _next_ganache_port
    block_num = request.node.get_closest_marker('ganache_block_num').args[0]

    mine_mode = request.node.get_closest_marker('ganache_mine_mode')

    if mine_mode is None:
        mine_mode = 'INSTANT'
    else:
        mine_mode = mine_mode.args[0]
    assert mine_mode in ['INSTANT', 'TIMED']

    old_block = mainnet_chain.eth.get_block(block_num)
    old_ts = old_block['timestamp']

    assert isinstance(block_num, int)
    assert block_num > 0

    extra_args = []
    if mine_mode == 'TIMED':
        extra_args += [
            '--miner.blockTime', '5',
        ]

    p = subprocess.Popen(
        [
            'yarn',
            'ganache-cli',
            '--fork.url', 'ws://172.17.0.1:8546',
            '--server.ws',
            '--server.port', str(_next_ganache_port),
            '--fork.blockNumber', str(block_num),
            '--wallet.accounts', f'{funded_deployer.key.hex()},{web3.Web3.toWei(100, "ether")}',
            '--chain.chainId', '1',
            '--chain.time', str(old_ts),
            '--miner.coinbase', web3.Web3.toChecksumAddress(b'\xa0' * 20) + ' ',
            *extra_args
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

    while not w3.isConnected():
        time.sleep(0.1)

    assert w3.isConnected()

    yield w3

    p.kill()
    p.wait()
    # NOTE sadly ganache is still running in the background here, oh well...


@pytest.fixture(scope='function')
def deployed_shooter(ganache_chain: web3.Web3, funded_deployer: LocalAccount):
    shooter_addr = shooter.deploy.deploy_shooter(
        ganache_chain,
        funded_deployer,
        max_priority=3,
        max_fee_total=web3.Web3.toWei(10, 'ether'),
    )
    assert web3.Web3.isChecksumAddress(shooter_addr)
    return shooter_addr
