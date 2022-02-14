import pytest
import web3
import random
import web3.constants
from eth_account.signers.local import LocalAccount

import shooter.deploy

@pytest.mark.ganache_block_num(14_000_000)
def test_basic_deploy(ganache_chain: web3.Web3, funded_deployer: LocalAccount):
    w3 = ganache_chain
    shooter_addr = shooter.deploy.deploy_shooter(
        w3,
        funded_deployer,
        max_priority=3,
        max_fee_total=w3.toWei(10, 'ether'),
    )
    assert w3.isChecksumAddress(shooter_addr)

