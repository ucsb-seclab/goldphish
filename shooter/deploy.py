"""
deploy.py

Module for deploying the shooter.
"""
import typing
import web3
import web3.contract
import logging
import pathlib
import json
from eth_account.signers.local import LocalAccount

l = logging.getLogger(__name__)

class MaxFeeExceededException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def retrieve_shooter():
    artifact = pathlib.Path(__file__).parent.parent / 'artifacts' / 'contracts' / 'shooter.sol' / 'Shooter.json'
    with open(artifact) as fin:
        return json.load(fin)


def deploy_shooter(
        w3: web3.Web3,
        deployer: LocalAccount,
        max_priority: int,
        max_fee_total: int
    ) -> str:
    """
    Deploy the patched shooter on-chain.

    Uses the deployer account to send the transaction, hard-coding the administrator with access.
    These accounts may be the same.

    Quits if max_fee_total would be exceeded.
    """
    l.debug(f'Attempting to deploy shooter with deployer account {deployer.address}')

    shooter_artifact = retrieve_shooter()
    shooter_contract: web3.contract.Contract = w3.eth.contract(
        bytecode=shooter_artifact['bytecode'],
        abi=shooter_artifact['abi']
    )
    txn = shooter_contract.constructor().buildTransaction({
        'chainId': w3.eth.chain_id,
        'from': deployer.address,
        'nonce': w3.eth.get_transaction_count(deployer.address),
    })

    est_gas_usage = w3.eth.estimate_gas(txn, block_identifier='latest')
    l.debug(f'Expect to use {est_gas_usage:,} gas')

    max_gas_price = max_fee_total // est_gas_usage

    l.debug(f'Max gas price {max_gas_price:,} ({max_gas_price / (10 ** 9):.3f})')

    block = w3.eth.get_block(block_identifier='latest')
    last_base_fee = block['baseFeePerGas']
    l.debug(f'Last base fee is {last_base_fee:,} ({last_base_fee / (10 ** 9):.3f} gwei) - block {block["number"]:,}')

    if max_gas_price < last_base_fee:
        raise MaxFeeExceededException(f'Max gas price ({max_gas_price / (10 ** 9):,.3f} gwei) is below last block base fee ({last_base_fee / (10 ** 9):,.3f} gwei)!')

    txn['maxPriorityFeePerGas'] = max_priority
    txn['maxFeePerGas'] = max_gas_price

    signed_txn = w3.eth.account.sign_transaction(txn, deployer.key)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    l.info(f'Submitted shooter deployment transaction {tx_hash.hex()}')
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    l.info(f'Deployed shooter in block {receipt["blockNumber"]} to address {receipt["contractAddress"]}')
    return receipt['contractAddress']
