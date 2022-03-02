"""
Uniswap v2 exchange pricer
"""

import web3
import web3.types
import web3.contract
import typing
import logging
from eth_utils import event_abi_to_log_topic


from .base import BaseExchangePricer
from utils import get_abi

generic_uv2 = web3.Web3().eth.contract(
    address = b'\x00' * 20,
    abi = get_abi('uniswap_v2/IUniswapV2Pair.json')['abi'],
)
UNIV2_SYNC_EVENT_TOPIC = event_abi_to_log_topic(generic_uv2.events.Sync().abi)


class UniswapV2Pricer(BaseExchangePricer):
    w3: web3.Web3
    address: str
    token0: str
    token1: str
    contract: web3.contract.Contract
    token0_contract: web3.contract.Contract
    token1_contract: web3.contract.Contract
    known_token0_bal: int
    known_token1_bal: int

    def __init__(self, w3: web3.Web3, address: str, token0: str, token1: str) -> None:
        assert web3.Web3.isChecksumAddress(address)
        self.address = address
        self.contract = w3.eth.contract(
            address = address,
            abi = get_abi('uniswap_v2/IUniswapV2Pair.json')['abi']
        )
        self.w3 = w3
        self.token0_contract = w3.eth.contract(
            address = token0,
            abi = get_abi('erc20.abi.json')
        )
        self.token1_contract = w3.eth.contract(
            address = token1,
            abi = get_abi('erc20.abi.json')
        )
        self.token0 = token0
        self.token1 = token1
        self.known_token0_bal = None
        self.known_token1_bal = None

    def get_balances(self, block_identifier) -> typing.Tuple[int, int]:
        if self.known_token0_bal is None or self.known_token1_bal is None:
            self.known_token0_bal = self.token0_contract.functions.balanceOf(self.address).call(block_identifier=block_identifier)
            self.known_token1_bal = self.token1_contract.functions.balanceOf(self.address).call(block_identifier=block_identifier)
        return (self.known_token0_bal, self.known_token1_bal)

    def quote_token0_to_token1(self, token0_amount, block_identifier=int) -> int:
        return self.exact_token0_to_token1(token0_amount, block_identifier)

    def quote_token1_to_token0(self, token1_amount, block_identifier=int) -> int:
        return self.exact_token1_to_token0(token1_amount, block_identifier)

    def exact_token0_to_token1(self, token0_amount, block_identifier=int) -> int:
        # based off https://github.com/Uniswap/v2-periphery/blob/master/contracts/libraries/UniswapV2Library.sol#L43
        bal0, bal1 = self.get_balances(block_identifier)
        amt_in_with_fee = token0_amount * 997
        numerator = amt_in_with_fee * bal1
        denominator = bal0 * 1000 + amt_in_with_fee
        return numerator // denominator

    def exact_token1_to_token0(self, token1_amount, block_identifier=int) -> int:
        # based off https://github.com/Uniswap/v2-periphery/blob/master/contracts/libraries/UniswapV2Library.sol#L43
        bal0, bal1 = self.get_balances(block_identifier)
        amt_in_with_fee = token1_amount * 997
        numerator = amt_in_with_fee * bal0
        denominator = bal1 * 1000 + amt_in_with_fee
        return numerator // denominator

    def observe_block(self, receipts: typing.List[web3.types.LogReceipt]):
        """
        Observe the logs emitted in a block and update internal state appropriately.
        NOTE: receipts _must_ be in sorted order of increasing log index
        """
        if len(receipts) == 0:
            return

        block_num = receipts[0]['blockNumber']
        # all we care about are Syncs
        for log in reversed(receipts):
            if len(log['topics']) > 0 and log['topics'][0] == UNIV2_SYNC_EVENT_TOPIC:
                sync = self.contract.events.Sync().processLog(log)
                bal0 = sync['args']['reserve0']
                assert bal0 >= 0
                bal1 = sync['args']['reserve1']
                assert bal1 >= 0
                self.known_token0_bal = bal0
                self.known_token1_bal = bal1
                return # this is all we care about -- the last Sync done
