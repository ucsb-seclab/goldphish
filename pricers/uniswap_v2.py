"""
Uniswap v2 exchange pricer
"""

import decimal
import web3
import web3.types
import web3.contract
import typing
import logging
from eth_utils import event_abi_to_log_topic
from pricers.block_observation_result import BlockObservationResult

from pricers.uniswap_v3 import UniswapV3Pricer

from .base import BaseExchangePricer
from utils import get_abi

l = logging.getLogger(__name__)

generic_uv2 = web3.Web3().eth.contract(
    address = b'\x00' * 20,
    abi = get_abi('uniswap_v2/IUniswapV2Pair.json')['abi'],
)
UNIV2_SYNC_EVENT_TOPIC = event_abi_to_log_topic(generic_uv2.events.Sync().abi)


class UniswapV2Pricer(BaseExchangePricer):
    RELEVANT_LOGS = [UNIV2_SYNC_EVENT_TOPIC]

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
        self.token0 = token0
        self.token1 = token1
        self.set_web3(w3)

    def get_tokens(self, _) -> typing.Set[str]:
        return set([self.token0, self.token1])

    def get_balances(self, block_identifier) -> typing.Tuple[int, int]:
        if self.known_token0_bal is None or self.known_token1_bal is None:
            self.known_token0_bal, self.known_token1_bal, _ = self.contract.functions.getReserves().call(block_identifier=block_identifier)
        return (self.known_token0_bal, self.known_token1_bal)

    def token_out_for_exact_in(self, token_in: str, token_out: str, amount_in: int, block_identifier: int, **_) -> typing.Tuple[int, float]:
        if token_in == self.token0 and token_out == self.token1:
            amt_out = self.exact_token0_to_token1(amount_in, block_identifier)
            new_reserve_in  = self.known_token0_bal + amount_in
            new_reserve_out = self.known_token1_bal - amt_out
        elif token_in == self.token1 and token_out == self.token0:
            amt_out = self.exact_token1_to_token0(amount_in, block_identifier)
            new_reserve_in  = self.known_token1_bal + amount_in
            new_reserve_out = self.known_token0_bal - amt_out
        else:
            raise NotImplementedError()

        if new_reserve_out == 0:
            spot = 0
        else:
            # how much out do we get for 1 unit in?
            # https://github.com/Uniswap/v2-periphery/blob/master/contracts/libraries/UniswapV2Library.sol#L43
            amount_in_with_fee = 1 * 997
            numerator = amount_in_with_fee * new_reserve_out
            denominator = new_reserve_in * 1_000 + amount_in_with_fee
            spot = numerator / denominator

        return (amt_out, spot)

    def exact_token0_to_token1(self, token0_amount, block_identifier: int) -> int:
        # based off https://github.com/Uniswap/v2-periphery/blob/master/contracts/libraries/UniswapV2Library.sol#L43
        bal0, bal1 = self.get_balances(block_identifier)
        if bal0 == 0 or bal1 == 0:
            return 0 # no amount can be taken out
        amt_in_with_fee = token0_amount * 997
        numerator = amt_in_with_fee * bal1
        denominator = bal0 * 1000 + amt_in_with_fee
        ret = numerator // denominator
        return ret

    def exact_token1_to_token0(self, token1_amount, block_identifier: int) -> int:
        # based off https://github.com/Uniswap/v2-periphery/blob/master/contracts/libraries/UniswapV2Library.sol#L43
        bal0, bal1 = self.get_balances(block_identifier)
        if bal0 == 0 or bal1 == 0:
            return 0 # no amount can be taken out
        amt_in_with_fee = token1_amount * 997
        numerator = amt_in_with_fee * bal0
        denominator = bal1 * 1000 + amt_in_with_fee
        return numerator // denominator

    def token1_out_to_exact_token0_in(self, token1_amount_out, block_identifier: int) -> int:
        return self.get_amount_in(token1_amount_out, True, block_identifier)

    def token0_out_to_exact_token1_in(self, token0_amount_out, block_identifier: int) -> int:
        return self.get_amount_in(token0_amount_out, False, block_identifier)

    def get_amount_in(self, amount_out: int, zero_for_one: bool, block_identifier=int) -> int:
        # out = ((in * 997 * b0) / (b1 * 1000 + in * 997))
        # out = 


        bal0, bal1 = self.get_balances(block_identifier)
        if bal0 == 0 or bal1 == 0:
            return 0 # no amount can be moved
        if not zero_for_one:
            bal0, bal1 = bal1, bal0
        assert amount_out <= bal1
        numerator = bal0 * amount_out * 1000
        denominator = (bal1 - amount_out) * 997
        return (numerator // denominator) + 1

    def get_value_locked(self, token_address: str, block_identifier: int) -> int:
        bal0, bal1 = self.get_balances(block_identifier)
        if token_address == self.token0:
            return bal0
        elif token_address == self.token1:
            return bal1

        raise Exception(f'Do not know about token {token_address}')

    def get_token_weight(self, token_address: str, block_identifier: int) -> decimal.Decimal:
        return decimal.Decimal('0.5')

    def observe_block(self, receipts: typing.List[web3.types.LogReceipt]) -> BlockObservationResult:
        """
        Observe the logs emitted in a block and update internal state appropriately.
        NOTE: receipts _must_ be in sorted order of increasing log index
        """
        if len(receipts) == 0:
            return

        block_num = receipts[0]['blockNumber']
        # all we care about are Syncs
        for log in reversed(receipts):
            if log['address'] == self.address and len(log['topics']) > 0 and log['topics'][0] == UNIV2_SYNC_EVENT_TOPIC:
                sync = self.contract.events.Sync().processLog(log)
                bal0 = sync['args']['reserve0']
                assert bal0 >= 0
                bal1 = sync['args']['reserve1']
                assert bal1 >= 0
                self.known_token0_bal = bal0
                self.known_token1_bal = bal1

                # balances updated
                return BlockObservationResult(
                    pair_prices_updated = [(self.token0, self.token1)],
                    swap_enabled = None,
                    gradual_weight_adjusting_scheduled = None,
                )

        return BlockObservationResult(
            pair_prices_updated = [],
            swap_enabled = None,
            gradual_weight_adjusting_scheduled = None,
        )

    def set_web3(self, w3: web3.Web3):
        self.contract = w3.eth.contract(
            address = self.address,
            abi = get_abi('uniswap_v2/IUniswapV2Pair.json')['abi']
        )
        self.w3 = w3
        self.token0_contract = w3.eth.contract(
            address = self.token0,
            abi = get_abi('erc20.abi.json')
        )
        self.token1_contract = w3.eth.contract(
            address = self.token1,
            abi = get_abi('erc20.abi.json')
        )
        self.known_token0_bal = None
        self.known_token1_bal = None

    def copy_without_cache(self) -> 'BaseExchangePricer':
        return UniswapV2Pricer(
            self.w3, self.address, self.token0, self.token1
        )

