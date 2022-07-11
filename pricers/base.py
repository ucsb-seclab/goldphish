"""
Interface for exchange pricers.
"""
import decimal
import web3
import web3.types
import typing

from pricers.block_observation_result import BlockObservationResult

class BaseExchangePricer:
    w3: web3.Web3
    address: str

    def __init__(self, w3: web3.Web3) -> None:
        self.w3 = w3

    def get_tokens(self, block_identifier: int) -> typing.Set[str]:
        raise NotImplementedError()

    def get_value_locked(self, token_address: str, block_identifier: int) -> int:
        raise NotImplementedError()

    def get_token_weight(self, token_address: str, block_identifier: int) -> decimal.Decimal:
        raise NotImplementedError()

    def token_out_for_exact_in(self, token_in: str, token_out: str, amount_in: int, block_identifier: int) -> int:
        raise NotImplementedError()

    def observe_block(self, logs: typing.List[web3.types.LogReceipt]) -> BlockObservationResult:
        pass

    def set_web3(self, w3: web3.Web3):
        """
        Used for moving a pricer onto a fork with minimal setup disruption

        Should also clear the running caches
        """
        raise NotImplementedError()
