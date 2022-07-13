"""
Interface for exchange pricers.
"""
import decimal
import web3
import web3.types
import typing

from pricers.block_observation_result import BlockObservationResult

class NotEnoughLiquidityException(Exception):
    """
    Thrown when there is not enough liquidity to complete the given swap.
    """
    amount_in: int
    remaining: int

    def __init__(self, amount_in, remaining, *args: object) -> None:
        super().__init__(*args)
        self.amount_in = amount_in
        self.remaining = remaining


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

    def token_out_for_exact_in(self, token_in: str, token_out: str, amount_in: int, block_identifier: int, **kwargs) -> typing.Tuple[int, float]:
        """
        Gets the amount of output the given swap would produce, and the spot price after the swap.

        Spot is in terms of token_out per token_in
        """
        raise NotImplementedError()

    def observe_block(self, logs: typing.List[web3.types.LogReceipt]) -> BlockObservationResult:
        pass

    def set_web3(self, w3: web3.Web3):
        """
        Used for moving a pricer onto a fork with minimal setup disruption

        Should also clear the running caches
        """
        raise NotImplementedError()

    def copy_without_cache(self) -> 'BaseExchangePricer':
        """
        Return a copy of this pricer absent its cached values, for ensuring cache-consistency
        """
        raise NotImplementedError()
