"""
Interface for exchange pricers.
"""
import web3
import web3.types
import typing

class BaseExchangePricer:
    w3: web3.Web3
    token0: str
    token1: str
    address: str

    def __init__(self, w3: web3.Web3) -> None:
        self.w3 = w3

    def quote_token0_to_token1(self, token0_amount, block_identifier: int) -> int:
        """
        Quickly use spot price to quote token transfer amount.
        Returned value is an over-estimate.
        """
        raise NotImplementedError()
    
    def quote_token1_to_token0(self, token1_amount, block_identifier: int) -> int:
        """
        Quickly use spot price to quote token transfer amount.
        Returned value is an over-estimate.
        """
        raise NotImplementedError()

    def exact_token0_to_token1(self, token0_amount, block_identifier: int) -> int:
        """
        Convert exact token0 to exact token1. SLOW.
        """
        raise NotImplementedError()
    
    def exact_token1_to_token0(self, token1_amount, block_identifier: int) -> int:
        """
        Convert exact token1 to exact token0. SLOW.
        """
        raise NotImplementedError()
    
    def observe_block(self, logs: typing.List[web3.types.LogReceipt]):
        pass

    def set_web3(self, w3: web3.Web3):
        """
        Used for moving a pricer onto a fork with minimal setup disruption

        Should also clear the running caches
        """
        raise NotImplementedError()
