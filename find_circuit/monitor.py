"""
find_circuit/monitor.py

Monitors arbitrage opportunities over time.
"""
import typing
import pricers
import logging

from .find import PricingCircuit, FoundArbitrage, detect_arbitrages_bisection

from utils import WETH_ADDRESS

l = logging.getLogger(__name__)


def profitable_circuits(
        modified_pairs_last_block: typing.Dict[typing.Tuple[str, str], typing.List[str]],
        pool: pricers.PricerPool,
        block_number: int,
        timestamp: typing.Optional[int] = None,
        only_weth_pivot = False,
    ) -> typing.Iterator[FoundArbitrage]:

    for circuit in propose_circuits(modified_pairs_last_block, pool, block_number):
        yield from detect_arbitrages_bisection(circuit, block_number, timestamp = timestamp, only_weth_pivot = only_weth_pivot)


def propose_circuits(
        modified_pairs_last_block: typing.Dict[typing.Tuple[str, str], typing.List[str]],
        pool: pricers.PricerPool,
        block_number: int
    ) -> typing.Iterator[PricingCircuit]:
    """
    Proposes arbitrage circuits to test for profitability.

    """
    for pair, addresses in modified_pairs_last_block.items():
        for address in addresses:
            yield from _propose_circuits_pair(pair, address, pool, block_number)


def _propose_circuits_pair(pair: typing.Tuple[str, str], address: str, pool: pricers.PricerPool, block_number: int) -> typing.Iterator[PricingCircuit]:
    # There are several situations here.

    # First, we need to find out whether this exchange has WETH or not;
    # if it does, we can yield some length-2 circuits and use special logic to find 3-length circuits
    token0, token1 = pair
    if WETH_ADDRESS in pair:
        if token0 == WETH_ADDRESS:
            other_token = token1
        else:
            other_token = token0

        # 2-length
        # (WETH -> other_token) (other_token -> WETH)    
        # 3-length
        # (WETH -> other_token) (other_token -> other_token2) (other_token2 -> WETH)
        for other_exchange in pool.get_exchanges_for(other_token, block_number):
            if other_exchange == address:
                continue

            tokens = pool.get_tokens_for(other_exchange)
            if WETH_ADDRESS in tokens:
                # stop here, we made a 2-length circuit
                yield PricingCircuit(
                    [
                        pool.get_pricer_for(address),
                        pool.get_pricer_for(other_exchange),
                    ],
                    [
                        (WETH_ADDRESS, other_token),
                        (other_token, WETH_ADDRESS),
                    ]
                )
            else:
                for other_token2 in tokens.difference([WETH_ADDRESS, other_token]):
                    # construct 3-length circuit

                    # find the remaining leg
                    for last_exchange in pool.get_exchanges_for_pair(WETH_ADDRESS, other_token2, block_number):
                        if last_exchange in [address, other_exchange]:
                            continue
                        assert len(set([address, other_exchange, last_exchange])) == 3, 'should not have duplicates'
                        yield PricingCircuit(
                            [
                                pool.get_pricer_for(address),
                                pool.get_pricer_for(other_exchange),
                                pool.get_pricer_for(last_exchange),
                            ],
                            [
                                (WETH_ADDRESS, other_token),
                                (other_token, other_token2),
                                (other_token2, WETH_ADDRESS),
                            ],
                        )
    else:
        # WETH is not in this pair, so we can only have length-3 exchanges
        for exchange_1 in pool.get_exchanges_for_pair(WETH_ADDRESS, token0, block_number):
            for exchange_2 in pool.get_exchanges_for_pair(WETH_ADDRESS, token1, block_number):
                if address == exchange_1 or address == exchange_2 or exchange_1 == exchange_2:
                    # no dupes allowed
                    continue

                yield PricingCircuit(
                    [
                        pool.get_pricer_for(exchange_1),
                        pool.get_pricer_for(address),
                        pool.get_pricer_for(exchange_2),
                    ],
                    [
                        (WETH_ADDRESS, token0),
                        (token0, token1),
                        (token1, WETH_ADDRESS),
                    ],
                )

