"""
find_circuit/monitor.py

Monitors arbitrage opportunities over time.
"""
import collections
import typing
import pricers
import logging
import web3
import time
import web3.types
from .find import detect_arbitrages, FoundArbitrage

from utils import WETH_ADDRESS

l = logging.getLogger(__name__)


def profitable_circuits(modified_exchanges_last_block: typing.Set[str], pool: pricers.PricerPool, block_number: int) -> typing.Iterator[FoundArbitrage]:
    found_circuits: typing.Dict[typing.Any, FoundArbitrage] = {}
    for circuit in propose_circuits(modified_exchanges_last_block, pool, block_number):
        if any(not pool.is_uniswap_v2(x) for x in circuit):
            l.debug(f'testing ' + str(circuit))
            # there MUST be a uniswap v3 in here to consider it
            circuit_pricers = [pool.get_pricer_for(a) for a in circuit]
            for fa in detect_arbitrages(circuit_pricers, block_number):
                key = tuple(sorted(p.address for p in fa.circuit))
                if key in found_circuits:
                    if found_circuits[key].profit < fa.profit:
                        found_circuits[key] = fa
                else:
                    found_circuits[key] = fa
    yield from found_circuits.values()


def propose_circuits(modified_exchanges_last_block: typing.Set[str], pool: pricers.PricerPool, block_number: int) -> typing.Iterator[typing.List[str]]:
    """
    Proposes arbitrage circuits to test for profitability.

    Uses DFS to find WETH-containing cycles.
    """

    for m in modified_exchanges_last_block:
        yield from _propose_circuits_exchange(m, pool, block_number)


def _propose_circuits_exchange(address: str, pool: pricers.PricerPool, block_number: int) -> typing.Iterator[typing.List[str]]:
    already_proposed = set()

    # There are several situations here.

    # First, we need to find out whether this exchange has WETH or not;
    # if it does, we can yield some length-2 circuits and use special logic to find 3-length circuits
    pair = pool.get_pair_for(address)
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

            pair2 = pool.get_pair_for(other_exchange)
            if WETH_ADDRESS in pair2:
                # stop here, we made a 2-length circuit
                yield [address, other_exchange]
            else:
                # 3-length circuit ... need to find other_token2
                if pair2[0] == other_token:
                    other_token2 = pair2[1]
                else:
                    other_token2 = pair2[0]
                
                if bytes.fromhex(other_token2[2:]) < bytes.fromhex(WETH_ADDRESS[2:]):
                    last_pair = (other_token2, WETH_ADDRESS)
                else:
                    last_pair = (WETH_ADDRESS, other_token2)

                # find the remaining leg
                for last_exchange in pool.get_exchanges_for_pair(last_pair[0], last_pair[1], block_number):
                    assert len(set([address, other_exchange, last_exchange])) == 3, 'should not have duplicates'
                    yield [address, other_exchange, last_exchange]
    else:
        # WETH is not in this pair, so we can only have length-3 exchanges
        for exchange_1 in pool.get_exchanges_for_pair(WETH_ADDRESS, token0, block_number):
            for exchange_2 in pool.get_exchanges_for_pair(WETH_ADDRESS, token1, block_number):
                assert exchange_1 != exchange_2
                yield [exchange_1, address, exchange_2]

