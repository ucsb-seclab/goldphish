"""
find_circuit/monitor.py

Monitors arbitrage opportunities over time.
"""
import time
import typing

import web3
import backtest.top_of_block.seek_candidates
import pricers
import logging
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.liquidity_bootstrapping_pool import BalancerV2LiquidityBootstrappingPoolPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer

from pricers.base import BaseExchangePricer
import utils
from utils.profiling import profile

from .find import PricingCircuit, FoundArbitrage, detect_arbitrages_bisection

from utils import TETHER_ADDRESS, UNI_ADDRESS, USDC_ADDRESS, WBTC_ADDRESS, WETH_ADDRESS

l = logging.getLogger(__name__)

# The threshold of eth locked for an exchange to be considered relevant
# for optimization. Set to just below 1 dollar (July '22).
DUST_THRESHOLD_WEI = (10 ** 18) // 1_000
THRESHOLDS = {
    WETH_ADDRESS: web3.Web3.toWei('0.01', 'ether'),
    USDC_ADDRESS: 10 * (10 ** 6),
    TETHER_ADDRESS: 10 * (10 ** 6),
    UNI_ADDRESS: web3.Web3.toWei('0.25', 'ether'), # UNI also uses 18 decimals and had a max price of 40, so this is about $400, optimistically
    WBTC_ADDRESS: 1 * (10 ** 8) // 10_000, # like $1-4 ish?
}



def profitable_circuits(
        modified_pairs_last_block: typing.Dict[typing.Tuple[str, str], typing.List[str]],
        pool: pricers.PricerPool,
        block_number: int,
        timestamp: typing.Optional[int] = None,
        only_weth_pivot = False,
    ) -> typing.Iterator[FoundArbitrage]:
    elapsed = 0
    it_pcs = propose_circuits(modified_pairs_last_block, pool, block_number)
    circuits_considered = set()

    while True:
        t_start = time.time()
        try:
            item = next(it_pcs)

            if backtest.top_of_block.seek_candidates.TMP_REMOVE_ME_FOR_FIXUP_ONLY:
                # if there's no Balancer (v1 or v2) in the circuit, don't bother
                has_balancer = any(isinstance(x, (BalancerPricer, BalancerV2WeightedPoolPricer, BalancerV2LiquidityBootstrappingPoolPricer)) for x in item._circuit)
                if not has_balancer:
                    continue

            # generate a unique key for this circuit to ensure we don't have to explore it more than once
            # since the detector works both forward, backward, and in all rotations.
            # Since we only deal with cycles of length 3 or 2, we disambiguate rotations of a cycle by simply
            # sorting the items.
            k = []
            for p, (t_in, t_out) in zip(item._circuit, item._directions):
                t1, t2 = sorted([t_in, t_out])
                k.append((p.address, t1, t2))
            k = tuple(sorted(k))

            if k in circuits_considered:
                # duplicate, don't bother
                continue
            circuits_considered.add(k)

            elapsed += time.time() - t_start
            yield from detect_arbitrages_bisection(item, block_number, timestamp = timestamp, only_weth_pivot = only_weth_pivot)
        except StopIteration:
            break
    utils.profiling.inc_measurement('propose-circuit', elapsed)



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


def _propose_circuits_pair(
        pair: typing.Tuple[str, str],
        address: str,
        pool: pricers.PricerPool,
        block_number: int,
    ) -> typing.Iterator[PricingCircuit]:
    # There are several situations here.

    # First, we need to find out whether this exchange has WETH or not;
    # if it does, we can yield some length-2 circuits and use special logic to find 3-length circuits
    token0, token1 = pair
    if WETH_ADDRESS in pair:
        pricer_1 = pool.get_pricer_for(address)

        if not meets_thresholds(pricer_1, block_number):
            return

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
                pricer_2 = pool.get_pricer_for(other_exchange)
                # we made a 2-length circuit
                if not meets_thresholds(pricer_2, block_number):
                    continue

                yield PricingCircuit(
                    [
                        pricer_1,
                        pricer_2,
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

                        pricer_2 = pool.get_pricer_for(other_exchange)
                        if not meets_thresholds(pricer_2, block_number):
                            continue

                        pricer_3 = pool.get_pricer_for(last_exchange)
                        if not meets_thresholds(pricer_3, block_number):
                            continue

                        assert len(set([address, other_exchange, last_exchange])) == 3, 'should not have duplicates'
                        yield PricingCircuit(
                            [
                                pricer_1,
                                pricer_2,
                                pricer_3,
                            ],
                            [
                                (WETH_ADDRESS, other_token),
                                (other_token, other_token2),
                                (other_token2, WETH_ADDRESS),
                            ],
                        )
    else:
        pricer_2 = pool.get_pricer_for(address)
        if not meets_thresholds(pricer_2, block_number):
            return

        # WETH is not in this pair, so we can only have length-3 exchanges
        for exchange_1 in pool.get_exchanges_for_pair(WETH_ADDRESS, token0, block_number):
            
            pricer_1 = pool.get_pricer_for(exchange_1)
            if not meets_thresholds(pricer_1, block_number):
                continue

            for exchange_3 in pool.get_exchanges_for_pair(WETH_ADDRESS, token1, block_number):
                if address == exchange_1 or address == exchange_3 or exchange_1 == exchange_3:
                    # no dupes allowed
                    continue
                
                pricer_3 = pool.get_pricer_for(exchange_3)
                if not meets_thresholds(pricer_3, block_number):
                    continue

                yield PricingCircuit(
                    [
                        pricer_1,
                        pricer_2,
                        pricer_3,
                    ],
                    [
                        (WETH_ADDRESS, token0),
                        (token0, token1),
                        (token1, WETH_ADDRESS),
                    ],
                )

def meets_thresholds(pricer: BaseExchangePricer, block_identifier: int) -> bool:
    with profile('propose-circuit.meets_thresholds'):
        thresh_tokens = pricer.get_tokens(block_identifier).intersection(THRESHOLDS.keys())
        for t in thresh_tokens:
            bal = pricer.get_value_locked(t, block_identifier)
            if bal < THRESHOLDS[t]:
                return False
        return True
