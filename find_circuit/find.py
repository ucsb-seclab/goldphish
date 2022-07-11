"""
find_circuit/find.py

Finds a profitable arbitrage circuit given point-in-time params.
"""
import itertools
import math
import typing

import logging
import scipy.optimize
from pricers.balancer import TooLittleInput
import pricers.token_transfer
from pricers.uniswap_v3 import NotEnoughLiqudityException

import pricers.base
from utils import WETH_ADDRESS
from utils import profiling
from utils.profiling import profile

l = logging.getLogger(__name__)

class FoundArbitrage(typing.NamedTuple):
    amount_in: int
    circuit: typing.List[pricers.base.BaseExchangePricer]
    directions: typing.List[typing.Tuple[str, str]]
    pivot_token: str
    profit: int

    @property
    def tokens(self) -> typing.Set[str]:
        ret = set()
        for exc in self.circuit:
            ret.add(exc.token0)
            ret.add(exc.token1)
        return ret

    def __hash__(self) -> int:
        return hash((self.amount_in, tuple(self.circuit), tuple(self.directions), self.pivot_token, self.profit))

class PricingCircuit:
    _circuit: typing.List[pricers.base.BaseExchangePricer]
    _directions: typing.List[typing.Tuple[str, str]]

    def __init__(self, _circuit: typing.List[pricers.base.BaseExchangePricer], _directions: typing.List[typing.Tuple[str, str]]) -> None:
        assert len(_circuit) == len(_directions)
        self._circuit = _circuit
        self._directions = _directions

    @property
    def pivot_token(self) -> str:
        return self._directions[0][0]

    @property
    def circuit(self) -> typing.List[pricers.base.BaseExchangePricer]:
        return list(self._circuit)

    @property
    def directions(self) -> typing.List[bool]:
        return list(self._directions)

    def sample(self, amount_in: int, block_identifier: int) -> int:
        """
        Run the circuit with the given amount_in, returning the amount_out
        """
        last_token = self.pivot_token
        curr_amt = amount_in
        for p, (t_in, t_out) in zip(self._circuit, self._directions):
            assert last_token == t_in
            curr_amt = p.token_out_for_exact_in(t_in, t_out, curr_amt, block_identifier)
            curr_amt = pricers.token_transfer.out_from_transfer(last_token, curr_amt)
            assert curr_amt >= 0, 'negative token balance is not possible'
            last_token = t_out
        assert last_token == self.pivot_token
        return curr_amt

    def rotate(self):
        """
        Rotate the cycle once, to use a new pivot token
        """
        self._circuit = self._circuit[1:] + [self._circuit[0]]
        self._directions = self._directions[1:] + [self._directions[0]]

    def flip(self):
        """
        Flip the cycle in the alternate direction
        a -> b -> c  ==> c -> b -> a
        """
        self._circuit = list(reversed(self._circuit))
        self._directions = list((t2, t1) for (t1, t2) in reversed(self._directions))


def detect_arbitrages(
        pc: PricingCircuit,
        block_identifier: int,
        only_weth_pivot = False
    ) -> typing.List[FoundArbitrage]:
    ret = []

    # for each rotation
    for _ in range(len(pc.circuit)):
        def run_exc(i):
            try:
                return pc.sample(math.ceil(i), block_identifier) if i >= 0 else -(i + 1000) # encourage optimizer to explore toward +direction
            except NotEnoughLiqudityException as e:
                # encourage going toward available liquidity
                return -(1000 + e.remaining)

        # try each direction
        for _ in range(2):

            if not (only_weth_pivot and pc.pivot_token != WETH_ADDRESS):
                # quickly try pushing 100 tokens -- if unprofitable, fail
                with profile('pricing_quick_check'):
                    quick_test_amount_in = 100
                    try:
                        for quick_test_amount_in_zeros in range(2, 22):
                            quick_test_amount_in = 10 ** quick_test_amount_in_zeros
                            try:
                                quick_test_amount_out1 = pc.sample(quick_test_amount_in, block_identifier)
                                break
                            except TooLittleInput:
                                # try the next largest amount
                                continue
                        else:
                            # exhausted quick_test_amount_in options -- probably there's no way to pump enough liquidity
                            # to this exchange just yet
                            continue
                    except NotEnoughLiqudityException:
                        # not profitable most likely
                        continue


                with profile('pricing_optimize'):
                    if quick_test_amount_out1 > quick_test_amount_in:
                        # this may be profitable

                        # search for crossing-point where liquidity does not run out
                        lower_bound = quick_test_amount_in
                        upper_bound = (100_000 * (10 ** 18)) # a shit-ton of ether
                        try:
                            pc.sample(upper_bound, block_identifier)
                        except NotEnoughLiqudityException:
                            # we need to adjust upper_bound down juuuust until it's in liquidity range
                            # do this by binary-search
                            search_lower = lower_bound
                            search_upper = upper_bound
                            while search_lower < search_upper - 1:
                                midpoint = (search_lower + search_upper) // 2
                                try:
                                    pc.sample(midpoint, block_identifier)
                                    search_lower = midpoint
                                except NotEnoughLiqudityException:
                                    search_upper = midpoint
                            upper_bound = search_lower


                        with profiling.profile('minimize_scalar'):
                            result = scipy.optimize.minimize_scalar(
                                fun = lambda x: - (run_exc(x) - x),
                                bounds = (
                                    lower_bound, # only a little
                                    upper_bound, # a shit-ton
                                ),
                                method='bounded',
                            )


                        if result.fun < 0:
                            amount_in = math.ceil(result.x)
                            expected_profit = pc.sample(amount_in, block_identifier) - amount_in

                            # if reducing input by 1 wei results in the same amount out, then use that value (rounding gets fucky)
                            for i in itertools.count(1):
                                token_in, token_out = pc.directions[0]
                                out_normal = pc.circuit[0].token_out_for_exact_in(token_in, token_out, amount_in, block_identifier=block_identifier)
                                out_reduced_by_1 = pc.circuit[0].token_out_for_exact_in(token_in, token_out, amount_in - 1, block_identifier=block_identifier)

                                if out_normal == out_reduced_by_1:
                                    amount_in -= 1
                                    expected_profit += 1
                                else:
                                    break

                            for i in itertools.count(1):
                                trying_amount_in = amount_in - i
                                expected_profit_new = pc.sample(trying_amount_in, block_identifier) - amount_in
                                if expected_profit_new < expected_profit:
                                    break # profit going down, give up
                            if expected_profit <= 0:
                                l.warning('fun indicated profit but expected profit did not!')
                            else:
                                to_add = FoundArbitrage(
                                    amount_in   = amount_in,
                                    directions  = pc.directions,
                                    circuit     = pc.circuit,
                                    pivot_token = pc.pivot_token,
                                    profit      = expected_profit,
                                )
                                ret.append(to_add)
            pc.flip()
        pc.rotate()
    return ret
