"""
find_circuit/find.py

Finds a profitable arbitrage circuit given point-in-time params.
"""
import itertools
import math
import typing

import logging
import numpy as np
import scipy.optimize
from pricers.balancer import TooLittleInput
import pricers.token_transfer
from pricers.base import NotEnoughLiquidityException

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
    def directions(self) -> typing.List[typing.Tuple[str, str]]:
        return list(self._directions)

    def copy(self) -> 'PricingCircuit':
        return PricingCircuit(
            self.circuit,
            self.directions
        )

    def sample(self, amount_in: int, block_identifier: int, timestamp: typing.Optional[int] = None, debug = False) -> int:
        """
        Run the circuit with the given amount_in, returning the amount_out
        """
        last_token = self.pivot_token
        curr_amt = amount_in
        for p, (t_in, t_out) in zip(self._circuit, self._directions):
            assert last_token == t_in
            _last_amt = curr_amt
            curr_amt, _ = p.token_out_for_exact_in(t_in, t_out, curr_amt, block_identifier, timestamp=timestamp)
            last_token = t_out
            curr_amt = pricers.token_transfer.out_from_transfer(last_token, curr_amt)
            assert curr_amt >= 0, 'negative token balance is not possible'
            if debug:
                l.debug(f'{p.address} ({t_in} -> {t_out}) : {_last_amt} -> {curr_amt}')
        assert last_token == self.pivot_token
        return curr_amt

    def sample_new_price_ratio(self, amount_in: int, block_identifier: int, timestamp: typing.Optional[int] = None, debug = False) -> float:
        """
        Run the circuit with the given amount_in, returning the new marginal price of this circuit
        """
        # some tokens charge a fee when you transfer -- attempt to account for this
        # in a hacky way by sending in 10 ** 18 units, assume 1:1 conversion at
        # each exchange point, and seeing how much you would get out the other end.

        # NOTE this does not work if the fee is not a simple percentage

        quantized_transfer_fee = 10 ** 18

        last_token = self.pivot_token
        curr_amt = amount_in
        new_mp = 1.0

        for p, (t_in, t_out) in zip(self._circuit, self._directions):
            assert last_token == t_in
            _last_amt = curr_amt
            curr_amt, curr_mp = p.token_out_for_exact_in(t_in, t_out, curr_amt, block_identifier, timestamp=timestamp)

            last_token = t_out
            new_mp *= curr_mp
            curr_amt = pricers.token_transfer.out_from_transfer(last_token, curr_amt)
            quantized_transfer_fee = pricers.token_transfer.out_from_transfer(last_token, quantized_transfer_fee)

            assert curr_amt >= 0, 'negative token balance is not possible'
            if debug:
                l.debug(f'{p.address} ({t_in} -> {t_out}) : {_last_amt} -> {curr_amt} @ {curr_mp}')

        assert last_token == self.pivot_token

        transfer_fee = quantized_transfer_fee / 10 ** 18

        return new_mp * transfer_fee

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


# def detect_arbitrages(
#         pc: PricingCircuit,
#         block_identifier: int,
#         timestamp: typing.Optional[int] = None,
#         only_weth_pivot = False
#     ) -> typing.List[FoundArbitrage]:
#     """
#     KEPT FOR LEGACY REFERENCE -- use the variant _bisection below
#     """
#     l.error('USING legacy arbitrage detector!!!!')


#     ret = []

#     # for each rotation
#     for _ in range(len(pc._circuit)):
#         def run_exc(i):
#             try:
#                 return pc.sample(math.ceil(i), block_identifier, timestamp=timestamp) if i >= 0 else -(i + 1000) # encourage optimizer to explore toward +direction
#             except NotEnoughLiquidityException as e:
#                 # encourage going toward available liquidity
#                 return -(1000 + e.remaining)

#         # try each direction
#         for _ in range(2):

#             if not (only_weth_pivot and pc.pivot_token != WETH_ADDRESS):
#                 # quickly try pushing 100 tokens -- if unprofitable, fail
#                 with profile('pricing_quick_check'):
#                     quick_test_amount_in = 100
#                     try:
#                         for quick_test_amount_in_zeros in range(2, 22):
#                             quick_test_amount_in = 10 ** quick_test_amount_in_zeros
#                             try:
#                                 quick_test_pr = pc.sample_new_price_ratio(quick_test_amount_in, block_identifier, timestamp=timestamp)
#                                 break
#                             except TooLittleInput:
#                                 # try the next largest amount
#                                 continue
#                         else:
#                             # exhausted quick_test_amount_in options -- probably there's no way to pump enough liquidity
#                             # to this exchange just yet
#                             continue
#                     except NotEnoughLiquidityException:
#                         # not profitable most likely
#                         continue


#                 with profile('pricing_optimize'):
#                     if quick_test_pr > 1:
#                         # this may be profitable

#                         with profile('bounds_seek'):
#                             # search for crossing-point where liquidity does not run out
#                             lower_bound = quick_test_amount_in
#                             upper_bound = (100_000 * (10 ** 18)) # a shit-ton of ether
#                             try:
#                                 pc.sample(upper_bound, block_identifier, timestamp=timestamp)
#                             except NotEnoughLiquidityException as e:                                
#                                 # we need to adjust upper_bound down juuuust until it's in liquidity range
#                                 # do this by binary-search
#                                 search_lower = lower_bound
#                                 search_upper = upper_bound
#                                 while True:
#                                     # rapidly reduce upper bound by orders of 10
#                                     x = search_upper // 10
#                                     try:
#                                         pc.sample(x, block_identifier, timestamp=timestamp)
#                                         break
#                                     except NotEnoughLiquidityException:
#                                         search_upper = x

#                                 while search_lower < search_upper - 1:
#                                     midpoint = search_lower + (search_upper - search_lower) * 3 // 10
#                                     midpoint = (search_lower + search_upper) // 2
#                                     try:
#                                         pc.sample(midpoint, block_identifier)
#                                         search_lower = midpoint
#                                     except NotEnoughLiquidityException as e:
#                                         search_upper = midpoint
                                    
#                                 upper_bound = search_lower


#                         with profiling.profile('minimize_scalar'):
#                             result = scipy.optimize.minimize_scalar(
#                                 fun = lambda x: - (run_exc(x) - x),
#                                 bounds = (
#                                     lower_bound, # only a little
#                                     upper_bound, # a shit-ton
#                                 ),
#                                 method='bounded',
#                             )

#                         if result.fun < 0:
#                             amount_in = math.ceil(result.x)
#                             expected_profit = pc.sample(amount_in, block_identifier, timestamp=timestamp) - amount_in

#                             # if reducing input by 1 wei results in the same amount out, then use that value (rounding gets fucky)
#                             for i in itertools.count(1):
#                                 token_in, token_out = pc.directions[0]
#                                 out_normal = pc.circuit[0].token_out_for_exact_in(token_in, token_out, amount_in, block_identifier=block_identifier)
#                                 out_reduced_by_1 = pc.circuit[0].token_out_for_exact_in(token_in, token_out, amount_in - 1, block_identifier=block_identifier)

#                                 if out_normal == out_reduced_by_1:
#                                     amount_in -= 1
#                                     expected_profit += 1
#                                 else:
#                                     break

#                             # for i in itertools.count(1):
#                             #     trying_amount_in = amount_in - i
#                             #     expected_profit_new = pc.sample(trying_amount_in, block_identifier, timestamp=timestamp) - amount_in
#                             #     if expected_profit_new < expected_profit:
#                             #         break # profit going down, give up

#                             if expected_profit <= 0:
#                                 l.warning('fun indicated profit but expected profit did not!')
#                             else:
#                                 to_add = FoundArbitrage(
#                                     amount_in   = amount_in,
#                                     directions  = pc.directions,
#                                     circuit     = pc.circuit,
#                                     pivot_token = pc.pivot_token,
#                                     profit      = expected_profit,
#                                 )
#                                 ret.append(to_add)
#             pc.flip()
#         pc.rotate()
#     return ret


def detect_arbitrages_bisection(
        pc: PricingCircuit,
        block_identifier: int,
        timestamp: typing.Optional[int] = None,
        only_weth_pivot = False
    ) -> typing.List[FoundArbitrage]:
    ret = []

    # for each rotation
    for _ in range(len(pc._circuit)):
        def run_exc(i):
            amt_in = math.ceil(i)
            price_ratio = pc.sample_new_price_ratio(amt_in, block_identifier, timestamp=timestamp)
            return price_ratio - 1

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
                                quick_test_pr = pc.sample_new_price_ratio(quick_test_amount_in, block_identifier, timestamp=timestamp)
                                break
                            except TooLittleInput:
                                # try the next largest amount
                                continue
                        else:
                            # exhausted quick_test_amount_in options -- probably there's no way to pump enough liquidity
                            # to this exchange just yet
                            continue
                    except NotEnoughLiquidityException:
                        # not profitable most likely
                        continue
                
                with profile('pricing_optimize'):
                    if quick_test_pr > 1:
                        # this may be profitable

                        # search for crossing-point where liquidity does not run out
                        lower_bound = quick_test_amount_in
                        upper_bound = (100_000 * (10 ** 18)) # a shit-ton of ether
                        try:
                            pc.sample(upper_bound, block_identifier, timestamp=timestamp)
                        except NotEnoughLiquidityException:
                            # we need to adjust upper_bound down juuuust until it's in liquidity range
                            # do this by binary-search
                            search_lower = lower_bound
                            search_upper = upper_bound

                            while True:
                                # rapidly reduce upper bound by orders of 10
                                x = search_upper // 10
                                try:
                                    pc.sample(x, block_identifier, timestamp=timestamp)
                                    search_lower = max(search_lower, x)
                                    break
                                except NotEnoughLiquidityException:
                                    search_upper = x

                            while search_lower < search_upper - 1:
                                midpoint = (search_lower + search_upper) // 2
                                try:
                                    pc.sample(midpoint, block_identifier, timestamp=timestamp)
                                    search_lower = midpoint
                                except NotEnoughLiquidityException:
                                    search_upper = midpoint
                            upper_bound = search_lower

                        out_lower_bound = pc.sample(lower_bound, block_identifier, timestamp=timestamp)
                        out_upper_bound = pc.sample(upper_bound, block_identifier, timestamp=timestamp)

                        if out_upper_bound < 100:
                            # haven't managed to get anything out with the most money we can pump through, abandon
                            continue

                        if out_lower_bound < 100:
                            # search for crossing-point where (some) positive tokens come out of lower bound
                            lower_bound_search_upper = upper_bound
                            while lower_bound < lower_bound_search_upper - 100:
                                midpoint = (lower_bound + lower_bound_search_upper) // 2
                                midpoint_out = pc.sample(midpoint, block_identifier, timestamp=timestamp)
                                if midpoint_out < 100:
                                    lower_bound = midpoint
                                else:
                                    lower_bound_search_upper = midpoint
                                    out_lower_bound = midpoint_out
                            lower_bound = lower_bound_search_upper

                        assert lower_bound <= upper_bound, f'expect {lower_bound} <= {upper_bound}'

                        if out_lower_bound <= lower_bound:
                            # lower bound is already not profitable
                            continue

                        mp_lower_bound = pc.sample_new_price_ratio(lower_bound, block_identifier, timestamp=timestamp)
                        mp_upper_bound = pc.sample_new_price_ratio(upper_bound, block_identifier, timestamp=timestamp)

                        if mp_lower_bound < 1:
                            amount_in = lower_bound
                        elif mp_upper_bound > 1:
                            amount_in = upper_bound
                        else:
                            if mp_lower_bound <= mp_upper_bound:
                                # about to fail, dump info
                                for p, (t_in, t_out) in zip(pc.circuit, pc.directions):
                                    print(type(p).__name__, p.address, t_in, t_out)
                                with open('/mnt/goldphish/pts.txt', mode='w') as fout:
                                    for amt_in in np.linspace(lower_bound, upper_bound, 200):
                                        amt_in = int(np.ceil(amt_in))
                                        profit = pc.sample(amt_in, block_identifier, timestamp=timestamp) - amt_in
                                        price = pc.sample_new_price_ratio(amt_in, block_identifier, timestamp=timestamp, debug=True)
                                        fout.write(f'{amt_in},{profit},{price}\n')
                                print('lower_bound', lower_bound)
                                print('upper_bound', upper_bound)
                                print('mp_lower_bound', mp_lower_bound)
                                print('mp_upper_bound', mp_upper_bound)
                            assert mp_lower_bound > mp_upper_bound
                            assert 1 < mp_lower_bound

                            # the root (marginal price = 1) lies somewhere within the bounds
                            with profiling.profile('root_find'):
                                # guess is linear midpoint between the two
                                # (y - y1) = m (x - x1) solve for x where y = 1
                                # 1 - y1 = m (x - x1)
                                # (1 - y1) / m = x - x1
                                # (1 - y1) / m + x1 = x

                                m_inv = (lower_bound - upper_bound) / (mp_lower_bound - mp_upper_bound)
                                pt1 = (1 - mp_lower_bound) * m_inv + lower_bound

                                assert lower_bound <= pt1 <= upper_bound

                                # cast upper bound to float so we can ensure it is STRICTLY LESS THAN the int val
                                fl_upper_bound = float(upper_bound)
                                while math.ceil(fl_upper_bound) > upper_bound:
                                    fl_upper_bound *= 0.99999999999999

                                result = scipy.optimize.root_scalar(
                                    f = run_exc,
                                    bracket = (
                                        lower_bound,
                                        fl_upper_bound,
                                    ),
                                    x0 = pt1
                                )
                            amount_in = math.ceil(result.root)

                        expected_profit = pc.sample(amount_in, block_identifier, timestamp=timestamp) - amount_in

                        # quickly reduce input amount (optimizes for rounding)
                        input_reduction = 0
                        first_token_in, first_token_out = pc.directions[0]
                        first_out_normal, _ = pc.circuit[0].token_out_for_exact_in(first_token_in, first_token_out, amount_in, block_identifier=block_identifier)

                        for i in range(0, 21):
                            attempting_reduction = 10 ** i
                            if attempting_reduction >= amount_in:
                                break

                            try:
                                out_reduced, _ = pc.circuit[0].token_out_for_exact_in(first_token_in, first_token_out, amount_in - attempting_reduction, block_identifier=block_identifier)
                            except NotEnoughLiquidityException:
                                l.critical(f'Ran out of liquidity while sampling {amount_in - attempting_reduction} on {pc.circuit[0].address}')
                                raise

                            if first_out_normal == out_reduced:
                                input_reduction = attempting_reduction
                            else:
                                break
                        if input_reduction > 0:
                            # l.debug(f'Reduced input by {input_reduction} to optimize')
                            amount_in -= input_reduction
                            expected_profit += input_reduction

                        # for i in itertools.count(1):
                        #     trying_amount_in = amount_in - i
                        #     expected_profit_new = pc.sample(trying_amount_in, block_identifier, timestamp=timestamp) - amount_in
                        #     if expected_profit_new < expected_profit:
                        #         break # profit going down, give up

                        if expected_profit <= 0:
                            # for p, (t_in, t_out) in zip(pc.circuit, pc.directions):
                            #     print(type(p).__name__, p.address, t_in, t_out)
                            # with open('/mnt/goldphish/pts.txt', mode='w') as fout:
                            #     for amt_in in np.linspace(lower_bound, amount_in * 1.1, 200):
                            #         amt_in = int(np.ceil(amt_in))
                            #         profit = pc.sample(amt_in, block_identifier, timestamp=timestamp) - amt_in
                            #         price = pc.sample_new_price_ratio(amt_in, block_identifier, timestamp=timestamp, debug=True)
                            #         fout.write(f'{amt_in},{profit},{price}\n')
                            # pc.sample(amount_in, block_identifier, timestamp=timestamp, debug=True)
                            # l.warning('fun indicated profit but expected profit did not!')
                            pass
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

