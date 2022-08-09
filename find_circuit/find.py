"""
find_circuit/find.py

Finds a profitable arbitrage circuit given point-in-time params.
"""
import math
import time
import typing

import logging
import numpy as np
import scipy.optimize
from pricers.balancer import TooLittleInput
import pricers.token_transfer
from pricers.base import NotEnoughLiquidityException

import pricers.base
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer
from utils import WETH_ADDRESS
from utils import profiling
from utils.profiling import profile, inc_measurement

l = logging.getLogger(__name__)

class FeeTransferCalculator:

    def out_from_transfer(self, token: str, from_: str, to_: str, amount: int) -> int:
        raise NotImplementedError()

class BuiltinFeeTransferCalculator(FeeTransferCalculator):

    def out_from_transfer(self, token: str, from_: str, to_: str, amount: int) -> int:
        return pricers.token_transfer.out_from_transfer(token, amount)

DEFAULT_FEE_TRANSFER_CALCULATOR: FeeTransferCalculator = BuiltinFeeTransferCalculator()

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

    def __str__(self) -> str:
        circuit_str = ', '.join(p.address for p in self.circuit)
        directions_str = ', '.join([x for x, _ in self.directions] + [self.directions[-1][1]])
        return f'<FoundArbitrage amount_in={self.amount_in} profit={self.profit} circuit=[{circuit_str}] directions=[{directions_str}]>'

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

    def sample(
            self,
            amount_in: int,
            block_identifier: int,
            timestamp: typing.Optional[int] = None,
            debug = False,
            fee_transfer_calculator: FeeTransferCalculator = DEFAULT_FEE_TRANSFER_CALCULATOR,
        ) -> int:
        """
        Run the circuit with the given amount_in, returning the amount_out
        """
        last_token = self.pivot_token
        curr_amt = amount_in
        for i, (p, (t_in, t_out)) in enumerate(zip(self._circuit, self._directions)):
            assert last_token == t_in
            _last_amt = curr_amt
            curr_amt, _ = p.token_out_for_exact_in(t_in, t_out, curr_amt, block_identifier, timestamp=timestamp)
            last_token = t_out

            if i + 1 < len(self._circuit):
                next_exchange_addr = self._circuit[i + 1].address
            else:
                next_exchange_addr = None

            curr_amt = fee_transfer_calculator.out_from_transfer(last_token, p.address, next_exchange_addr, curr_amt)

            assert curr_amt >= 0, 'negative token balance is not possible'
            if debug:
                l.debug(f'{p.address} ({t_in} -> {t_out}) : {_last_amt} -> {curr_amt}')
        assert last_token == self.pivot_token
        return curr_amt

    def sample_new_price_ratio(
            self,
            amount_in: int,
            block_identifier: int,
            timestamp: typing.Optional[int] = None,
            debug = False,
            fee_transfer_calculator: FeeTransferCalculator = DEFAULT_FEE_TRANSFER_CALCULATOR
        ) -> float:
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

        for i, (p, (t_in, t_out)) in enumerate(zip(self._circuit, self._directions)):
            assert last_token == t_in
            _last_amt = curr_amt
            curr_amt, curr_mp = p.token_out_for_exact_in(t_in, t_out, curr_amt, block_identifier, timestamp=timestamp)

            last_token = t_out
            new_mp *= curr_mp

            if i + 1 < len(self._circuit):
                next_exchange_addr = self._circuit[i + 1].address
            else:
                next_exchange_addr = None

            curr_amt = fee_transfer_calculator.out_from_transfer(last_token, p.address, next_exchange_addr, curr_amt)

            quantized_transfer_fee = fee_transfer_calculator.out_from_transfer(last_token, p.address, next_exchange_addr, quantized_transfer_fee)

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


def detect_arbitrages_bisection(
        pc: PricingCircuit,
        block_identifier: int,
        timestamp: typing.Optional[int] = None,
        only_weth_pivot = False,
        try_all_directions = True,
        fee_transfer_calculator: FeeTransferCalculator = DEFAULT_FEE_TRANSFER_CALCULATOR
    ) -> typing.List[FoundArbitrage]:
    ret = []

    t_start = time.time()

    # for each rotation
    for _ in range(len(pc._circuit) if try_all_directions else 1):
        def run_exc(i):
            amt_in = math.ceil(i)
            price_ratio = pc.sample_new_price_ratio(amt_in, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
            return price_ratio - 1

        # try each direction
        for _ in range(2 if try_all_directions else 1):

            if not (only_weth_pivot and pc.pivot_token != WETH_ADDRESS):
                # quickly try pushing 100 tokens -- if unprofitable, fail

                with profile('pricing.quick_check'):
                    try:
                        for quick_test_amount_in_zeros in range(5, 25): # start quick test at about 10^-10 dollars (July '22)
                            quick_test_amount_in = 10 ** quick_test_amount_in_zeros
                            try:
                                quick_test_pr = pc.sample_new_price_ratio(quick_test_amount_in, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
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

                with profile('pricing.optimize'):
                    if quick_test_pr > 1:
                        # this may be profitable

                        # search for crossing-point where liquidity does not run out
                        lower_bound = quick_test_amount_in
                        upper_bound = (100_000 * (10 ** 18)) # a shit-ton of ether

                        with profile('pricing.opti6mize.bounds.upper'):
                            upper_bound = find_upper_bound(pc, lower_bound, upper_bound, block_identifier, fee_transfer_calculator, timestamp=timestamp)

                        out_lower_bound = pc.sample(lower_bound, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                        out_upper_bound = pc.sample(upper_bound, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)

                        if out_upper_bound < 100:
                            # haven't managed to get anything out with the most money we can pump through, abandon
                            continue

                        if out_lower_bound < 100:
                            # search for crossing-point where (some) positive tokens come out of lower bound
                            lower_bound_search_upper = upper_bound
                            with profile('pricing.optimize.bounds.lower'):
                                while lower_bound < lower_bound_search_upper - 1000:
                                    midpoint = (lower_bound + lower_bound_search_upper) // 2
                                    midpoint_out = pc.sample(midpoint, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                                    if midpoint_out < 100:
                                        lower_bound = midpoint
                                    else:
                                        lower_bound_search_upper = midpoint
                                        out_lower_bound = midpoint_out
                                lower_bound = lower_bound_search_upper

                        assert lower_bound <= upper_bound, f'expect {lower_bound} <= {upper_bound}'

                        # NOTE: it may be the case here that out_lower_bound - lower_bound < 0
                        # i.e, the lower bound is not profitable. This can occur if there is significant
                        # input required to get the first units of output produced -- those are essentially a flat
                        # fee which pushes the pricing "parabola" downward

                        mp_lower_bound = pc.sample_new_price_ratio(lower_bound, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                        mp_upper_bound = pc.sample_new_price_ratio(upper_bound, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)

                        if mp_lower_bound < 1:
                            amount_in = lower_bound
                        elif mp_upper_bound > 1:
                            amount_in = upper_bound
                        else:
                            if mp_lower_bound <= mp_upper_bound:
                                # about to fail, dump info
                                for p, (t_in, t_out) in zip(pc.circuit, pc.directions):
                                    l.critical(type(p).__name__, p.address, t_in, t_out)
                                with open('/mnt/goldphish/pts.txt', mode='w') as fout:
                                    for amt_in in np.linspace(lower_bound, upper_bound, 200):
                                        amt_in = int(np.ceil(amt_in))
                                        profit = pc.sample(amt_in, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator) - amt_in
                                        price = pc.sample_new_price_ratio(amt_in, block_identifier, timestamp=timestamp, debug=True, fee_transfer_calculator=fee_transfer_calculator)
                                        fout.write(f'{amt_in},{profit},{price}\n')
                                l.critical(f'lower_bound {lower_bound}')
                                l.critical(f'upper_bound {upper_bound}')
                                l.critical(f'mp_lower_bound {mp_lower_bound}')
                                l.critical(f'mp_upper_bound {mp_upper_bound}')
                            assert mp_lower_bound > mp_upper_bound
                            assert 1 < mp_lower_bound

                            # the root (marginal price = 1) lies somewhere within the bounds
                            with profiling.profile('pricing.optimize.root_find'):
                                # guess is linear midpoint between the two
                                # (y - y1) = m (x - x1) solve for x where y = 1
                                # 1 - y1 = m (x - x1)
                                # (1 - y1) / m = x - x1
                                # (1 - y1) / m + x1 = x

                                # cast upper bound to float so we can ensure it is STRICTLY LESS THAN the int val
                                fl_upper_bound = float(upper_bound)
                                while math.ceil(fl_upper_bound) > upper_bound:
                                    fl_upper_bound *= 0.99999999999999

                                fl_lower_bound = float(lower_bound)
                                while math.ceil(fl_lower_bound) < lower_bound:
                                    fl_lower_bound *= 1.00000000000001

                                if fl_lower_bound < fl_upper_bound:
                                    try:
                                        result = scipy.optimize.root_scalar(
                                            f = run_exc,
                                            bracket = (
                                                fl_lower_bound,
                                                fl_upper_bound,
                                            ),
                                        )
                                        amount_in = math.ceil(result.root)
                                    except ValueError:
                                        # probably the upper bound is juuuuust above 1 -- use that as amount_in
                                        mp_fl_upper_bound = pc.sample_new_price_ratio(math.ceil(fl_upper_bound), block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                                        if mp_fl_upper_bound > 1:
                                            amount_in = math.ceil(mp_fl_upper_bound)
                                        else:
                                            # this should not happen, log generously if it does
                                            mp_fl_lower_bound = pc.sample_new_price_ratio(math.ceil(fl_lower_bound), block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)

                                            l.critical('about to fail')
                                            for p in pc._circuit:
                                                l.critical(str(p))
    
                                            l.critical(f"fl_upper_bound {fl_upper_bound}")
                                            l.critical(f'fl_lower_bound {fl_lower_bound}')
                                            l.critical(f'mp_lower_bound {mp_fl_lower_bound}')
                                            l.critical(f'mp_upper_bound {mp_fl_upper_bound}')
                                            l.critical(f'upper_bound {upper_bound}')
                                            l.critical(f'lower_bound {lower_bound}')
                                            l.critical(f'block_identifier {block_identifier}')
                                            raise
                                else:
                                    l.warning('fl_lower_bound crossed fl_upper_bound')
                                    amount_in = math.ceil(fl_lower_bound)

                        expected_profit = pc.sample(amount_in, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator) - amount_in

                        # quickly reduce input amount (optimizes for rounding)
                        input_reduction = 0
                        first_token_in, first_token_out = pc.directions[0]
                        first_out_normal, _ = pc.circuit[0].token_out_for_exact_in(first_token_in, first_token_out, amount_in, block_identifier=block_identifier)

                        with profile('pricing.optimize.reduce_input'):
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

                            amount_in -= input_reduction
                            expected_profit += input_reduction

                        if expected_profit > 0:
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

    if all(isinstance(x, UniswapV2Pricer) for x in pc._circuit):
        inc_measurement(f'optimize_uv2_{len(pc._circuit)}', time.time() - t_start)

    return ret


def find_upper_bound_binary_search(
        pc: PricingCircuit,
        lower_bound: int,
        upper_bound: int,
        block_identifier: int,
        fee_transfer_calculator: FeeTransferCalculator,
        timestamp: typing.Optional[int] = None,
    ) -> int:
    """
    Use binary search to find the maximum amount of input amount that can be put through this circuit.

    May also return a value lower than the maximum amount if it found to have marginal price below 1, as
    this is also a binidng upper bound for arbitrage search.
    """
    if lower_bound == upper_bound:
        return lower_bound

    assert lower_bound < upper_bound

    t_start = time.time()

    try:
        pc.sample(upper_bound, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
    except NotEnoughLiquidityException:
        # we need to adjust upper_bound down juuuust until it's in liquidity range
        # do this by binary-search
        search_lower = lower_bound
        search_upper = upper_bound

        while True:
            # rapidly reduce upper bound by orders of 10
            x = search_upper // 10
            try:
                pc.sample(x, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                search_lower = max(search_lower, x)
                break
            except NotEnoughLiquidityException:
                search_upper = x

        while search_lower < search_upper - 1000:
            midpoint = (search_lower + search_upper) // 2
            try:
                pr = pc.sample_new_price_ratio(midpoint, block_identifier, timestamp=timestamp, fee_transfer_calculator=fee_transfer_calculator)
                search_lower = midpoint
                if pr < 0.9:
                    # we can exit the search early because marginal price is not optimal
                    # at this bound already, no need to further refine
                    break
            except NotEnoughLiquidityException:
                search_upper = midpoint
        upper_bound = search_lower

    elapsed = time.time() - t_start
    inc_measurement('pricing.optimize.bounds.upper.binary_search', elapsed)
    return upper_bound

def find_upper_bound(
        pc: PricingCircuit,
        lower_bound: int,
        upper_bound: int,
        block_identifier: int,
        fee_transfer_calculator: FeeTransferCalculator,
        timestamp: typing.Optional[int] = None,
        last_sticking_point = None,
    ) -> int:
    """
    Complex upper-bound finder.

    Whenever possible, uses reverse-flow to find the correct amount_in.

    lower_bound _must_ be low enough to not trigger out of liquidity error
    """
    if lower_bound == upper_bound:
        return lower_bound

    assert lower_bound < upper_bound

    if not all(isinstance(p, (UniswapV2Pricer, UniswapV3Pricer)) for p in pc._circuit):
        return find_upper_bound_binary_search(pc, lower_bound, upper_bound, block_identifier, fee_transfer_calculator, timestamp = timestamp)

    # we may be able to use new method, attempt to run upper_bound and find the first exchange
    # that trips up on it

    t_start = time.time()

    curr_amt = upper_bound
    last_token = pc.pivot_token
    for pricer_idx, (p, (t_in, t_out)) in enumerate(zip(pc._circuit, pc._directions)):
        assert last_token == t_in

        try:
            amt_out, _ = p.token_out_for_exact_in(
                t_in,
                t_out,
                curr_amt,
                block_identifier=block_identifier,
                timestamp=timestamp
            )
        except NotEnoughLiquidityException as not_enough_liq_exc:
            too_much_in  = not_enough_liq_exc.amount_in
            remaining_in = not_enough_liq_exc.remaining
            break

        if pricer_idx + 1 < len(pc._circuit):
            next_exchange_addr = pc._circuit[pricer_idx + 1].address
        else:
            next_exchange_addr = None

        last_token = t_out
        curr_amt = fee_transfer_calculator.out_from_transfer(last_token, p.address, next_exchange_addr, amt_out)
    else:
        assert last_token == pc.pivot_token
        return upper_bound

    if last_sticking_point is not None and last_sticking_point >= pricer_idx:
        # this failed, we're stuck at the same point as last recursive iteration.
        # log enough info to diangose and fail early.
        l.critical('ABOUT TO FAIL')
        l.critical(f'pricer_idx   {pricer_idx}')
        l.critical(f'block_number {block_identifier}')
        l.critical(f'lower_bound  {lower_bound}')
        l.critical(f'upper_bound  {upper_bound}')
        l.critical(f'timestamp    {timestamp}')
        l.critical(f'too_much_in  {too_much_in}')
        l.critical(f'remaining_in {remaining_in}')
        for p in pc._circuit:
            l.critical(str(p))
        pc.sample(upper_bound, block_identifier, timestamp=timestamp, debug=True, fee_transfer_calculator=fee_transfer_calculator)
        raise Exception(f'failed')

    reverse_amt = (too_much_in - remaining_in)
    reverse_last_token = pc._directions[pricer_idx][0]
    if pricer_idx > 0:
        sender = pc._circuit[pricer_idx - 1].address
        recipient = pc._circuit[pricer_idx].address
        reverse_amt = crude_in_for_out_transfer_round_down(reverse_last_token, sender, recipient, reverse_amt, fee_transfer_calculator)

    assert reverse_amt >= 0

    for p, (t_in, t_out) in reversed(list(zip(pc._circuit[:pricer_idx], pc._directions[:pricer_idx]))):
        p: typing.Union[UniswapV2Pricer, UniswapV3Pricer]
        assert reverse_last_token == t_out
        assert reverse_amt >= 0

        zero_for_one = bytes.fromhex(t_in[2:]) < bytes.fromhex(t_out[2:])

        # we need to find the correct amount to put in WITHOUT EXCEEDING target output amount

        if zero_for_one:
            amt_in = p.token1_out_to_exact_token0_in(reverse_amt, block_identifier)
        else:
            amt_in = p.token0_out_to_exact_token1_in(reverse_amt, block_identifier)

        out_for_in, _ = p.token_out_for_exact_in(t_in, t_out, amt_in, block_identifier, timestamp=timestamp)
        while out_for_in > reverse_amt:
            # we need to gradually reduce input until its within limits
            # (this typically runs at most once)
            amt_in = max(0, min(amt_in - 100, amt_in * 995 // 1_000))
            out_for_in, _ = p.token_out_for_exact_in(t_in, t_out, amt_in, block_identifier, timestamp=timestamp)

        # crude attempt at reversing amount out for in
        idx = pc._circuit.index(p)
        if idx > 0:
            sender = pc._circuit[idx - 1].address
            recipient = p.address
            reverse_amt = crude_in_for_out_transfer_round_down(t_in, sender, recipient, amt_in, fee_transfer_calculator)
        else:
            reverse_amt = amt_in
        reverse_last_token = t_in

    reverse_amt = max(lower_bound, reverse_amt)

    assert lower_bound <= reverse_amt, f'expected {lower_bound} <= {reverse_amt} for exchanges {[p.address for p in pc._circuit]} block {block_identifier}'
    assert reverse_amt <= upper_bound, f'expected {reverse_amt} < {upper_bound} for exchanges {[p.address for p in pc._circuit]} block {block_identifier}'

    elapsed = time.time() - t_start
    inc_measurement('pricing.optimize.bounds.upper.new', elapsed)

    # call once again because we may need to blow past another blocking exchange
    return find_upper_bound(pc, lower_bound, reverse_amt, block_identifier, fee_transfer_calculator, timestamp=timestamp, last_sticking_point=pricer_idx)

def crude_in_for_out_transfer_round_down(
        token: str,
        from_: str,
        to: str,
        amount_out: int,
        fee_transfer_calculator: FeeTransferCalculator,
    ):
    assert amount_out >= 0
    ratio = fee_transfer_calculator.out_from_transfer(token, from_, to, 10 ** 10)
    ret = amount_out * (10 ** 10) // ratio
    ret_out = fee_transfer_calculator.out_from_transfer(token, from_, to, ret)

    n_subs = 0
    while ret_out > amount_out:
        # sometimes fucked due to rounding, that's okay
        ret = ret - 1
        ret_out = fee_transfer_calculator.out_from_transfer(token, from_, to, ret)
        n_subs += 1
        assert n_subs < 5, f'expect n_subs to be low for address={token} amount_out={amount_out}'

    return ret
