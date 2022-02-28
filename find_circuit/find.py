"""
find_circuit/find.py

Finds a profitable arbitrage circuit
"""
import math
import typing

import scipy.optimize

import pricers.base

class FoundArbitrage(typing.NamedTuple):
    amount_in: int
    circuit: typing.List[typing.Tuple[pricers.base.BaseExchangePricer]]
    pivot_token: str
    profit: int


class PricingCircuit:
    _circuit: typing.List[pricers.base.BaseExchangePricer]
    _directions: typing.List[bool]

    def __init__(self, _circuit: typing.List[pricers.base.BaseExchangePricer]) -> None:
        self._circuit = _circuit
        _circuit_pairs = list(zip(_circuit, _circuit[1:] + [_circuit[0]]))

        # assume we start zeroForOne, if that doesn't hold
        # (this is only possible when circuit is len 3), then assume not zeroForOne
        _directions = [True]
        if _circuit_pairs[0][0].token1 not in [_circuit_pairs[0][1].token0, _circuit_pairs[0][1].token1]:
            assert _circuit_pairs[0][0].token0 in [_circuit_pairs[0][1].token0, _circuit_pairs[0][1].token1]
            _directions = [False]
        
        for i, (p1, p2) in list(enumerate(_circuit_pairs))[1:]:
            prev_exc = _circuit_pairs[i - 1][0]
            if _directions[i - 1] == True:
                # previous direction was zeroForOne
                prev_token = prev_exc.token1
            else:
                prev_token = prev_exc.token0

            if prev_token == p1.token0:
                next_token = p1.token1
                _directions.append(True)
            else:
                next_token = p1.token0
                _directions.append(False)

            assert next_token in [p2.token0, p2.token1]

        self._directions = _directions
        assert len(self._circuit) == len(self._directions)


    @property
    def pivot_token(self) -> str:
        if self._directions[0] == True:
            # zeroForOne for first exchange; zero token is pivot
            return self._circuit[0].token0
        else:
            return self._circuit[0].token1

    @property
    def circuit(self) -> typing.List[pricers.base.BaseExchangePricer]:
        return list(self._circuit)

    def sample(self, amount_in: int, block_identifier: int) -> int:
        """
        Run the circuit with the given amount_in, returning the amount_out
        """
        last_token = self.pivot_token
        curr_amt = amount_in
        for p, dxn in zip(self._circuit, self._directions):
            if dxn == True:
                # zeroForOne
                assert last_token == p.token0
                curr_amt = p.exact_token0_to_token1(amount_in, block_identifier)
                last_token = p.token1
            else:
                assert last_token == p.token1
                curr_amt = p.exact_token1_to_token0(amount_in, block_identifier)
                last_token = p.token0
        return curr_amt

    def rotate(self):
        """
        Rotate the cycle once, to use a new pivot token
        """
        self._circuit = self._circuit[1:] + [self._circuit[0]]
        self._directions = self._directions

    def flip(self):
        """
        Flip the cycle in the alternate direction
        a -> b -> c  ==> c -> b -> a
        """
        self._circuit = list(reversed(self._circuit))
        self._directions = list(not x for x in reversed(self._directions))


def detect_arbitrages(exchanges: typing.List[pricers.base.BaseExchangePricer], block_identifier: int) -> typing.List[FoundArbitrage]:
    assert len(exchanges) in [2, 3]

    # for all exchange pairs, ensure they share either (1) token (if len 3) or (2) tokens (if len 2)
    for a, b in zip(exchanges, exchanges[1:] + [exchanges[0]]):
        excs_a = set([a.token0, a.token1])
        excs_b = set([b.token0, b.token1])
        if len(exchanges) == 2:
            assert len(excs_a.intersection(excs_b)) == 2
        else:
            assert len(exchanges) == 3
            assert len(excs_a.intersection(excs_b)) == 1

    ret = []
    pc = PricingCircuit(exchanges)

    # for each rotation
    for _ in range(len(exchanges) - 1):
        run_exc = lambda i: pc.sample(math.ceil(i), block_identifier)
        # try each direction
        for _ in range(2):
            result = scipy.optimize.minimize_scalar(
                fun = lambda x: - (run_exc(x) - x),
                bounds = (
                    100, # only a little
                    (1_000 * (10 ** 18)) # a shit-ton
                ),
                method='bounded'
            )
            if result.fun < 0:
                # profit reported
                ret.append(
                    FoundArbitrage(
                        amount_in   = math.ceil(result.x),
                        circuit     = pc.circuit,
                        pivot_token = pc.pivot_token,
                        profit      = -result.fun,
                    )
                )
            pc.flip()
    return ret
