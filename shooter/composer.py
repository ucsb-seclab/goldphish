"""
Given a descriptive series of an arbitrage flow, orders the exchanges properly
and synthesizes the input data for the shooter.
"""

import typing
import logging
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.liquidity_bootstrapping_pool import BalancerV2LiquidityBootstrappingPoolPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.base import BaseExchangePricer, NotEnoughLiquidityException
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer

import shooter.encoder
import pricers
import find_circuit.find
from utils import BALANCER_VAULT_ADDRESS, WETH_ADDRESS

l = logging.getLogger(__name__)


def construct_arbitrage(
        fa: find_circuit.find.FoundArbitrage,
        shooter_addr: str,
        block_identifier: int,
        fee_transfer_calculator: find_circuit.find.FeeTransferCalculator,
        timestamp: int = None,
    ) -> typing.Tuple[typing.List, typing.List[typing.Tuple[str, str]]]:
    ret = []
    approvals_required: typing.List[typing.Tuple[str, str]] = []

    amount_in = fa.amount_in

    for i, (p, (token_in, token_out)) in enumerate(zip(fa.circuit, fa.directions)):
        amount_out, _ = p.token_out_for_exact_in(
            token_in,
            token_out,
            amount_in,
            block_identifier,
            timestamp=timestamp
        )

        if isinstance(p, UniswapV2Pricer):
            ret.append(shooter.encoder.UniswapV2Swap(
                amount_in=None,
                amount_out=amount_out,
                exchange=p.address,
                to=None,
                zero_for_one=(bytes.fromhex(token_in[2:]) < bytes.fromhex(token_out[2:]))
            ))
        elif isinstance(p, UniswapV3Pricer):
            ret.append(shooter.encoder.UniswapV3Swap(
                amount_in=amount_in,
                exchange=p.address,
                to=[],
                zero_for_one=(bytes.fromhex(token_in[2:]) < bytes.fromhex(token_out[2:])),
                leading_exchanges=None,
                must_send_input=False
            ))
        elif isinstance(p, BalancerPricer):
            ret.append(shooter.encoder.BalancerV1Swap(
                amount_in=amount_in,
                exchange=p.address,
                token_in=token_in,
                token_out=token_out,
                to=None,
                requires_approval=False,
            ))
            approvals_required.append((p.address, token_in))
        elif isinstance(p, (BalancerV2WeightedPoolPricer, BalancerV2LiquidityBootstrappingPoolPricer)):
            ret.append(shooter.encoder.BalancerV2Swap(
                pool_id=p.pool_id,
                amount_in=amount_in,
                amount_out=amount_out,
                token_in=token_in,
                token_out=token_out,
                to=None
            ))
            approvals_required.append((BALANCER_VAULT_ADDRESS, token_in))

        if i + 1 < len(fa.circuit):
            if isinstance(fa.circuit[i + 1], (BalancerV2WeightedPoolPricer, BalancerV2LiquidityBootstrappingPoolPricer)):
                next_exchange_addr = BALANCER_VAULT_ADDRESS
            else:
                next_exchange_addr = fa.circuit[i + 1].address
        else:
            next_exchange_addr = shooter_addr

        if isinstance(p, (BalancerV2WeightedPoolPricer, BalancerV2LiquidityBootstrappingPoolPricer)):
            sender = BALANCER_VAULT_ADDRESS
        else:
            sender = p.address

        # If the next exchange is Balancer V1 or V2, or if the current exchange is Balancer V1, then we need to account for two fees:
        # (1) to self, (2) to next exchange
        if isinstance(p, BalancerPricer) or \
            (i + 1 < len(fa.circuit) and \
                isinstance(fa.circuit[i + 1], (BalancerPricer, BalancerV2WeightedPoolPricer, BalancerV2LiquidityBootstrappingPoolPricer))
            ):
            amount_in = fee_transfer_calculator.out_from_transfer(token_out, sender, shooter_addr, amount_out)
            amount_in = fee_transfer_calculator.out_from_transfer(token_out, shooter_addr, next_exchange_addr, amount_in)
        else:
            amount_in = fee_transfer_calculator.out_from_transfer(token_out, sender, next_exchange_addr, amount_out)


    if isinstance(ret[0], shooter.encoder.UniswapV2Swap):
        ret[0] = ret[0]._replace(amount_in = fa.amount_in)

    if isinstance(ret[0], shooter.encoder.UniswapV3Swap):
        ret[0]  = ret[0]._replace(must_send_input = True)

    ret[-1] = ret[-1]._replace(to = shooter_addr)

    for i in range(len(ret) - 1):
        p1 = ret[i]
        p2 = ret[i + 1]

        if isinstance(p2, (shooter.encoder.BalancerV1Swap, shooter.encoder.BalancerV2Swap)):
            ret[i] = p1._replace(to = shooter_addr)
        else:
            ret[i] = p1._replace(to = p2.exchange)


    gathered = _recurse_gather_uniswap_v3(ret, [])
    assert isinstance(gathered, list)
    assert len(gathered) > 0
    return gathered, approvals_required


def _recurse_gather_uniswap_v3(l, acc):
    if len(l) == 0:
        return acc

    if isinstance(l[0], shooter.encoder.UniswapV3Swap):
        uv3 = l[0]._replace(leading_exchanges=acc)
        return _recurse_gather_uniswap_v3(l[1:], [uv3])

    return _recurse_gather_uniswap_v3(l[1:], acc + [l[0]])
