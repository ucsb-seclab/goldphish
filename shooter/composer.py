"""
Given a descriptive series of an arbitrage flow, orders the exchanges properly
and synthesizes the input data for the shooter.
"""

import typing
import logging
from pricers.base import NotEnoughLiquidityException

import shooter.encoder
import pricers
import find_circuit.find

l = logging.getLogger(__name__)


class ExchangeRecord(typing.NamedTuple):
    is_uniswap_v2: bool
    address: str
    amount_in: int
    amount_out: int
    token_in: str
    token_out: str

    @property
    def zero_for_one(self) -> bool:
        return bytes.fromhex(self.token_in[2:]) < bytes.fromhex(self.token_out[2:])

    @property
    def amount_out_less_token_fee(self) -> int:
        return pricers.out_from_transfer(self.token_out, self.amount_out)

class ConstructionException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class NotEnoughOutException(ConstructionException):
    """
    When the arbitrage doesn't have enough output
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CouldSaveOneWeiException(ConstructionException):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def construct_from_found_arbitrage(fa: find_circuit.find.FoundArbitrage, coinbase_xfer: int, target_block: int):
    """
    Simple wrapper over construct() which transforms the input into expected named tuple.
    """
    assert len(fa.circuit) == len(fa.directions)

    exchanges: typing.List[ExchangeRecord] = []

    last_out = fa.amount_in
    # this should be taken care-of earlier, but check for it now.
    # you might be able to save 1 wei
    if fa.directions[0] == True:
        assert fa.circuit[0].token0 == WETH_ADDRESS
        out_normal = fa.circuit[0].exact_token0_to_token1(fa.amount_in, block_identifier=target_block - 1)
        out_reduced_by_1 = fa.circuit[0].exact_token0_to_token1(fa.amount_in - 1, block_identifier=target_block - 1)
    else:
        assert fa.circuit[0].token1 == WETH_ADDRESS
        out_normal = fa.circuit[0].exact_token1_to_token0(fa.amount_in, block_identifier=target_block - 1)
        out_reduced_by_1 = fa.circuit[0].exact_token1_to_token0(fa.amount_in - 1, block_identifier=target_block - 1)

    if out_normal == out_reduced_by_1:
        raise CouldSaveOneWeiException('could save 1 wei by reducing amount_in by 1')

    try:
        for dxn, p in zip(fa.directions, fa.circuit):
            if dxn == True:
                # zeroForOne
                amt_out = p.exact_token0_to_token1(last_out, target_block - 1)
                token_in = p.token0
                token_out = p.token1
            else:
                assert dxn == False
                amt_out = p.exact_token1_to_token0(last_out, target_block - 1)
                token_in = p.token1
                token_out = p.token0
            # if p.address == '0x7289bA7C7B3d82FaF5D8800EE9dD2Fa7ABA918C3':
            #     fresh_pricer = pricers.UniswapV3Pricer(p.w3, p.address, p.token0, p.token1, p.fee)
            #     if dxn == True:
            #         new_amt_out = fresh_pricer.exact_token0_to_token1(last_out, target_block - 1)
            #     else:
            #         new_amt_out = fresh_pricer.exact_token1_to_token0(last_out, target_block - 1)
            #     if new_amt_out == amt_out:
            #         l.info('amount_outs match')
            #     else:
            #         l.error(f'amount_outs do not match, got {new_amt_out}')
            #     if dxn == True:
            #         alt_amt_out = p.exact_token0_to_token1(1010959910380, target_block - 1)
            #     else:
            #         alt_amt_out = p.exact_token1_to_token0(1010959910380, target_block - 1)
            #     l.info(f'expected amt_out from 1010959910380 in is {alt_amt_out} (expected {new_amt_out} previously)')
            exchanges.append(ExchangeRecord(
                is_uniswap_v2 = isinstance(p, pricers.uniswap_v2.UniswapV2Pricer),
                address = p.address,
                amount_in = last_out,
                amount_out = amt_out,
                token_in = token_in,
                token_out = token_out,
            ))
            last_out = pricers.out_from_transfer(token_out, amt_out)
        if last_out <= fa.amount_in:
            raise NotEnoughOutException('arbitrage did not result in profit')
    except NotEnoughLiquidityException:
        l.warning('Not enough liquidity to construct this....')
        raise ConstructionException('not enough liquidity')

    # see if we can take advantage of rounding error to save 1 wei

    return construct(exchanges, coinbase_xfer, target_block)    


def construct(
        exchanges: typing.List[ExchangeRecord],
        coinbase_xfer: int,
        target_block: int,
    ):
    assert 2 <= len(exchanges) <= 3
    assert 0 <= coinbase_xfer <= MAX_COINBASE_XFER

    for e in exchanges:
        l.debug(e)

    univ3_chain: typing.List[ExchangeRecord] = []
    univ2_chain: typing.List[ExchangeRecord] = []
    for exchange in exchanges:
        if not exchange.is_uniswap_v2:
            if len(univ3_chain) == 0:
                univ3_chain.append(exchange)
            else:
                for i, other_exchange in enumerate(univ3_chain):
                    if other_exchange.token_in == exchange.token_out:
                        # insert after other
                        univ3_chain.insert(i+1, exchange)
                        break
                    elif other_exchange.token_out == exchange.token_in:
                        # insert before other
                        univ3_chain.insert(i, exchange)
                        break
                else:
                    raise Exception('unreachable')
        else:
            if len(univ2_chain) == 0:
                univ2_chain.append(exchange)
            else:
                assert len(univ2_chain) == 1
                if univ2_chain[0].token_out == exchange.token_in:
                    # insert after
                    univ2_chain.append(exchange)
                elif univ2_chain[0].token_in == exchange.token_out:
                    # insert before
                    univ2_chain.insert(0, exchange)
                else:
                    raise Exception('unreachable')

    if len(univ2_chain) > 0:
        # sanity check -- uniswap v2s must pay for each other
        assert univ2_chain[-1].token_out == univ3_chain[-1].token_in
        assert univ2_chain[-1].amount_out_less_token_fee >= univ3_chain[-1].amount_in
        # sanity check -- the token input should be available from the first v3 exchange
        assert univ3_chain[0].token_out == univ2_chain[0].token_in

    # sanity check
    assert all(not x.is_uniswap_v2 for x in univ3_chain)
    assert all(x.is_uniswap_v2 for x in univ2_chain)
    assert len(set(univ3_chain).union(univ2_chain)) == len(univ3_chain) + len(univ2_chain), 'should not have duplicates'
    assert len(univ3_chain) + len(univ2_chain) == len(exchanges), 'no exchange left behind'

    excs = []

    # Handle each situation separately, I suppose. This makes the logic perhaps a bit more readable (but repeated).
    print('univ3_chain', univ3_chain)
    print('univ2_chain', univ2_chain)

    if len(univ3_chain) == 1 and len(univ2_chain) == 1:
        if univ3_chain[0].amount_out_less_token_fee > univ2_chain[0].amount_in:
            # SITUATION
            #   (uv3) -- profit --> (uv2) --> (uv3)
            #
            #   uv3 -> self
            #     self -> uv2
            #     uv2 -> uv3
            l.debug('Cycle situation 1')
            excs = [
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[0].address,
                    amount_in = univ3_chain[0].amount_in,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.SHOOTER,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[0].address,
                    amount_in_explicit = univ2_chain[0].amount_in,
                    amount_out = univ2_chain[0].amount_out,
                    zero_for_one = univ2_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.MSG_SENDER,
                )
            ]
        else:
            l.debug('Cycle situation 2')
            # SITUATION
            #   (uv3) --> (uv2) -- profit --> (uv3)
            #
            #   uv3 -> self
            #     self -> uv2
            #     uv2 -> uv3
            assert univ2_chain[0].amount_out_less_token_fee > univ3_chain[0].amount_in
            excs = [
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[0].address,
                    amount_in = univ3_chain[0].amount_in,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[0].address,
                    amount_in_explicit = 0,
                    amount_out = univ2_chain[0].amount_out,
                    zero_for_one = univ2_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.SHOOTER,
                )
            ]
    elif len(univ3_chain) == 1 and len(univ2_chain) == 2:
        if univ3_chain[0].amount_out_less_token_fee > univ2_chain[0].amount_in:
            l.debug('Cycle situation 3')
            # SITUATION
            #   (uv3) -- profit --> (uv2a) --> (uv2b) --> (uv3)
            #
            #   uv3 -> self
            #     self -> uv2a
            #     uv2a -> uv2b
            #     uv2b -> uv3
            excs = [
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[0].address,
                    amount_in = univ3_chain[0].amount_in,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.SHOOTER,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[0].address,
                    amount_in_explicit = univ2_chain[0].amount_in,
                    amount_out = univ2_chain[0].amount_out,
                    zero_for_one = univ2_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[1].address,
                    amount_in_explicit = 0,
                    amount_out = univ2_chain[1].amount_out,
                    zero_for_one = univ2_chain[1].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.MSG_SENDER,
                )
            ]
        elif univ2_chain[0].amount_out_less_token_fee > univ2_chain[1].amount_in:
            l.debug('Cycle situation 4')
            # SITUATION
            #   (uv3) --> (uv2a) -- profit --> (uv2b) --> (uv3)
            #
            #   uv3 -> uv2a
            #     uv2a -> self
            #     self -> uv2b
            #     uv2b -> uv3
            excs = [
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[0].address,
                    amount_in = univ3_chain[0].amount_in,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[0].address,
                    amount_in_explicit = 0,
                    amount_out = univ2_chain[0].amount_out,
                    zero_for_one = univ2_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.SHOOTER,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[1].address,
                    amount_in_explicit = univ2_chain[1].amount_in,
                    amount_out = univ2_chain[1].amount_out,
                    zero_for_one = univ2_chain[1].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.MSG_SENDER,
                )
            ]
        else:
            assert univ2_chain[1].amount_out_less_token_fee > univ3_chain[0].amount_in
            l.debug('Cycle situation 5')
            # SITUATION
            #   (uv3) --> (uv2a) --> (uv2b) -- profit --> (uv3)
            #
            #   uv3 -> uv2a
            #     uv2a -> uv2b
            #     uv2b -> self
            #     self -> uv3
            excs = [
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[0].address,
                    amount_in = univ3_chain[0].amount_in,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[0].address,
                    amount_in_explicit = 0,
                    amount_out = univ2_chain[0].amount_out,
                    zero_for_one = univ2_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[1].address,
                    amount_in_explicit = 0,
                    amount_out = univ2_chain[1].amount_out,
                    zero_for_one = univ2_chain[1].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.SHOOTER,
                ),
            ]
    elif len(univ3_chain) == 2 and len(univ2_chain) == 1:
        if univ3_chain[0].amount_in < univ3_chain[1].amount_out_less_token_fee:
            l.debug('Cycle situation 6')
            # SITUATION
            #   (uv3_1) -- profit --> (uv3_0) --> (uv2) --> (uv3_1)
            #
            #   uv3_0 -> uv2
            #     uv3_1 -> self
            #       uv2 -> uv3_1
            #     self -> uv3_0
            excs = [
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[0].address,
                    amount_in = univ3_chain[0].amount_in,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE, # here, means 'next uniswap v2 exchange'
                ),
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[1].address,
                    amount_in = univ3_chain[1].amount_in,
                    zero_for_one = univ3_chain[1].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.SHOOTER,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[0].address,
                    amount_in_explicit = 0,
                    amount_out = univ2_chain[0].amount_out,
                    zero_for_one = univ2_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.MSG_SENDER,
                ),
            ]
        elif univ3_chain[0].amount_out_less_token_fee > univ2_chain[0].amount_in:
            l.debug('Cycle situation 7')
            # SITUATION
            #   (uv3_1) --> (uv3_0) -- profit --> (uv2) --> (uv3_1)
            #
            #   uv3_0 -> self
            #     uv3_1 -> uv3_0
            #       self -> uv2
            #       uv2 -> uv3_1
            excs = [
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[0].address,
                    amount_in = univ3_chain[0].amount_in,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.SHOOTER,
                ),
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[1].address,
                    amount_in = univ3_chain[1].amount_in,
                    zero_for_one = univ3_chain[1].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.MSG_SENDER,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[0].address,
                    amount_in_explicit = univ2_chain[0].amount_in,
                    amount_out = univ2_chain[0].amount_out,
                    zero_for_one = univ2_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.MSG_SENDER,
                ),
            ]
        else:
            assert univ2_chain[0].amount_out_less_token_fee > univ3_chain[1].amount_in
            l.debug('Cycle situation 8')
            # SITUATION
            #   (uv3_1) --> (uv3_0) --> (uv2) -- profit --> (uv3_1)
            #
            #   uv3_0 -> uv2
            #     uv3_1 -> uv3_0
            #       uv2 -> self
            #       self -> uv3_1
            excs = [
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[0].address,
                    amount_in = univ3_chain[0].amount_in,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE,
                ),
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[1].address,
                    amount_in = univ3_chain[1].amount_in,
                    zero_for_one = univ3_chain[1].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.MSG_SENDER,
                ),
                shooter.encoder.UniswapV2Record(
                    address = univ2_chain[0].address,
                    amount_in_explicit = 0,
                    amount_out = univ2_chain[0].amount_out,
                    zero_for_one = univ2_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.SHOOTER,
                ),
            ]
    else:
        assert 2 <= len(univ3_chain) <= 3
        assert len(univ2_chain) == 0
        l.debug(f'Cycle situation 9')
        # fortunately this is the easy one
        # SITUATION
        # (uv3_2) --> (uv3_1) --> (uv3_0) --> (uv3_2) // implicit profit possible along any of these
        
        # rotate so that profit is at the beginning
        while not (univ3_chain[0].amount_out_less_token_fee > univ3_chain[-1].amount_in):
            univ3_chain = univ3_chain[1:] + univ3_chain[:1]

        # ensure invariant(s) hold
        for ex1, ex2 in zip(univ3_chain, (univ3_chain[1:] + univ3_chain[:1])):
            assert ex1.token_in == ex2.token_out
        assert univ3_chain[0].amount_out_less_token_fee > univ3_chain[-1].amount_in

        # NORMALIZED SITUATION
        # (uv3_2) --> (uv3_1) --> (uv3_0) -- profit --> (uv3_2)
        #
        #   uv3_0 -> self
        #     uv3_1 -> uv3_0
        #       uv3_2 -> uv3_1
        #         self -> uv3_2

        for i, exchange in list(enumerate(univ3_chain)):
            assert exchange.token_out == univ3_chain[i-1].token_in
            if exchange.amount_out_less_token_fee > univ3_chain[i-1].amount_in:
                # we're taking profit, must send to self
                recipient = shooter.encoder.FundsRecipient.SHOOTER
            else:
                # we're not taking profit, pay previous exchange directly
                recipient = shooter.encoder.FundsRecipient.MSG_SENDER
            excs.append(shooter.encoder.UniswapV3Record(
                address = exchange.address,
                amount_in = exchange.amount_in,
                zero_for_one = exchange.zero_for_one,
                recipient = recipient
            ))

    uv3_flash_loaner = next(filter(lambda x: x.address == excs[0].address, univ3_chain))

    l.debug(str(excs))
    l.debug(f'in_amount {uv3_flash_loaner.amount_in}')

    encoded = shooter.encoder.encode_basic(
        target_block,
        uv3_flash_loaner.amount_in,
        coinbase_xfer,
        excs,
    )

    return encoded

