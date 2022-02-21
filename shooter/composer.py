"""
Given a descriptive series of an arbitrage flow, orders the exchanges properly
and synthesizes the input data for the shooter.
"""

import typing
import logging

import shooter.encoder

from .constants import MAX_COINBASE_XFER

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

def construct(
        exchanges: typing.List[ExchangeRecord],
        coinbase_xfer: int,
        target_block: int,
    ):
    assert 2 <= len(exchanges) <= 3
    assert 0 <= coinbase_xfer <= MAX_COINBASE_XFER

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
        assert univ2_chain[-1].amount_out >= univ3_chain[-1].amount_in
        # sanity check -- the token input should be available from the first v3 exchange
        assert univ3_chain[0].token_out == univ2_chain[0].token_in

    # sanity check
    assert all(not x.is_uniswap_v2 for x in univ3_chain)
    assert all(x.is_uniswap_v2 for x in univ2_chain)
    assert len(set(univ3_chain).union(univ2_chain)) == len(univ3_chain) + len(univ2_chain), 'should not have duplicates'
    assert len(univ3_chain) + len(univ2_chain) == len(exchanges), 'no exchange left behind'

    excs = []

    # Handle each situation separately, I suppose. This makes the logic perhaps a bit more readable (but repeated).

    if len(univ3_chain) == 1 and len(univ2_chain) == 1:
        if univ3_chain[0].amount_out > univ2_chain[0].amount_in:
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
                    amount_out = univ3_chain[0].amount_out,
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
            assert univ2_chain[0].amount_out > univ3_chain[0].amount_in
            excs = [
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[0].address,
                    amount_out = univ3_chain[0].amount_out,
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
        if univ3_chain[0].amount_out > univ2_chain[0].amount_in:
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
                    amount_out = univ3_chain[0].amount_out,
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
        elif univ2_chain[0].amount_out > univ2_chain[1].amount_in:
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
                    amount_out = univ3_chain[0].amount_out,
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
            assert univ2_chain[1].amount_out > univ3_chain[0].amount_in
            l.debug('Cycle situation 4')
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
                    amount_out = univ3_chain[0].amount_out,
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
                ),
            ]
    elif len(univ3_chain) == 2 and len(univ2_chain) == 1:
        if univ3_chain[0].amount_in < univ3_chain[1].amount_out:
            l.debug('Cycle situation 5')
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
                    amount_out = univ3_chain[0].amount_out,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE, # here, means 'next uniswap v2 exchange'
                ),
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[1].address,
                    amount_out = univ3_chain[1].amount_out,
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
        elif univ3_chain[0].amount_out > univ2_chain[0].amount_in:
            l.debug('Cycle situation 6')
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
                    amount_out = univ3_chain[0].amount_out,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.SHOOTER,
                ),
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[1].address,
                    amount_out = univ3_chain[1].amount_out,
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
            assert univ2_chain[0].amount_out > univ3_chain[1].amount_in
            l.debug('Cycle situation 7')
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
                    amount_out = univ3_chain[0].amount_out,
                    zero_for_one = univ3_chain[0].zero_for_one,
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE,
                ),
                shooter.encoder.UniswapV3Record(
                    address = univ3_chain[1].address,
                    amount_out = univ3_chain[1].amount_out,
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
        assert len(univ3_chain) == 3
        assert len(univ2_chain) == 0
        l.debug(f'Cycle situation 8')
        # fortunately this is the easy one
        # SITUATION
        # (uv3_2) --> (uv3_1) --> (uv3_0) --> (uv3_2) // implicit profit possible along any of these
        
        # rotate so that profit is at the beginning
        while not (univ3_chain[0].amount_out > univ3_chain[-1].amount_in):
            univ3_chain = univ3_chain[1:] + univ3_chain[:1]

        # ensure invariant(s) hold
        for ex1, ex2 in zip(univ3_chain, (univ3_chain[1:] + univ3_chain[:1])):
            assert ex1.token_in == ex2.token_out
        assert univ3_chain[0].amount_out > univ3_chain[-1].amount_in

        # NORMALIZED SITUATION
        # (uv3_2) --> (uv3_1) --> (uv3_0) -- profit --> (uv3_2)
        #
        #   uv3_0 -> self
        #     uv3_1 -> uv3_0
        #       uv3_2 -> uv3_1
        #         self -> uv3_2

        for i, exchange in list(enumerate(univ3_chain)):
            assert exchange.token_out == univ3_chain[i-1].token_in
            if exchange.amount_out > univ3_chain[i-1].amount_in:
                # we're taking profit, must send to self
                recipient = shooter.encoder.FundsRecipient.SHOOTER
            else:
                # we're not taking profit, pay previous exchange directly
                recipient = shooter.encoder.FundsRecipient.MSG_SENDER
            excs.append(shooter.encoder.UniswapV3Record(
                exchange.address,
                exchange.amount_out,
                exchange.zero_for_one,
                recipient
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

