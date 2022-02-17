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

    # handle first exchange, which is univ3
    # figure out where to send funds
    if len(univ2_chain) == 0:
        send_funds = shooter.encoder.FundsRecipient.SHOOTER
    else:
        # if we need to take profit before giving to v2, send to self
        if univ3_chain[0].amount_out > univ2_chain[0].amount_in:
            send_funds = shooter.encoder.FundsRecipient.SHOOTER
        else:
            if len(univ3_chain) == 1:
                send_funds = shooter.encoder.FundsRecipient.NEXT_EXCHANGE
            else:
                raise NotImplementedError('Not sure how to handle this yet, ideally sends funds directly to the v2 exchange')
    excs.append(shooter.encoder.UniswapV3Record(
        univ3_chain[0].address,
        univ3_chain[0].amount_out,
        bytes.fromhex(univ3_chain[0].token_in[2:]) < bytes.fromhex(univ3_chain[0].token_out[2:]),
        send_funds
    ))

    for i, exchange in list(enumerate(univ3_chain))[1:]:
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
            bytes.fromhex(exchange.token_in[2:]) < bytes.fromhex(exchange.token_out[2:]),
            recipient
        ))
    
    if len(univ2_chain) > 0:
        # handle first uniswap v2 separately
        if univ3_chain[0].amount_out > univ2_chain[0].amount_in:
            # must forward payment from self manually
            excs.append(shooter.encoder.UniswapV2Record(
                univ2_chain[0].address,
                univ2_chain[0].amount_out,
                amount_in_explicit=univ2_chain[0].amount_in,
                recipient=shooter.encoder.FundsRecipient.MSG_SENDER if len(univ2_chain) == 1 else shooter.encoder.FundsRecipient.NEXT_EXCHANGE,
                zero_for_one=bytes.fromhex(univ2_chain[0].token_in[2:]) < bytes.fromhex(univ2_chain[0].token_out[2:]),
            ))
        else:
            # assume exchange is already paid; if so, find out whether we're taking profit _after_ this exchange
            if len(univ2_chain) == 1:
                next_in = univ3_chain[-1].amount_in
                if univ2_chain[0].amount_out > next_in:
                    recipient = shooter.encoder.FundsRecipient.SHOOTER
                else:
                    recipient = shooter.encoder.FundsRecipient.MSG_SENDER
            else:
                next_in = univ2_chain[1].amount_in
                if univ2_chain[0].amount_out > next_in:
                    recipient = shooter.encoder.FundsRecipient.SHOOTER
                else:
                    recipient = shooter.encoder.FundsRecipient.NEXT_EXCHANGE
            excs.append(shooter.encoder.UniswapV2Record(
                univ2_chain[0].address,
                univ2_chain[0].amount_out,
                amount_in_explicit=0,
                recipient=recipient,
                zero_for_one=bytes.fromhex(univ2_chain[0].token_in[2:]) < bytes.fromhex(univ2_chain[0].token_out[2:]),
            ))
        
        # handle second (and last) uniswap v2
        if len(univ2_chain) > 1:
            # if we just took profit, send payment manually
            if univ2_chain[0].amount_out > univ2_chain[1].amount_in:
                amount_in_explicit = univ2_chain[1].amount_in
                recipient = shooter.encoder.FundsRecipient.MSG_SENDER
            else:
                amount_in_explicit = 0
                # if we take profit after this, then send to self; otherwise send to msg_sender
                if univ2_chain[1].amount_out > univ3_chain[-1].amount_in:
                    recipient = shooter.encoder.FundsRecipient.SHOOTER
                else:
                    recipient = shooter.encoder.FundsRecipient.MSG_SENDER
            excs.append(shooter.encoder.UniswapV2Record(
                univ2_chain[0].address,
                univ2_chain[0].amount_out,
                amount_in_explicit=amount_in_explicit,
                recipient=recipient,
                zero_for_one=bytes.fromhex(univ2_chain[1].token_in[2:]) < bytes.fromhex(univ2_chain[1].token_out[2:]),
            ))

    print(excs)

    encoded = shooter.encoder.encode_basic(
        target_block,
        univ3_chain[0].amount_in,
        coinbase_xfer,
        excs,
    )

    return encoded

