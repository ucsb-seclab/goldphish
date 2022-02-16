"""
encoder.py

Packs parameters for the shooter.
"""

import itertools
import typing
import web3
import web3.types
import enum

from .constants import MAX_AMOUNT_IN_BRIEF, MAX_AMOUNT_IN_UNISWAP_V2, MAX_AMOUNT_OUT, MAX_COINBASE_XFER, MAX_TARGET_BLOCK, MAX_AMOUNT_IN_SWAP

class FundsRecipient(enum.Enum):
    SHOOTER = 0x0
    MSG_SENDER = 0x1
    NEXT_EXCHANGE = 0x2

class UniswapV2Record(typing.NamedTuple):
    address: str
    amount_out: int
    amount_in_explicit: int
    recipient: FundsRecipient
    zero_for_one: bool

class UniswapV3Record(typing.NamedTuple):
    address: str
    amount_out: int
    zero_for_one: bool
    recipient: FundsRecipient

def encode_basic(
        target_block: int,
        amount_in: int,
        coinbase_xfer: int,
        exchanges: typing.List[typing.Union[UniswapV2Record, UniswapV3Record]],
    ) -> bytes:
    """
    Encodes the shooter contract input.

    NOTE: `mode` is fixed at 0 for now
    """
    assert isinstance(amount_in, int)
    assert 0 <= amount_in
    assert amount_in <= MAX_AMOUNT_IN_SWAP
    assert isinstance(target_block, int)
    assert 0 <= target_block

    assert isinstance(coinbase_xfer, int)
    assert 0 <= coinbase_xfer
    assert coinbase_xfer <= MAX_COINBASE_XFER

    assert len(exchanges) >= 2
    assert isinstance(exchanges[0], UniswapV3Record)
    assert exchanges[0].recipient in [FundsRecipient.NEXT_EXCHANGE, FundsRecipient.SHOOTER]

    ret =  b''
    # method selector is always zero
    ret += b'\x00' * 4

    first_line = 0
    first_line |= (target_block & MAX_TARGET_BLOCK) << 240
    if exchanges[0].zero_for_one:
        first_line |= (0x1 << 239)
    if exchanges[0].recipient == FundsRecipient.NEXT_EXCHANGE:
        first_line |= (0x1 << 238)
    first_line |= int(exchanges[0].address[2:], base=16)

    # do we need extradata?
    need_extradata = coinbase_xfer > 0 or amount_in > MAX_AMOUNT_IN_BRIEF
    if not need_extradata:
        first_line |= (amount_in << 160)
        ret += int.to_bytes(first_line, length=32, byteorder='big', signed=False)
    else:
        second_line = coinbase_xfer
        second_line |= amount_in << 64
        ret += int.to_bytes(first_line, length=32, byteorder='big', signed=False)
        ret += int.to_bytes(second_line, length=32, byteorder='big', signed=False)

    for i, ex in zip(itertools.count(1), exchanges[1:]):
        assert ex.amount_out > 0
        assert ex.amount_out <= MAX_AMOUNT_OUT
        assert len(ex.address) == 42
        assert web3.Web3.isChecksumAddress(ex.address)
        assert isinstance(ex.zero_for_one, bool)

        first_line = 0

        # handle flags
        if isinstance(ex, UniswapV3Record):
            first_line |= 0x1 << 252
        if ex.zero_for_one:
            first_line |= 0x1 << 253
        first_line |= int(ex.recipient.value) << 254
 
        # add address
        first_line |= int(ex.address[2:], base=16)

        can_infer_amount_out = isinstance(ex, UniswapV2Record) and i == len(exchanges) - 1 and ex.recipient == FundsRecipient.MSG_SENDER
        if can_infer_amount_out:
            assert ex.amount_in_explicit < MAX_AMOUNT_IN_UNISWAP_V2
            first_line |= ex.amount_in_explicit << 160
        else:
            # add amount out
            first_line |= ex.amount_out << 160

        ret += int.to_bytes(first_line, length=32, byteorder='big', signed=False)

        if not can_infer_amount_out and isinstance(ex, UniswapV2Record) and ex.amount_in_explicit > 0:
            assert ex.amount_in_explicit < MAX_AMOUNT_IN_UNISWAP_V2
            ret += int.to_bytes(
                ex.amount_in_explicit << 160,
                length=32,
                byteorder='big',
                signed=False
            )

    return ret
