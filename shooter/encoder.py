"""
encoder.py

Packs parameters for the shooter.
"""

import typing
import web3
import web3.types
import enum

from .constants import MAX_AMOUNT_IN_UNISWAP_V2, MAX_AMOUNT_OUT, MAX_COINBASE_XFER, MAX_DEADLINE

class UniswapV2Record(typing.NamedTuple):
    address: str
    amount_out: int
    amount_in_explicit: int
    recipient: typing.Optional[str]
    zero_for_one: bool

class UniswapV3Record(typing.NamedTuple):
    address: str
    amount_out: int
    zero_for_one: bool
    auto_funded: bool

def encode_basic(
        deadline: int,
        coinbase_xfer: int,
        exchanges: typing.List[typing.Union[UniswapV2Record, UniswapV3Record]],
    ) -> bytes:
    """
    Encodes the shooter contract input.

    NOTE: `mode` is fixed at 0 for now
    """
    assert isinstance(deadline, int)
    assert 0 < deadline
    assert deadline <= MAX_DEADLINE

    assert isinstance(coinbase_xfer, int)
    assert 0 <= coinbase_xfer
    assert coinbase_xfer <= MAX_COINBASE_XFER

    assert len(exchanges) >= 2
    assert isinstance(exchanges[0], UniswapV3Record)

    ret =  b''
    # method selector is always zero
    ret += b'\x00' * 4

    deadline_and_flags = deadline << 4
    if exchanges[0].zero_for_one:
        deadline_and_flags |= 0x1
    if exchanges[0].auto_funded:
        deadline_and_flags |= (0x1 << 1)
    ret += int.to_bytes(deadline_and_flags, length=4, byteorder='big', signed = False)
    ret += int.to_bytes(coinbase_xfer, length=8, byteorder='big', signed=False)
    ret += bytes.fromhex(exchanges[0].address[2:])
    
    assert len(ret) == 4 + 32
    ret += int.to_bytes(exchanges[0].amount_out, length=32, byteorder='big', signed=False)

    for ex in exchanges[1:]:
        assert ex.amount_out > 0
        assert ex.amount_out <= MAX_AMOUNT_OUT
        assert len(ex.address) == 42
        assert web3.Web3.isChecksumAddress(ex.address)
        assert isinstance(ex.zero_for_one, bool)

        flags = 0
        if isinstance(ex, UniswapV3Record):
            assert isinstance(ex.auto_funded, bool)
            flags |= (0x1 << 4)
            if not ex.auto_funded:
                flags |= (0x1 << 6)
        if ex.zero_for_one:
            flags |= (0x1 << 5)
 
        addr_int = int.from_bytes(bytes.fromhex(ex.address[2:]), byteorder='big', signed=False)

        action = (flags << (256 - 8)) | (ex.amount_out << 160) | addr_int
        if isinstance(ex, UniswapV2Record):
            extra_details = 0x0
            if ex.recipient is not None:
                assert web3.Web3.isChecksumAddress(ex.recipient)
                extra_details |= int.from_bytes(bytes.fromhex(ex.recipient[2:]), byteorder='big', signed=False)
            if ex.amount_in_explicit > 0:
                assert isinstance(ex.amount_in_explicit, int)
                assert ex.amount_in_explicit <= MAX_AMOUNT_IN_UNISWAP_V2
                extra_details |= (ex.amount_in_explicit << 160)
            if extra_details != 0x0:
                action |= (0x1 << 6) << (256 - 8)
                ret += int.to_bytes(action, length=32, byteorder='big', signed=False)
                ret += int.to_bytes(extra_details, length=32, byteorder='big', signed=False)

    return ret
