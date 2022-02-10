"""
encoder.py

Packs parameters for the shooter.
"""

import typing
import web3
import web3.types
import enum

class ExchangeType(enum.Enum):
    UNISWAP_V2: 0x1
    UNISWAP_V3: 0x2


class ExchangeRecord(typing.NamedTuple):
    address: str
    send_to_self: bool
    zero_for_one: bool
    profit_taking: typing.Optional[int]
    exchange_type: ExchangeType

class ProfitMethod(enum.Enum):
    TAKE_EXCHANGE: 0x0
    TAKE_DIRECTLY: 0x1

def encode_basic(
        deadline: int,
        coinbase_xfer: int,
        profit_method: ProfitMethod,
        input_amt: int,
        exchanges: typing.List[ExchangeRecord]
    ) -> bytes:
    """
    Encodes the shooter contract input.

    NOTE: `mode` is fixed at 0 for now
    """
    assert isinstance(deadline, int)
    assert 0 < deadline
    assert deadline <= 0xfffffff

    assert isinstance(coinbase_xfer, int)
    assert 0 <= coinbase_xfer
    assert coinbase_xfer <= 0xfffffffffffffff

    assert len(exchanges) > 2
    assert exchanges[0].exchange_type == ExchangeType.UNISWAP_V3

    ret =  b''
    # method selector is always zero
    ret += b'\x00' * 4

    # again, the mode nybble is zero, so just leave the empty space there empty
    deadline_and_mode = deadline << 4
    ret += int.to_bytes(deadline_and_mode, length=4, byteorder='big', signed = False)

    coinbase_xfer_and_flags = coinbase_xfer << 4

    if exchanges[0].send_to_self:
        coinbase_xfer_and_flags |= 1 << 2
    
    if exchanges[0].zero_for_one:
        coinbase_xfer_and_flags |= 1 << 3
    
    ret += int.to_bytes(coinbase_xfer_and_flags, length=8, byteorder='big', signed=False)

    ret += bytes.fromhex(exchanges[0].address[2:])
    
    assert len(ret) == 4 + 32
    ret += int.to_bytes(input_amt, length=32, byteorder='big', signed=False)

    for ex in exchanges[1:]:
        addr_and_flags = int.from_bytes(bytes.fromhex(ex.address[2:]), byteorder='big', signed=False)
        if profit_method == ProfitMethod.TAKE_DIRECTLY:
            addr_and_flags |= (1 << 252)
        if ex.exchange_type == ExchangeType.UNISWAP_V3:
            addr_and_flags |= (1 << 253)
        if ex.profit_taking is not None:
            addr_and_flags |= (1 << 254)
        if ex.send_to_self:
            addr_and_flags |= (1 << 255)

        ret += int.to_bytes(addr_and_flags, length=32, byteorder='big', signed=False)
        if ex.profit_taking is not None:
            assert isinstance(ex.profit_taking, int)
            assert 0 < ex.profit_taking
            assert ex.profit_taking <= ((1 << 256) - 1)
            ret += int.to_bytes(ex.profit_taking, length=32, byteorder='big', signed=False)

    return ret
