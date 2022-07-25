"""
encoder.py

Packs parameters for the shooter.
"""

import itertools
import typing
import numpy as np
import web3
import web3.types

class UniswapV2Swap(typing.NamedTuple):
    amount_in: int
    amount_out: int
    exchange: str
    to: str
    zero_for_one: bool

    def serialize(self) -> bytes:
        builder = [
            int.to_bytes(self.amount_in if self.amount_in else 0, length=32, byteorder='big', signed=False),
            int.to_bytes(self.amount_out, length=32, byteorder='big', signed=False),
            bytes.fromhex(self.exchange[2:]),
            bytes.fromhex(self.to[2:]),
            b'\x01' if self.zero_for_one else b'\x00',
        ]
        return b''.join(builder)

class UniswapV3Swap(typing.NamedTuple):
    amount_in: int
    exchange: str
    to: str
    zero_for_one: bool
    leading_exchanges: typing.List
    must_send_input: bool

    def serialize(self) -> bytes:
        assert self.amount_in > 0

        if len(self.leading_exchanges) > 0:
            extradata = serialize(self.leading_exchanges)
        else:
            extradata = b''
        extradata = (b'\x01' if self.must_send_input else b'\x00') + extradata

        builder = [
            int.to_bytes(self.amount_in, length=32, byteorder='big', signed=True),
            bytes.fromhex(self.exchange[2:]),
            bytes.fromhex(self.to[2:]),
            b'\x01' if self.zero_for_one else b'\x00',
            int.to_bytes(len(extradata), length=2, byteorder='big', signed=False),
            extradata,
        ]
        return b''.join(builder)


class BalancerV1Swap(typing.NamedTuple):
    amount_in: int
    exchange: str
    token_in: str
    token_out: str
    to: str
    requires_approval: bool

    def serialize(self) -> bytes:
        assert self.amount_in > 0
        builder = [
            int.to_bytes(self.amount_in, length=32, byteorder='big', signed=False),
            bytes.fromhex(self.exchange[2:]),
            bytes.fromhex(self.token_in[2:]),
            bytes.fromhex(self.token_out[2:]),
            bytes.fromhex(self.to[2:]),
            b'\x01' if self.requires_approval else b'\x00',
        ]
        return b''.join(builder)


class BalancerV2Swap(typing.NamedTuple):
    pool_id: bytes
    amount_in: int
    amount_out: int
    token_in: str
    token_out: str
    to: str

    def serialize(self) -> bytes:
        assert self.amount_in > 0
        assert self.amount_out > 0
        assert len(self.pool_id) == 32
        builder = [
            self.pool_id,
            int.to_bytes(self.amount_in, length=32, byteorder='big', signed=False),
            int.to_bytes(self.amount_out, length=32, byteorder='big', signed=False),
            bytes.fromhex(self.token_in[2:]),
            bytes.fromhex(self.token_out[2:]),
            bytes.fromhex(self.to[2:]),
        ]

        return b''.join(builder)


def serialize(l: typing.List[typing.Union[UniswapV2Swap, UniswapV3Swap, BalancerV1Swap, BalancerV2Swap]]) -> bytes:
    # assert 2 <= len(l) <= 3
    builder = []
    builder.append(
        int.to_bytes(len(l), length=1, byteorder='big', signed=False),
    )
    serialized = [x.serialize() for x in l]
    offsets = [0] + list(np.cumsum([len(x) for x in serialized])[:-1])
    offsets = [o + (1 + 3 * 3) for o in offsets]
    assert len(offsets) == len(l)

    # pad with 0 offset to 3
    offsets.extend(0 for _ in range(3 - len(offsets)))

    for i, o in enumerate(offsets):
        if i < len(l):
            type_ = {
                UniswapV2Swap:  1,
                UniswapV3Swap:  2,
                BalancerV1Swap: 3,
                BalancerV2Swap: 4,
            }[type(l[i])]
            builder.append(
                int.to_bytes(type_, length=1, byteorder='big', signed=False)
            )
            builder.append(
                int.to_bytes(int(o), length=2, byteorder='big', signed=False)
            )
        else:
            builder.append(
                b'\x00\x00\x00'
            )

    builder.extend(serialized)
    return b''.join(builder)

