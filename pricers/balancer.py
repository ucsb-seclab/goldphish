"""
Pricer based on original Balancer

SEE: https://github.com/balancer-labs/balancer-core/blob/master/contracts/BPool.sol
"""

import decimal
import typing

from pricers.block_observation_result import BlockObservationResult
from .base import BaseExchangePricer, NotEnoughLiquidityException
import web3
import web3.types
import web3.contract
from eth_utils import event_abi_to_log_topic
from utils import get_abi
import logging

l = logging.getLogger(__name__)

_base_balancer: web3.contract.Contract = web3.Web3().eth.contract(address=b'\x00'*20, abi=get_abi('balancer_v1/bpool.abi.json'))

# logs that modify balance
BIND_SELECTOR         = bytes.fromhex(_base_balancer.functions.bind(web3.Web3.toChecksumAddress(b'\x00' * 20), 0, 0).selector[2:])
UNBIND_SELECTOR       = bytes.fromhex(_base_balancer.functions.unbind(web3.Web3.toChecksumAddress(b'\x00' * 20)).selector[2:])
REBIND_SELECTOR       = bytes.fromhex(_base_balancer.functions.rebind(web3.Web3.toChecksumAddress(b'\x00' * 20), 0, 0).selector[2:])
GULP_SELECTOR         = bytes.fromhex(_base_balancer.functions.gulp(web3.Web3.toChecksumAddress(b'\x00' * 20)).selector[2:])
FINALIZE_SELECTOR     = bytes.fromhex(_base_balancer.functions.finalize().selector[2:])
PUBLIC_SWAP_SELECTOR  = bytes.fromhex(_base_balancer.functions.setPublicSwap(False).selector[2:])
SET_SWAP_FEE_SELECTOR = bytes.fromhex(_base_balancer.functions.setSwapFee(10).selector[2:])


BIND_TOPIC         = BIND_SELECTOR.ljust(32, b'\x00')
UNBIND_TOPIC       = UNBIND_SELECTOR.ljust(32, b'\x00')
REBIND_TOPIC       = REBIND_SELECTOR.ljust(32, b'\x00')
GULP_TOPIC         = GULP_SELECTOR.ljust(32, b'\x00')
FINALIZE_TOPIC     = FINALIZE_SELECTOR.ljust(32, b'\x00')
PUBLIC_SWAP_TOPIC  = PUBLIC_SWAP_SELECTOR.ljust(32, b'\x00')
SET_SWAP_FEE_TOPIC = SET_SWAP_FEE_SELECTOR.ljust(32, b'\x00')

LOG_JOIN_TOPIC = event_abi_to_log_topic(_base_balancer.events.LOG_JOIN().abi)
LOG_EXIT_TOPIC = event_abi_to_log_topic(_base_balancer.events.LOG_EXIT().abi)
LOG_SWAP_TOPIC = event_abi_to_log_topic(_base_balancer.events.LOG_SWAP().abi)

TOKEN_BASE_SLOT = int.from_bytes(bytes.fromhex('6e1540171b6c0c960b71a7020d9f60077f6af931a8bbf590da0223dacf75c7af'), byteorder='big', signed=False)

class NotFinalizedException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class TooLittleInput(Exception):

    def __init__(self, *args) -> None:
        super().__init__(*args)


class BalancerPricer(BaseExchangePricer):
    RELEVANT_LOGS = [
                        BIND_TOPIC, UNBIND_TOPIC, REBIND_TOPIC, \
                        GULP_TOPIC, FINALIZE_TOPIC, PUBLIC_SWAP_TOPIC, SET_SWAP_FEE_TOPIC, \
                        LOG_JOIN_TOPIC, LOG_EXIT_TOPIC, LOG_SWAP_TOPIC, \
                    ]

    w3: web3.Web3
    finalized: typing.Optional[bool]
    tokens: typing.Optional[typing.Set[str]]
    address: str
    swap_fee: typing.Optional[int]
    token_denorms: typing.Dict[str, int]

    _public_swap: typing.Optional[bool]
    _balance_cache: typing.Dict[str, int]


    def __init__(self, w3: web3.Web3, address: str) -> None:
        self.address = address
        self.w3 = w3

        self.finalized = None
        self.tokens = None
        self.swap_fee = None
        self.token_denorms = {}
        self._balance_cache = {}
        self._public_swap = None

    def get_tokens(self, block_identifier: int) -> typing.Set[str]:
        if self.tokens is None:
            # no cache, get the list of tokens
            bn_tokens = self.w3.eth.get_storage_at(self.address, '0x9', block_identifier)
            n_tokens = int.from_bytes(bn_tokens, byteorder='big', signed=False)

            tokens = set()

            for i in range(n_tokens):
                token_slot = int.to_bytes(TOKEN_BASE_SLOT + i, length=32, byteorder='big', signed=False)
                hex_token_slot = '0x' + token_slot.hex()
                btoken = self.w3.eth.get_storage_at(self.address, hex_token_slot, block_identifier).rjust(32, b'\x00')
                assert btoken[0:12] == b'\x00'*12

                token = self.w3.toChecksumAddress(btoken[12:])
                tokens.add(token)

            self.tokens = tokens
        return self.tokens

    def get_finalized(self, block_identifier: int) -> bool:
        if self.finalized is None:
            bfinalized = self.w3.eth.get_storage_at(self.address, '0x8', block_identifier)
            self.finalized = bfinalized[-1] != 0
        return self.finalized

    def get_balance(self, address: str, block_identifier: int) -> int:
        assert address in self.tokens
        if address not in self._balance_cache:
            slot_base = self.w3.keccak(
                bytes.fromhex(address[2:]).rjust(32, b'\x00') +
                b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0a'
            )
            slot = int.to_bytes(int.from_bytes(slot_base, byteorder='big', signed=False) + 0x3, length=32, byteorder='big', signed=False)

            bbal = self.w3.eth.get_storage_at(self.address, slot.hex(), block_identifier)
            balance = int.from_bytes(bbal, byteorder='big', signed=False)

            self._balance_cache[address] = balance
        return self._balance_cache[address]

    def get_public_swap(self, block_identifier: int) -> bool:
        if self._public_swap is None:
            if self.finalized == True:
                # finalized implies _publicSwap
                self._public_swap = True
            else:
                bpublic_swap = self.w3.eth.get_storage_at(self.address, '0x6', block_identifier)
                self._public_swap = (int.from_bytes(bpublic_swap, byteorder='big', signed=False) >> 0xa0) != 0
        return self._public_swap

    def get_swap_fee(self, block_identifier: int) -> int:
        if self.swap_fee is None:
            bswap_fee = self.w3.eth.get_storage_at(self.address, '0x7', block_identifier)
            self.swap_fee = int.from_bytes(bswap_fee, byteorder='big', signed=False)
        return self.swap_fee

    def get_denorm_weight(self, address: str, block_identifier: int) -> int:
        if address not in self.token_denorms:
            slot_base = self.w3.keccak(
                bytes.fromhex(address[2:]).rjust(32, b'\x00') +
                b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0a'
            )
            slot = int.to_bytes(int.from_bytes(slot_base, byteorder='big', signed=False) + 0x2, length=32, byteorder='big', signed=False)

            bweight = self.w3.eth.get_storage_at(self.address, slot.hex(), block_identifier)
            weight = int.from_bytes(bweight, byteorder='big', signed=False)

            self.token_denorms[address] = weight
        return self.token_denorms[address]

    def token_out_for_exact_in(self, token_in: str, token_out: str, token_amount_in: int, block_identifier: int, **_):
        # modeled based off swapExactAmountIn
        # neglects minAmountOut and maxPrice
        if not self.get_public_swap(block_identifier):
            l.warning(f'Not yet public swap: {self.address}')
            return 0, 0.0

        _tokens = self.get_tokens(block_identifier)

        if token_in not in _tokens:
            l.warning(f'Token not available: {self.address} {token_in}')
            return 0, 0.0
        elif token_out not in _tokens:
            l.warning(f'Token not available: {self.address} {token_out}')
            return 0, 0.0

        token_weight_in  = self.get_denorm_weight(token_in,  block_identifier)
        token_weight_out = self.get_denorm_weight(token_out, block_identifier)

        token_balance_in  = self.get_balance(token_in, block_identifier)
        token_balance_out = self.get_balance(token_out, block_identifier)

        if token_balance_out == 0:
            # this fucks up some calculations, so just call it 0
            return 0, 0.0

        max_in = BalancerPricer.bmul(token_balance_in, MAX_IN_RATIO)
        if token_amount_in > max_in:
            raise NotEnoughLiquidityException(token_amount_in, remaining=token_amount_in - max_in)

        swap_fee = self.get_swap_fee(block_identifier)

        spot_price_before = BalancerPricer.calc_spot_price(
            token_balance_in,
            token_weight_in,
            token_balance_out,
            token_weight_out,
            swap_fee,
        )

        # calcOutGivenIn

        weight_ratio = BalancerPricer.bdiv(token_weight_in, token_weight_out)
        adjusted_in = BalancerPricer.bsub(BalancerPricer.BONE, swap_fee)
        adjusted_in = BalancerPricer.bmul(token_amount_in, adjusted_in)
        assert adjusted_in <= token_amount_in, f'expect {adjusted_in} <= {token_amount_in}'

        denominator = BalancerPricer.badd(token_balance_in, adjusted_in)
        if denominator == 0:
            # The following division is about to blow up. Probably we asked for 0 input
            # with 0 original balance. Just return reporting that no amount is available
            # for purchase.
            assert token_balance_in == 0
            assert adjusted_in == 0
            return 0, 0.0
        y = BalancerPricer.bdiv(token_balance_in, denominator)
        foo = BalancerPricer.bpow(y, weight_ratio) # their var name, not mine
        bar = BalancerPricer.bsub(BalancerPricer.BONE, foo)

        token_amount_out = BalancerPricer.bmul(token_balance_out, bar)

        new_balance_in  = token_balance_in + token_amount_in
        new_balance_out = token_balance_out - token_amount_out

        assert new_balance_out >= 0

        if new_balance_out > 0:

            spot_price_after = BalancerPricer.calc_spot_price(
                new_balance_in,
                token_weight_in,
                new_balance_out,
                token_weight_out,
                swap_fee,
            )

            assert spot_price_after >= spot_price_before

            # if token_amount_out > 100:
            #     # I guess just a double-check that we didn't fuck up rounding?
            #     divd = BalancerPricer.bdiv(token_amount_in, token_amount_out)
            #     print('spot_price_before', spot_price_before)
            #     print('divd             ', divd)
            #     if spot_price_before > divd:
            #         raise TooLittleInput(self.address, block_identifier, f'BalancerV1 {self.address} ({token_in} -> {token_out}) @ {block_identifier} needs more input than {token_amount_in}')

            return token_amount_out, BalancerPricer.BONE / spot_price_after
        else:
            # new_balance_out == 0
            return token_amount_out, 0.0 # marginal price is now 0

    def get_value_locked(self, token_address: str, block_identifier: int) -> int:
        assert token_address in self.get_tokens(block_identifier)

        return self.get_balance(token_address, block_identifier)

    def get_token_weight(self, token_address: str, block_identifier: int) -> decimal.Decimal:
        _tot_weight = 0
        for t in self.get_tokens(block_identifier):
            _tot_weight += self.get_denorm_weight(t, block_identifier)

        return decimal.Decimal(self.get_denorm_weight(token_address, block_identifier)) / decimal.Decimal(_tot_weight)

    def observe_block(self, logs: typing.List[web3.types.LogReceipt], force_load: bool = False) -> BlockObservationResult:
        just_finalized = False
        tokens_modified = set()


        # LOG_JOIN, LOG_EXIT, LOG_SWAP
        for log in logs:
            if log['address'] == self.address and len(log['topics']) > 0:
                block_number = log['blockNumber']

                # add liquidity
                if log['topics'][0] == LOG_JOIN_TOPIC:
                    parsed = _base_balancer.events.LOG_JOIN().processLog(log)
                    token = parsed['args']['tokenIn']
                    amount = parsed['args']['tokenAmountIn']

                    if force_load and self.tokens is None:
                        self.get_tokens(block_number - 1)
                    if force_load and token not in self._balance_cache:
                        self.get_balance(token, block_number - 1)

                    if token in self._balance_cache:
                        self._balance_cache[token] += amount
                    
                    tokens_modified.add(token)

                # remove liquidity
                elif log['topics'][0] == LOG_EXIT_TOPIC:
                    parsed = _base_balancer.events.LOG_EXIT().processLog(log)
                    token = parsed['args']['tokenOut']
                    amount = parsed['args']['tokenAmountOut']

                    if force_load and self.tokens is None:
                        self.get_tokens(block_number - 1)
                    if force_load and token not in self._balance_cache:
                        self.get_balance(token, block_number - 1)

                    if token in self._balance_cache:
                        old_bal = self._balance_cache[token]
                        assert old_bal >= amount, 'exiting more token than available'
                        self._balance_cache[token] = old_bal - amount

                    tokens_modified.add(token)

                # perform a swap
                elif log['topics'][0] == LOG_SWAP_TOPIC:
                    parsed = _base_balancer.events.LOG_SWAP().processLog(log)
                    token_in = parsed['args']['tokenIn']
                    token_out = parsed['args']['tokenOut']
                    amount_in = parsed['args']['tokenAmountIn']
                    amount_out = parsed['args']['tokenAmountOut']

                    if force_load and self.tokens is None:
                        self.get_tokens(block_number - 1)

                    if force_load and token_in not in self._balance_cache:
                        self.get_balance(token_in, block_number - 1)
                    if force_load and token_out not in self._balance_cache:
                        self.get_balance(token_out, block_number - 1)

                    if token_in in self._balance_cache:
                        self._balance_cache[token_in] += amount_in
                    
                    if token_out in self._balance_cache:
                        old_bal = self._balance_cache[token_out]
                        assert old_bal >= amount_out, 'swapping out more token than available'
                        self._balance_cache[token_out] = old_bal - amount_out

                    tokens_modified.add(token_in)
                    tokens_modified.add(token_out)

                elif log['topics'][0] == GULP_TOPIC:
                    payload = bytes.fromhex(log['data'][136+2:])
                    token_address = web3.Web3.toChecksumAddress(payload[12:32])

                    if force_load:
                        raise Exception('cannot force load on GULP')

                    if token_address in self._balance_cache:
                        # this cache is invalidated, unfortunately we don't know what balance is now
                        # because it isnt logged
                        del self._balance_cache[token_address]

                    tokens_modified.add(token_address)

                elif log['topics'][0] == REBIND_TOPIC:
                    payload = bytes.fromhex(log['data'][136+2:])

                    token_address = web3.Web3.toChecksumAddress(payload[12:32])
                    balance = int.from_bytes(payload[32:64], byteorder='big', signed=False)
                    denorm = int.from_bytes(payload[64:96], byteorder='big', signed=False)

                    self.token_denorms[token_address] = denorm
                    self._balance_cache[token_address] = balance

                    self.tokens = None

                elif log['topics'][0] == BIND_TOPIC:
                    # a token-bind event
                    payload = bytes.fromhex(log['data'][136+2:])

                    token_address = web3.Web3.toChecksumAddress(payload[12:32])
                    balance = int.from_bytes(payload[32:64], byteorder='big', signed=False)
                    denorm = int.from_bytes(payload[64:96], byteorder='big', signed=False)

                    self.token_denorms[token_address] = denorm
                    self._balance_cache[token_address] = balance

                    self.tokens = None

                elif log['topics'] == PUBLIC_SWAP_TOPIC:
                    raise NotImplementedError('hmmm public swap....')

                elif log['topics'] == UNBIND_TOPIC:
                    raise NotImplementedError('hmmm unbind ')

                elif log['topics'][0] == FINALIZE_TOPIC:
                    # these are now immutable
                    if self.tokens is None:
                        self.tokens = None
                        self.get_tokens(block_identifier=log['blockNumber'])
                    if self.swap_fee is None:
                        self.swap_fee = None
                        self.get_swap_fee(block_identifier=log['blockNumber'])

                    self.finalized = True
                    self._public_swap = True

                    just_finalized = True

                elif log['topics'][0] == SET_SWAP_FEE_TOPIC:
                    assert self.finalized != True, 'Cannot update swap fee after finalization'
                    payload = bytes.fromhex(log['data'][136+2:])

                    swap_fee = int.from_bytes(payload[0:32], byteorder='big', signed=False)
                    self.swap_fee = swap_fee

        if just_finalized:
            block_number = logs[0]['blockNumber']
            tokens = self.get_tokens(block_number)
            all_pairs = []
            for t1 in tokens:
                for t2 in tokens:
                    if bytes.fromhex(t1[2:]) < bytes.fromhex(t2[2:]):
                        all_pairs.append((t1, t2))
            assert len(all_pairs) >= 1
            return BlockObservationResult(
                pair_prices_updated = all_pairs,
                swap_enabled = True,
                gradual_weight_adjusting_scheduled = None,
            )
        
        updated_pairs = []
        if len(tokens_modified) > 0:
            block_number = logs[0]['blockNumber']
            tokens = self.get_tokens(block_number)
            for t1 in tokens_modified:
                for t2 in tokens:
                    if bytes.fromhex(t1[2:]) < bytes.fromhex(t2[2:]):
                        updated_pairs.append((t1, t2))
        
        return BlockObservationResult(
            pair_prices_updated = updated_pairs,
            swap_enabled = None,
            gradual_weight_adjusting_scheduled = None,
        )


    def set_web3(self, w3: web3.Web3):
        """
        Used for moving a pricer onto a fork with minimal setup disruption

        Should also clear the running caches
        """
        raise NotImplementedError()

    #
    # math helpers
    #

    BONE = 10 ** 18
    MIN_BPOW_BASE = 1
    MAX_BPOW_BASE = (2 * BONE) - 1
    BPOW_PRECISION = BONE // (10 ** 10)

    @staticmethod
    def calc_spot_price(
            token_balance_in: int,
            token_weight_in: int,
            token_balance_out: int,
            token_weight_out: int,
            swap_fee: int,
        ):
        assert token_balance_in >= 0
        assert token_balance_out >= 0
        assert token_weight_in > 0
        assert token_weight_out > 0
        assert swap_fee >= 0
        
        numer = BalancerPricer.bdiv(token_balance_in, token_weight_in)
        denom = BalancerPricer.bdiv(token_balance_out, token_weight_out)
        ratio = BalancerPricer.bdiv(numer, denom)
        scale = BalancerPricer.bdiv(BalancerPricer.BONE, BalancerPricer.bsub(BalancerPricer.BONE, swap_fee))
        return BalancerPricer.bmul(ratio, scale)

    @staticmethod
    def badd(a: int, b: int) -> int:
        # overflow not possible on python
        return a + b

    @staticmethod
    def bsub(a: int, b: int) -> int:
        # just subtraction with underflow protection
        assert a >= b
        return a - b

    @staticmethod
    def bmul(a: int, b: int) -> int:
        c0 = a * b

        # overflow not possible

        c1 = c0 + (BalancerPricer.BONE // 2)

        c2 = c1 // BalancerPricer.BONE

        return c2

    @staticmethod
    def bdiv(a: int, b: int) -> int:
        assert a >= 0
        assert b > 0
        
        c0 = a * BalancerPricer.BONE

        # overflow not possible on python

        c1 = c0 + (b // 2)

        assert c1 >= c0 # idk what this is for but it's in Balancer code

        c2 = c1 // b

        return c2

    @staticmethod
    def bpow(base: int, exp: int) -> int:
        assert base >= 0
        assert exp >= 0

        assert base >= BalancerPricer.MIN_BPOW_BASE
        assert base <= BalancerPricer.MAX_BPOW_BASE

        whole = BalancerPricer.bfloor(exp)
        remain = BalancerPricer.bsub(exp, whole)

        whole_pow = BalancerPricer.bpowi(base, whole // BalancerPricer.BONE)

        if remain == 0:
            return whole_pow

        partial_result = BalancerPricer.bpow_approx(base, remain, BalancerPricer.BPOW_PRECISION)

        return BalancerPricer.bmul(whole_pow, partial_result)

    @staticmethod
    def bpowi(a: int, n: int) -> int:
        assert a >= 0
        assert n >= 0

        z = a if (n % 2 != 0) else BalancerPricer.BONE

        n = n // 2
        while n != 0:

            a = BalancerPricer.bmul(a, a)

            if n % 2 != 0:
                z = BalancerPricer.bmul(z, a)

            n = n // 2
        
        return z

    @staticmethod
    def bfloor(a: int) -> int:
        return (a // BalancerPricer.BONE) * BalancerPricer.BONE

    @staticmethod
    def bpow_approx(base: int, exp: int, precision: int) -> int:
        assert base >= 0
        assert exp >= 0
        assert precision >= 0

        a = exp

        # bSubSign in-line
        if base >= BalancerPricer.BONE:
            x = base - BalancerPricer.BONE
            xneg = False
        else:
            x = BalancerPricer.BONE - base
            xneg = True
        
        term = BalancerPricer.BONE
        sum_ = term
        negative = False

        i = 1
        while term >= precision:
            bigK = i * BalancerPricer.BONE

            # bSubSign in-line
            tmp_ = BalancerPricer.bsub(bigK, BalancerPricer.BONE)
            if a >= tmp_:
                c = a - tmp_
                cneg = False
            else:
                c = tmp_ - a
                cneg = True
            
            term = BalancerPricer.bmul(term, BalancerPricer.bmul(c, x))
            term = BalancerPricer.bdiv(term, bigK)

            if term == 0:
                break
            
            if xneg:
                negative = not negative
            if cneg:
                negative = not negative
            
            if negative:
                sum_ = BalancerPricer.bsub(sum_, term)
            else:
                sum_ = BalancerPricer.badd(sum_, term)

            i += 1

        return sum_

    def copy_without_cache(self) -> 'BaseExchangePricer':
        return BalancerPricer(
            self.w3, self.address
        )

    def __str__(self) -> str:
        return f'<BalancerPricer {self.address} tokens={self.tokens}>'

MAX_IN_RATIO = BalancerPricer.BONE // 2
