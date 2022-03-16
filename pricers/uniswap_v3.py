import typing
import web3
import web3.contract
import web3.types
from eth_utils import event_abi_to_log_topic
from utils import get_abi, profile
import logging

from pricers.base import BaseExchangePricer

l = logging.getLogger(__name__)

Tick = typing.NamedTuple('Tick', [
    ('id', int),
    ('liquidity_gross', int),
    ('liquidity_net', int),
    ('x_', int),
    ('x__', int),
    ('x___', int),
    ('x____', int),
    ('x_____', int),
    ('initialized', bool)
])

generic_uv3 = web3.Web3().eth.contract(
    address = b'\x00' * 20,
    abi = get_abi('uniswap_v3/IUniswapV3Pool.json')['abi'],
)
UNIV3_SWAP_EVENT_TOPIC = event_abi_to_log_topic(generic_uv3.events.Swap().abi)
UNIV3_BURN_EVENT_TOPIC = event_abi_to_log_topic(generic_uv3.events.Burn().abi)
UNIV3_MINT_EVENT_TOPIC = event_abi_to_log_topic(generic_uv3.events.Mint().abi)


class UniswapV3Pricer(BaseExchangePricer):
    MIN_TICK = -887272
    MAX_TICK = 887272
    MIN_SQRT_RATIO = 4295128739
    MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

    w3: web3.Web3
    address: str
    contract: web3.contract.Contract
    token0: str
    token1: str
    fee: int
    tick_spacing: int
    tick_cache: typing.Dict[int, Tick]
    tick_bitmap_cache: typing.Dict[int, int]
    slot0_cache: typing.Optional[typing.Tuple[int, int]]
    liquidity_cache: typing.Optional[int]
    last_block_observed: int

    def __init__(self, w3: web3.Web3, address: str, token0: str, token1: str, fee: int) -> None:
        assert web3.Web3.isChecksumAddress(address)
        assert fee in [100, 500, 3_000, 10_000]
        self.address = address
        self.token0 = token0
        self.token1 = token1
        self.fee = fee
        # tick_spacing is constant throughout contract's life
        self.tick_spacing = {
            # for mapping see: https://github.com/Uniswap/v3-core/blob/main/contracts/UniswapV3Factory.sol#L26
            100:     1,
            500:    10,
            3_000:  60,
            10_000: 200,
        }[fee]
        self.set_web3(w3)

    def get_slot0(self, block_identifier, use_cache = True) -> typing.Tuple[int, int]:
        assert self.last_block_observed is None or self.last_block_observed <= block_identifier
        if use_cache == False or self.slot0_cache is None:
            with profile('uniswap_v3_fetch'):
                (sqrt_ratio_x96, tick, _, _, _, _, _) = self.contract.functions.slot0().call(block_identifier=block_identifier)
            if use_cache == False:
                return (sqrt_ratio_x96, tick)
            self.slot0_cache = (sqrt_ratio_x96, tick)
        return self.slot0_cache

    def get_liquidity(self, block_identifier, use_cache = True) -> int:
        assert self.last_block_observed is None or self.last_block_observed <= block_identifier
        if use_cache == False or self.liquidity_cache is None:
            with profile('uniswap_v3_fetch'):
                got = self.contract.functions.liquidity().call(block_identifier=block_identifier)
            if use_cache == False:
                l.debug(f'not using cache')
                return got
            self.liquidity_cache = got
        return self.liquidity_cache

    def quote_token0_to_token1(self, amount0, block_identifier) -> int:
        sqrtPricex96, _ = self.get_slot0(block_identifier)
        assert isinstance(sqrtPricex96, int)
        assert isinstance(amount0, int)
        ratiox192 = sqrtPricex96 ** 2
        return ((ratiox192 * amount0 // (1 << 192)) * ((10 ** 6) - self.fee)) // (10 ** 6)

    def quote_token1_to_token0(self, amount1, block_identifier) -> int:
        sqrtPricex96, _ = self.get_slot0(block_identifier)
        assert isinstance(sqrtPricex96, int)
        assert isinstance(amount1, int)
        ratiox192 = sqrtPricex96 ** 2
        return (1 << 192) * (amount1) // ratiox192 * ((10 ** 6) - self.fee) // (10 ** 6)

    def exact_token0_to_token1(self, token0_in: int, block_identifier) -> int:
        (_, ret) = self.swap(zero_for_one=True, amount_specified=token0_in, sqrt_price_limitX96=None, block_identifier=block_identifier)
        return -ret

    def exact_token1_to_token0(self, token1_in: int, block_identifier) -> int:
        (ret, _) = self.swap(zero_for_one=False, amount_specified=token1_in, sqrt_price_limitX96=None, block_identifier=block_identifier)
        return -ret

    def swap(self, zero_for_one: bool, amount_specified: int, sqrt_price_limitX96: typing.Optional[int], block_identifier) -> typing.Tuple[int, int]:
        """
        returns: (amount0, amount1)
        """
        assert amount_specified >= 0
        assert isinstance(zero_for_one, bool)

        (sqrt_price_x96, tick) = self.get_slot0(block_identifier)
        if sqrt_price_x96 == 0:
            # this is not initialized; you cannot get any token out
            return (0, 0)
        liquidity = self.get_liquidity(block_identifier)

        if zero_for_one:
            if sqrt_price_limitX96 == None:
                sqrt_price_limitX96 = UniswapV3Pricer.MIN_SQRT_RATIO + 1
            assert sqrt_price_limitX96 <= sqrt_price_x96, f'expected {sqrt_price_limitX96} <= {sqrt_price_x96}'
            assert sqrt_price_limitX96 > UniswapV3Pricer.MIN_SQRT_RATIO
        else:
            if sqrt_price_limitX96 == None:
                sqrt_price_limitX96 = UniswapV3Pricer.MAX_SQRT_RATIO - 1
            assert sqrt_price_limitX96 >= sqrt_price_x96, f'expected {sqrt_price_limitX96} <= {sqrt_price_x96}'
            assert sqrt_price_limitX96 < UniswapV3Pricer.MAX_SQRT_RATIO

        amount_specified_remaining = amount_specified
        amount_calculated = 0

        while amount_specified_remaining != 0 and sqrt_price_x96 != sqrt_price_limitX96:
            # print(f'amount_specified_remaining={amount_specified_remaining} sqrt_price_limit={sqrt_price_limitX96} sqrt_price={sqrt_price_x96} liquidity={liquidity} tick={tick}')
            sqrt_price_start_x96 = sqrt_price_x96
            # compute tickNext
            next_tick_num, initialized = self.next_initialized_tick_within_one_word(
                tick, zero_for_one,
                block_identifier
            )

            if next_tick_num < UniswapV3Pricer.MIN_TICK:
                next_tick_num = UniswapV3Pricer.MIN_TICK
            elif next_tick_num > UniswapV3Pricer.MAX_TICK:
                next_tick_num = UniswapV3Pricer.MAX_TICK

            sqrt_price_next_X96 = UniswapV3Pricer.get_sqrt_ratio_at_tick(next_tick_num)

            if zero_for_one:
                use_limit = sqrt_price_next_X96 < sqrt_price_limitX96
            else:
                use_limit = sqrt_price_next_X96 > sqrt_price_limitX96
            if use_limit:
                limit_to_use = sqrt_price_limitX96
            else:
                limit_to_use = sqrt_price_next_X96
            sqrt_price_x96, amount_in, amount_out, fee_amount = UniswapV3Pricer.compute_swap_step(
                sqrt_price_x96,
                limit_to_use,
                liquidity,
                amount_specified_remaining,
                self.fee
            )

            # assume exact_input = true
            amount_specified_remaining -= (amount_in + fee_amount)
            assert amount_specified_remaining >= 0
            amount_calculated = amount_calculated - amount_out

            if sqrt_price_next_X96 == sqrt_price_next_X96:
                if initialized:
                    tick_obj = self.tick_at(next_tick_num, block_identifier)
                    if zero_for_one:
                        liquidity -= tick_obj.liquidity_net
                    else:
                        liquidity += tick_obj.liquidity_net
                    assert liquidity >= 0
                tick = next_tick_num - 1 if zero_for_one else next_tick_num
            elif sqrt_price_x96 != sqrt_price_start_x96:
                raise NotImplementedError('never got around to this')
                # tick = UniswapV3Pricer.get_tick_at_sqrt_ratio(sqrt_price_x96)

        if zero_for_one:
            amount0 = amount_specified - amount_specified_remaining
            amount1 = amount_calculated
        else:
            amount0 = amount_calculated
            amount1 = amount_specified - amount_specified_remaining

        return (amount0, amount1)


    @staticmethod
    def compute_swap_step(
            sqrt_ratio_currentx96,
            sqrt_ratio_targetx96,
            liquidity,
            amount_remaining,
            fee_pips
        ) -> typing.Tuple[int, int, int, int]:
        # note: assume exactIn = true
        assert amount_remaining >= 0
        assert liquidity >= 0
        zero_for_one = sqrt_ratio_currentx96 >= sqrt_ratio_targetx96
        amount_remaining_less_fee = amount_remaining * ((10 ** 6) - fee_pips) // (10 ** 6)

        if zero_for_one:
            amount_in = UniswapV3Pricer.get_amount0_delta(
                sqrt_ratio_targetx96, sqrt_ratio_currentx96, liquidity, True
            )
        else:
            amount_in = UniswapV3Pricer.get_amount1_delta(
                sqrt_ratio_currentx96, sqrt_ratio_targetx96, liquidity, True
            )

        if amount_remaining_less_fee >= amount_in:
            sqrt_ratio_nextX96 = sqrt_ratio_targetx96
        else:
            sqrt_ratio_nextX96 = UniswapV3Pricer.get_next_sqrt_price_from_input(
                sqrt_ratio_currentx96,
                liquidity,
                amount_remaining_less_fee,
                zero_for_one
            )

        max_: bool = sqrt_ratio_targetx96 == sqrt_ratio_nextX96

        if zero_for_one:
            amount_in = amount_in if max_ else UniswapV3Pricer.get_amount0_delta(sqrt_ratio_nextX96, sqrt_ratio_currentx96, liquidity, True)
            amount_out = UniswapV3Pricer.get_amount1_delta(sqrt_ratio_nextX96, sqrt_ratio_currentx96, liquidity, False)
        else:
            amount_in = amount_in if max_ else UniswapV3Pricer.get_amount1_delta(sqrt_ratio_currentx96, sqrt_ratio_nextX96, liquidity, True)
            amount_out = UniswapV3Pricer.get_amount0_delta(sqrt_ratio_currentx96, sqrt_ratio_nextX96, liquidity, False)

        if sqrt_ratio_nextX96 != sqrt_ratio_targetx96:
            fee_amount = amount_remaining - amount_in
        else:
            fee_amount = UniswapV3Pricer.mul_div_rounding_up(
                amount_in,
                fee_pips,
                (10 ** 6 - fee_pips)
            )

        return (sqrt_ratio_nextX96, amount_in, amount_out, fee_amount)

    @staticmethod
    def get_next_sqrt_price_from_input(sqrt_pX96: int, liquidity: int, amount_in: int, zero_for_one: bool) -> int:
        assert sqrt_pX96 > 0
        assert liquidity > 0
        assert amount_in >= 0

        if zero_for_one:
            # getNextSqrtPriceFromAmount0RoundingUp
            # add = true
            if amount_in == 0:
                return sqrt_pX96
            numerator1 = liquidity << 96
            product = amount_in * sqrt_pX96
            if product // amount_in == sqrt_pX96:
                denominator = numerator1 + product
                if denominator >= numerator1:
                    return UniswapV3Pricer.mul_div_rounding_up(numerator1, sqrt_pX96, denominator)
            return UniswapV3Pricer.div_rounding_up(numerator1, ((numerator1 // sqrt_pX96) + amount_in))
        else:
            # getNextSqrtPriceFromAmount1RoundingDown
            # add = true
            quotient = (amount_in << 96) // liquidity
            return sqrt_pX96 + quotient

    @staticmethod
    def get_amount0_delta(sqrt_ratio_aX96: int, sqrt_ratio_bX96: int, liquidity: int, roundUp: bool = None) -> int:
        sqrt_ratio_aX96, sqrt_ratio_bX96 = sorted([sqrt_ratio_aX96, sqrt_ratio_bX96])
        assert sqrt_ratio_aX96 > 0

        numerator1 = liquidity << 96
        numerator2 = sqrt_ratio_bX96 - sqrt_ratio_aX96

        if roundUp is None:
            roundUp = not (liquidity < 0)

        if roundUp == True:
            # roundUp = true
            return UniswapV3Pricer.div_rounding_up(
                UniswapV3Pricer.mul_div_rounding_up(numerator1, numerator2, sqrt_ratio_bX96),
                sqrt_ratio_aX96
            )
        else:
            # roundUp = false
            assert roundUp == False
            return UniswapV3Pricer.mul_div(numerator1, numerator2, sqrt_ratio_bX96) // sqrt_ratio_aX96

    @staticmethod
    def get_amount1_delta(sqrt_ratio_aX96: int, sqrt_ratio_bX96: int, liquidity: int, roundUp: bool = None):
        sqrt_ratio_aX96, sqrt_ratio_bX96 = sorted([sqrt_ratio_aX96, sqrt_ratio_bX96])

        if roundUp is None:
            roundUp = not (liquidity < 0)

        if roundUp:
            # roundUp = true
            return UniswapV3Pricer.mul_div_rounding_up(liquidity, sqrt_ratio_bX96 - sqrt_ratio_aX96, 0x1000000000000000000000000)
        else:
            # roundUp = false
            return UniswapV3Pricer.mul_div(liquidity, sqrt_ratio_bX96 - sqrt_ratio_aX96, 0x1000000000000000000000000)

    @staticmethod
    def mul_div(a, b, d):
        assert a >= 0
        assert b >= 0
        assert d > 0
        return (a * b) // d

    @staticmethod
    def mul_div_rounding_up(a, b, d):
        result = UniswapV3Pricer.mul_div(a, b, d)
        if (a * b) % d > 0:
            return result + 1
        return result

    @staticmethod
    def div_rounding_up(x, y):
        # UnsafeMath
        assert x >= 0
        assert y > 0
        tmp = (x // y)
        if x % y > 0:
             return tmp + 1
        return tmp

    @staticmethod
    def div(x, y):
        """
        Returns x / y as done by EVM semantics (rounds toward zero)
        """
        assert isinstance(x, int)
        assert isinstance(y, int)
        result = x // y
        if result < 0 and x % y != 0:
            # python rounded to -inf, we need to bump up by 1 to round toward zero
            return result + 1
        return result


    @staticmethod
    def mod(x: int, y: int) -> int:
        return x % y


    def next_initialized_tick_within_one_word(self, tick: int, lte: bool, block_identifier) -> typing.Tuple[int, bool]:
        compressed = tick // self.tick_spacing
        # actually no need to round down, this is python behavior
        # if tick < 0 and tick % self.tick_spacing != 0:
        #     compressed -= 1 # round down, idk, just says to do so

        if lte:
            word_pos = compressed >> 8
            bit_pos = compressed % 256
            mask = (1 << bit_pos) - 1 + (1 << bit_pos)
            masked = self.get_tick_bitmap_word(word_pos, block_identifier) & mask

            initialized = masked != 0
            if initialized:
                ret = (compressed - (bit_pos - UniswapV3Pricer.most_significant_bit(masked))) * self.tick_spacing
            else:
                ret = (compressed - bit_pos) * self.tick_spacing
        else:
            word_pos = (compressed + 1) >> 8
            bit_pos = (compressed + 1) % 256
            # mask = ~((1 << bit_pos) - 1)
            mask_pre_negate = (1 << bit_pos) - 1
            mask = int.from_bytes(
                bytes(0xff ^ x for x in int.to_bytes(mask_pre_negate, length = 256 // 8, byteorder='big', signed=False)),
                byteorder='big',
                signed=False
            )
            masked = self.get_tick_bitmap_word(word_pos, block_identifier) & mask

            initialized = masked != 0
            if initialized:
                ret = (compressed + 1 + UniswapV3Pricer.least_significant_bit(masked) - bit_pos) * self.tick_spacing
            else:
                ret = (compressed + 1 + ((1 << 8) - 1) - bit_pos) * self.tick_spacing

        return ret, initialized


    def get_tick_bitmap_word(self, word_idx, block_identifier) -> int:
        if word_idx not in self.tick_bitmap_cache:
            with profile('uniswap_v3_fetch'):
                self.tick_bitmap_cache[word_idx] = \
                    self.contract.functions.tickBitmap(word_idx).call(block_identifier=block_identifier)
        return self.tick_bitmap_cache[word_idx]

    def tick_at(self, tick: int, block_identifier) -> Tick:
        assert UniswapV3Pricer.MIN_TICK <= tick
        assert tick <= UniswapV3Pricer.MAX_TICK
        if tick not in self.tick_cache:
            with profile('uniswap_v3_fetch'):
                self.tick_cache[tick] = Tick(
                    tick,
                    *self.contract.functions.ticks(tick).call(block_identifier=block_identifier)
                )
        return self.tick_cache[tick]

    @staticmethod
    def least_significant_bit(x: int) -> int:
        assert x > 0

        r = 255
        if x & ((1 << 128) - 1) > 0:
            r -= 128
        else:
            x >>= 128
        if x & ((1 << 64) - 1) > 0:
            r -= 64
        else:
            x >>= 64
        if x & ((1 << 32) - 1) > 0:
            r -= 32
        else:
            x >>= 32
        if x & ((1 << 16) - 1) > 0:
            r -= 16
        else:
            x >>= 16
        if x & ((1 << 8) - 1) > 0:
            r -= 8
        else:
            x >>= 8
        if x & 0xf > 0:
            r -= 4
        else:
            x >>= 4
        if x & 0x3 > 0:
            r -= 2
        else:
            x >>= 2
        if x & 0x1 > 0:
            r -= 1
        return r

    @staticmethod
    def most_significant_bit(x: int) -> int:
        r = 0
        assert x > 0

        if x >= 0x100000000000000000000000000000000:
            x >>= 128
            r += 128
        if x >= 0x10000000000000000:
            x >>= 64
            r += 64
        if x >= 0x100000000:
            x >>= 32
            r += 32
        if x >= 0x10000:
            x >>= 16
            r += 16
        if x >= 0x100:
            x >>= 8
            r += 8
        if x >= 0x10:
            x >>= 4
            r += 4
        if x >= 0x4:
            x >>= 2
            r += 2
        if x >= 0x2:
            r += 1

        return r

    @staticmethod
    def get_sqrt_ratio_at_tick(tick_num: int) -> int:
        abs_tick = abs(tick_num)
        assert abs_tick <= UniswapV3Pricer.MAX_TICK
        # idk, taken from https://github.com/Uniswap/v3-core/blob/main/contracts/libraries/TickMath.sol#L24

        if abs_tick & 0x1 == 0:
            ratio = 0x100000000000000000000000000000000
        else:
            ratio = 0xfffcb933bd6fad37aa2d162d1a594001
        if abs_tick & 0x2 != 0:
            ratio = (ratio * 0xfff97272373d413259a46990580e213a) >> 128
        if abs_tick & 0x4 != 0:
            ratio = (ratio * 0xfff2e50f5f656932ef12357cf3c7fdcc) >> 128
        if abs_tick & 0x8 != 0:
            ratio = (ratio * 0xffe5caca7e10e4e61c3624eaa0941cd0) >> 128
        if abs_tick & 0x10 != 0:
            ratio = (ratio * 0xffcb9843d60f6159c9db58835c926644) >> 128
        if abs_tick & 0x20 != 0:
            ratio = (ratio * 0xff973b41fa98c081472e6896dfb254c0) >> 128
        if abs_tick & 0x40 != 0:
            ratio = (ratio * 0xff2ea16466c96a3843ec78b326b52861) >> 128
        if abs_tick & 0x80 != 0:
            ratio = (ratio * 0xfe5dee046a99a2a811c461f1969c3053) >> 128
        if abs_tick & 0x100 != 0:
            ratio = (ratio * 0xfcbe86c7900a88aedcffc83b479aa3a4) >> 128
        if abs_tick & 0x200 != 0:
            ratio = (ratio * 0xf987a7253ac413176f2b074cf7815e54) >> 128
        if abs_tick & 0x400 != 0:
            ratio = (ratio * 0xf3392b0822b70005940c7a398e4b70f3) >> 128
        if abs_tick & 0x800 != 0:
            ratio = (ratio * 0xe7159475a2c29b7443b29c7fa6e889d9) >> 128
        if abs_tick & 0x1000 != 0:
            ratio = (ratio * 0xd097f3bdfd2022b8845ad8f792aa5825) >> 128
        if abs_tick & 0x2000 != 0:
            ratio = (ratio * 0xa9f746462d870fdf8a65dc1f90e061e5) >> 128
        if abs_tick & 0x4000 != 0:
            ratio = (ratio * 0x70d869a156d2a1b890bb3df62baf32f7) >> 128
        if abs_tick & 0x8000 != 0:
            ratio = (ratio * 0x31be135f97d08fd981231505542fcfa6) >> 128
        if abs_tick & 0x10000 != 0:
            ratio = (ratio * 0x9aa508b5b7a84e1c677de54f3e99bc9) >> 128
        if abs_tick & 0x20000 != 0:
            ratio = (ratio * 0x5d6af8dedb81196699c329225ee604) >> 128
        if abs_tick & 0x40000 != 0:
            ratio = (ratio * 0x2216e584f5fa1ea926041bedfe98) >> 128
        if abs_tick & 0x80000 != 0:
            ratio = (ratio * 0x48a170391f7dc42444e8fa2) >> 128

        if tick_num > 0:
            uint256_max = (1 << 256) - 1
            ratio = (uint256_max) // ratio

        if ratio % (1 << 32) == 0:
            to_add = 0
        else:
            to_add = 1

        return (ratio >> 32) + to_add

    def observe_block(self, receipts: typing.List[web3.types.LogReceipt]):
        """
        Observe the logs emitted in a block and update internal state appropriately.
        NOTE: receipts _must_ be in sorted order of increasing log index
        """
        assert self.address == receipts[0]['address']

        if len(receipts) == 0:
            return

        block_num = receipts[0]['blockNumber']
        assert self.last_block_observed is None or self.last_block_observed < block_num
        self.last_block_observed = block_num

        # if any of the logs are a burn or mint, dump the liquidity cache
        for log in receipts:
            if len(log['topics']) > 0 and log['topics'][0] == UNIV3_SWAP_EVENT_TOPIC:
                swap = self.contract.events.Swap().processLog(log)
                sqrt_price_x96 = swap['args']['sqrtPriceX96']
                liquidity = swap['args']['liquidity']
                tick = swap['args']['tick']
                self.slot0_cache = (
                    sqrt_price_x96, tick
                )
                self.liquidity_cache = liquidity
            elif len(log['topics']) > 0 and log['topics'][0] in [UNIV3_BURN_EVENT_TOPIC, UNIV3_MINT_EVENT_TOPIC]:
                if log['topics'][0] == UNIV3_BURN_EVENT_TOPIC:
                    event = self.contract.events.Burn().processLog(log)
                else:
                    event = self.contract.events.Mint().processLog(log)
                tick_lower = event['args']['tickLower']
                tick_upper = event['args']['tickUpper']
                word_lower = (tick_lower // self.tick_spacing) >> 8
                word_upper = (tick_upper // self.tick_spacing) >> 8
                self.tick_bitmap_cache.pop(word_lower, None)
                self.tick_bitmap_cache.pop(word_upper, None)
                self.tick_cache.pop(tick_lower, None)
                self.tick_cache.pop(tick_upper, None)
                if self.liquidity_cache is not None and event['args']['amount'] != 0:
                    if self.slot0_cache is not None:
                        if tick_lower <= self.slot0_cache[1] < tick_upper:
                            if event['event'] == 'Burn':
                                self.liquidity_cache -= event['args']['amount']
                            else:
                                self.liquidity_cache += event['args']['amount']
                        else:
                            pass # nothing to do here, liquidity does not change
                    else:
                        # not sure what to do here bc we dont know the current tick so we don't know if we're in
                        # or out of range
                        self.liquidity_cache = None


    def set_web3(self, w3: web3.Web3):
        self.contract = w3.eth.contract(
            address = self.address,
            abi = get_abi('uniswap_v3/IUniswapV3Pool.json')['abi']
        )
        self.w3 = w3
        self.known_token0_bal = None
        self.known_token1_bal = None
        self.tick_cache = {}
        self.tick_bitmap_cache = {}
        self.slot0_cache = None
        self.liquidity_cache = None
        self.last_block_observed = None

