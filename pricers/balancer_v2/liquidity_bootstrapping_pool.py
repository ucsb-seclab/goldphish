import decimal
import web3
import web3.types
import web3.contract
import typing
import logging
from eth_utils import event_abi_to_log_topic
from pricers.balancer import BalancerPricer
from pricers.block_observation_result import BlockObservationResult

from utils import get_abi, get_block_timestamp

from pricers.base import BaseExchangePricer, NotEnoughLiquidityException
from pricers.balancer_v2.common import ONE, POOL_BALANCE_CHANGED_TOPIC, POOL_REGISTERED_TOPIC, SWAP_TOPIC, TOKENS_DEREGISTERED_TOPIC, TOKENS_REGISTERED_TOPIC, _vault, complement, div_down, div_up, downscale_down, mul_down, mul_up, pow_up, pow_up_legacy, spot, upscale


l = logging.getLogger(__name__)

_pool: web3.contract.Contract = web3.Web3().eth.contract(address=b'\x00'*20, abi=get_abi('balancer_v2/LiquidityBootstrappingPool.json'))

SWAP_FEE_CHANGED_TOPIC = event_abi_to_log_topic(_pool.events.SwapFeePercentageChanged().abi)
SWAP_ENABLED_SET_TOPIC = event_abi_to_log_topic(_pool.events.SwapEnabledSet().abi)
GRADUAL_WEIGHT_UPDATE_SCHEDULED = event_abi_to_log_topic(_pool.events.GradualWeightUpdateScheduled().abi)

def compress(x: int, bits: int) -> int:
    max_compressed_value = (1 << bits) - 1
    return div_up(mul_up(x, max_compressed_value), ONE)


def decompress(x: int, bits: int) -> int:
    max_compressed_value = (1 << bits) - 1
    return div_up(mul_up(x, ONE), max_compressed_value)

class DecodedPoolState(typing.NamedTuple):
    start_ts: int
    end_ts: int
    start_weights: typing.List[int]
    end_weights: typing.List[int]
    public_swap: typing.Optional[bool]

def decode(x: int) -> DecodedPoolState:
    end_ts   = x >> (256 - 32)
    start_ts = ((1 << 32) - 1) & (x >> (256 - 64))

    end_weights_bits = ((1 << 64) - 1) & (x >> (256 - 64 - 64))
    end_weights = []

    for i in range(4):
        ew_encoded = (end_weights_bits >> (16 * i)) & ((1 << 16) - 1)
        ew = decompress(ew_encoded, 16)
        end_weights.append(ew)

    start_weights_bits = (x >> 4) & ((1 << 124) - 1)
    start_weights = []

    for i in range(4):
        sw_encoded = (start_weights_bits >> (31 * i)) & ((1 << 31) - 1)
        sw = decompress(sw_encoded, 31)
        start_weights.append(sw)

    public_swap = x & 0x1 == 1

    return DecodedPoolState(
        start_ts = start_ts,
        end_ts   = end_ts,
        start_weights = start_weights,
        end_weights = end_weights,
        public_swap=public_swap,
    )

def get_interpolated_value(start_value: int, end_value: int, start_time: int, end_time: int, current_time: int) -> int:
    assert start_value >= 0
    assert end_value >= 0
    assert start_time >= 0
    assert end_time >= 0
    assert current_time >= 0

    assert start_time <= end_time

    # _calculateValueChangeProgress
    if current_time > end_time:
        pct_progress = ONE
    elif current_time < start_time:
        pct_progress = 0
    else:
        total_seconds = end_time - start_time
        seconds_elapsed = current_time - start_time

        if total_seconds == 0:
            pct_progress = ONE
        else:
            pct_progress = div_down(seconds_elapsed, total_seconds)
    
    # _interpolateValue
    if pct_progress == 0 or start_value == end_value:
        return start_value
    elif pct_progress >= ONE:
        return end_value
    elif start_value > end_value:
        delta = mul_down(pct_progress, start_value - end_value)
        return start_value - delta
    else:
        delta = mul_down(pct_progress, end_value - start_value)
        return start_value + delta


class BalancerV2LiquidityBootstrappingPoolPricer(BaseExchangePricer):
    w3: web3.Web3
    address: str
    vault: web3.contract.Contract
    contract: web3.contract.Contract
    pool_id: bytes
    _balance_cache: typing.Dict[str, int]

    swap_fee: int
    swap_enabled: typing.Optional[bool]
    tokens: typing.Tuple[str]
    pool_state: typing.Optional[DecodedPoolState]


    def __init__(self, w3: web3.Web3, vault: web3.contract.Contract, address: str, pool_id: typing.Optional[bytes] = None) -> None:
        self.address = address
        self.vault = vault
        self.w3 = w3
        self.contract: web3.contract.Contract = w3.eth.contract(
            address = address,
            abi = get_abi('balancer_v2/LiquidityBootstrappingPool.json'),
        )
        if pool_id is None:
            self.pool_id = self.contract.functions.getPoolId().call()
        else:
            self.pool_id = pool_id

        self._balance_cache = {}

        tokens, _, _ = self.vault.functions.getPoolTokens(self.pool_id).call()
        self.tokens = tuple(tokens)

        self.swap_fee = None
        self.pool_state = None
        self.swap_enabled = None

    def is_in_adjustment_range(self, timestamp: int, block_identifier: int) -> bool:
        """
        Returns True when the given timestamp is within adjustment
        range as of the given block identifier
        """
        ps = self.get_pool_state(block_identifier)
        return ps.start_ts < timestamp <= ps.end_ts

    def get_tokens(self, _) -> typing.Set[str]:
        return self.tokens

    def get_swap_enabled(self, block_identifier: int) -> bool:
        if self.swap_enabled is None:
            swap_enabled = self.contract.functions.getSwapEnabled().call(block_identifier=block_identifier)
            self.swap_enabled = swap_enabled
        return self.swap_enabled

    def get_swap_fee(self, block_identifier: int) -> int:
        if self.swap_fee is None:
            swap_fee = self.contract.functions.getSwapFeePercentage().call(block_identifier=block_identifier)
            self.swap_fee = swap_fee
        return self.swap_fee

    def get_balance(self, address: str, block_identifier: int) -> int:
        if address not in self._balance_cache:
            tokens, balances, _ = self.vault.functions.getPoolTokens(self.pool_id).call(block_identifier=block_identifier)
            self.tokens = tuple(tokens)

            for t, b in zip(tokens, balances):
                assert isinstance(b, int)
                self._balance_cache[t] = b

        return self._balance_cache[address]

    def get_pool_state(self, block_identifier: int) -> DecodedPoolState:
        if self.pool_state is None:
            storage = self.w3.eth.get_storage_at(self.address, 0xb, block_identifier=block_identifier)
            storage_i = int.from_bytes(storage, byteorder='big', signed=False)
            decoded = decode(storage_i)
            self.pool_state = decoded
        return self.pool_state

    def get_weight(self, address: str, block_identifier: int, ts_override: typing.Optional[int] = None) -> int:
        token_idx = self.tokens.index(address)
        pool_state = self.get_pool_state(block_identifier)

        if ts_override is None:
            ts = get_block_timestamp(self.w3, block_identifier)
        else:
            ts = ts_override

        return get_interpolated_value(
            pool_state.start_weights[token_idx],
            pool_state.end_weights[token_idx],
            pool_state.start_ts,
            pool_state.end_ts,
            ts,
        )

    MAX_IN_RATIO = 3 * 10 ** 17 # 0.3e18

    def token_out_for_exact_in(
            self,
            token_in: str,
            token_out: str,
            token_amount_in: int,
            block_identifier: int,
            timestamp: typing.Optional[int] = None,
            **_,
        ) -> typing.Tuple[int, float]:
        assert token_in in self.tokens, f'expected {token_in} in {self.tokens}'
        assert token_out in self.tokens, f'expected {token_out} in {self.tokens}'

        token_amount_in_not_scaled = token_amount_in

        swap_fee = self.get_swap_fee(block_identifier)

        fee_amount = mul_up(token_amount_in, swap_fee)
        token_amount_in = token_amount_in - fee_amount

        balance_in_not_scaled = self.get_balance(token_in, block_identifier)
        balance_out_not_scaled = self.get_balance(token_out, block_identifier)

        # now we must upscale
        token_amount_in = upscale(self.w3, token_in, token_amount_in)
        balance_in = upscale(self.w3, token_in, balance_in_not_scaled)
        balance_out = upscale(self.w3, token_out, balance_out_not_scaled)

        weight_in  = self.get_weight(token_in, block_identifier=block_identifier, ts_override=timestamp)
        weight_out = self.get_weight(token_out, block_identifier=block_identifier, ts_override=timestamp)

        max_in = mul_down(balance_in, BalancerV2LiquidityBootstrappingPoolPricer.MAX_IN_RATIO)
        if token_amount_in > max_in:
            raise NotEnoughLiquidityException(None, None, token_amount_in - max_in)

        denominator = balance_in + token_amount_in
        base = div_up(balance_in, denominator)
        exponent = div_down(weight_in, weight_out)
        power_ = pow_up_legacy(base, exponent)

        ret = mul_down(balance_out, complement(power_))
        ret = downscale_down(self.w3, token_out, ret)

        spot_out = spot(
            balance_in_not_scaled + token_amount_in_not_scaled,
            weight_in,
            balance_out_not_scaled - ret,
            weight_out,
            swap_fee
        )

        # if ret > 100 and token_amount_in_not_scaled > 0:
        #     curr_price = ret / token_amount_in_not_scaled
        #     # There's some weirdness here with marginal price when it takes a certain minimum amount to get any out
        #     # in the first place. Then we end up with a hockey-stick shape on a graph like so:
        #     #
        #     #            |     /
        #     #            |    /
        #     # amount_out |   /
        #     #            |  /
        #     #            +-----------------
        #     #                 amount_in
        #     #
        #     # which means that measuring current price as out / in isn't /really/ accurate since there's some "flat fee"
        #     # at the start there -- warn anyway when this situation occurs

        #     percent_diff = (spot_out - curr_price) / curr_price * 100
        #     if percent_diff >= 5:
        #         # only give a warning for this -- 
        #         l.warning(f'diff       {percent_diff:.08}%')
        #         l.warning(f'spot_out   {spot_out}')
        #         l.warning(f'curr_price {curr_price}')
        #         l.warning(f'address    {self.address}')
        #         l.warning(f'pool_id    {self.pool_id.hex()}')
        #         l.warning(f'block      {block_identifier}')
        #         l.warning(f'token_in   {token_in}')
        #         l.warning(f'token_out  {token_out}')
        #         l.warning(f'amount_in  {token_amount_in_not_scaled}')
        #         l.warning(f'amount_out {ret}')
        #         # raise Exception('this should not occur')
        #     return ret, spot_out

        return ret, spot_out

    def get_value_locked(self, token_address: str, block_identifier: int) -> int:
        assert token_address in self.get_tokens(block_identifier)

        return self.get_balance(token_address, block_identifier)

    def get_token_weight(self, token_address: str, block_identifier: int) -> decimal.Decimal:
        norm = self.get_token_weight(token_address, block_identifier)
        return decimal.Decimal(norm) / decimal.Decimal(ONE)

    def observe_block(self, logs: typing.List[web3.types.LogReceipt]) -> BlockObservationResult:
        tokens_modified = set()
        swap_enabled = None
        gradual_weight_update_scheduled = False

        for log in logs:

            if log['address'] == self.address:
                if log['topics'][0] == SWAP_FEE_CHANGED_TOPIC:
                    parsed = self.contract.events.SwapFeePercentageChanged().processLog(log)
                    self.swap_fee = parsed['args']['swapFeePercentage']

                    # all exchange rates just updated
                    # tokens are immutable past initialization (WeightedPool + LiquditiyBootstrappingPool)
                    # so we can mark them all as updated
                    tokens = self.get_tokens(log['blockNumber'])
                    tokens_modified.update(tokens)

                elif log['topics'][0] == GRADUAL_WEIGHT_UPDATE_SCHEDULED:
                    parsed = self.contract.events.GradualWeightUpdateScheduled().processLog(log)

                    start_time = parsed['args']['startTime']
                    end_time = parsed['args']['endTime']
                    start_weights = parsed['args']['startWeights']
                    end_weights = parsed['args']['endWeights']

                    start_weights_real = [decompress(compress(x, 31), 31) for x in start_weights]
                    end_weights_real = [decompress(compress(x, 16), 16) for x in end_weights]

                    self.pool_state = DecodedPoolState(
                        start_ts      = start_time,
                        end_ts        = end_time,
                        start_weights = start_weights_real,
                        end_weights   = end_weights_real,
                        public_swap   = None,
                    )
                    gradual_weight_update_scheduled = True
                
                elif log['topics'][0] == SWAP_ENABLED_SET_TOPIC:
                    parsed = self.contract.events.SwapEnabledSet().processLog(log)
                    swap_enabled = parsed['args']['swapEnabled']
                    self.swap_enabled = swap_enabled

            if log['address'] == self.vault.address:
                if log['topics'][0] == SWAP_TOPIC and log['topics'][1] == self.pool_id:
                    parsed = _vault.events.Swap().processLog(log)
                    token_in   = parsed['args']['tokenIn']
                    token_out  = parsed['args']['tokenOut']
                    amount_in  = parsed['args']['amountIn']
                    amount_out = parsed['args']['amountOut']

                    if token_in in self._balance_cache:
                        self._balance_cache[token_in] += amount_in
                    
                    if token_out in self._balance_cache:
                        old_bal = self._balance_cache[token_out]
                        assert amount_out <= old_bal
                        self._balance_cache[token_out] = old_bal - amount_out

                    tokens_modified.add(token_in)
                    tokens_modified.add(token_out)

                elif log['topics'][0] == POOL_REGISTERED_TOPIC and log['topics'][1] == self.pool_id:
                    parsed = _vault.events.PoolRegistered().processLog(log)
                    assert parsed['args']['poolAddress'] == self.address
                    self.tokens = tuple()

                elif log['topics'][0] == TOKENS_REGISTERED_TOPIC and log['topics'][1] == self.pool_id:
                    print(f'txn {log["transactionHash"].hex()}')
                    parsed = _vault.events.TokensRegistered().processLog(log)

                    if self.tokens is not None:
                        self.tokens = tuple(parsed['args']['tokens'])
                    else:
                        assert self.tokens == tuple(parsed['args']['tokens'])

                    for t in parsed['args']['tokens']:
                        assert t not in self._balance_cache or self._balance_cache[t] == 0
                        self._balance_cache[t] = 0

                elif log['topics'][0] == TOKENS_DEREGISTERED_TOPIC and log['topics'][1] == self.pool_id:
                    raise NotImplementedError('deregister')
                    parsed = _vault.events.TokensDeregistered().processLog(log)
                    
                    if self.tokens is not None:
                        self.tokens.difference_update(parsed['args']['tokens'])

                    for t in parsed['args']['tokens']:
                        self._balance_cache.pop(t, None)

                elif log['topics'][0] == POOL_BALANCE_CHANGED_TOPIC and log['topics'][1] == self.pool_id:
                    parsed = _vault.events.PoolBalanceChanged().processLog(log)

                    for t, b_delta in zip(parsed['args']['tokens'], parsed['args']['deltas']):
                        if self.tokens is not None:
                            assert t in self.tokens

                        if t in self._balance_cache:
                            self._balance_cache[t] += b_delta

                    tokens_modified.update(parsed['args']['tokens'])

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
            swap_enabled = swap_enabled,
            gradual_weight_adjusting_scheduled = gradual_weight_update_scheduled,
        )

    def copy_without_cache(self) -> 'BaseExchangePricer':
        return BalancerV2LiquidityBootstrappingPoolPricer(
            self.w3, self.address, pool_id = self.pool_id
        )
