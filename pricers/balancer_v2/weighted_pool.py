import decimal
import web3
import web3.types
import web3.contract
import typing
import logging
from eth_utils import event_abi_to_log_topic
from pricers.balancer import BalancerPricer
from pricers.block_observation_result import BlockObservationResult

from utils import get_abi

from pricers.base import BaseExchangePricer, NotEnoughLiquidityException
from pricers.balancer_v2.common import ONE, POOL_BALANCE_CHANGED_TOPIC, POOL_REGISTERED_TOPIC, SWAP_TOPIC, TOKENS_DEREGISTERED_TOPIC, TOKENS_REGISTERED_TOPIC, _vault, complement, div_down, div_up, downscale_down, mul_down, mul_up, pow_up, pow_up_legacy, upscale


l = logging.getLogger(__name__)

_pool: web3.contract.Contract = web3.Web3().eth.contract(address=b'\x00'*20, abi=get_abi('balancer_v2/WeightedPool.json'))

SWAP_FEE_CHANGED_TOPIC = event_abi_to_log_topic(_pool.events.SwapFeePercentageChanged().abi)

class BalancerV2WeightedPoolPricer(BaseExchangePricer):
    w3: web3.Web3
    address: str
    vault: web3.contract.Contract
    contract: web3.contract.Contract
    pool_id: bytes
    _balance_cache: typing.Dict[str, int]

    swap_fee: int
    tokens: typing.Optional[typing.Set[str]]
    token_weights: typing.Dict[str, int]

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
        self.tokens = set(tokens)

        self.token_weights = {}
        weights = self.contract.functions.getNormalizedWeights().call()
        assert len(weights) == len(self.tokens)
        for t, w in zip(sorted(self.tokens, key=lambda x: bytes.fromhex(x[2:])), weights):
            self.token_weights[w3.toChecksumAddress(t)] = w

        self.swap_fee = None

    def get_tokens(self, _) -> typing.Set[str]:
        return self.tokens

    def get_swap_fee(self, block_identifier: int) -> int:
        if self.swap_fee is None:
            swap_fee = self.contract.functions.getSwapFeePercentage().call(block_identifier=block_identifier)
            self.swap_fee = swap_fee
        return self.swap_fee

    def get_balance(self, address: str, block_identifier: int) -> typing.Set[str]:
        if address not in self._balance_cache:
            tokens, balances, _ = self.vault.functions.getPoolTokens(self.pool_id).call(block_identifier=block_identifier)
            self.tokens = set(tokens)

            for t, b in zip(tokens, balances):
                assert isinstance(b, int)
                self._balance_cache[t] = b

        return self._balance_cache[address]            

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

        weight_in  = self.token_weights[token_in]
        weight_out = self.token_weights[token_out]

        max_in = mul_down(balance_in, BalancerV2WeightedPoolPricer.MAX_IN_RATIO)
        if token_amount_in > max_in:
            raise NotEnoughLiquidityException(None, None, token_amount_in - max_in)

        denominator = balance_in + token_amount_in
        base = div_up(balance_in, denominator)
        exponent = div_down(weight_in, weight_out)
        power_ = pow_up_legacy(base, exponent)

        ret = mul_down(balance_out, complement(power_))
        ret = downscale_down(self.w3, token_out, ret)

        spot_out = BalancerPricer.BONE / BalancerPricer.calc_spot_price(
            token_balance_in  = balance_in_not_scaled + token_amount_in_not_scaled,
            token_weight_in   = weight_in,
            token_balance_out = balance_out_not_scaled - ret,
            token_weight_out  = weight_out,
            swap_fee          = swap_fee
        )
        curr_price = ret / token_amount_in_not_scaled
        if ret > 100:
            assert curr_price > spot_out

        return ret, spot_out

    @staticmethod
    def spot(balance_in, weight_in, balance_out, weight_out) -> float:
        exponent = weight_in / weight_out
        power = pow(balance_in, exponent)
        complement = 1 - power
        return balance_out * complement

    def get_value_locked(self, token_address: str, block_identifier: int) -> int:
        assert token_address in self.get_tokens(block_identifier)

        return self.get_balance(token_address, block_identifier)

    def get_token_weight(self, token_address: str, _: int) -> decimal.Decimal:
        norm = self.token_weights[token_address]

        return decimal.Decimal(norm) / decimal.Decimal(ONE)

    def observe_block(self, logs: typing.List[web3.types.LogReceipt]) -> BlockObservationResult:
        tokens_modified = set()

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
                    self.tokens = set()

                elif log['topics'][0] == TOKENS_REGISTERED_TOPIC and log['topics'][1] == self.pool_id:
                    parsed = _vault.events.TokensRegistered().processLog(log)

                    if self.tokens is not None:
                        self.tokens.update(parsed['args']['tokens'])

                    for t in parsed['args']['tokens']:
                        assert t not in self._balance_cache or self._balance_cache[t] == 0
                        self._balance_cache[t] = 0

                elif log['topics'][0] == TOKENS_DEREGISTERED_TOPIC and log['topics'][1] == self.pool_id:
                    raise NotImplementedError('deregister is disallowed, afaik')

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
            swap_enabled = None,
            gradual_weight_adjusting_scheduled = None,
        )

    def copy_without_cache(self) -> 'BaseExchangePricer':
        return BalancerV2WeightedPoolPricer(
            self.w3, self.address, pool_id = self.pool_id
        )
