import web3
import web3.types
import web3.contract
import typing
import logging
from eth_utils import event_abi_to_log_topic

from utils import get_abi

from pricers.base import BaseExchangePricer
from pricers.balancer_v2.common import POOL_BALANCE_CHANGED_TOPIC, POOL_REGISTERED_TOPIC, SWAP_TOPIC, TOKENS_DEREGISTERED_TOPIC, TOKENS_REGISTERED_TOPIC, PoolSpecialization, _vault, complement, div_down, div_up, downscale_down, get_pool_specialization, mul_down, mul_up, pow_up, pow_up_legacy, upscale


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

    def __init__(self, w3: web3.Web3, vault: web3.contract.Contract, address: str) -> None:
        self.address = address
        self.vault = vault
        self.w3 = w3
        self.contract: web3.contract.Contract = w3.eth.contract(
            address = address,
            abi = get_abi('balancer_v2/WeightedPool.json'),
        )
        self.pool_id = self.contract.functions.getPoolId().call()

        self._balance_cache = {}

        tokens, _, _ = self.vault.functions.getPoolTokens(self.pool_id).call()
        self.tokens = set(tokens)

        self.token_weights = {}
        weights = self.contract.functions.getNormalizedWeights().call()
        assert len(weights) == len(self.tokens)
        for t, w in zip(sorted(self.tokens, key=lambda x: bytes.fromhex(x[2:])), weights):
            self.token_weights[w3.toChecksumAddress(t)] = w

        self.swap_fee = None

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

    def swap_exact_amount_in(self, token_in: str, token_amount_in: int, token_out: str, block_identifier: int):
        assert token_in in self.tokens
        assert token_out in self.tokens

        swap_fee = self.get_swap_fee(block_identifier)
        assert swap_fee == self.contract.functions.getSwapFeePercentage().call(block_identifier=block_identifier)
        # assert swap_fee == self.contract.functions.getSwapFeePercentage().call(block_identifier=block_identifier + 1)
        # assert swap_fee == self.contract.functions.getSwapFeePercentage().call(block_identifier=block_identifier - 1)

        print('token_in before fee', token_amount_in)
        print('swap_fee', swap_fee)
        fee_amount = mul_up(token_amount_in, swap_fee)
        print('fee_amount', fee_amount)
        token_amount_in = token_amount_in - fee_amount

        balance_in = self.get_balance(token_in, block_identifier)
        balance_out = self.get_balance(token_out, block_identifier)

        # now we must upscale
        token_amount_in = upscale(self.w3, token_in, token_amount_in)
        balance_in = upscale(self.w3, token_in, balance_in)
        balance_out = upscale(self.w3, token_out, balance_out)

        weight_in  = self.token_weights[token_in]
        weight_out = self.token_weights[token_out]


        print('amount_in', token_amount_in)
        print('balance_in', balance_in)
        print('balance_out', balance_out)
        print('weight_in', weight_in)
        print('weight_out', weight_out)

        assert token_amount_in <= mul_down(balance_in, BalancerV2WeightedPoolPricer.MAX_IN_RATIO)

        denominator = balance_in + token_amount_in
        print('denominator', denominator)
        base = div_up(balance_in, denominator)
        print('base', base) # ok
        exponent = div_down(weight_in, weight_out)
        print('exponent', exponent) # ok I think?
        power_ = pow_up_legacy(base, exponent)
        print('power', power_) # wrong, should be 0xdc390e97f6c096f 991795675491207535

        print('power.complement', complement(power_))
        ret = mul_down(balance_out, complement(power_))
        print('ret', ret)
        
        return downscale_down(self.w3, token_out, ret)

    def observe_block(self, logs: typing.List[web3.types.LogReceipt]):
        for log in logs:

            if log['address'] == self.address and log['topics'][0] == SWAP_FEE_CHANGED_TOPIC:
                parsed = self.contract.events.SwapFeePercentageChanged().processLog(log)
                self.swap_fee = parsed['args']['swapFeePercentage']

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
