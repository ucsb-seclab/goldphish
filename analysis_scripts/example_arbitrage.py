import matplotlib.pyplot as plt
import decimal
import numpy as np
import web3
import web3.types
import web3.contract
import typing
import logging
import scipy.optimize
from eth_utils import event_abi_to_log_topic

l = logging.getLogger(__name__)

class UniswapV2Pricer():
    address: str
    token0: str
    token1: str
    known_token0_bal: int
    known_token1_bal: int

    def __init__(self, address: str, token0: str, token1: str) -> None:
        self.address = address
        self.token0 = token0
        self.token1 = token1
        self.known_token0_bal = None
        self.known_token1_bal = None

    def __getstate__(self):
        return (self.address, self.token0, self.token1, self.known_token0_bal, self.known_token1_bal)

    def __setstate__(self, state):
        self.address, self.token0, self.token1, self.known_token0_bal, self.known_token1_bal = state

    def get_tokens(self, _) -> typing.Set[str]:
        return set([self.token0, self.token1])

    def get_balances(self, block_identifier) -> typing.Tuple[int, int]:
        if self.known_token0_bal is None or self.known_token1_bal is None:
            print('what')
            breserves = self.w3.eth.get_storage_at(self.address, block_identifier=block_identifier)

            reserve1 = int.from_bytes(breserves[4:18], byteorder='big', signed=False)
            reserve0 = int.from_bytes(breserves[18:32], byteorder='big', signed=False)

            self.known_token0_bal = reserve0
            self.known_token1_bal = reserve1
        return (self.known_token0_bal, self.known_token1_bal)

    def token_out_for_exact_in(self, token_in: str, token_out: str, amount_in: int, block_identifier: int, **_) -> typing.Tuple[int, float]:
        if token_in == self.token0 and token_out == self.token1:
            amt_out = self.exact_token0_to_token1(amount_in, block_identifier)
            new_reserve_in  = self.known_token0_bal + amount_in
            new_reserve_out = self.known_token1_bal - amt_out
        elif token_in == self.token1 and token_out == self.token0:
            amt_out = self.exact_token1_to_token0(amount_in, block_identifier)
            new_reserve_in  = self.known_token1_bal + amount_in
            new_reserve_out = self.known_token0_bal - amt_out
        else:
            raise NotImplementedError()

        if new_reserve_out == 0:
            spot = 0
        else:
            # how much out do we get for 1 unit in?
            # https://github.com/Uniswap/v2-periphery/blob/master/contracts/libraries/UniswapV2Library.sol#L43
            amount_in_with_fee = 1 * 997
            numerator = amount_in_with_fee * new_reserve_out
            denominator = new_reserve_in * 1_000 + amount_in_with_fee
            spot = numerator / denominator

        return (amt_out, spot)

    def exact_token0_to_token1(self, token0_amount, block_identifier: int) -> int:
        # based off https://github.com/Uniswap/v2-periphery/blob/master/contracts/libraries/UniswapV2Library.sol#L43
        bal0, bal1 = self.get_balances(block_identifier)
        if bal0 == 0 or bal1 == 0:
            return 0 # no amount can be taken out
        amt_in_with_fee = token0_amount * 997
        numerator = amt_in_with_fee * bal1
        denominator = bal0 * 1000 + amt_in_with_fee
        ret = numerator // denominator
        return ret

    def exact_token1_to_token0(self, token1_amount, block_identifier: int) -> int:
        # based off https://github.com/Uniswap/v2-periphery/blob/master/contracts/libraries/UniswapV2Library.sol#L43
        bal0, bal1 = self.get_balances(block_identifier)
        if bal0 == 0 or bal1 == 0:
            return 0 # no amount can be taken out
        amt_in_with_fee = token1_amount * 997
        numerator = amt_in_with_fee * bal0
        denominator = bal1 * 1000 + amt_in_with_fee
        return numerator // denominator

    def token1_out_to_exact_token0_in(self, token1_amount_out, block_identifier: int) -> int:
        return self.get_amount_in(token1_amount_out, True, block_identifier)

    def token0_out_to_exact_token1_in(self, token0_amount_out, block_identifier: int) -> int:
        return self.get_amount_in(token0_amount_out, False, block_identifier)

    def get_amount_in(self, amount_out: int, zero_for_one: bool, block_identifier=int) -> int:
        # out = ((in * 997 * b0) / (b1 * 1000 + in * 997))
        # out = 


        bal0, bal1 = self.get_balances(block_identifier)
        if bal0 == 0 or bal1 == 0:
            return 0 # no amount can be moved
        if not zero_for_one:
            bal0, bal1 = bal1, bal0
        assert amount_out <= bal1
        numerator = bal0 * amount_out * 1000
        denominator = (bal1 - amount_out) * 997
        return (numerator // denominator) + 1

    def get_value_locked(self, token_address: str, block_identifier: int, **_) -> int:
        bal0, bal1 = self.get_balances(block_identifier)
        if token_address == self.token0:
            return bal0
        elif token_address == self.token1:
            return bal1

        raise Exception(f'Do not know about token {token_address} in {self.address}')

    def get_token_weight(self, token_address: str, block_identifier: int) -> decimal.Decimal:
        return decimal.Decimal('0.5')

    def __str__(self) -> str:
        return f'<UniswapV2Pricer {self.address} token0={self.token0} token1={self.token1}>'


amm1 = UniswapV2Pricer(None, None, None)
amm1.token0 = 'token0'
amm1.token1 = 'token1'
amm1.known_token0_bal = 1_000_000
amm1.known_token1_bal = 2_000_000
amm2 = UniswapV2Pricer(None, None, None)
amm2.token0 = 'token0'
amm2.token1 = 'token1'
amm2.known_token0_bal = 1_500_000
amm2.known_token1_bal = 2_000_000

def sample(x):
    if x < 0:
        return 0
    out1 = amm1.exact_token0_to_token1(x, None)
    return amm2.exact_token1_to_token0(out1, None) - x

def sample_mp(x):
    out1, f1 = amm1.token_out_for_exact_in('token0', 'token1', x, None)
    _, f2 = amm2.token_out_for_exact_in('token1', 'token0', out1, None)
    return f1 * f2

opt = scipy.optimize.minimize_scalar(lambda x: -sample(int(x)), method='bounded', bounds=(2, 200_000))


xs = np.linspace(0, 300_000, 100)
ys = [sample(x) for x in xs]
ys_mp = [sample_mp(x) for x in xs]

fig, (p1, p2) = plt.subplots(nrows=2, ncols=1)
p1.plot(xs, ys)
p1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
p1.set_ylabel('Revenue')
p1.axvline(opt.x, color='black')
p2.plot(xs, ys_mp)
p2.set_ylabel('Marginal Revenue')
p2.axvline(opt.x, color='black')
p2.axhline(1, color='black')
plt.xlabel('Amount input')
fig.tight_layout()
plt.savefig('simple_example_optimization.png', format='png', dpi=300)
# plt.axhline(y=0, color='black')
plt.show()
