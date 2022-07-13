import functools
import typing
import enum
import web3
import web3.contract

from eth_utils import event_abi_to_log_topic
from utils import get_abi

VAULT_ADDRESS = '0xBA12222222228d8Ba445958a75a0704d566BF2C8'

_vault: web3.contract.Contract = web3.Web3().eth.contract(
    address = VAULT_ADDRESS,
    abi=get_abi('balancer_v2/Vault.json'),
)

SWAP_TOPIC                 = event_abi_to_log_topic(_vault.events.Swap().abi)
POOL_BALANCE_CHANGED_TOPIC = event_abi_to_log_topic(_vault.events.PoolBalanceChanged().abi)
TOKENS_REGISTERED_TOPIC    = event_abi_to_log_topic(_vault.events.TokensRegistered().abi)
TOKENS_DEREGISTERED_TOPIC  = event_abi_to_log_topic(_vault.events.TokensDeregistered().abi)
POOL_REGISTERED_TOPIC      = event_abi_to_log_topic(_vault.events.PoolRegistered().abi)


ONE  = 1 * 10 ** 18
TWO  = 1 * 10 ** 18
FOUR = 4 * 10 ** 18

ONE_20 = 1 * 10 ** 20

ONE_36 = 1 * 10 ** 36

MAX_POW_RELATIVE_ERROR = 10000

MAX_NATURAL_EXPONENT = 130 * 10 ** 18
MIN_NATURAL_EXPONENT = -41 * 10 ** 18

LN_36_LOWER_BOUND = ONE - 1 * 10 ** 17
LN_36_UPPER_BOUND = ONE + 1 * 10 ** 17

a0 = 38877084059945950922200000000000000000000000000000000000 # eˆ(x0) (no decimals)
x0 = 128000000000000000000 # 2ˆ7
x1 = 64000000000000000000 # 2ˆ6
a1 = 6235149080811616882910000000 # eˆ(x1) (no decimals)

x2 = 3200000000000000000000 # 2ˆ5
a2 = 7896296018268069516100000000000000 # eˆ(x2)
x3 = 1600000000000000000000 # 2ˆ4
a3 = 888611052050787263676000000 # eˆ(x3)
x4 = 800000000000000000000 # 2ˆ3
a4 = 298095798704172827474000 # eˆ(x4)
x5 = 400000000000000000000 # 2ˆ2
a5 = 5459815003314423907810 # eˆ(x5)
x6 = 200000000000000000000 # 2ˆ1
a6 = 738905609893065022723 # eˆ(x6)
x7 = 100000000000000000000 # 2ˆ0
a7 = 271828182845904523536 # eˆ(x7)
x8 = 50000000000000000000 # 2ˆ-1
a8 = 164872127070012814685 # eˆ(x8)
x9 = 25000000000000000000 # 2ˆ-2
a9 = 128402541668774148407 # eˆ(x9)
x10 = 12500000000000000000 # 2ˆ-3
a10 = 113314845306682631683 # eˆ(x10)
x11 = 6250000000000000000 # 2ˆ-4
a11 = 106449445891785942956 # eˆ(x11)


_sc_cache = {}
def get_scaling_factor(w3: web3.Web3, token: str) -> int:
    sc = _sc_cache.get(token, None)
    if sc is not None:
        return sc
    # need to load into cache
    erc20: web3.contract.Contract = w3.eth.contract(
        address=token,
        abi = get_abi('erc20.abi.json'),
    )
    d = erc20.functions.decimals().call()
    assert d <= 18
    diff = 18 - d
    
    sc = ONE * (10 ** diff)
    _sc_cache[token] = sc
    return sc


def upscale(w3: web3.Web3, token: str, amount: int) -> int:
    sc = get_scaling_factor(w3, token)
    return mul_down(amount, sc)

def downscale_down(w3: web3.Web3, token: str, amount: int) -> int:
    sc = get_scaling_factor(w3, token)
    return div_down(amount, sc)

def sol_signed_div(a: int, b: int) -> int:
    # solidity division rounds toward zero, but python rounds toward -inf
    if a < 0 and b > 0:
        return - ((-a) // b)
    if a > 0 and b < 0:
        return - (a // (-b))
    return a // b


def complement(x: int) -> int:
    assert x >= 0
    if x < ONE:
        return ONE - x
    return 0

def mul_down(a: int, b: int) -> int:
    assert a >= 0
    assert b >= 0
    product = a * b

    return product // ONE

def mul_up(a: int, b: int) -> int:
    assert a >= 0
    assert b >= 0

    product = a * b
    if product == 0:
        return 0

    return ((product - 1) // ONE) + 1

def div_up(a: int, b: int) -> int:
    assert a >= 0
    assert b > 0

    if a == 0:
        return 0
    
    a_inflated = a * ONE
    return ((a_inflated - 1) // b) + 1

def div_down(a: int, b: int) -> int:
    assert a >= 0
    assert b > 0

    if a == 0:
        return 0
    
    a_inflated = a * ONE
    return a_inflated // b

def pow_up(x: int, y: int) -> int:
    if y == ONE:
        return x
    
    if y == TWO:
        return mul_up(x, x)
    
    if y == FOUR:
        square = mul_up(x, x)
        return mul_up(square, square)

    raw = pow(x, y)
    max_error = mul_up(raw, MAX_POW_RELATIVE_ERROR) + 1

    if raw < max_error:
        return 0

    return raw + max_error

def pow_up_legacy(x: int, y: int) -> int:
    raw = pow(x, y)
    max_error = mul_up(raw, MAX_POW_RELATIVE_ERROR) + 1

    if raw < max_error:
        return 0

    return raw + max_error

def pow(x: int, y: int) -> int:
    assert x >= 0
    assert y >= 0

    if y == 0:
        return ONE

    if x == 0:
        return 0
    
    if LN_36_LOWER_BOUND <= x < LN_36_UPPER_BOUND:
        ln_36_x = _ln_36(x)
        logx_times_y = ((ln_36_x // ONE) * y + ((ln_36_x % ONE) * y) // ONE)
    else:
        logx_times_y = _ln(x) * y

    logx_times_y = sol_signed_div(logx_times_y, ONE)

    ret = exp(logx_times_y)
    return ret


def exp(x: int) -> int:
    assert x >= MIN_NATURAL_EXPONENT and x <= MAX_NATURAL_EXPONENT

    if x < 0:
        # We only handle positive exponents: e^(-x) is computed as 1 / e^x. We can safely make x positive since it
        # fits in the signed 256 bit range (as it is larger than MIN_NATURAL_EXPONENT).
        # Fixed point division requires multiplying by ONE_18.
        return sol_signed_div((ONE * ONE), exp(-x))


    # First, we use the fact that e^(x+y) = e^x * e^y to decompose x into a sum of powers of two, which we call x_n,
    # where x_n == 2^(7 - n), and e^x_n = a_n has been precomputed. We choose the first x_n, x0, to equal 2^7
    # because all larger powers are larger than MAX_NATURAL_EXPONENT, and therefore not present in the
    # decomposition.
    # At the end of this process we will have the product of all e^x_n = a_n that apply, and the remainder of this
    # decomposition, which will be lower than the smallest x_n.
    # exp(x) = k_0 * a_0 * k_1 * a_1 * ... + k_n * a_n * exp(remainder), where each k_n equals either 0 or 1.
    # We mutate x by subtracting x_n, making it the remainder of the decomposition.

    # The first two a_n (e^(2^7) and e^(2^6)) are too large if stored as 18 decimal numbers, and could cause
    # intermediate overflows. Instead we store them as plain integers, with 0 decimals.
    # Additionally, x0 + x1 is larger than MAX_NATURAL_EXPONENT, which means they will not both be present in the
    # decomposition.

    # For each x_n, we test if that term is present in the decomposition (if x is larger than it), and if so deduct
    # it and compute the accumulated product.

    if x >= x0:
        x -= x0
        firstAN = a0
    elif x >= x1:
        x -= x1
        firstAN = a1
    else:
        firstAN = 1 # One with no decimal places

    # We now transform x into a 20 decimal fixed point number, to have enhanced precision when computing the
    # smaller terms.
    x *= 100

    # `product` is the accumulated product of all a_n (except a0 and a1), which starts at 20 decimal fixed point
    # one. Recall that fixed point multiplication requires dividing by ONE_20.
    product = ONE_20

    if x >= x2:
        x -= x2
        product = sol_signed_div((product * a2), ONE_20)

    if x >= x3:
        x -= x3
        product = sol_signed_div((product * a3), ONE_20)

    if x >= x4:
        x -= x4
        product = sol_signed_div((product * a4), ONE_20)

    if x >= x5:
        x -= x5
        product = sol_signed_div((product * a5), ONE_20)

    if x >= x6:
        x -= x6
        product = sol_signed_div((product * a6), ONE_20)

    if x >= x7:
        x -= x7
        product = sol_signed_div((product * a7), ONE_20)

    if x >= x8:
        x -= x8
        product = sol_signed_div((product * a8), ONE_20)

    if x >= x9:
        x -= x9
        product = sol_signed_div((product * a9), ONE_20)


    # x10 and x11 are unnecessary here since we have high enough precision already.

    # Now we need to compute e^x, where x is small (in particular, it is smaller than x9). We use the Taylor series
    # expansion for e^x: 1 + x + (x^2 / 2!) + (x^3 / 3!) + ... + (x^n / n!).

    seriesSum = ONE_20; # The initial one in the sum, with 20 decimal places.
    # term; # Each term in the sum, where the nth term is (x^n / n!).

    # The first term is simply x.
    term = x
    seriesSum += term

    # Each term (x^n / n!) equals the previous one times x, divided by n. Since x is a fixed point number,
    # multiplying by it requires dividing by ONE_20, but dividing by the non-fixed point n values does not.

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 2)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 3)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 4)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 5)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 6)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 7)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 8)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 9)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 10)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 11)
    seriesSum += term

    term = sol_signed_div(sol_signed_div((term * x), ONE_20), 12)
    seriesSum += term

    # 12 Taylor terms are sufficient for 18 decimal precision.

    # We now have the first a_n (with no decimals), and the product of all other a_n present, and the Taylor
    # approximation of the exponentiation of the remainder (both with 20 decimals). All that remains is to multiply
    # all three (one 20 decimal fixed point multiplication, dividing by ONE_20, and one integer multiplication),
    # and then drop two digits to return an 18 decimal value.

    return sol_signed_div(sol_signed_div((product * seriesSum), ONE_20) * firstAN, 100)


def _ln(a: int) -> int:
        if a < ONE:
            # Since ln(a^k) = k * ln(a), we can compute ln(a) as ln(a) = ln((1/a)^(-1)) = - ln((1/a)). If a is less
            # than one, 1/a will be greater than one, and this if statement will not be entered in the recursive call.
            # Fixed point division requires multiplying by ONE_18.
            return -_ln(sol_signed_div((ONE * ONE), a))

        # First, we use the fact that ln^(a * b) = ln(a) + ln(b) to decompose ln(a) into a sum of powers of two, which
        # we call x_n, where x_n == 2^(7 - n), which are the natural logarithm of precomputed quantities a_n (that is,
        # ln(a_n) = x_n). We choose the first x_n, x0, to equal 2^7 because the exponential of all larger powers cannot
        # be represented as 18 fixed point decimal numbers in 256 bits, and are therefore larger than a.
        # At the end of this process we will have the sum of all x_n = ln(a_n) that apply, and the remainder of this
        # decomposition, which will be lower than the smallest a_n.
        # ln(a) = k_0 * x_0 + k_1 * x_1 + ... + k_n * x_n + ln(remainder), where each k_n equals either 0 or 1.
        # We mutate a by subtracting a_n, making it the remainder of the decomposition.

        # For reasons related to how `exp` works, the first two a_n (e^(2^7) and e^(2^6)) are not stored as fixed point
        # numbers with 18 decimals, but instead as plain integers with 0 decimals, so we need to multiply them by
        # ONE_18 to convert them to fixed point.
        # For each a_n, we test if that term is present in the decomposition (if a is larger than it), and if so divide
        # by it and compute the accumulated sum.

        sum = 0

        if a >= a0 * ONE:
            a = sol_signed_div(a, a0) # Integer, not fixed point division
            sum += x0

        if a >= a1 * ONE:
            a = sol_signed_div(a, a1) # Integer, not fixed point division
            sum += x1

        # All other a_n and x_n are stored as 20 digit fixed point numbers, so we convert the sum and a to this format.
        sum *= 100
        a *= 100

        # Because further a_n are  20 digit fixed point numbers, we multiply by ONE_20 when dividing by them.

        if a >= a2:
            a = sol_signed_div(a * ONE_20, a2)
            sum += x2


        if a >= a3:
            a = sol_signed_div(a * ONE_20, a3)
            sum += x3


        if a >= a4:
            a = sol_signed_div(a * ONE_20, a4)
            sum += x4


        if a >= a5:
            a = sol_signed_div(a * ONE_20, a5)
            sum += x5


        if a >= a6:
            a = sol_signed_div(a * ONE_20, a6)
            sum += x6


        if a >= a7:
            a = sol_signed_div(a * ONE_20, a7)
            sum += x7


        if a >= a8:
            a = sol_signed_div(a * ONE_20, a8)
            sum += x8


        if a >= a9:
            a = sol_signed_div(a * ONE_20, a9)
            sum += x9


        if a >= a10:
            a = sol_signed_div(a * ONE_20, a10)
            sum += x10


        if a >= a11:
            a = sol_signed_div(a * ONE_20, a11)
            sum += x11


        # a is now a small number (smaller than a_11, which roughly equals 1.06). This means we can use a Taylor series
        # that converges rapidly for values of `a` close to one - the same one used in ln_36.
        # Let z = (a - 1) / (a + 1).
        # ln(a) = 2 * (z + z^3 / 3 + z^5 / 5 + z^7 / 7 + ... + z^(2 * n + 1) / (2 * n + 1))

        # Recall that 20 digit fixed point division requires multiplying by ONE_20, and multiplication requires
        # division by ONE_20.
        z = sol_signed_div((a - ONE_20) * ONE_20, a + ONE_20)
        z_squared = sol_signed_div(z * z, ONE_20)

        # num is the numerator of the series: the z^(2 * n + 1) term
        num = z

        # seriesSum holds the accumulated sum of each term in the series, starting with the initial z
        seriesSum = num

        # In each step, the numerator is multiplied by z^2
        num = sol_signed_div(num * z_squared, ONE_20)
        seriesSum += sol_signed_div(num, 3)

        num = sol_signed_div(num * z_squared, ONE_20)
        seriesSum += sol_signed_div(num, 5)

        num = sol_signed_div(num * z_squared, ONE_20)
        seriesSum += sol_signed_div(num, 7)

        num = sol_signed_div(num * z_squared, ONE_20)
        seriesSum += sol_signed_div(num, 9)

        num = sol_signed_div(num * z_squared, ONE_20)
        seriesSum += sol_signed_div(num, 11)

        # 6 Taylor terms are sufficient for 36 decimal precision.

        # Finally, we multiply by 2 (non fixed point) to compute ln(remainder)
        seriesSum *= 2

        # We now have the sum of all x_n present, and the Taylor approximation of the logarithm of the remainder (both
        # with 20 decimals). All that remains is to sum these two, and then drop two digits to return a 18 decimal
        # value.

        return sol_signed_div(sum + seriesSum, 100)


def _ln_36(x: int) -> int:
    x *= ONE

    z = sol_signed_div(((x - ONE_36) * ONE_36), (x + ONE_36))
    z_squared = (z * z) // ONE_36

    # num is the numerator of the series: the z^(2 * n + 1) term
    num = z

    # seriesSum holds the accumulated sum of each term in the series, starting with the initial z
    seriesSum = num

    # In each step, the numerator is multiplied by z^2
    num = sol_signed_div((num * z_squared), ONE_36)
    seriesSum += sol_signed_div(num, 3)

    num = sol_signed_div((num * z_squared), ONE_36)
    seriesSum += sol_signed_div(num, 5)

    num = sol_signed_div((num * z_squared), ONE_36)
    seriesSum += sol_signed_div(num, 7)

    num = (num * z_squared) // ONE_36
    seriesSum += sol_signed_div(num, 9)

    num = sol_signed_div((num * z_squared), ONE_36)
    seriesSum += sol_signed_div(num, 11)

    num = sol_signed_div((num * z_squared), ONE_36)
    seriesSum += sol_signed_div(num, 13)

    num = sol_signed_div((num * z_squared), ONE_36)
    seriesSum += sol_signed_div(num, 15)

    # 8 Taylor terms are sufficient for 36 decimal precision.

    # All that remains is multiplying by 2 (non fixed point).

    return seriesSum * 2
