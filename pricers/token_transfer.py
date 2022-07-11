"""
pricers/token_transfer.py

Some tokens burn-on-transfer. Handle that logic here.
"""

SAITAMA_TOKEN = '0x8B3192f5eEBD8579568A2Ed41E6FEB402f93f73F'
SANSHU_INU_TOKEN = '0xc73C167E7a4Ba109e4052f70D5466D0C312A344D'

def out_from_transfer(address: str, amount: int) -> int:
    """
    Get the amount actually sent to the recipient if transfer(.., amount) is called.
    """
    if address in [SAITAMA_TOKEN, SANSHU_INU_TOKEN]:
        t_fee = amount // 100 * 2
        return amount - t_fee
    return amount

def in_from_transfer(address: str, amount: int, debug = False) -> int:
    """
    Get the amount needed to be sent to the recipient if `amount` is desired to be actually received.
    """
    if address in [SAITAMA_TOKEN, SANSHU_INU_TOKEN]:
        # out = amount - (amount / 100) * 2
        # out = amount - amount * 2 / 100
        # out = amount * 100 / 100 - (amount * 2) / 100
        # out = (amount * 100 - amount * 2) / 100
        # out = (amount * 98) / 100
        # out * 100 / 98 = amount
        ret = amount * 100 // 98

        while True:
            out_ret = out_from_transfer(address, ret)
            assert out_ret >= amount
            if out_ret != amount:
                out_ret -= 1
            else:
                break

        assert out_ret == amount, f'expected {out_ret} == {amount} (amount={amount})'
        assert out_from_transfer(address, ret - 1) != amount

        if debug:
            print(f'adjusting reverse.... {out_ret} -> {amount}')
        return out_ret

    return amount

