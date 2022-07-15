"""
pricers/token_transfer.py

Some tokens burn-on-transfer. Handle that logic here.
"""

SAITAMA_TOKEN = '0x8B3192f5eEBD8579568A2Ed41E6FEB402f93f73F'
SANSHU_INU_TOKEN = '0xc73C167E7a4Ba109e4052f70D5466D0C312A344D'
KISHU_INU_TOKEN = '0xA2b4C0Af19cC16a6CfAcCe81F192B024d625817D'
PAXOS_GOLD_TOKEN = '0x45804880De22913dAFE09f4980848ECE6EcbAf78'
DEGO_FINANCE_TOKEN = '0x88EF27e69108B2633F8E1C184CC37940A075cC02'
HOKKAIDU_INU_TOKEN = '0xC40AF1E4fEcFA05Ce6BAb79DcD8B373d2E436c4E'
FEG_TOKEN = '0x389999216860AB8E0175387A0c90E5c52522C945'
CULT_DAO_TOKEN = '0xf0f9D895aCa5c8678f706FB8216fa22957685A13'
OPEN_ALEXA_TOKEN = '0x1788430620960F9a70e3DC14202a3A35ddE1A316'
RYOSHIS_VISION_TOKEN = '0x777E2ae845272a2F540ebf6a3D03734A5a8f618e'
TENSET_TOKEN = '0x7FF4169a6B5122b664c51c95727d87750eC07c84'


def out_from_transfer(address: str, amount: int) -> int:
    """
    Get the amount actually sent to the recipient if transfer(.., amount) is called.
    """

    # NOTE if this ever becomes anything other than a flat percentage, then
    # we need to re-visit PricingCircuit.sample_new_price_ratio, which makes
    # exactly that assumption

    if address in [SAITAMA_TOKEN, SANSHU_INU_TOKEN, KISHU_INU_TOKEN, FEG_TOKEN, RYOSHIS_VISION_TOKEN, TENSET_TOKEN]:
        t_fee = amount // 100 * 2
        return amount - t_fee
    elif address in [OPEN_ALEXA_TOKEN]:
        return amount * 999 // 1_000
    elif address in [CULT_DAO_TOKEN]:
        return amount * 996 // 1_000
    elif address in [PAXOS_GOLD_TOKEN]:
        return amount * 9998 // 10_000
    elif address in [DEGO_FINANCE_TOKEN]:
        return amount * 998 // 1_000
    return amount

def in_from_transfer(address: str, amount: int, debug = False) -> int:
    """
    Get the amount needed to be sent to the recipient if `amount` is desired to be actually received.
    """
    raise NotImplementedError('not finished')

    if address in [SAITAMA_TOKEN, SANSHU_INU_TOKEN, KISHU_INU_TOKEN]:
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

