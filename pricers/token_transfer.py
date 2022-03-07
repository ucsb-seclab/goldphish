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
