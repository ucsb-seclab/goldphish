"""
shooter/constants.py

Constants for the shooter contract
"""


# target block is identified (non-uniquely) by lower 2 bytes
MAX_TARGET_BLOCK = 0xffff

# Maximum wei transfer to coinbase supported, inclusive
MAX_COINBASE_XFER = 0xffffffffffffffff

# Maximum amountIn for the uniswap v3 flash swap that fits in the 'brief' segment
MAX_AMOUNT_IN_BRIEF = 0x3fffffffffffffffffff
# Maximum amountIn for the uniswap v3 flash that fits in the extended segment
MAX_AMOUNT_IN_SWAP = 0xffffffffffffffffffffffffffffffffffffffffffffffff


MAX_AMOUNT_OUT = 0xfffffffffffffffffffffff
MAX_AMOUNT_IN_UNISWAP_V2 = 0xffffffffffffffffffffffff
