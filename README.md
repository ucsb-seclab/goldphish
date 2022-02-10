# GoldPhish - an ethereum arbitrage searcher & shooter

GoldPhish is an arbitrage shooter for the Ethereum blockchain.

It is planned to support Uniswap v3, v2, and SushiSwap (which is uniswap v2 abi-compatible).

## Planning

### Invocation ABI

```
Offset: 0
function selector, kept at zero for obfuscation purposes
> 00000000

-- uint256 [4:36] --

Offset: 228 bits
block deadline, 28 bits (3 bytes + 1 nybble)
reverts if transaction occurs at block height higher than this value
example, block 14_000_000

Offset: 224 bits
MODE selector (1 nybble)

Offset: 164 bits
coinbase transfer amount, wei (7 bytes + 1 nybble)

----- WHEN MODE & 0x1 = 0 ----

Offset: 0 bits
Exchange 1 address: must be Uniswap v3, 20 bytes
example, uniswap v3 wbtc-usdc
> 99ac8ca7087fa4a2a1fb6357269965a2014abc35

Offset: 163 bits
Exchange 1: 0-for-1 (1 bit)

Offset: 162 bits
Exchange 1: send to self? (1 bit)

Offset: 161 bits (1 bit)
  reserved

Offset: 160 bits (1 bit)
  reserved

-- int[36:68] --

Offset: 0 bits
exact input to exchange 1, uint256

-- reset offset --

List of further exchange interactions, as follows:

  Interaction:

  Offset 0: exchange address (160 bits)
  92 bits / reserved (for use among callbacks)
  Offset 252: profit taking mode (1 bit)
    0 = take profits by transfer to weth
    1 = take profits by subtraction
  Offset 253: exchange type (0 = uniswap v2, 1 = uniswap v3)
  Offset 254: take profits now (1 bit)
  Offset 255: send to self (1 bit)
  IF take profits == 1:
  -- reset offset --
  Offset 0: uint256, amount to take
```
