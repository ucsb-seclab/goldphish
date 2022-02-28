# GoldPhish - an ethereum arbitrage searcher & shooter

GoldPhish is an arbitrage shooter for the Ethereum blockchain.

It is planned to support Uniswap v3, v2, and SushiSwap (which is uniswap v2 abi-compatible).

## Planning

### Invocation ABI

```
ARBITRAGE WITH UNISWAP V3
==========================

function selector = 00000000

# First 32 bytes:
ffff # block target
    / # FLAGS
    /fffffffffffffffffff # amount in (brief)
                        ffffffffffffffffffffffffffffffffffffffff # flash swap addr

FLAGS:
_ _ x x
| | +-+--- belongs to amount in (brief)
| +------- 0 = send to self, 1 = send to next exchange addr
+--------- flash swap: zeroForOne

IF amount in == 0, use extradata below

Extradata, if included (following 32 bytes)
ffffffffffffffffffffffffffffffffffffffffffffffff # amount in, (extended) flash swap
                                                ffffffffffffffff # coinbase xfer (wei)

# Following: series of Action records

f # FLAGS
 fffffffffffffffffffffff # amountOut or, when amountOut can be inferred in uniswap v2, amountIn (see NOTE about amount in below)
                        ffffffffffffffffffffffffffffffffffffffff # exchange address
FLAGS:
_ _ _ _ x x x x
+-+ | +--- uniswap v2/v3 (0 = v2)
 |  +----- zeroForOne
 +-------- recipient (0 = self, 1 = msg.sender, 2 = next exchange, 3 = separate_profits)

When Uniswap v2, check if next exchange address is zero; if so, this is the amount to forward from self
ffffffffffffffffffffffff # amountInFromSelf
                        0000000000000000000000000000000000000000 # must = 0

When recipient = 3
f # FLAGS
 fffffffffffffffffffffff # the amount of WETH to take as profit
                        ffffffffffffffffffffffffffffffffffffffff # exchange address (uniswap v2)
FLAGS:
1 1 _ 0 x x x x
+-+ |
 |  |
 |  +---------- zeroForOne (in order to sell other token and buy WETH)
 +------------- indicates this splits-out profits

NOTE:
  When splitting out funds, uniswap v2 address is assumed to be already funded completely.
  We assume forwarding the unspent remainder of funds to the next uniswap v2 address; if none is found,
  we send them to msg.sender.


ARBITRAGE WITH ONLY UNISWAP V2
==============================

NOTE: this does NOT support flash-swaps

Function selector: 00000001



```
