# GoldPhish - an ethereum arbitrage searcher & shooter

GoldPhish is an arbitrage shooter for the Ethereum blockchain.

It is planned to support Uniswap v3, v2, and SushiSwap (which is uniswap v2 abi-compatible).

## Planning

### Invocation ABI

```
Offset: 0
function selector, kept at zero for obfuscation purposes
> 00000000

# First 32 bytes:
fffffff # block deadline
       f # FLAGS
        ffffffffffffffff # coinbase transfer
                        ffffffffffffffffffffffffffffffffffffffff # flash swap addr

FLAGS:
_ _ _ _ _ _ _ _
< RESRVED > | +------- flash swap: zeroForOne
            +--------- 1 = auto-funded, 0 = manually-funded

# Next 32 bytes:
<int256 amountOut, flash swap>

# Following: series of Action records

f # FLAGS
 fffffffffffffffffffffff # amountOut
                        ffffffffffffffffffffffffffffffffffffffff # exchange address
FLAGS:
_ _ _ _ x x x x
| | | +--- uniswap v2/v3 (0 = v2)
| | +----- zeroForOne
| +------- when uniswap v3, 1 = auto-funded, 0 = manually funded; when uniswap v2, 1 = extra data follows
+--------- (reserved for internal use; indicates auto-funding status of previous univ3 call)

When Uniswap v2, followed by 32 bytes
ffffffffffffffffffffffff # amountIn (NOTE: high bit must be zero if specifying this, otherwise default of 0, address(this) is used)
                        ffffffffffffffffffffffffffffffffffffffff # address (recipient)

```
