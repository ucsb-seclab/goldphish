pragma solidity ^0.8.0;

import './interfaces/uniswap_v3/callback/IUniswapV3SwapCallback.sol';
import './interfaces/uniswap_v3/IUniswapV3Pool.sol';

contract Shooter is
    IUniswapV3SwapCallback
{

    uint160 internal constant MIN_SQRT_RATIO = 4295128739;
    uint160 internal constant MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342;
    address internal constant deployer = 0x0000000000c26A74238a3F53Acf4348a868605FB;

    function uniswapV3SwapCallback(
        int256 amount0Delta,
        int256 amount1Delta,
        bytes calldata data
    ) external {
        revert();
    }

    fallback(bytes calldata input) external returns (bytes memory) {
        {
            bytes4 sig = bytes4(input[:4]);

            if (sig != 0x00000000)
            {
                revert();
            }
        }

        {
            if (msg.sender != deployer)
            {
                revert();
            }
        }

        uint input0 = uint(bytes32(input[4:36]));
        
        {
            uint deadline = input0 >> 228;
            if (deadline > block.number)
            {
                revert();
            }
        }

        if ((input0 & (0xf << 224)) == 0x0)
        {
            // mode is zero
            uint coinbase_xfer = 0xfffffffffffffff & (input0 >> 164);

            // unpack parameters
            address exchange1 = address(uint160(input0));
            uint8 flags = uint8(input0 >> 160);
            bool zero_for_one  = (flags & (0x1 << 3)) != 0;
            bool send_to_self  = (flags & (0x1 << 2)) != 0;
            int256 amount_in = int256(uint256(bytes32(input[36:68])));

            address recipient;
            if (send_to_self) {
                recipient = address(this);
            }
            else {
                recipient = address(uint160(bytes20(input[80:100])));
            }

            // we need to copy the remaining exchange info into memory
            bytes memory cdata = input[68:];

            (bool success, bytes memory data) = exchange1.call(
                abi.encodeWithSelector(
                    IUniswapV3PoolActions.swap.selector,
                    recipient,
                    zero_for_one,
                    amount_in,
                    (
                        zero_for_one
                            ? MIN_SQRT_RATIO + 1
                            : MAX_SQRT_RATIO - 1
                    ),
                    cdata
                )
            );

            require(success && data.length == 64);

            if (coinbase_xfer > 0)
            {
                require(block.coinbase.send(coinbase_xfer));
            }
        }
    }
}
