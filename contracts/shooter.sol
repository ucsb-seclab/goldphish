pragma solidity ^0.8.0;

import './interfaces/uniswap_v2/IUniswapV2Pair.sol';
import './interfaces/uniswap_v3/callback/IUniswapV3SwapCallback.sol';
import './interfaces/uniswap_v3/IUniswapV3Pool.sol';
import './libraries/uniswap_v3/CallbackValidation.sol';
import './interfaces/IERC20.sol';

contract Shooter is
    IUniswapV3SwapCallback
{

    uint160 internal constant MIN_SQRT_RATIO = 4295128739;
    uint160 internal constant MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342;
    address internal constant UNISWAP_V3_FACTORY = 0x1F98431c8aD98523631AE4a59f267346ea31F984;
    // below deployer is for example purposes; private key = 0xab1179084d3336336d60b2ed654d99a21c2644cadd89fd3034ee592e931e4a77
    address internal constant deployer = 0x23E7D87AFF47ba3D65D7Ab2F8cbc9F1BB3DDD32d;

    constructor() {
        require(msg.sender == deployer);
    }


    /// @notice Transfers tokens from msg.sender to a recipient
    /// @dev Errors with ST if transfer fails
    /// @param token The contract address of the token which will be transferred
    /// @param to The recipient of the transfer
    /// @param value The value of the transfer
    function safeTransfer(
        address token,
        address to,
        uint256 value
    ) internal {
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(IERC20.transfer.selector, to, value));
        require(success && (data.length == 0 || abi.decode(data, (bool))), 'ST');
    }

    function doUniswapV3Swap(
            address exchange,
            uint256 amountOut,
            bool zeroForOne,
            uint8 recipientCode,
            uint256 cdata_idx,
            bytes calldata data
        ) internal {
        // uniswap v3
        // we need to copy the remaining exchange info into memory
        address recipient;
        if (recipientCode == 0x0)
        {
            // send to self
            recipient = address(this);
        }
        else if (recipientCode == 0x1)
        {
            // send to msg.sender
            recipient = msg.sender;
        }
        else
        {
            // send to next exchange address
            recipient = address(uint160(bytes20(data[cdata_idx+(32+12):cdata_idx+(32+12+20)])));
        }

        (bool success,) = exchange.call(
            abi.encodeWithSelector(
                IUniswapV3PoolActions.swap.selector,
                recipient,
                zeroForOne,
                -int256(amountOut),
                (
                    zeroForOne
                        ? MIN_SQRT_RATIO + 1
                        : MAX_SQRT_RATIO - 1
                ),
                data[cdata_idx+32:]
            )
        );

        require(success);
    }

    function uniswapV3SwapCallback(
        int256 amount0Delta,
        int256 amount1Delta,
        bytes calldata data
    ) external {
        address repaymentToken;
        address lastTokenSentToSelf;
        {
            address token0;
            address token1;
            IUniswapV3Pool pool = IUniswapV3Pool(msg.sender);
            token0 = pool.token0();
            token1 = pool.token1();
            uint24 fee = pool.fee();
            CallbackValidation.verifyCallback(
                UNISWAP_V3_FACTORY,
                token0,
                token1,
                fee
            );
            (lastTokenSentToSelf, repaymentToken) = amount0Delta < 0
                ? (token0, token1)
                : (token1, token0);
        }

        uint8 recipientCode = 0;
        uint256 cdata_idx = 0;
        while (cdata_idx < data.length)
        {
            uint256 cdataNext = uint256(bytes32(data[cdata_idx:cdata_idx+32]));

            // decode flags
            recipientCode = uint8(cdataNext >> 254);
            bool zeroForOne = (cdataNext & (0x1 << 253)) != 0;
            address exchange = address(uint160(cdataNext));
            uint256 amountOut = (cdataNext >> 160) & 0xfffffffffffffffffffffff;

            if (cdataNext & (0x1 << 252) != 0)
            {
                // broken out to avoid 'stack too deep' error (I guess)
                doUniswapV3Swap(exchange, amountOut, zeroForOne, recipientCode, cdata_idx, data);
                // assume remainder was handled recursively
                break;
            }
            else
            {
                address recipient;

                if (cdata_idx + 64 <= data.length)
                {
                    uint256 maybeExtraData = uint256(bytes32(data[cdata_idx + 32 : cdata_idx + 64]));
                    if (maybeExtraData & uint160(0x00ffffffffffffffffffffffffffffffffffffffff /* leading zero is deliberate */) == 0)
                    {
                        // using extradata mark
                        uint256 requiredInput = maybeExtraData >> 160;
                        safeTransfer(lastTokenSentToSelf, exchange, requiredInput);
                        
                        cdata_idx += 64;
                    }
                    else
                    {
                        cdata_idx += 32;
                    }
                }
                else
                {
                    // this is the last exchange; we can infer the amountOut so actually we need to swap some stuff
                    safeTransfer(lastTokenSentToSelf, exchange, amountOut);
                    amountOut = uint256(amount0Delta > 0 ? amount0Delta : amount1Delta);
                }

                if (recipientCode == 0x0)
                {
                    // send to self
                    recipient = address(this);
                    // only update if we need this info for forwarding payment to another uniswap v2 address
                    if (cdata_idx < data.length)
                    {
                        lastTokenSentToSelf = zeroForOne ? IUniswapV2Pair(exchange).token1() : IUniswapV2Pair(exchange).token0();
                    }
                }
                else if (recipientCode == 0x1)
                {
                    // send to msg.sender
                    recipient = msg.sender;
                }
                else
                {
                    // send to next exchange address directly
                    recipient = address(uint160(bytes20(data[cdata_idx+12:cdata_idx+32])));
                }

                (bool success, ) = exchange.call(
                    abi.encodeWithSelector(
                        IUniswapV2Pair.swap.selector,
                        zeroForOne ? 0 : amountOut,
                        zeroForOne ? amountOut : 0,
                        recipient,
                        new bytes(0)
                    )
                );
                require(success);
                cdata_idx += 32;
            }
        }

        if (recipientCode == 0x0) {
            // must manually forward payment to msg.sender
            int256 neededValue = int256(amount0Delta > 0 ? amount0Delta : amount1Delta);
            safeTransfer(repaymentToken, msg.sender, uint256(neededValue));
        }
    }

    fallback(bytes calldata input) external payable returns (bytes memory) {
        {
            bytes4 method_sel = bytes4(input[:4]);

            if (method_sel != 0x00000000)
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

        uint256 input0 = uint256(bytes32(input[4:36]));
        
        {
            uint blockTarget = input0 >> 240;
            if ((block.number) & 0xffff != blockTarget)
            {
                revert();
            }
        }

        // unpack flags
        address exchange1 = address(uint160(input0));
        bool zeroForOne  = (input0 & (0x1 << 239)) != 0;
        int256 amountIn = int256((input0 >> 160) & 0x3fffffffffffffffffff);
        uint256 coinbase_xfer = 0;
        address recipient;
        bytes memory cdata;
        if (amountIn == 0)
        {
            uint256 input1 = uint256(bytes32(input[36:68]));
            // use extradata for amountIn (and, potentially, coinbase xfer)
            coinbase_xfer = input1 & 0xffffffffffffffff;
            amountIn = int256(input1 >> 64);
            cdata = input[68:];
            recipient = 
                (input0 & (0x1 << 238)) == 0
                ? address(this)
                : address(uint160(bytes20(input[80:100])));
        }
        else
        {
            cdata = input[36:];
            recipient = 
                (input0 & (0x1 << 238)) == 0
                ? address(this)
                : address(uint160(bytes20(input[48:68])));
        }
        require(amountIn > 0);


        (bool success, bytes memory data) = exchange1.call(
            abi.encodeWithSelector(
                IUniswapV3PoolActions.swap.selector,
                address(this),
                zeroForOne,
                amountIn,
                (
                    zeroForOne
                        ? MIN_SQRT_RATIO + 1
                        : MAX_SQRT_RATIO - 1
                ),
                cdata
            )
        );

        require(success && data.length == 64);

        if (coinbase_xfer > 0)
        {
            block.coinbase.transfer(coinbase_xfer);
        }
    }

    /** fallback to accept payments of Ether */
    receive() external payable {}

    function withdraw(uint256 wad) external {
        require(msg.sender == deployer);
        payable(msg.sender).transfer(wad);
    }

    function withdrawToken(address token, uint256 amount) external {
        require(msg.sender == deployer);
        safeTransfer(token, msg.sender, amount);
    }

    // function sellAndWithdrawToken(address token, uint256 amount, address uniswapV3Exchange, )
    // {

    // }

}
