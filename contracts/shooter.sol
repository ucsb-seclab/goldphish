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
    // below deployer is for example purposes; private key = 0xf5987d356f3dfbf28c0cd1ba4d3ae438cd2a115918f9150d09a180991ea2803b
    address internal constant deployer = 0xA3dC6e48Ee8aBF19D3ee5f6aB799D566CC78F93e;

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

    function maybeDecodeUniswapV2ExtraData(uint256 data) view internal returns (address recipient, uint256 requiredIn) {
        requiredIn = (data >> 160);
        uint160 irecipient = uint160(data);
        if (irecipient == 0x0)
        {
            recipient = address(this);
        }
        else
        {
            recipient = address(irecipient);
        }
    }

    function uniswapV3SwapCallback(
        int256 amount0Delta,
        int256 amount1Delta,
        bytes calldata data
    ) external {
        address token0;
        address token1;
        {
            IUniswapV3Pool pool = IUniswapV3Pool(msg.sender);
            token0 = pool.token0();
            token1 = pool.token1();
            uint24  fee    = pool.fee();
            CallbackValidation.verifyCallback(
                UNISWAP_V3_FACTORY,
                token0,
                token1,
                fee
            );
        }

        address lastExchange = msg.sender;
        bool lastZeroForOne = amount0Delta > 0;

        uint256 cdata_idx = 0;
        while (cdata_idx < data.length)
        {
            uint256 cdataNext = uint256(bytes32(data[cdata_idx:cdata_idx+32]));
            bool zeroForOne = (cdataNext & (0x1 << 249)) != 0;
            address exchange = address(uint160(cdataNext));
            uint256 amountOut = (cdataNext >> 160) & 0xfffffffffffffffffffffff;

            if ((cdata_idx & (0x1 << 248)) != 0)
            {
                // uniswap v3
                // we need to copy the remaining exchange info into memory
                bytes memory cdata = data[cdata_idx+32:];

                // forward the uniswap v3 funding method info into callback
                cdata[0] = cdata[0] | bytes1(uint8((cdataNext >> 254) & 0x1) << 7);

                (bool success, bytes memory totallyunused) = exchange.call(
                    abi.encodeWithSelector(
                        IUniswapV3PoolActions.swap.selector,
                        address(this),
                        zeroForOne,
                        -int256(amountOut),
                        (
                            zeroForOne
                                ? MIN_SQRT_RATIO + 1
                                : MAX_SQRT_RATIO - 1
                        ),
                        cdata
                    )
                );

                require(success && totallyunused.length == 64);
                // assume remainder was handled recursively
                break;
            }
            else
            {
                // address recipient = address(this);

                // uniswap v2
                address recipient = address(this);
                if (cdataNext & (0x1 << 254) != 0)
                {
                    // extradata follows
                    uint256 requiredInput;
                    (recipient, requiredInput) = maybeDecodeUniswapV2ExtraData(uint256(bytes32(data[cdata_idx + 32 : cdata_idx + 2 * 32])));
                    if (requiredInput > 0)
                    {
                        address lastOutputToken;
                        // figure out what token we need to send
                        if (lastExchange == msg.sender)
                        {
                            lastOutputToken = lastZeroForOne ? token1 : token0;
                        }
                        else
                        {
                            // last exchange must necessarily have been v2, since we recurse to do more v3s
                            lastOutputToken = lastZeroForOne ? IUniswapV2Pair(lastExchange).token1() : IUniswapV2Pair(lastExchange).token0();
                        }
                        safeTransfer(lastOutputToken, exchange, requiredInput);
                    }

                    cdata_idx += 32;
                }
                cdata_idx += 32;
                (bool success, ) = exchange.call(
                    abi.encodeWithSelector(
                        IUniswapV2Pair.swap.selector,
                        zeroForOne ? amountOut : 0,
                        zeroForOne ? 0 : amountOut,
                        recipient,
                        new bytes(0)
                    )
                );
                require(success);


                lastExchange = exchange;
                lastZeroForOne = zeroForOne;
            }
        }

        if (uint8(data[0]) & (0x1 << 7) == 0) {
            // manually funded
            address neededToken = amount0Delta > 0 ? token0 : token1;
            int256 neededValue = int256(amount0Delta > 0 ? amount0Delta : amount1Delta);
            require(neededValue > 0);
            safeTransfer(neededToken, msg.sender, uint256(neededValue));
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
            uint deadline = input0 >> 228;
            if (block.number > deadline)
            {
                revert();
            }
        }

        uint256 coinbase_xfer = 0xffffffffffffffff & (input0 >> 160);

        // unpack parameters
        address exchange1 = address(uint160(input0));
        bool zeroForOne  = (input0 & (0x1 << 160)) != 0;
        int256 amount_in = int256(uint256(bytes32(input[36:68])));
        require(amount_in > 0);

        // we need to copy the remaining exchange info into memory
        bytes memory cdata = input[68:];

        // forward the uniswap v3 payment method info into callback
        cdata[0] = cdata[0] | bytes1(uint8((input0 >> (160 + 64 + 1)) & 0x1) << 7);

        (bool success, bytes memory data) = exchange1.call(
            abi.encodeWithSelector(
                IUniswapV3PoolActions.swap.selector,
                address(this),
                zeroForOne,
                amount_in,
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
        
        return new bytes(0);
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

}
