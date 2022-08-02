pragma solidity ^0.8.0;

import './interfaces/uniswap_v2/IUniswapV2Pair.sol';
import './interfaces/uniswap_v3/callback/IUniswapV3SwapCallback.sol';
import './interfaces/uniswap_v3/IUniswapV3Pool.sol';
import './interfaces/balancer_v1/IBPool.sol';
import './libraries/uniswap_v3/CallbackValidation.sol';
import './interfaces/IERC20.sol';
import './interfaces/balancer_v2/IVault.sol';


contract Shooter // is
    // IUniswapV3SwapCallback
{

    IVault internal constant BALANCER_VAULT = IVault(0xBA12222222228d8Ba445958a75a0704d566BF2C8);
    uint160 internal constant MIN_SQRT_RATIO = 4295128739;
    uint160 internal constant MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342;
    address internal constant UNISWAP_V3_FACTORY = 0x1F98431c8aD98523631AE4a59f267346ea31F984;
    // below deployer is for example purposes; private key = 0xf96003b86ed95cb86eae15653bf4b0bc88691506141a1a9ae23afd383415c268; contract address = 0x000000005bac821ef13ddd9573288e28c74695eb
    address internal constant deployer = 0xf637c126339E83E8073e912Cb11726764a56dd2a;
    address internal constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;

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

    fallback(bytes calldata input) external payable returns (bytes memory) {
        _doArbitrage(input, 4);
        return new bytes(0);
    }

    function _doArbitrage(
        bytes calldata data,
        uint offset
    ) internal {
        if (data.length <= offset) {
            return;
        }
        // first 2 or 3 bytes indicate arbitrage type
        // 0 = ignore
        // 1 = uniswap v2
        // 2 = uniswap v3
        // 3 = balancer v1
        // 4 = balancer v2

        uint header = uint256(bytes32(data[:32])) >> (22 - offset) * 8;
        uint n_exchanges = header >> (9 * 8);


        for (uint i=0; i < n_exchanges; i += 1) {
            uint type_ = (header >> (2 - i) * 3 * 8 + 16) & 0xff;
            uint loc_start = (header >> (2 - i) * 3 * 8) & 0xffff;

            if (type_ == 1)
            {
                doUniswapV2Swap(data, loc_start + offset);
            }
            else if (type_ == 2)
            {
                doUniswapV3Swap(data, loc_start + offset);
            }
            else if (type_ == 3)
            {
                doBalancerV1Swap(data, loc_start + offset);
            }
            else if (type_ == 4)
            {
                doBalancerV2Swap(data, loc_start + offset);
            }
            else
            {
                require(type_ == 0, "type_");
            }
        }
    }

    function doUniswapV2Swap(
        bytes calldata data,
        uint loc_start
    ) internal {
        uint256 amountIn   = uint256(bytes32(data[loc_start:loc_start+32]));
        uint256 amountOut  = uint256(bytes32(data[loc_start+32:loc_start+64]));
        address exchange   = address(bytes20(data[loc_start+64:loc_start+84]));
        address to         = address(bytes20(data[loc_start+84:loc_start+104]));
        bool    zeroForOne = uint8(bytes1(data[loc_start+104:loc_start+105])) != 0;

        if (amountIn > 0) {
            safeTransfer(WETH, exchange, amountIn);
        }

        // do the swap
        if (zeroForOne)
        {
            ((IUniswapV2Pair)(exchange)).swap(0, amountOut, to, new bytes(0));
        }
        else
        {
            ((IUniswapV2Pair)(exchange)).swap(amountOut, 0, to, new bytes(0));
        }
    }

    function doUniswapV3Swap(
        bytes calldata data,
        uint loc_start
    ) internal {
        int256  amountIn   = int256(uint256(bytes32(data[loc_start:loc_start+32])));
        address exchange   = address(bytes20(data[loc_start+32:loc_start+52]));
        // len(to) + len(zeroForOne) + len(extradata_len) = 20 + 1 + 2 = 23
        // 43 + 9 = 32
        uint256 extradataLen = uint256(bytes32(data[loc_start+43:loc_start+75]));
        address to           = address(uint160(extradataLen >> 8 * 3));
        bool    zeroForOne   = uint8(extradataLen >> 8 * 2) != 0;
        extradataLen &= 0xffff;

        bytes memory extradata = data[loc_start+75:loc_start+75+extradataLen];

        ((IUniswapV3Pool)(exchange)).swap(
            to,
            zeroForOne,
            amountIn,
            (zeroForOne ? MIN_SQRT_RATIO + 1 : MAX_SQRT_RATIO - 1),
            extradata
        );
    }

    function doBalancerV1Swap(
        bytes calldata data,
        uint loc_start
    ) internal {
        uint256 amountIn      = uint256(bytes32(data[loc_start:loc_start+32]));
        address exchange      = address(bytes20(data[loc_start+32:loc_start+52]));
        address tokenIn       = address(bytes20(data[loc_start+52:loc_start+72]));
        address tokenOut      = address(bytes20(data[loc_start+72:loc_start+92]));
        address to            = address(bytes20(data[loc_start+92:loc_start+112]));
        bool requiresApproval = uint8(data[loc_start+112]) != 0;

        if (requiresApproval) {
            ((IERC20)(tokenIn)).approve(exchange, 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff);
        }

        (uint amountOut, ) = ((IBPool)(exchange)).swapExactAmountIn(
            tokenIn,
            amountIn,
            tokenOut,
            1,
            0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        );
        if (to != address(this)) {
            safeTransfer(tokenOut, to, amountOut);
        }
    }

    function doBalancerV2Swap(
        bytes calldata data,
        uint loc_start
    ) internal {
        bytes32 poolId        = bytes32(data[loc_start:loc_start+32]);
        uint256 amountIn      = uint256(bytes32(data[loc_start+32:loc_start+64]));
        uint256 amountOut     = uint256(bytes32(data[loc_start+64:loc_start+96]));
        address tokenIn       = address(bytes20(data[loc_start+96:loc_start+116]));
        address tokenOut      = address(bytes20(data[loc_start+116:loc_start+136]));
        address payable to    = payable(address(bytes20(data[loc_start+136:loc_start+156])));

        IVault.FundManagement memory funds = IVault.FundManagement(
            address(this),
            false,
            to,
            false
        );

        IVault.SingleSwap memory swap = IVault.SingleSwap(
            poolId,
            IVault.SwapKind.GIVEN_IN,
            IAsset(tokenIn),
            IAsset(tokenOut),
            amountIn,
            new bytes(0)
        );

        BALANCER_VAULT.swap(
            swap,
            funds,
            amountOut,
            0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        );
    }

    function uniswapV3SwapCallback(
        int256 amount0Delta,
        int256 amount1Delta,
        bytes calldata data
    ) external {
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
        }

        _doArbitrage(data, 1);

        if (data[0] != 0) {
            // must send amount in weth to caller
            if (amount0Delta > 0) {
                safeTransfer(WETH, msg.sender, uint256(amount0Delta));
            }
            else if (amount1Delta > 0)
            {
                safeTransfer(WETH, msg.sender, uint256(amount1Delta));
            }
        }

        return;
    }

    function doApprove(IERC20 token, address delegate) external {
        require(msg.sender == deployer);
        (bool success, bytes memory data) = address(token).call(abi.encodeWithSelector(IERC20.approve.selector, delegate, 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff));
        require(success && (data.length == 0 || abi.decode(data, (bool))), 'Approve');
    }

    /** fallback to accept payments of Ether */
    receive() external payable {}

}
