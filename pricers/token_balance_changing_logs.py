# the token USDN emits a Reward() event whenever it gives out staking rewards, which can change
# the balance of uniswap without emitting a Swap() event.
# If we see this after a sync, we must invalidate caches.
NEUTRINO_REWARD_TOPIC_HEX = '0x45cad8c10023de80f4c0672ff6c283b671e11aa93c92b9380cdf060d2790da52'
NEUTRINO_REWARD_TOPIC = bytes.fromhex(NEUTRINO_REWARD_TOPIC_HEX[2:])

# maps token addresses to log topics which invalidate
# balance caches
CACHE_INVALIDATING_TOKEN_LOGS = {
    '0x674C6Ad92Fd080e4004b2312b45f796a192D27a0': [NEUTRINO_REWARD_TOPIC],
}
