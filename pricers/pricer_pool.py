import collections
import typing
import time
import logging
import web3
import web3.types
from .base import BaseExchangePricer
from .uniswap_v2 import UniswapV2Pricer
from .uniswap_v3 import UniswapV3Pricer
import cachetools


l = logging.getLogger(__name__)


class PricerPool:
    """
    Contains a pool of pricers and some utility methods.
    Pricers may be materialized in memory, or not.
    LRU caching is in place.
    """
    STAT_LOG_PERIOD_SECONDS = 60 * 2

    _w3: web3.Web3
    _cache: cachetools.LRUCache
    _token_to_pools: typing.Dict[str, typing.List[str]]
    _token_pairs_to_pools: typing.Dict[typing.Tuple[str, str], typing.List[str]]
    _uniswap_v2_pools: typing.Dict[str, typing.Tuple[str, str]]
    _uniswap_v3_pools: typing.Dict[str, typing.Tuple[str, str, int]]
    _origin_blocks: typing.Dict[str, int]
    _cache_hits: int
    _cache_misses: int
    _last_stat_log_ts: float

    def __init__(self, w3: web3.Web3) -> None:
        self._cache = cachetools.LRUCache(maxsize=5_000)
        self._token_to_pools = collections.defaultdict(lambda: [])
        self._token_pairs_to_pools = collections.defaultdict(lambda: [])
        self._uniswap_v2_pools = {}
        self._uniswap_v3_pools = {}
        self._w3 = w3
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_stat_log_ts = time.time()
        self._origin_blocks = {}

    def add_uniswap_v2(self, address: str, token0: str, token1: str, origin_block: int):
        """
        Add the given uniswap v2 exchange details to the pricer pool.
        """
        assert web3.Web3.isChecksumAddress(address)
        assert web3.Web3.isChecksumAddress(token0)
        assert web3.Web3.isChecksumAddress(token1)
        assert origin_block > 0 # sanity check
        assert bytes.fromhex(token0[2:]) < bytes.fromhex(token1[2:])
        self._uniswap_v2_pools[address] = (token0, token1)
        self._token_to_pools[token0].append(address)
        self._token_to_pools[token1].append(address)
        self._token_pairs_to_pools[(token0, token1)].append(address)
        self._origin_blocks[address] = origin_block

    def add_uniswap_v3(self, address: str, token0: str, token1: str, fee: int, origin_block: int):
        """
        Add the given uniswap v2 exchange details to the pricer pool.
        """
        assert web3.Web3.isChecksumAddress(address)
        assert web3.Web3.isChecksumAddress(token0)
        assert web3.Web3.isChecksumAddress(token1)
        assert fee in [100, 500, 3_000, 10_000]
        assert bytes.fromhex(token0[2:]) < bytes.fromhex(token1[2:])
        assert origin_block > 0
        self._uniswap_v3_pools[address] = (token0, token1, fee)
        self._token_to_pools[token0].append(address)
        self._token_to_pools[token1].append(address)
        self._token_pairs_to_pools[(token0, token1)].append(address)
        self._origin_blocks[address] = origin_block

    def is_uniswap_v2(self, address: str) -> bool:
        if address in self._uniswap_v2_pools:
            return True
        assert address in self._uniswap_v3_pools, 'address was neither uniswap v2 nor v3'
        return False

    def get_exchanges_for(self, token_address: str, block_number: typing.Optional[int] = None) -> typing.Iterable[str]:
        """
        Gets an interable over all exchange addresses that pair this token.

        Optionally filter by block_num, which returns only exchanges available as of block_num + 1
        """
        if token_address not in self._token_to_pools:
            return

        if block_number is None:
            yield from self._token_to_pools[token_address]
        else:
            for address in self._token_to_pools[token_address]:
                if self._origin_blocks[address] <= block_number:
                    yield address

    def get_exchanges_for_pair(self, token0: str, token1: str, block_number: typing.Optional[int] = None) -> typing.Iterable[str]:
        assert token0 != token1
        if bytes.fromhex(token0[2:]) > bytes.fromhex(token1[2:]):
            token0, token1 = token1, token0

        if block_number is None:
            yield from self._token_pairs_to_pools[(token0, token1)]
        else:
            for address in self._token_pairs_to_pools[(token0, token1)]:
                if self._origin_blocks[address] <= block_number:
                    yield address

    def observe_block(self, logs: typing.List[web3.types.LogReceipt]) -> typing.Set[str]:
        """
        Update any (cached) pricers based on logs found in this new block.

        Returns a set of all exchange addresses that received updates
        """
        ret = set()

        # gather block logs by address
        gathered = collections.defaultdict(lambda: [])
        for log in logs:
            gathered[log['address']].append(log)

        # if we have a pricer in the cache, force it to observe the block
        for address in gathered:
            if address in self._cache:
                self._cache[address].observe_block(gathered[address])
            if address in self._uniswap_v2_pools or address in self._uniswap_v3_pools:
                ret.add(address)

        return ret

    def get_pricer_for(self, address: str) -> typing.Optional[BaseExchangePricer]:
        ret = None

        maybe_uv2 = self._uniswap_v2_pools.get(address)
        if maybe_uv2 is not None:
            token0, token1 = maybe_uv2
            ret = self._get_uniswap_v2_pricer(address, token0, token1)
        else:
            maybe_uv3 = self._uniswap_v3_pools.get(address)
            if maybe_uv3 is not None:
                token0, token1, fee = maybe_uv3
                ret = self._get_uniswap_v3_pricer(address, token0, token1, fee)

        self._maybe_log_stats()
        return ret

    def get_pair_for(self, address: str) -> typing.Tuple[str, str]:
        if address in self._uniswap_v2_pools:
            return self._uniswap_v2_pools[address]
        elif address in self._uniswap_v3_pools:
            return self._uniswap_v3_pools[address][:2]

    def origin_block_for(self, address: str) -> int:
        return self._origin_blocks[address]

    def _get_uniswap_v2_pricer(self, address: str, token0: str, token1: str) -> BaseExchangePricer:
        if address in self._cache:
            self._cache_hits += 1
            return self._cache[address]
        self._cache_misses += 1
        ret = UniswapV2Pricer(self._w3, address, token0, token1)
        self._cache[address] = ret
        return ret

    def _get_uniswap_v3_pricer(self, address: str, token0: str, token1: str, fee: int) -> BaseExchangePricer:
        if address in self._cache:
            self._cache_hits += 1
            return self._cache[address]
        self._cache_misses += 1
        ret = UniswapV3Pricer(self._w3, address, token0, token1, fee)
        self._cache[address] = ret
        return ret

    def _maybe_log_stats(self):
        if time.time() > self._last_stat_log_ts + self.__class__.STAT_LOG_PERIOD_SECONDS:
            # do log
            hits = self._cache_hits
            misses = self._cache_misses
            if hits + misses == 0:
                return
            hit_percent = hits / (hits + misses) * 100
            l.debug(f'Pricer pool cache stats: hits={hits} misses={misses} hit_percent={hit_percent:.2f}%')
            self._last_stat_log_ts = time.time()

    @property
    def exchange_count(self) -> int:
        return len(self._uniswap_v2_pools) + len(self._uniswap_v3_pools)
