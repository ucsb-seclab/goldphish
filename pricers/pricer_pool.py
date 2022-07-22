import collections
import pickle
import typing
import time
import logging
import os
import web3
import web3.contract
import web3.types

from utils.profiling import profile, inc_measurement
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.liquidity_bootstrapping_pool import BalancerV2LiquidityBootstrappingPoolPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.block_observation_result import BlockObservationResult
from utils import get_abi, BALANCER_VAULT_ADDRESS, get_block_timestamp
from .base import BaseExchangePricer
from .uniswap_v2 import UniswapV2Pricer
from .uniswap_v3 import UniswapV3Pricer
from .token_balance_changing_logs import CACHE_INVALIDATING_TOKEN_LOGS

import cachetools
import leveldb


l = logging.getLogger(__name__)


class MyLRUCacher(cachetools.LRUCache):

    def __init__(self, pool: 'PricerPool', maxsize: int, *args, **kwargs):
        super().__init__(maxsize, *args, **kwargs)
        self.pool = pool

    def popitem(self):
        k, v = super().popitem()
        self.pool._evicted(k, v)

_pool_id = 0

class PricerPool:
    """
    Contains a pool of pricers and some utility methods.
    Pricers may be materialized in memory, or not.
    LRU caching is in place.
    """
    STAT_LOG_PERIOD_SECONDS = 60 * 10

    _w3: web3.Web3
    _cache: typing.Dict[str, BaseExchangePricer]
    _evictable_cache: cachetools.LRUCache

    _token_to_pools: typing.Dict[str, typing.List[str]]
    _token_pairs_to_pools: typing.Dict[typing.Tuple[str, str], typing.List[str]]
    _uniswap_v2_pools: typing.Dict[str, typing.Tuple[str, str]]
    _sushiswap_v2_pools: typing.Dict[str, typing.Tuple[str, str]]
    _shibaswap_pools: typing.Dict[str, typing.Tuple[str, str]]
    _uniswap_v3_pools: typing.Dict[str, typing.Tuple[str, str, int]]
    _balancer_v1_pools: typing.Dict[str, typing.List[str]]
    _balancer_v2_pools: typing.Dict[str, typing.Tuple[typing.List[str], bytes, str]]
    _balancer_v2_pool_id_to_addr: typing.Dict[bytes, str]
    _balancer_v2_updating_pools: typing.List[BalancerV2LiquidityBootstrappingPoolPricer]
    _origin_blocks: typing.Dict[str, int]
    _cache_hits: int
    _soft_cache_hits: int
    _cache_misses: int
    _last_stat_log_ts: float
    _balancer_v2_vault: web3.contract.Contract

    def __init__(self, w3: web3.Web3, tmpdir: typing.Optional[str] = None) -> None:
        global _pool_id
        my_pool_id = _pool_id
        _pool_id += 1

        if tmpdir is not None:
            assert os.path.isdir(tmpdir)
            my_dir = os.path.join(tmpdir, str(my_pool_id))
            os.mkdir(my_dir)
            self._db = leveldb.LevelDB(filename=my_dir)
            self._evictable_cache = MyLRUCacher(self, 1_000)
            l.debug(f'Initialized pricing pool leveldb at {tmpdir}')
        else:
            self._db = None
            self._evictable_cache = cachetools.LRUCache(maxsize=1_000)

        self._cache = {} # infinite size cache

        self._token_to_pools = collections.defaultdict(lambda: [])
        self._token_pairs_to_pools = collections.defaultdict(lambda: [])
        self._uniswap_v2_pools = {}
        self._sushiswap_v2_pools = {}
        self._shibaswap_pools = {}
        self._uniswap_v3_pools = {}
        self._balancer_v1_pools = {}
        self._balancer_v2_pools = {}
        self._balancer_v2_pool_id_to_addr = {}
        self._balancer_v2_updating_pools = []
        self._w3 = w3
        self._cache_hits = 0
        self._soft_cache_hits = 0
        self._cache_misses = 0
        self._last_stat_log_ts = time.time()
        self._origin_blocks = {}
        self._balancer_v2_vault = w3.eth.contract(
            address=BALANCER_VAULT_ADDRESS,
            abi=get_abi('balancer_v2/Vault.json'),
        )

    def clear(self):
        """
        Reset the pricer pool
        """
        self._evictable_cache.clear()
        self._cache.clear()

    def monitored_addresses(self) -> typing.Set[str]:
        """
        Gets all addresses which must be monitored for logs
        """
        ret = set()
        ret.update(self._uniswap_v2_pools.keys())
        ret.update(self._sushiswap_v2_pools.keys())
        ret.update(self._shibaswap_pools.keys())
        ret.update(self._uniswap_v3_pools.keys())
        ret.update(self._balancer_v1_pools.keys())
        ret.update(self._balancer_v2_pools.keys())
        ret.add(BALANCER_VAULT_ADDRESS)
        return ret

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

    def add_sushiswap_v2(self, address: str, token0: str, token1: str, origin_block: int):
        """
        Add the given uniswap v2 exchange details to the pricer pool.
        """
        assert web3.Web3.isChecksumAddress(address)
        assert web3.Web3.isChecksumAddress(token0)
        assert web3.Web3.isChecksumAddress(token1)
        assert origin_block > 0 # sanity check
        assert bytes.fromhex(token0[2:]) < bytes.fromhex(token1[2:])
        self._sushiswap_v2_pools[address] = (token0, token1)
        self._token_to_pools[token0].append(address)
        self._token_to_pools[token1].append(address)
        self._token_pairs_to_pools[(token0, token1)].append(address)
        self._origin_blocks[address] = origin_block

    def add_shibaswap(self, address: str, token0: str, token1: str, origin_block: int):
        """
        Add the given uniswap v2 exchange details to the pricer pool.
        """
        assert web3.Web3.isChecksumAddress(address)
        assert web3.Web3.isChecksumAddress(token0)
        assert web3.Web3.isChecksumAddress(token1)
        assert origin_block > 0 # sanity check
        assert bytes.fromhex(token0[2:]) < bytes.fromhex(token1[2:])
        self._shibaswap_pools[address] = (token0, token1)
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

    def add_balancer_v1(self, address: str, origin_block: int):
        """
        Add the given balancer v1 exchange to the pricer pool
        """
        assert web3.Web3.isChecksumAddress(address)
        self._balancer_v1_pools[address] = []
        self._origin_blocks[address] = origin_block

    def add_balancer_v2(self, address: str, pool_id: bytes, pool_type: str, origin_block: int):
        """
        Add the given balancer v2 exchange to the pricer pool
        """
        assert pool_type in ['WeightedPool', 'WeightedPool2Tokens', 'LiquidityBootstrappingPool', 'NoProtocolFeeLiquidityBootstrappingPool']

        assert web3.Web3.isChecksumAddress(address)
        self._balancer_v2_pools[address] = ([], pool_id, pool_type)
        self._origin_blocks[address] = origin_block
        self._balancer_v2_pool_id_to_addr[pool_id] = address

    def warm(self, block_identifier: int):
        """
        Warm cache in prep for scrape starting at given block
        """
        l.debug('warming balancer v1 token addresses')
        for i, addr in enumerate(sorted(self._balancer_v1_pools)):
            
            origin_block = self._origin_blocks[addr]
            if origin_block > block_identifier:
                # not created yet
                continue

            b = BalancerPricer(self._w3, addr)
            if b.get_finalized(block_identifier):
                tokens = b.get_tokens(block_identifier)
                self._set_tokens(addr, tokens)

        l.debug('warming balancer v2 token addresses')
        vault: web3.contract.Contract = self._w3.eth.contract(
            address=BALANCER_VAULT_ADDRESS,
            abi=get_abi('balancer_v2/Vault.json'),
        )

        for i, addr in enumerate(sorted(self._balancer_v2_pools)):
            origin_block = self._origin_blocks[addr]
            if origin_block > block_identifier:
                # not created yet
                continue

            _, pool_id, pool_type = self._balancer_v2_pools[addr]
            if pool_type in ['WeightedPool', 'WeightedPool2Tokens']:
                b = BalancerV2WeightedPoolPricer(self._w3, vault, addr, pool_id)
                tokens = b.get_tokens(block_identifier)
                self._set_tokens(addr, tokens)
            elif pool_type in ['LiquidityBootstrappingPool', 'NoProtocolFeeLiquidityBootstrappingPool']:
                b = BalancerV2LiquidityBootstrappingPoolPricer(self._w3, vault, addr, pool_id)
                if b.get_swap_enabled(block_identifier):
                    tokens = b.get_tokens(block_identifier)
                    self._set_tokens(addr, tokens)
                self._balancer_v2_updating_pools(b)


    def _set_tokens(self, address: str, tokens: typing.List[str]):
        """
        For a multi-token pool, set all token-pairs
        """
        if address in self._balancer_v1_pools:
            old_tokens = self._balancer_v1_pools[address]
        elif address in self._balancer_v2_pools:
            old_tokens, _, _ = self._balancer_v2_pools[address]
        else:
            raise NotImplementedError(f'not sure how to handle {address}')

        # remove from self._token_pairs_to_pools
        for t0 in old_tokens:
            for t1 in old_tokens:
                if bytes.fromhex(t0[2:]) < bytes.fromhex(t1[2:]):
                    pool = (t0, t1)
                    self._token_pairs_to_pools[pool].remove(address)

        # remove from self._token_to_pools
        for t in old_tokens:
            self._token_to_pools[t].remove(address)

        # add to self._token_pairs_to_pools
        for t0 in tokens:
            for t1 in tokens:
                if bytes.fromhex(t0[2:]) < bytes.fromhex(t1[2:]):
                    pool = (t0, t1)
                    self._token_pairs_to_pools[pool].append(address)
        
        # add to self._token_to_pools
        for t in tokens:
            self._token_to_pools[t].append(address)
        
        old_tokens.clear()
        old_tokens.extend(tokens)

    def get_exchanges_for(self, token_address: str, block_number: typing.Optional[int] = None) -> typing.Iterable[str]:
        """
        Gets an iterable over all exchange addresses that pair this token.

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

    def observe_block(self, block_number: int, logs: typing.List[web3.types.LogReceipt]) -> typing.Dict[typing.Tuple[str, str], typing.List[str]]:
        """
        Update any (cached) pricers based on logs found in this new block.

        Returns a set of all token price-pairs that got updates
        """
        assert all(log['blockNumber'] == block_number for log in logs)

        ret = collections.defaultdict(lambda: [])

        # look for balancer v2 pricers that are currently updating prices based
        # on timestamp, regardless of logs
        ts = get_block_timestamp(self._w3, block_number)
        n_updating = 0
        for p in self._balancer_v2_updating_pools:
            if p.is_in_adjustment_range(ts + 13, block_number):
                n_updating += 1
                # price will update in next block most likely
                tokens = p.get_tokens(block_number)
                for t1 in tokens:
                    for t2 in tokens:
                        if bytes.fromhex(t1[2:]) < bytes.fromhex(t2[2:]):
                            ret[(t1, t2)].append(p.address)
        if n_updating > 0:
            l.debug(f'Have {n_updating} balancer v2 exchanges in dynamic update in block {block_number}')

        # gather block logs by address
        gathered: typing.Dict[str, typing.List[web3.types.LogReceipt]] = collections.defaultdict(lambda: [])
        for log in logs:
            gathered[log['address']].append(log)

        # materialize pricers on-demand
        update_results: typing.List[typing.Tuple[BaseExchangePricer, BlockObservationResult]] = []
        for address in gathered:
            if address == BALANCER_VAULT_ADDRESS:
                # gather by pool_id
                gathered_by_pool_id: typing.Dict[str, typing.List[web3.types.LogReceipt]] = collections.defaultdict(lambda: [])
                for log in gathered[BALANCER_VAULT_ADDRESS]:
                    if len(log['topics']) < 2:
                        continue
                    gathered_by_pool_id[log['topics'][1]].append(log)
                
                for maybe_poolid in gathered_by_pool_id:
                    maybe_addr = self._balancer_v2_pool_id_to_addr.get(maybe_poolid, None)
                    if maybe_addr is None:
                        continue

                    p = self.get_pricer_for(maybe_addr)

                    result = p.observe_block(gathered[address])
                    update_results.append((p, result))

            else:
                p = self.get_pricer_for(address)

                result = p.observe_block(gathered[address])
                update_results.append((p, result))

        for p, result in update_results:
            if result.swap_enabled == True and isinstance(p, BalancerPricer):
                # _just_ enabled swap
                block_number = logs[0]['blockNumber']
                self._set_tokens(p.address, p.get_tokens(block_number))
            elif result.swap_enabled == False:
                self._set_tokens(p.address, set())

            for pair in result.pair_prices_updated:
                ret[pair].append(p.address)

        return dict(ret)

        # For now -- do not care about misbehaving tokens that change balances

        # invalidated_tokens = set()
        # for address in gathered:
        #     if address in CACHE_INVALIDATING_TOKEN_LOGS:
        #         important_topics = CACHE_INVALIDATING_TOKEN_LOGS[address]
        #         for log in gathered[address]:
        #             if len(log['topics']) > 0 and log['topics'][0] in important_topics:
        #                 # ugh, we need to do invalidation
        #                 invalidated_tokens.add(address)
        #                 break

        # for address in invalidated_tokens:
        #     for pool_address in self._token_to_pools[address]:
        #         # invalidate only if uniswap v2 (v3 doesn't care about balance shuffling?)
        #         if pool_address in self._uniswap_v2_pools:
        #             # mark exchange as changed
        #             ret.add(pool_address)
        #             if pool_address in self._cache:
        #                 # invalidate cache
        #                 del self._cache[pool_address]

        # return ret        

    def get_pricer_for(self, address: str) -> typing.Optional[BaseExchangePricer]:
        with profile('get_pricer_for'):
            self._maybe_log_stats()

            maybe_cached_pricer = self._evictable_cache.get(address, None) or self._cache.get(address, None)
            if maybe_cached_pricer is not None:
                self._cache_hits += 1
                return maybe_cached_pricer

            ret = None

            maybe_uv2 = self._uniswap_v2_pools.get(address)
            if maybe_uv2 is not None:
                token0, token1 = maybe_uv2
                ret = self._get_uniswap_v2_pricer(address, token0, token1)
                return ret

            maybe_sushi = self._sushiswap_v2_pools.get(address)
            if maybe_sushi is not None:
                token0, token1 = maybe_sushi
                ret = self._get_sushiswap_v2_pricer(address, token0, token1)
                return ret

            maybe_shiba = self._shibaswap_pools.get(address)
            if maybe_shiba is not None:
                token0, token1 = maybe_shiba
                ret = self._get_shibaswap_pricer(address, token0, token1)
                return ret

            maybe_uv3 = self._uniswap_v3_pools.get(address)
            if maybe_uv3 is not None:
                token0, token1, fee = maybe_uv3
                return self._get_uniswap_v3_pricer(address, token0, token1, fee)

            maybe_balv1 = self._balancer_v1_pools.get(address)
            if maybe_balv1 is not None:
                return self._get_balancer_v1_pricer(address)

            maybe_balv2 = self._balancer_v2_pools.get(address)
            if maybe_balv2 is not None:
                _, pool_id, pool_type = maybe_balv2
                return self._get_balancer_v2_pricer(address, pool_id, pool_type)

        raise NotImplementedError(f'Not sure which pool {address} belongs to')

    def get_tokens_for(self, address: str) -> typing.Set[str]:
        if address in self._uniswap_v2_pools:
            return set(self._uniswap_v2_pools[address])
        elif address in self._sushiswap_v2_pools:
            return set(self._sushiswap_v2_pools[address])
        elif address in self._shibaswap_pools:
            return set(self._shibaswap_pools[address])
        elif address in self._uniswap_v3_pools:
            return set(self._uniswap_v3_pools[address][:2])
        elif address in self._balancer_v1_pools:
            return set(self._balancer_v1_pools[address])
        elif address in self._balancer_v2_pools:
            return set(self._balancer_v2_pools[address][0])
        raise Exception(f'could not find tokens for {address}')

    def origin_block_for(self, address: str) -> int:
        return self._origin_blocks[address]

    def _get_uniswap_v2_pricer(self, address: str, token0: str, token1: str) -> BaseExchangePricer:
        maybe_uv2 = self._hydrate_pricer(address)
        if maybe_uv2 is not None:
            return maybe_uv2

        self._cache_misses += 1
        ret = UniswapV2Pricer(self._w3, address, token0, token1)
        self._evictable_cache[address] = ret
        return ret

    def _get_sushiswap_v2_pricer(self, address: str, token0: str, token1: str) -> BaseExchangePricer:
        maybe_sv2 = self._hydrate_pricer(address)
        if maybe_sv2 is not None:
            return maybe_sv2

        self._cache_misses += 1
        ret = UniswapV2Pricer(self._w3, address, token0, token1)
        self._evictable_cache[address] = ret
        return ret

    def _get_shibaswap_pricer(self, address: str, token0: str, token1: str) -> BaseExchangePricer:
        maybe_shib = self._hydrate_pricer(address)
        if maybe_shib is not None:
            return maybe_shib

        self._cache_misses += 1
        ret = UniswapV2Pricer(self._w3, address, token0, token1)
        self._evictable_cache[address] = ret
        return ret

    def _get_uniswap_v3_pricer(self, address: str, token0: str, token1: str, fee: int) -> BaseExchangePricer:
        maybe_uv3 = self._hydrate_pricer(address)
        if maybe_uv3 is not None:
            return maybe_uv3

        self._cache_misses += 1
        ret = UniswapV3Pricer(self._w3, address, token0, token1, fee)
        self._evictable_cache[address] = ret
        return ret

    def _get_balancer_v1_pricer(self, address: str) -> BaseExchangePricer:
        self._cache_misses += 1
        ret = BalancerPricer(self._w3, address)
        self._cache[address] = ret
        return ret

    def _get_balancer_v2_pricer(self, address: str, pool_id: bytes, pool_type: str) -> BaseExchangePricer:
        if pool_type in ['WeightedPool', 'WeightedPool2Tokens']:
            b = BalancerV2WeightedPoolPricer(self._w3, self._balancer_v2_vault, address, pool_id)
        elif pool_type in ['LiquidityBootstrappingPool', 'NoProtocolFeeLiquidityBootstrappingPool']:
            b = BalancerV2LiquidityBootstrappingPoolPricer(self._w3, self._balancer_v2_vault, address, pool_id)
        else:
            raise Exception(f'not sure how to make {pool_type} pool (address = {address})')

        self._cache_misses += 1
        self._cache[address] = b
        return b

    def _hydrate_pricer(self, key: str) -> typing.Optional[BaseExchangePricer]:
        if self._db is None:
            return None

        try:
            with profile('ldb.read'):
                bs = self._db.Get(key.encode('ascii'))
                unpickled = pickle.loads(bs)
            unpickled.set_web3(self._w3)
            self._evictable_cache[key] = unpickled

            self._soft_cache_hits += 1
            return unpickled
        except KeyError:
            return None

    def _evicted(self, k: str, v: typing.Union[UniswapV2Pricer, UniswapV3Pricer]):
        with profile('ldb.write'):
            assert self._db is not None
            # cache out to leveldb
            bs = pickle.dumps(v)
            self._db.Put(k.encode('ascii'), bs)

    def _maybe_log_stats(self):
        if time.time() > self._last_stat_log_ts + self.__class__.STAT_LOG_PERIOD_SECONDS:
            # do log
            hits = self._cache_hits
            soft_hits = self._soft_cache_hits
            misses = self._cache_misses

            n_queries = hits + misses + soft_hits
            if n_queries == 0:
                return
            hit_percent = hits / (n_queries) * 100
            soft_hit_percent = soft_hits / n_queries * 100
            miss_percent = misses / n_queries * 100

            l.debug(f'Pricer pool cache stats: size_evictable={len(self._evictable_cache):,} hits={hit_percent:.2f}% soft_hit_percent={soft_hit_percent:.2f}% misses={miss_percent:.2f}%')

            self._last_stat_log_ts = time.time()
            self._cache_hits = 0
            self._soft_cache_hits = 0
            self._cache_misses = 0

