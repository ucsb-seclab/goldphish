import collections
import itertools
import os
import web3
import web3._utils.filters

from utils import get_abi
import pricers

web3_host = os.getenv('WEB3_HOST', 'ws://172.17.0.1:8546')

w3 = web3.Web3(web3.WebsocketProvider(
    web3_host,
    websocket_timeout=60 * 5,
    websocket_kwargs={
        'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
    },
))

if not w3.isConnected():
    print('Could not connect to web3')
    exit(1)

TARGET_EXCHANGE = '0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8'
contract = w3.eth.contract(address=TARGET_EXCHANGE, abi=get_abi('uniswap_v3/IUniswapV3Pool.json')['abi'])

start_block = 12370625

pricer = pricers.UniswapV3Pricer(w3, TARGET_EXCHANGE, w3.toChecksumAddress('0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'), w3.toChecksumAddress('0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'), 3000)

def process_block(block_number: int, logs):
    # gather all relevant logs for the block
    f: web3._utils.filters.Filter = w3.eth.filter({'address': TARGET_EXCHANGE, 'fromBlock': block_number, 'toBlock': block_number})
    logs = f.get_all_entries()
    if len(logs) == 0:
        return

    pricer.observe_block(logs, force_load = True)
    # ensure that everything is in order
    s0 = pricer.slot0_cache
    assert s0 == pricer.get_slot0(block_number, use_cache=False)
    lq = pricer.liquidity_cache
    assert lq == pricer.get_liquidity(block_number, use_cache=False)

    for tick in pricer.tick_cache.values():
        got = pricer.tick_at(tick.id, block_number, use_cache=False)
        assert got.id == tick.id
        assert got.liquidity_gross == tick.liquidity_gross
        assert got.liquidity_net == tick.liquidity_net
        assert got.initialized == tick.initialized
    
    for word, val in list(pricer.tick_bitmap_cache.items()):
        assert val == pricer.get_tick_bitmap_word(word, block_number, use_cache = False)

    print(f'verified up to {block_number:,}')

batch_size = 100
for i in itertools.count():
    this_start_block = start_block + batch_size * i
    this_end_block = start_block + batch_size * (i + 1) - 1
    f: web3._utils.filters.Filter = w3.eth.filter({'address': TARGET_EXCHANGE, 'fromBlock': this_start_block, 'toBlock': this_end_block})
    logs = f.get_all_entries()

    logs_by_block = collections.defaultdict(lambda: [])
    for log in logs:
        logs_by_block[log['blockNumber']] += log

    for block_number in sorted(logs_by_block.keys()):
        process_block(block_number, logs_by_block[block_number])

