import os
import web3
from pricers.balancer import BalancerPricer

web3_host = os.getenv('WEB3_HOST', 'ws://172.17.0.1:8546')

w3 = web3.Web3(web3.WebsocketProvider(
    web3_host,
    websocket_timeout=60 * 5,
    websocket_kwargs={
        'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
    },
))

if not w3.isConnected():
    exit(1)

bp = BalancerPricer(w3, address='0x69d460e01070A7BA1bc363885bC8F4F0daa19Bf5')

print(bp.get_tokens(13_005_166))

with open('tab.csv', mode='w') as fout:
    for amt_in in range(11781, 28105, 100):
        got_out = bp.token_out_for_exact_in(
            '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
            '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            amt_in,
            block_identifier=13_005_172,
        )
        fout.write(f'{amt_in},{got_out}\n')
        

bp.token_out_for_exact_in(
    '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    11781,
    block_identifier=13_005_172,
)

print('ok!!!!')

bp.token_out_for_exact_in(
    '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    28005, # 28163,
    block_identifier=13_005_172,
)
