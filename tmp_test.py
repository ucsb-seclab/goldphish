import os
import web3
import web3.contract
import web3._utils.filters
from pricers.balancer import BalancerPricer
from pricers.uniswap_v2 import UniswapV2Pricer
from utils import get_abi

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


f: web3._utils.filters.Filter = w3.eth.filter({
    'address': '0x5Fa464CEfe8901d66C09b85d5Fcdc55b3738c688',
    'fromBlock': 13_005_100,
    'toBlock': 13_005_176
})

logs = f.get_all_entries()
print(f'got {len(logs)} logs')
for log in logs:
    print(log)

c: web3.contract.Contract = w3.eth.contract(
    address = '0x5Fa464CEfe8901d66C09b85d5Fcdc55b3738c688',
    abi = get_abi('uniswap_v2/IUniswapV2Pair.json')['abi']
)

for block_number in range(13_005_145, 13_005_176):
    b0, b1, _ = c.functions.getReserves().call(block_identifier=block_number)
    print(b0, b1)


f: web3._utils.filters.Filter = w3.eth.filter({
    'address': ['0x5Fa464CEfe8901d66C09b85d5Fcdc55b3738c688'],
    'topics': [['0x' + x.hex() for x in UniswapV2Pricer.RELEVANT_LOGS]],
    'fromBlock': 13_005_153,
    'toBlock': 13_005_252,
})

logs = f.get_all_entries()
for log in logs:
    print(log)
