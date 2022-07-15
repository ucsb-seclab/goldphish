import os
import time
import numpy as np
import web3
import web3.contract
import web3._utils.filters
from backtest.utils import connect_db
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.uniswap_v2 import UniswapV2Pricer
from utils import BALANCER_VAULT_ADDRESS, get_abi, connect_web3, setup_logging


db = connect_db()
curr = db.cursor()

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

print('getting uniswap v2 exchanges')
curr.execute(
    '''
    SELECT address FROM uniswap_v2_exchanges
    '''
)
addresses = [web3.Web3.toChecksumAddress(x.tobytes()) for (x,) in curr]
print(f'Loaded {len(addresses):,} uniswap v2 addresses')

start_block = 13_000_000

t_start = time.time()
f: web3._utils.filters.Filter = w3.eth.filter({'fromBlock': start_block, 'toBlock': start_block + 99})

logs = f.get_all_entries()
elapsed_noaddr = time.time() - t_start
print(f'{elapsed_noaddr} seconds elapsed without address')
print(f'got {len(logs):,} logs from that')

t_start = time.time()
f: web3._utils.filters.Filter = w3.eth.filter({'address': addresses, 'fromBlock': start_block, 'toBlock': start_block + 99})

logs = f.get_all_entries()
elapsed_addr = time.time() - t_start
print(f'{elapsed_addr} seconds elapsed with address')
print(f'got {len(logs):,} logs from that')


exit()

vault: web3.contract.Contract = w3.eth.contract(
    address=BALANCER_VAULT_ADDRESS,
    abi=get_abi('balancer_v2/Vault.json'),
)


p = BalancerV2WeightedPoolPricer(w3, vault, '0x5c6Ee304399DBdB9C8Ef030aB642B10820DB8F56', bytes.fromhex('5c6ee304399dbdb9c8ef030ab642b10820db8f56000200000000000000000014'))

# p.token_out_for_exact_in('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', '0xba100000625a3754423978a60c9317c58a424e3D', 25261521867, block_identifier=13093825)


diffs = []
for x in np.linspace(463014849, 463015043, 100):
    amt = int(np.ceil(x))
    try:
        out, spot = p.token_out_for_exact_in('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', '0xba100000625a3754423978a60c9317c58a424e3D', amt, block_identifier=13093825)
        print(f'{amt},{out},{spot}')
    except:
        print(f'failed at {amt}')
        raise

# max_diff = np.max(diffs)
# min_diff = np.min(diffs)
# median_diff = np.median(diffs)
# avg_diff = np.mean(diffs)

# print(f'max_diff    {max_diff}')
# print(f'min_diff    {min_diff}')
# print(f'avg_diff    {avg_diff}')
# print(f'median_diff {median_diff}')

# print('done')

# f: web3._utils.filters.Filter = w3.eth.filter({
#     'address': '0x5Fa464CEfe8901d66C09b85d5Fcdc55b3738c688',
#     'fromBlock': 13_005_100,
#     'toBlock': 13_005_176
# })

# logs = f.get_all_entries()
# print(f'got {len(logs)} logs')
# for log in logs:
#     print(log)

# c: web3.contract.Contract = w3.eth.contract(
#     address = '0x5Fa464CEfe8901d66C09b85d5Fcdc55b3738c688',
#     abi = get_abi('uniswap_v2/IUniswapV2Pair.json')['abi']
# )

# for block_number in range(13_005_145, 13_005_176):
#     b0, b1, _ = c.functions.getReserves().call(block_identifier=block_number)
#     print(b0, b1)


# f: web3._utils.filters.Filter = w3.eth.filter({
#     'address': ['0x5Fa464CEfe8901d66C09b85d5Fcdc55b3738c688'],
#     'topics': [['0x' + x.hex() for x in UniswapV2Pricer.RELEVANT_LOGS]],
#     'fromBlock': 13_005_153,
#     'toBlock': 13_005_252,
# })

# logs = f.get_all_entries()
# for log in logs:
#     print(log)
