from collections import deque
import collections
import datetime
import psycopg2
import psycopg2.extras
import numpy as np
from matplotlib import pyplot as plt
import tabulate
import web3
import scipy.stats
import sqlite3

from common import setup_only_uniswap_tables, setup_weth_arb_tables

SAMPLE_SIZE = 5_000
print(f'SAMPLING WITH SIZE {SAMPLE_SIZE}')

db = psycopg2.connect(
    host='10.10.111.111',
    port=5432,
    user = 'measure',
    password = 'password',
    database = 'eth_measure_db',
)
print('connected to postgresql')
db.autocommit = False

curr = db.cursor()

curr.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')

curr.execute('SELECT MAX(end_block) FROM block_samples')
(end_block,) = curr.fetchone()

curr.execute(
    '''
    SELECT COUNT(*)
    FROM uniswap_v2_exchanges WHERE origin_block <= %s
    ''',
    (end_block,)
)
(n_uv2,) = curr.fetchone()
print(f'Have {n_uv2:,} uniswap v2')

curr.execute(
    '''
    SELECT COUNT(*)
    FROM uniswap_v3_exchanges WHERE origin_block <= %s
    ''',
    (end_block,)
)
(n_uv3,) = curr.fetchone()
print(f'Have {n_uv3:,} uniswap v3')

curr.execute(
    '''
    SELECT COUNT(*)
    FROM sushiv2_swap_exchanges WHERE origin_block <= %s
    ''',
    (end_block,)
)
(n_sushi,) = curr.fetchone()
print(f'Have {n_sushi:,} sushiswap')

curr.execute(
    '''
    SELECT COUNT(*)
    FROM shibaswap_exchanges WHERE origin_block <= %s
    ''',
    (end_block,)
)
(n_shib,) = curr.fetchone()
print(f'Have {n_shib:,} shibaswap')

curr.execute(
    '''
    SELECT COUNT(*)
    FROM balancer_exchanges WHERE origin_block <= %s
    ''',
    (end_block,)
)
(n_bal,) = curr.fetchone()
print(f'Have {n_bal:,} Balancer v1')

curr.execute(
    '''
    SELECT COUNT(*)
    FROM balancer_v2_exchanges WHERE origin_block <= %s
    ''',
    (end_block,)
)
(n_balv2,) = curr.fetchone()
print(f'Have {n_balv2:,} Balancer v2')

tot_exchanges = n_uv2 + n_uv3 + n_sushi + n_shib + n_bal + n_balv2
print(f'Total exchanges: {tot_exchanges:,}')


curr.execute(
    '''
    CREATE TEMP TABLE tmp_all_tokens(token_id) AS
    SELECT token0_id
    FROM uniswap_v2_exchanges WHERE origin_block <= %(origin_block)s
    UNION
    SELECT token1_id
    FROM uniswap_v2_exchanges WHERE origin_block <= %(origin_block)s
    UNION
    SELECT token0_id
    FROM uniswap_v3_exchanges WHERE origin_block <= %(origin_block)s
    UNION
    SELECT token1_id
    FROM uniswap_v3_exchanges WHERE origin_block <= %(origin_block)s
    UNION
    SELECT token0_id
    FROM sushiv2_swap_exchanges WHERE origin_block <= %(origin_block)s
    UNION
    SELECT token1_id
    FROM sushiv2_swap_exchanges WHERE origin_block <= %(origin_block)s
    UNION
    SELECT token0_id
    FROM shibaswap_exchanges WHERE origin_block <= %(origin_block)s
    UNION
    SELECT token1_id
    FROM shibaswap_exchanges WHERE origin_block <= %(origin_block)s
    ''',
    {
        'origin_block': end_block,
    }
)

w3 = web3.Web3(web3.WebsocketProvider('ws://10.10.111.111:8546',
        websocket_timeout=60 * 5,
        websocket_kwargs={
            'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
        },
))


curr.execute('SELECT address FROM balancer_exchanges WHERE origin_block <= %s', (end_block,))

block = w3.eth.block_number
TOKEN_BASE_SLOT = int.from_bytes(bytes.fromhex('6e1540171b6c0c960b71a7020d9f60077f6af931a8bbf590da0223dacf75c7af'), byteorder='big', signed=False)

tokens = set()

for i, (addr,) in enumerate(curr):
    if i % 100 == 0:
        print(f'{i} / {curr.rowcount}')

    addr = web3.Web3.toChecksumAddress(addr.tobytes())

    # no cache, get the list of tokens
    bn_tokens = w3.eth.get_storage_at(addr, '0x9', block)
    n_tokens = int.from_bytes(bn_tokens, byteorder='big', signed=False)

    tokens = set()

    for i in range(n_tokens):
        token_slot = int.to_bytes(TOKEN_BASE_SLOT + i, length=32, byteorder='big', signed=False)
        hex_token_slot = '0x' + token_slot.hex()
        btoken = w3.eth.get_storage_at(addr, hex_token_slot, block).rjust(32, b'\x00')
        assert btoken[0:12] == b'\x00'*12

        token = w3.toChecksumAddress(btoken[12:])
        tokens.add(token)
print('loaded balancer v1')

vault = w3.eth.contract(
    address = '0xBA12222222228d8Ba445958a75a0704d566BF2C8',
    abi = [    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "poolId",
                "type": "bytes32"
            }
        ],
        "name": "getPoolTokens",
        "outputs": [
            {
                "internalType": "contract IERC20[]",
                "name": "tokens",
                "type": "address[]"
            },
            {
                "internalType": "uint256[]",
                "name": "balances",
                "type": "uint256[]"
            },
            {
                "internalType": "uint256",
                "name": "lastChangeBlock",
                "type": "uint256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
]
)

curr.execute('SELECT pool_id FROM balancer_v2_exchanges WHERE origin_block <= %s', (end_block,))
for (pool_id,) in curr:
    for token in vault.functions.getPoolTokens(pool_id.tobytes()).call()[0]:
        assert w3.isChecksumAddress(token)
        tokens.add(token)


curr.execute(
    '''
    CREATE TEMP TABLE tmp_all_tokens_addrs (address bytea not null);
    '''
)
psycopg2.extras.execute_values(
    curr,
    '''
    INSERT INTO tmp_all_tokens_addrs (address) VALUES %s
    ''',
    [(bytes.fromhex(x[2:]),) for x in tokens],
)
print('inserted, querying....')

curr.execute(
    '''
    SELECT COUNT(DISTINCT address)
    FROM (
        SELECT address
        FROM tmp_all_tokens_addrs
        UNION (
            SELECT address
            FROM tmp_all_tokens
            JOIN tokens ON tokens.id = token_id
        )
    ) x
    '''
)
(n_tokens,) = curr.fetchone()

print(f'Have {n_tokens:,} tokens')
