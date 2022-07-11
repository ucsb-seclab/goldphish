import collections
import datetime
import itertools
import os
import time
import typing
import psycopg2
import tabulate
import web3
import web3.types
import web3.contract
import web3._utils.filters
import pricers.balancer
from pricers.balancer import LOG_SWAP_TOPIC, BalancerPricer
from eth_utils import event_abi_to_log_topic
from pricers.balancer_v2.common import SWAP_TOPIC
from pricers.balancer_v2.liquidity_bootstrapping_pool import BalancerV2LiquidityBootstrappingPoolPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer


from utils import get_abi, get_block_timestamp

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


pg_host = os.getenv('PSQL_HOST', 'ethereum-measurement-pg')
pg_port = int(os.getenv('PSQL_PORT', '5432'))
db = psycopg2.connect(
    host = pg_host,
    port = pg_port,
    user = 'measure',
    password = 'password',
    database = 'eth_measure_db',
)
db.autocommit = False
print('connected to postgresql')

curr = db.cursor()


vault: web3.contract.Contract = w3.eth.contract(address='0xBA12222222228d8Ba445958a75a0704d566BF2C8', abi=get_abi('balancer_v2/Vault.json'))

top_pools_by_id = [
'96646936b91d6b9d7d0c47c496afbf3d6ec7b6f8000200000000000000000019',
'0b09dea16768f0799065c475be02919503cb2a3500020000000000000000001a',
'5c6ee304399dbdb9c8ef030ab642b10820db8f56000200000000000000000014',
'a6f548df93de924d73be7d25dc02554c6bd66db500020000000000000000000e',
'06df3b2bbb68adc8b0e302443692037ed9f91b42000000000000000000000063',
'a30ac4a3bf3f680a29eb02238280c75acbb89d6d0002000000000000000000d3',
'3e5fa9518ea95c3e533eb377c001702a9aacaa32000200000000000000000052',
'072f14b85add63488ddad88f855fda4a99d6ac9b000200000000000000000027',
'231e687c9961d3a27e6e266ac5c433ce4f8253e4000200000000000000000023',
'10a2f8bd81ee2898d7ed18fb8f114034a549fa59000200000000000000000090',
'e99481dc77691d8e2456e5f3f61c1810adfc1503000200000000000000000018',
'4eebc19e5f29dec3dea07f66b9e707afc8f28c060002000000000000000000b3',
'4ddf308520864ecfc759c49e72acfc96c023ed900002000000000000000000e1',
'a02e4b3d18d4e6b8d18ac421fbc3dfff8933c40a00020000000000000000004b',
'efaa1604e82e1b3af8430b90192c1b9e8197e377000200000000000000000021',
'ff083f57a556bfb3bbe46ea1b4fa154b2b1fbe88000200000000000000000030',
'9212b088d48fc749c5adc573b445bc0d0a289a340002000000000000000000b1',
'ec60a5fef79a92c741cb74fdd6bfc340c0279b01000200000000000000000015',
'f4c0dd9b82da36c07605df83c8a416f11724d88b000200000000000000000026',
'aac98ee71d4f8a156b6abaa6844cdb7789d086ce00020000000000000000001b',
'd153e1de63b478213b7b62bf47dcc4099608b1ae0002000000000000000000d8',
'f659abd8bc89590389e4f8c0959285677128232a00020000000000000000010c',
'6aa8a7b23f7b3875a966ddcc83d5b675cc9af54b00020000000000000000008e',
'5d6e3d7632d6719e04ca162be652164bec1eaa6b000200000000000000000048',
'7edde0cb05ed19e03a9a47cd5e53fc57fde1c80c0002000000000000000000c8',
'5f7fa48d765053f8dd85e052843e12d23e3d7bc50002000000000000000000c0',
'6d68d7b0ca469bd1171f81a895e649d86d523c200002000000000000000000cc',
'47a2121f2781ad48ec2b6f705ac040cf5fe3beaa0002000000000000000000b8',
'29d7a7e0d781c957696697b94d4bc18c651e358e000200000000000000000049',
'960f4ecd3533b0aabf355ff36f79d747d2ca51e7000200000000000000000113',
'64e2c43ca952ba01e32e8cfa05c1e009bc92e06c00020000000000000000009b',
'89d4a55ca51192109bb85083ff7d9a13ab24c8a10002000000000000000000bc',
'702605f43471183158938c1a3e5f5a359d7b31ba00020000000000000000009f',
'787546bf2c05e3e19e2b6bde57a203da7f682eff00020000000000000000007c',
'56b2811bf75bb258d2234af4f43b479bb55c3b46000200000000000000000091',
'a47d1251cf21ad42685cc6b8b3a186a73dbd06cf000200000000000000000097',
'90ca5cef5b29342b229fb8ae2db5d8f4f894d6520002000000000000000000b5',
'838de9a9e4fd604be653ef61e63792512e0a0c030002000000000000000000ae',
'186084ff790c65088ba694df11758fae4943ee9e000200000000000000000013',
'148ce9b50be946a96e94a4f5479b771bab9b1c59000100000000000000000054',
'6836bfa9b9d000036d41dbd44b40688c45055037000200000000000000000119',
'01abc00e86c7e258823b9a055fd62ca6cf61a16300010000000000000000003b',
'c1382fe6e17bcdbc3d35f73f5317fbf261ebeecd0002000000000000000000a9',
'f5aaf7ee8c39b651cebf5f1f50c10631e78e0ef9000200000000000000000069',
'1050f901a307e7e71471ca3d12dfcea01d0a0a1c00020000000000000000004c',
'61d5dc44849c9c87b0856a2a311536205c96c7fd000100000000000000000001',
'5b1c06c4923dbba4b27cfa270ffb2e60aa28615900020000000000000000004a',
'ede4efcc5492cf41ed3f0109d60bc0543cfad23a0002000000000000000000bb',
'bf96189eee9357a95c7719f4f5047f76bde804e5000200000000000000000087',
'e5769603af1c9ec809dd5cfbc7fee36e7f09a3e60002000000000000000000bf',
]

top_pools_by_id_bytes = [bytes.fromhex(x) for x in top_pools_by_id]

if False:
    # collect swaps and find which pools are most popular
    counts = collections.defaultdict(lambda: 0)

    start_block = 12_500_000
    end_block   = 14_000_000
    batch_size = 1_000

    SWAP_TOPIC = event_abi_to_log_topic(vault.events.Swap().abi)
    SWAP_TOPIC_HEX = '0x' + SWAP_TOPIC.hex()

    for i in itertools.count():
        batch_start_block = start_block + i * batch_size
        batch_end_block = min(end_block, batch_start_block + batch_size - 1)


        if batch_start_block > end_block:
            break

        if i % 10 == 1:
            print(f'{batch_start_block:,}')
            tab = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:50]
            print(tabulate.tabulate(tab, headers=['PoolId', 'Count']))
            print()

        f: web3._utils.filters.Filter = w3.eth.filter({
            'topics': [SWAP_TOPIC_HEX],
            'address': vault.address,
            'fromBlock': batch_start_block,
            'toBlock': batch_end_block
        })

        for log in f.get_all_entries():
            parsed = vault.events.Swap().processLog(log)
            counts[parsed['args']['poolId'].hex()] += 1

bs: typing.List[BalancerV2WeightedPoolPricer] = []
for pool_id in top_pools_by_id_bytes:
    (addr, _) = vault.functions.getPool(pool_id).call()

    curr.execute(
        '''
        SELECT pool_type
        FROM balancer_v2_exchanges
        WHERE address = %s
        ''',
        (bytes.fromhex(addr[2:]),)
    )
    assert curr.rowcount <= 1

    if curr.rowcount == 1:
        (pool_type,) = curr.fetchone()
    else:
        pool_type = 'UNKNOWN'

    print(f'{pool_id.hex()} -> {addr} | {pool_type}')

    if pool_type in ['WeightedPool', 'WeightedPool2Tokens']:
        bs.append(BalancerV2WeightedPoolPricer(w3, vault, addr))
    elif pool_type in ['LiquidityBootstrappingPool', 'NoProtocolFeeLiquidityBootstrappingPool']:
        bs.append(BalancerV2LiquidityBootstrappingPoolPricer(w3, vault, addr))

print(f'Have {len(bs)} balancer v2 pools to check')

start_block = 13506756 # 13_502_446 # 12_413_346 # 12_366_246 # 12_272_146
batch_size = 10

latest_block = w3.eth.get_block('latest')['number'] - 10

print(f'Scanning from {start_block:,} to {latest_block:,} ({latest_block - start_block:,} blocks)')

t_start = time.time()

# warm some caches
for b in bs:
    if isinstance(b, BalancerV2LiquidityBootstrappingPoolPricer):
        b.get_pool_state(start_block - 1)
        print('warmed', b.address)

for i in itertools.count():
    batch_start_block = start_block + i * batch_size
    batch_end_block = min(latest_block, batch_start_block + batch_size - 1)

    if batch_start_block > latest_block:
        break

    if i % 10 == 1:
        # do an ETA update
        n_blocks = batch_start_block - start_block
        s_elapsed = time.time() - t_start
        bps = n_blocks / s_elapsed

        n_remaining = latest_block - batch_start_block
        s_remaining = n_remaining / bps
        eta_td = datetime.timedelta(seconds=s_remaining)
        print(f'Processed up to {batch_start_block:,}, ETA {eta_td}')


    f: web3._utils.filters.Filter = w3.eth.filter({
        'address': vault.address,
        'fromBlock': batch_start_block,
        'toBlock': batch_end_block
    })

    logs = f.get_all_entries()

    f: web3._utils.filters.Filter = w3.eth.filter({
        'address': [b.address for b in bs],
        'fromBlock': batch_start_block,
        'toBlock': batch_end_block
    })

    logs += f.get_all_entries()

    logs = sorted(logs, key=lambda x: (x['blockNumber'], x['logIndex']))

    # separate it into blocks
    logs_by_block = collections.defaultdict(lambda: [])

    for log in logs:
        logs_by_block[log['blockNumber']].append(log)

    for block_number in sorted(logs_by_block.keys()):
        logs_by_pool_id: typing.Dict[bytes, typing.List[web3.types.LogReceipt]] = collections.defaultdict(lambda: [])

        for log in logs_by_block[block_number]:
            if log['address'] == vault.address:
                logs_by_pool_id[log['topics'][1]].append(log)

        swaps_by_txn: typing.Dict[bytes, typing.List[web3.types.LogReceipt]] = collections.defaultdict(lambda: [])
        for log in logs_by_block[block_number]:
            if log['topics'][0] == SWAP_TOPIC:
                swaps_by_txn[log['transactionHash']].append(log)

        for b in bs:
            pool_logs = logs_by_pool_id[b.pool_id]

            if len(pool_logs) > 0 and pool_logs[0]['topics'][0] == SWAP_TOPIC and len(swaps_by_txn[pool_logs[0]['transactionHash']]) == 1:
                # attempt to re-create a swap

                trace = w3.provider.make_request('debug_traceTransaction', [pool_logs[0]['transactionHash'].hex(), {'tracer': 'callTracer'}])

                is_swap_given_in = None

                q = [trace['result']]
                while len(q) > 0:
                    item = q.pop()

                    if 'calls' in q:
                        q += list(item['calls'])

                    if w3.toChecksumAddress(item['to']) == vault.address:
                        decoded = vault.decode_function_input(item['input'])

                        if decoded[0].abi['name'] == 'swap':
                            is_swap_given_in = decoded[1]['singleSwap'][1] == 0
                            break
                else:
                    # yeah we don't know what happened here
                    is_swap_given_in = False

                if is_swap_given_in:

                    parsed = vault.events.Swap().processLog(pool_logs[0])

                    token_in   = parsed['args']['tokenIn']
                    token_out  = parsed['args']['tokenOut']
                    amount_in  = parsed['args']['amountIn']
                    amount_out = parsed['args']['amountOut']

                    ts = get_block_timestamp(w3, block_number)

                    expected_out = b.swap_exact_amount_in(token_in, amount_in, token_out, block_number - 1, timestamp=ts)

                    if amount_out != expected_out:
                        tr = w3.provider.make_request('debug_traceTransaction', [pool_logs[0]['transactionHash'].hex()])
                        with open('/mnt/goldphish/trace.txt', mode='w') as fout:
                            for sl in tr['result']['structLogs']:
                                fout.write(str(sl) + '\n')

                    assert amount_out == expected_out, f'expected {amount_out} == {expected_out} https://etherscan.io/tx/{pool_logs[0]["transactionHash"].hex()}'

                    if isinstance(b, BalancerV2LiquidityBootstrappingPoolPricer):

                        print('priced properly!!!')

                # raise Exception(f'found a top swap! {pool_logs[0]["transactionHash"].hex()}')

            b.observe_block(logs_by_block[block_number])

            try:
                tokens, balances, _ = vault.functions.getPoolTokens(b.pool_id).call(block_identifier=block_number)
            except Exception as e:
                if 'BAL#500' in str(e):
                    # not registered yet
                    continue

            actual_balances = dict(zip(tokens, balances))

            # check consistency
            if b.tokens is not None:
                tokens = set(tokens)

                assert tokens == set(b.tokens), f'expected {tokens} == {b.tokens}'

            for t in b._balance_cache.keys():
                assert actual_balances[t] == b._balance_cache[t], f'expected {actual_balances[t]} == {b._balance_cache[t]}'

            if isinstance(b, BalancerV2LiquidityBootstrappingPoolPricer):
                if b.pool_state is not None:
                    weights = b.contract.functions.getNormalizedWeights().call(block_identifier=block_number)
                    for tok, w in zip(b.tokens, weights):
                        my_weight = b.get_weight(tok, block_number)

                        assert w == my_weight, f'expected {w} == {my_weight}'
