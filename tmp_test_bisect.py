# test scalar-optimizing a function

import datetime
import os
import time
import typing
import random
import numpy as np
import web3
import web3.types
import web3.contract
from pricers.balancer_v2.common import SWAP_TOPIC
import utils.profiling

from backtest.utils import connect_db
from find_circuit.find import FoundArbitrage, PricingCircuit, detect_arbitrages, detect_arbitrages_bisection
from pricers import UniswapV2Pricer
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.balancer_v2.liquidity_bootstrapping_pool import BalancerV2LiquidityBootstrappingPoolPricer
from pricers.base import BaseExchangePricer
from pricers.uniswap_v3 import UniswapV3Pricer
from utils import BALANCER_VAULT_ADDRESS, WETH_ADDRESS, get_abi, get_block_timestamp, setup_logging

setup_logging(root_dir='/tmp')

#
# Connect to web3
#

web3_host = os.getenv('WEB3_HOST', 'ws://172.17.0.1:8546')

w3 = web3.Web3(web3.WebsocketProvider(
    web3_host,
    websocket_timeout=60 * 5,
    websocket_kwargs={
        'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
    },
))

assert w3.isConnected()

print(f'Connected to web3 endpoint')

db = connect_db()
db.autocommit = False

curr = db.cursor()
curr.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')

N_CYCLE_SAMPLES = 1000
# read some random arbitrage cycles

vault: web3.contract.Contract = w3.eth.contract(
    address=BALANCER_VAULT_ADDRESS,
    abi=get_abi('balancer_v2/Vault.json'),
)

curr.execute('SELECT COUNT(*) FROM candidate_arbitrages')
(n_candidate_arb,) = curr.fetchone()

# detect arbitrage 0x305177ee6c77dd17025acacc3724558613bb8d113e78bf9476bac191e833596b

# with open('trace.txt', mode='w') as fout:
#     tr = w3.provider.make_request('debug_traceTransaction', ['0x305177ee6c77dd17025acacc3724558613bb8d113e78bf9476bac191e833596b'])
#     for sl in tr['result']['structLogs']:
#         fout.write(str(sl) + '\n')

# p1 = UniswapV3Pricer(w3, '0xD1D5A4c0eA98971894772Dcd6D2f1dc71083C44E', '0x6DEA81C8171D0bA574754EF6F8b412F2Ed88c54D', '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', 3000)
# p2 = BalancerV2WeightedPoolPricer(w3, vault, '0xdB3e5Cf969c05625Db344deA9C8b12515e235Df3', bytes.fromhex('db3e5cf969c05625db344dea9c8b12515e235df300010000000000000000006a'))
# p3 = UniswapV3Pricer(w3, '0xcD83055557536EFf25FD0eAfbC56e74a1b4260B3', '0xbC396689893D065F41bc2C6EcbeE5e0085233447', '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', 3000)

# tr = w3.provider.make_request('debug_traceTransaction', ['0x305177ee6c77dd17025acacc3724558613bb8d113e78bf9476bac191e833596b', {'tracer': 'callTracer'}])

# q = [tr['result']]
# while len(q) > 0:
#     item = q.pop()

#     if 'calls' in item:
#         q.extend(item['calls'])
    
#     if w3.toChecksumAddress(item['to']) == '0xcD83055557536EFf25FD0eAfbC56e74a1b4260B3':
#         # print(item)
#         f, decoded = p1.contract.decode_function_input(item['input'])
#         print('found', decoded)
#         break

# f = w3.eth.filter({
#     'address': p1.address,
#     'fromBlock': 13_005_118,
#     'toBlock': 13_005_118
# })
# logs = f.get_all_entries()
# for l in logs:
#     print(l)


# pc = PricingCircuit(
#     [p1, p2, p3],
#     [
#         ('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', '0x6DEA81C8171D0bA574754EF6F8b412F2Ed88c54D'),
#         ('0x6DEA81C8171D0bA574754EF6F8b412F2Ed88c54D', '0xbC396689893D065F41bc2C6EcbeE5e0085233447'),
#         ('0xbC396689893D065F41bc2C6EcbeE5e0085233447', '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'),
#     ]
# )

# # ratio = pc.sample_new_price_ratio(100, block_identifier=13_005_036 - 1)
# # print('ratio', ratio)

# print('---------------')
# amt = pc.sample_new_price_ratio(294749464192638503, 13_005_118 - 1, debug=True)
# print('amt', amt)
# print('---------------')

# det = detect_arbitrages(pc, 13_005_118 - 1, only_weth_pivot=True)

# amt_out, _ = p1.token_out_for_exact_in('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', '0x6DEA81C8171D0bA574754EF6F8b412F2Ed88c54D', 294749464192638503, 13_005_118 - 1)

# print('amt_out', amt_out)
# print('       ', 128336196178668743173)

# # 302343750000000000
# # amt_out = 1059196448

# amt_out, _ = p2.token_out_for_exact_in('0x6DEA81C8171D0bA574754EF6F8b412F2Ed88c54D', '0xbC396689893D065F41bc2C6EcbeE5e0085233447', amt_out, 13_005_118 - 1)
# print('amt_out', amt_out)

# amt_out, _ = p3.token_out_for_exact_in('0xbC396689893D065F41bc2C6EcbeE5e0085233447', '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', 60504461649832927232, 13_005_118 - 1)
# print(amt_out)

# print(det)

# exit()

# # find a candidate arbitrage with balancer v2
# balancer_exchanges = [BALANCER_VAULT_ADDRESS]
# for exc in balancer_exchanges:
#     curr.execute(
#         '''
#         SELECT id
#         FROM sample_arbitrage_exchanges
#         WHERE address = %s
#         ''',
#         (bytes.fromhex(exc[2:]),)
#     )
#     assert curr.rowcount <= 1
#     if curr.rowcount < 1:
#         continue

#     (exc_id,) = curr.fetchone()

#     # see if there's any uses with this
#     curr.execute(
#         '''
#         SELECT block_number, txn_hash
#         FROM sample_arbitrages sa
#         JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
#         JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
#         JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
#         WHERE sacei.exchange_id = %s AND
#               13005000 < sa.block_number AND sa.block_number < 13100000
#         ''',
#         (exc_id,)
#     )
#     for bn, txn_hash in list(curr):
#         # See if there's a balancer swap log and if so get the pool id
#         receipt = w3.eth.get_transaction_receipt(txn_hash.tobytes())
#         found = False
#         for log in receipt['logs']:
#             if log['address'] == BALANCER_VAULT_ADDRESS and log['topics'][0] == SWAP_TOPIC:
#                 swap = vault.events.Swap().processLog(log)
#                 pool_id = swap['args']['poolId']
#                 curr.execute('SELECT address, pool_type FROM balancer_v2_exchanges WHERE pool_id = %s', (pool_id,))
#                 (address, pool_type,) = curr.fetchone()
#                 if 'boot' in pool_type.lower():
#                     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#                     found = True
#                 print(w3.toChecksumAddress(address.tobytes()), pool_type)
#         print(f'{bn:,} https://etherscan.io/tx/0x{txn_hash.tobytes().hex()}')
#         if found:
#             exit()

    # curr.execute(
    #     '''
    #     SELECT id
    #     FROM candidate_arbitrages
    #     WHERE %s = ANY(exchanges)
    #     ''',
    #     (bytes.fromhex(exc[2:]),),
    # )
    # if curr.rowcount > 0:
    #     (id_,) = curr.fetchone()
    #     print(f'found balancer v1 arb with id={id_}')
    #     exit()

# print('found none.....')
# exit()

curr.execute(
    '''
    SELECT id
    FROM candidate_arbitrages
    -- WHERE id = 163906
    ORDER BY RANDOM()
    LIMIT %s
    ''',
    (N_CYCLE_SAMPLES,)
)
arb_ids = [x for (x,) in curr]
# assert len(arb_ids) == N_CYCLE_SAMPLES

print(f'Sampled {N_CYCLE_SAMPLES:,} of {n_candidate_arb:,} arbitrages ({N_CYCLE_SAMPLES / n_candidate_arb * 100:.2f}%)')


class ArbToTest(typing.NamedTuple):
    block_number: int
    fa: FoundArbitrage
    pc: PricingCircuit

# load the arbitrage details

t_started = time.time()
marks = [N_CYCLE_SAMPLES * i // 100 for i in range(5, 100, 5)]

arbs_to_test: typing.List[ArbToTest] = []

for i, id_ in enumerate(arb_ids):

    if i in marks and i > 0:
        elapsed = time.time() - t_started
        nps = i / elapsed
        remain = len(arb_ids) - i
        eta_seconds = remain / nps
        eta = datetime.timedelta(seconds = eta_seconds)

        print(f'Loading samples, {i} / {N_CYCLE_SAMPLES} ({i / N_CYCLE_SAMPLES * 100 :.2f}%) ETA {eta}')

    curr.execute(
        '''
        SELECT block_number, exchanges, directions, amount_in, profit_no_fee
        FROM candidate_arbitrages
        WHERE id = %s
        ''',
        (id_,)
    )
    assert curr.rowcount == 1
    block_number, exchanges, directions, amount_in, profit_no_fee = curr.fetchone()

    amount_in = int(amount_in)
    profit_no_fee = int(profit_no_fee)

    # attempt to load a pricer for each exchange
    exchange_pricers: typing.List[BaseExchangePricer] = []

    for exchange in exchanges:
        exchange_address = w3.toChecksumAddress(exchange.tobytes())
        curr.execute(
            '''
            SELECT t0.address, t1.address
            FROM uniswap_v2_exchanges uv2
            JOIN tokens t0 ON uv2.token0_id = t0.id
            JOIN tokens t1 ON uv2.token1_id = t1.id
            WHERE uv2.address = %s
            ''',
            (exchange,)
        )
        if curr.rowcount > 0:
            assert curr.rowcount == 1

            token0, token1 = curr.fetchone()
            token0 = w3.toChecksumAddress(token0.tobytes())
            token1 = w3.toChecksumAddress(token1.tobytes())
            p = UniswapV2Pricer(w3, exchange_address, token0, token1)

            exchange_pricers.append(p)
            continue

        curr.execute(
            '''
            SELECT t0.address, t1.address
            FROM sushiv2_swap_exchanges sv2
            JOIN tokens t0 ON sv2.token0_id = t0.id
            JOIN tokens t1 ON sv2.token1_id = t1.id
            WHERE sv2.address = %s
            ''',
            (exchange,)
        )
        if curr.rowcount > 0:
            assert curr.rowcount == 1

            token0, token1 = curr.fetchone()
            token0 = w3.toChecksumAddress(token0.tobytes())
            token1 = w3.toChecksumAddress(token1.tobytes())
            p = UniswapV2Pricer(w3, exchange_address, token0, token1)

            exchange_pricers.append(p)
            continue

        curr.execute(
            '''
            SELECT t0.address, t1.address, originalfee
            FROM uniswap_v3_exchanges uv3
            JOIN tokens t0 ON uv3.token0_id = t0.id
            JOIN tokens t1 ON uv3.token1_id = t1.id
            WHERE uv3.address = %s            
            ''',
            (exchange,)
        )
        if curr.rowcount > 0:
            assert curr.rowcount == 1
            token0, token1, fee = curr.fetchone()
            token0 = w3.toChecksumAddress(token0.tobytes())
            token1 = w3.toChecksumAddress(token1.tobytes())
            p = UniswapV3Pricer(w3, exchange_address, token0, token1, fee)

            exchange_pricers.append(p)
            continue

        curr.execute(
            '''
            SELECT EXISTS(SELECT 1 FROM balancer_exchanges WHERE address = %s)
            ''',
            (exchange,)
        )
        (is_balancerv1,) = curr.fetchone()
        if is_balancerv1:
            p = BalancerPricer(w3, exchange_address)

            print(f'Found balancer v1 in {id_} !!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            exchange_pricers.append(p)
            continue

        curr.execute(
            '''
            SELECT pool_id, pool_type
            FROM balancer_v2_exchanges
            WHERE address = %s
            ''',
            (exchange,)
        )
        if curr.rowcount > 0:
            assert curr.rowcount == 1
            (pool_id, pool_type) = curr.fetchone()
            pool_id = pool_id.tobytes()

            print(f'Found balancer in {id_} !!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            if pool_type in ['WeightedPool', 'WeightedPool2Tokens']:
                b = BalancerV2WeightedPoolPricer(w3, vault, exchange_address, pool_id)
            elif pool_type in ['LiquidityBootstrappingPool', 'NoProtocolFeeLiquidityBootstrappingPool']:
                b = BalancerV2LiquidityBootstrappingPoolPricer(w3, vault, exchange_address, pool_id)
            else:
                raise Exception(f'not sure how to make {pool_type} pool (address = {exchange_address})')

            exchange_pricers.append(p)
            continue

        raise NotImplementedError(f'what is this? {exchange_address}')

    assert len(exchange_pricers) == len(exchanges)

    directions_tuples = []
    for t0, t1 in zip(directions, directions[1:] + [directions[0]]):
        t0 = web3.Web3.toChecksumAddress(t0.tobytes())
        t1 = web3.Web3.toChecksumAddress(t1.tobytes())
        directions_tuples.append((t0, t1))

    assert len(directions_tuples) == len(exchange_pricers)
    assert WETH_ADDRESS == directions_tuples[0][0]
    assert WETH_ADDRESS == directions_tuples[-1][1]

    pc = PricingCircuit(
        exchange_pricers,
        directions_tuples
    )

    fa = FoundArbitrage(
        amount_in = amount_in,
        circuit = exchange_pricers,
        directions = directions_tuples,
        pivot_token = WETH_ADDRESS,
        profit = profit_no_fee,
    )

    to_test = ArbToTest(block_number, fa, pc)
    arbs_to_test.append(to_test)

# assert len(arbs_to_test) == N_CYCLE_SAMPLES

print('Loaded all arbitrages')

# attempt to optimize these guys

n_warm_iters = 4
n_timed_iters = 100
tot_old_method = 0
tot_new_method = 0

for arb_id, arb in zip(arb_ids, arbs_to_test):
    # if not all(isinstance(x, (UniswapV2Pricer, UniswapV3Pricer)) for x in arb.pc.circuit):
    #     continue

    print('examining', arb_id)

    ts = get_block_timestamp(w3, arb.block_number + 1)

    # with open('pts.csv', mode='w') as fout:
    #     for amt_in in np.linspace(100, 100_000_000, 500):
    #         amt_in = int(amt_in)
    #         new_spot = arb.pc.sample_new_price_ratio(amt_in, block_identifier=arb.block_number, timestamp=ts)
    #         fout.write(f'{amt_in},{new_spot}\n')

    # warm up a few times
    # for _ in range(n_warm_iters):
    #     new_fa = detect_arbitrages(arb.pc.copy(), arb.block_number, timestamp = ts, only_weth_pivot=True)
    #     assert len(new_fa) > 0

    # xs = [0, 1]
    # random.shuffle(xs)
    # for i in xs:
    #     # time how long it takes to optimize, warmed
    #     if i == 0:
    #         t_start = time.time()
    #         for _ in range(n_timed_iters):
    #             detect_arbitrages(arb.pc.copy(), arb.block_number, timestamp = ts, only_weth_pivot=True)
    #         t_end = time.time()
    #         elapsed_old_method = t_end - t_start
    #         tot_old_method += elapsed_old_method
    #     else:
    #         assert i == 1
    #         t_start = time.time()
    #         for _ in range(n_timed_iters):
    #             detect_arbitrages_bisection(arb.pc.copy(), arb.block_number, timestamp = ts, only_weth_pivot=True)
    #         t_end = time.time()
    #         elapsed_new_method = t_end - t_start
    #         tot_new_method += elapsed_new_method

    # print('old method', elapsed_old_method)
    # print('new method', elapsed_new_method)
    # print('diff      ', f'{(elapsed_new_method - elapsed_old_method) / elapsed_old_method * 100:.4f}%')

    # print('passed')

    # for p in arb.pc.circuit:
    #     print(type(p).__name__, p.address)

    # for t0, t1 in arb.pc.directions:
    #     print(f'{t0} -> {t1}')

    # arb.pc.sample_new_price_ratio()

    old_profit = 0
    new_profit = 0

    result = detect_arbitrages(arb.pc.copy(), arb.block_number, timestamp = ts, only_weth_pivot=True)
    for r in result:
        if r.directions == arb.pc.directions:
            old_profit = r.profit
            print(r.amount_in)

    result = detect_arbitrages_bisection(arb.pc.copy(), arb.block_number, timestamp = ts, only_weth_pivot=True)
    for r in result:
        if r.directions == arb.pc.directions:
            new_profit = r.profit
            print(r.amount_in)

    if old_profit > 10 and new_profit > 10:
        diff_pct = (new_profit - old_profit) / old_profit * 100
        print('old profit', old_profit)
        print('new profit', new_profit)
        print('diff      ', f'{diff_pct:.4f}%')

        # if abs(diff_pct) > 1:
        #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        if abs(diff_pct) > 0.1:
            exit()

# print('old method', tot_old_method)
# print('new method', tot_new_method)
# print('diff      ', f'{(tot_new_method - tot_old_method) / tot_old_method * 100:.4f}%')

# utils.profiling.log()
print('done')