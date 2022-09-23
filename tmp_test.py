import asyncio
import json
import os
import random
import time
import typing
import numpy as np
import web3
import web3.contract
import web3._utils.filters
import find_circuit.find
from backtest.utils import connect_db
from find_circuit.find import PricingCircuit
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.base import NotEnoughLiquidityException
from pricers.token_transfer import out_from_transfer
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer
from utils import BALANCER_VAULT_ADDRESS, get_abi, connect_web3, setup_logging, RetryingProvider
from eth_hash.auto import keccak
import logging

setup_logging()

l = logging.getLogger(__name__)

w3 = connect_web3()
db = connect_db()

curr = db.cursor()

curr.execute('SELECT address FROM balancer_v2_exchanges ORDER BY RANDOM() LIMIT 100')

for (baddr,) in curr.fetchall():
    address = w3.toChecksumAddress(baddr.tobytes())
    contract: web3.contract.Contract = w3.eth.contract(
        address=address,
        abi = get_abi('balancer_v2/LiquidityBootstrappingPool.json'),
    )

sel = contract.functions.getSwapFeePercentage().selector
print('getSwapFeePercentage', contract.functions.getSwapFeePercentage().call(block_identifier='latest'))
print(sel)

txn = {
    'from': w3.toChecksumAddress(b'\x00'*20),
    'to': contract.address,
    'data': sel # + (bytes.fromhex(token[2:])).rjust(32, b'\x00').hex()
}
resp = w3.provider.make_request('debug_traceCall', [txn, "latest", {"enableMemory": True}])

with open('trace.txt', mode='w') as fout:
    for sl in resp['result']['structLogs']:
        fout.write(str(sl) + '\n')


exit()

curr.execute('SELECT address FROM balancer_exchanges ORDER BY RANDOM() LIMIT 100')

for (baddr,) in curr.fetchall():
    address = w3.toChecksumAddress(baddr.tobytes())
    contract: web3.contract.Contract = w3.eth.contract(
        address=address,
        abi = get_abi('balancer_v1/bpool.abi.json'),
    )

    for token in contract.functions.getCurrentTokens().call(block_identifier='latest')[:1]:
        slot_base = w3.keccak(bytes.fromhex(token[2:]).rjust(32, b'\x00') + b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0a')
        slot = int.to_bytes(int.from_bytes(slot_base, byteorder='big', signed=False) + 0x2, length=32, byteorder='big', signed=False)
        
        bbal = w3.eth.get_storage_at(contract.address, slot.hex(), 'latest')
        weight = int.from_bytes(bbal, byteorder='big', signed=False)

        got_weight = contract.functions.getDenormalizedWeight(token).call(block_identifier='latest')
        assert weight == got_weight

sel = contract.functions.getDenormalizedWeight(token).selector
print(sel)

txn = {
    'from': w3.toChecksumAddress(b'\x00'*20),
    'to': contract.address,
    'data': sel + (bytes.fromhex(token[2:])).rjust(32, b'\x00').hex()
}
resp = w3.provider.make_request('debug_traceCall', [txn, "latest", {"enableMemory": True}])

with open('trace.txt', mode='w') as fout:
    for sl in resp['result']['structLogs']:
        fout.write(str(sl) + '\n')


exit()

for (baddr,) in curr.fetchall():
    address = w3.toChecksumAddress(baddr.tobytes())
    contract: web3.contract.Contract = w3.eth.contract(
        address=address,
        abi = get_abi('balancer_v1/bpool.abi.json'),
    )

    for token in contract.functions.getCurrentTokens().call(block_identifier='latest')[:1]:
        slot_base = w3.keccak(bytes.fromhex(token[2:]).rjust(32, b'\x00') + b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0a')
        slot = int.to_bytes(int.from_bytes(slot_base, byteorder='big', signed=False) + 0x3, length=32, byteorder='big', signed=False)
        
        bbal = w3.eth.get_storage_at(contract.address, slot.hex(), 'latest')
        balance = int.from_bytes(bbal, byteorder='big', signed=False)

        got_balance = contract.functions.getBalance(token).call(block_identifier='latest')
        assert balance == got_balance

sel = contract.functions.getBalance(token).selector
print(sel)

txn = {
    'from': w3.toChecksumAddress(b'\x00'*20),
    'to': contract.address,
    'data': sel + (bytes.fromhex(token[2:])).rjust(32, b'\x00').hex()
}
resp = w3.provider.make_request('debug_traceCall', [txn, "latest", {"enableMemory": True}])

with open('trace.txt', mode='w') as fout:
    for sl in resp['result']['structLogs']:
        fout.write(str(sl) + '\n')



exit()

for (baddr,) in curr.fetchall():
    address = w3.toChecksumAddress(baddr.tobytes())
    contract: web3.contract.Contract = w3.eth.contract(
        address=address,
        abi = get_abi('balancer_v1/bpool.abi.json'),
    )

    bpublic_swap = w3.eth.get_storage_at(contract.address, '0x6', 'latest')
    public_swap = (int.from_bytes(bpublic_swap, byteorder='big', signed=False) >> 0xa0) != 0

    got_public_swap = contract.functions.isPublicSwap().call(block_identifier='latest')
    if got_public_swap == False:
        print('got false!!!')
    assert got_public_swap == public_swap

sel = contract.functions.isPublicSwap().selector
print(sel)

txn = {
    'from': w3.toChecksumAddress(b'\x00'*20),
    'to': contract.address,
    'data': sel
}
resp = w3.provider.make_request('debug_traceCall', [txn, "latest", {"enableMemory": True}])

with open('trace.txt', mode='w') as fout:
    for sl in resp['result']['structLogs']:
        fout.write(str(sl) + '\n')


exit()
curr.execute('SELECT address FROM balancer_exchanges ORDER BY RANDOM() LIMIT 100')

for (baddr,) in curr.fetchall():
    address = w3.toChecksumAddress(baddr.tobytes())
    contract: web3.contract.Contract = w3.eth.contract(
        address=address,
        abi = get_abi('balancer_v1/bpool.abi.json'),
    )

    bswap_fee = w3.eth.get_storage_at(contract.address, '0x7', 'latest')
    swap_fee = int.from_bytes(bswap_fee, byteorder='big', signed=False)
    print('swap_fee?', swap_fee)
    
    got_swap_fee = contract.functions.getSwapFee().call(block_identifier='latest')
    print('swap fee ', got_swap_fee)

    assert swap_fee == got_swap_fee

sel = contract.functions.getSwapFee().selector
print(sel)

txn = {
    'from': w3.toChecksumAddress(b'\x00'*20),
    'to': contract.address,
    'data': sel
}
resp = w3.provider.make_request('debug_traceCall', [txn, "latest", {"enableMemory": True}])

with open('trace.txt', mode='w') as fout:
    for sl in resp['result']['structLogs']:
        fout.write(str(sl) + '\n')


exit()

for (baddr,) in curr.fetchall():
    address = w3.toChecksumAddress(baddr.tobytes())
    contract: web3.contract.Contract = w3.eth.contract(
        address=address,
        abi = get_abi('balancer_v1/bpool.abi.json'),
    )

    bfinalized = w3.eth.get_storage_at(contract.address, '0x8', 'latest')
    print(bfinalized)
    finalized = bfinalized[-1] != 0

    got_finalized = contract.functions.isFinalized().call(block_identifier='latest')
    assert finalized == got_finalized, f'{finalized} != {got_finalized}'

sel = contract.functions.isFinalized().selector
print(sel)

txn = {
    'from': w3.toChecksumAddress(b'\x00'*20),
    'to': contract.address,
    'data': sel
}
resp = w3.provider.make_request('debug_traceCall', [txn, "latest", {"enableMemory": True}])

with open('trace.txt', mode='w') as fout:
    for sl in resp['result']['structLogs']:
        fout.write(str(sl) + '\n')

exit()

for (baddr,) in curr.fetchall():
    address = w3.toChecksumAddress(baddr.tobytes())
    contract: web3.contract.Contract = w3.eth.contract(
        address=address,
        abi = get_abi('balancer_v1/bpool.abi.json'),
    )

    bn_tokens = w3.eth.get_storage_at(contract.address, '0x9', 'latest')
    n_tokens = int.from_bytes(bn_tokens, byteorder='big', signed=False)
    print('n_tokens=', n_tokens)
    base = int.from_bytes(bytes.fromhex('6e1540171b6c0c960b71a7020d9f60077f6af931a8bbf590da0223dacf75c7af'), byteorder='big', signed=False)

    tokens = []

    for i in range(n_tokens):
        token_slot = int.to_bytes(base + i, length=32, byteorder='big', signed=False)
        hex_token_slot = '0x' + token_slot.hex()
        btoken = w3.eth.get_storage_at(contract.address, hex_token_slot, 'latest')
        assert btoken[0:12] == b'\x00'*12
        token = w3.toChecksumAddress(btoken[12:])
        tokens.append(token)
        print('token', token)
    
    got = contract.functions.getCurrentTokens().call(block_identifier='latest')
    assert set(got) == set(tokens)


# sel = contract.functions.getCurrentTokens().selector
# print(sel)

# txn = {
#     'from': w3.toChecksumAddress(b'\x00'*20),
#     'to': contract.address,
#     'data': sel
# }
# resp = w3.provider.make_request('debug_traceCall', [txn, "latest", {"enableMemory": True}])

# with open('trace.txt', mode='w') as fout:
#     for sl in resp['result']['structLogs']:
#         fout.write(str(sl) + '\n')

exit()

p1 = UniswapV2Pricer(
    w3,
    '0xF82d8Ec196Fb0D56c6B82a8B1870F09502A49F88',
    '0xA2b4C0Af19cC16a6CfAcCe81F192B024d625817D',
    '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
)
# p2 = UniswapV3Pricer(
#     w3,
#     '0x3D71021345AeD9FFab1Efd805e58aEc9c857D525',
#     '0xA0335820dC549dBfae5b8D691331CadfCA7026E0',
#     '0xdAC17F958D2ee523a2206206994597C13D831ec7',
#     10000,
# )
p3 = UniswapV3Pricer(
    w3,
    '0xCdC0F4092086452d8F980acEfFD6BC077C7e656B',
    '0xA2b4C0Af19cC16a6CfAcCe81F192B024d625817D',
    '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    10000,
)
directions = [
    ('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', '0xA2b4C0Af19cC16a6CfAcCe81F192B024d625817D'),
    ('0xA2b4C0Af19cC16a6CfAcCe81F192B024d625817D', '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'),
    # ('0xdAC17F958D2ee523a2206206994597C13D831ec7', '0xC02aaA39b223/E8D0A0e5C4F27eAD9083C756Cc2'),
]

pc_ = PricingCircuit([p1, p3], directions)


find_circuit.find.find_upper_bound(pc_, 100, 100000000000000000000000, 13483123)

exit()

def old_find_upper_bound(pc: PricingCircuit, lower_bound: int, upper_bound: int, block_identifier: int, timestamp: typing.Optional[int] = None) -> int:
    try:
        pc.sample(upper_bound, block_identifier, timestamp=timestamp)
    except NotEnoughLiquidityException:
        # we need to adjust upper_bound down juuuust until it's in liquidity range
        # do this by binary-search
        search_lower = lower_bound
        search_upper = upper_bound

        while True:
            # rapidly reduce upper bound by orders of 10
            x = search_upper // 10
            try:
                pc.sample(x, block_identifier, timestamp=timestamp)
                search_lower = max(search_lower, x)
                break
            except NotEnoughLiquidityException:
                search_upper = x

        while search_lower < search_upper - 1000:
            midpoint = (search_lower + search_upper) // 2
            try:
                pr = pc.sample_new_price_ratio(midpoint, block_identifier, timestamp=timestamp)
                search_lower = midpoint
                if pr < 0.9:
                    # we can exit the search early because marginal price is not optimal
                    # at this bound already, no need to further refine
                    break
            except NotEnoughLiquidityException:
                search_upper = midpoint
        upper_bound = search_lower

    return upper_bound

def new_find_upper_bound(pc: PricingCircuit, lower_bound: int, upper_bound: int, block_identifier: int, timestamp: typing.Optional[int] = None) -> int:
    if not all(isinstance(p, (UniswapV2Pricer, UniswapV3Pricer)) for p in pc._circuit):
        print('using old upper bound method')
        old_find_upper_bound()

    # we may be able to use new method, attempt to run upper_bound and find the first exchange
    # that trips up on it

    curr_amt = upper_bound
    last_token = pc.pivot_token
    for pricer_idx, (p, (t_in, t_out)) in enumerate(zip(pc._circuit, directions)):
        assert last_token == t_in

        try:
            amt_out, _ = p.token_out_for_exact_in(
                t_in,
                t_out,
                curr_amt,
                block_identifier=block_identifier,
                timestamp=timestamp
            )
        except NotEnoughLiquidityException as not_enough_liq_exc:
            too_much_in  = not_enough_liq_exc.amount_in
            remaining_in = not_enough_liq_exc.remaining
            break

        curr_amt = out_from_transfer(t_out, amt_out)
        last_token = t_out
    else:
        assert last_token == pc.pivot_token
        return upper_bound

    reverse_amt = too_much_in - remaining_in
    reverse_last_token = directions[pricer_idx][0]

    # amt_out = pc._circuit[pricer_idx].token_out_for_exact_in(
    #     directions[pricer_idx][0],
    #     directions[pricer_idx][1],
    #     reverse_amt,
    #     block_identifier
    # )

    for p, (t_in, t_out) in reversed(list(zip(pc._circuit[:pricer_idx], pc._directions[:pricer_idx]))):
        p: typing.Union[UniswapV2Pricer, UniswapV3Pricer]
        assert reverse_last_token == t_out

        zero_for_one = bytes.fromhex(t_in[2:]) < bytes.fromhex(t_out[2:])

        try:
            if zero_for_one:
                amt_in = p.token1_out_to_exact_token0_in(reverse_amt, block_identifier)
            else:
                amt_in = p.token0_out_to_exact_token1_in(reverse_amt, block_identifier)
        except NotEnoughLiquidityException as e:
            print('e.amount_in', e.amount_in)
            print('e.remaining', e.remaining)
            print('diff', e.amount_in - e.remaining)
            raise

        # crude attempt at reversing amount out for in
        ratio = out_from_transfer(t_in, 10 ** 10)
        reverse_amt = amt_in * ratio // (10 ** 10)
        reverse_last_token = t_in

    return new_find_upper_bound(pc, lower_bound, reverse_amt, block_identifier, timestamp=timestamp)

upper_bound = old_find_upper_bound(pc_, 100, 100000000000000000000000, 14789678)
print('upper_bound_old', upper_bound)

upper_bound = new_find_upper_bound(pc_, 100, 100000000000000000000000, 14789678)
print('upper_bound_new', upper_bound)


# warm up
print('warming')
for _ in range(4):
    old_find_upper_bound(pc_, 100, 100000000000000000000000, 14789678)

print('warmed')

t_start = time.time()
for _ in range(100):
    old_find_upper_bound(pc_, 100, 100000000000000000000000, 14789678)
elapsed_old = time.time() - t_start
print(f'Old method, {elapsed_old} seconds elapsed')

t_start = time.time()
for _ in range(100):
    new_find_upper_bound(pc_, 100, 100000000000000000000000, 14789678)
elapsed_new = time.time() - t_start
print(f'New method, {elapsed_new} seconds elapsed')

diff = elapsed_new - elapsed_old
diff_pct = diff / elapsed_old * 100
print(f'Percent difference: {diff_pct:.2f}%')

exit()

uv3 = UniswapV3Pricer(
    w3,
    '0x3840d56cfE2c80Ec2D6555dCda986A0982B0D2db',
    token0='0x6c28AeF8977c9B773996d0e8376d2EE379446F2f',
    token1='0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    fee=3000
)

data = uv3.contract.functions.liquidity().buildTransaction()['data']
print('data', data)

resp = w3.provider.make_request('debug_traceCall', [{
    "from": "0xdeadbeef29292929192939494959594933929292",
    "to":   uv3.address,
    "gas":  "0x7a120",
    "data": data,
}, "latest", {"enableMemory": True}])

with open('trace.txt', mode='w') as fout:
    for sl in resp['result']['structLogs']:
        fout.write(str(sl) + '\n')

# exit()

def new_get_liquidity():
    bliquidity = w3.eth.get_storage_at(uv3.address, '0x4', 'latest')
    liquidity = int.from_bytes(bliquidity[16:32], byteorder='big', signed=False)
    print(liquidity)

l = uv3.get_liquidity('latest')
print('liquidity', l)

new_get_liquidity()

# RESERVES_SLOT = '0x0000000000000000000000000000000000000000000000000000000000000008'
# def new_get_reserves():
#     breserves = w3.eth.get_storage_at(uv2.address, RESERVES_SLOT, 'latest')

#     reserve1 = int.from_bytes(breserves[4:18], byteorder='big', signed=False)
#     reserve0 = int.from_bytes(breserves[18:32], byteorder='big', signed=False)
#     print('reserve0', reserve0)
#     print('reserve1', reserve1)

# r0, r1, _ = uv2.contract.functions.getReserves().call()
# print('actual r0', r0)
# print('actual r1', r1)
# new_get_reserves()

# six = int.to_bytes(6, length=32, byteorder='big', signed=False)
# def get_bmap_advanced(idx):
#     bword_idx = int.to_bytes(idx, length=32, byteorder='big', signed=True)

#     h = keccak(bword_idx + six)
#     result = w3.eth.get_storage_at(uv3.address, h)
#     return int.from_bytes(result, byteorder='big', signed=False)



# print('making request')
# provider: RetryingProvider = w3.provider

# reqs = [("eth_chainId", {}), ("eth_chainId", {})]

# print(provider.make_request_batch(reqs))

# five = int.to_bytes(5, length=32, byteorder='big', signed=False)
# def new_get_tick(tick):
#     btick = int.to_bytes(tick, length=32, byteorder='big', signed=True)
#     h = keccak(btick + five)

#     reqs = []
#     reqs.append(('eth_getStorageAt', [uv3.address, '0x' + h.hex(), 'latest']))

#     h_int = int.from_bytes(h, byteorder='big', signed=False)
#     slot = int.to_bytes(h_int + 3, length=32, byteorder='big', signed=False)
#     reqs.append(('eth_getStorageAt', [uv3.address, '0x' + slot.hex(), 'latest']))

#     resp = provider.make_request_batch(reqs)
#     assert len(resp) == 2

#     bresp_0 = bytes.fromhex(resp[0]['result'][2:])
#     liquidity_gross = int.from_bytes(bresp_0[16:32], byteorder='big', signed=False)
#     liquidity_net = int.from_bytes(bresp_0[0:16], byteorder='big', signed=True)

#     bresp_1 = bytes.fromhex(resp[1]['result'][2:])
#     initialized = bool(bresp_1[0])

#     return (liquidity_gross, liquidity_net, initialized)

# def old_get_tick(tick):
#     lg, ln, _, __, ___, ____, _____, initialized = uv3.contract.functions.ticks(tick).call()
#     return (lg, ln, initialized)

# data = uv3.contract.functions.ticks(0).buildTransaction()['data']
# print('data', data)

# resp = w3.provider.make_request('debug_traceCall', [{
#     "from": "0xdeadbeef29292929192939494959594933929292",
#     "to":   uv3.address,
#     "gas":  "0x7a120",
#     "data": data,
# }, "latest", {"enableMemory": True}])

# with open('trace.txt', mode='w') as fout:
#     for sl in resp['result']['structLogs']:
#         fout.write(str(sl) + '\n')

# ticks = list(range(UniswapV3Pricer.MIN_TICK, UniswapV3Pricer.MAX_TICK))
# random.shuffle(ticks)

# next = uv3.next_initialized_tick_within_one_word(1, True, 'latest')
# old = old_get_tick(0)
# new_ = new_get_tick(0)
# print(old, new_)
# assert old == new_

# elapsed_old = 0
# elapsed_new = 0

# for i, tick in enumerate(ticks):
#     if i % 100 == 1:
#         print('elapsed_old', elapsed_old)
#         print('elapsed_new', elapsed_new)
#     b = [True, False]
#     random.shuffle(b)
#     for use_old in b:
#         if use_old:
#             t_start = time.time()
#             old = old_get_tick(tick)
#             elapsed_old += time.time() - t_start
#         else:
#             t_start = time.time()
#             new_ = new_get_tick(tick)
#             elapsed_new += time.time() - t_start
#     assert old == new_

# print('elapsed_old', elapsed_old)
# print('elapsed_new', elapsed_new)


# exit()

# for block_number in range(13_016_237, 13_016_249 + 2):
#     f: web3._utils.filters.Filter = w3.eth.filter({
#         'address': '0x3840d56cfE2c80Ec2D6555dCda986A0982B0D2db',
#         'fromBlock': block_number,
#         'toBlock': block_number,
#     })
#     logs = f.get_all_entries()
#     print(block_number, len(logs), logs)
#     uv3.observe_block(logs)
#     for amt_in in np.linspace(100, 100000000000000000000000, 1000):
#         amt_in = int(np.ceil(amt_in))
#         try:
#             uv3.token_out_for_exact_in(
#                 '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
#                 '0x6c28AeF8977c9B773996d0e8376d2EE379446F2f',
#                 amt_in,
#                 block_number
#             )
#         except NotEnoughLiquidityException:
#             pass


# uv3.token_out_for_exact_in('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', '0x6c28AeF8977c9B773996d0e8376d2EE379446F2f', 100000000000000000000000)

exit()

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
