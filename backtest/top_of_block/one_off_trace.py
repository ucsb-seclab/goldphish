import typing
import psycopg2
import psycopg2.extensions
from backtest.utils import connect_db
from backtest.top_of_block.common import TraceMode, load_exchanges, shoot
import find_circuit
import pricers
import web3

from utils import WETH_ADDRESS, decode_trace_calls, pretty_print_trace

def print_trace(w3: web3.Web3, candidate_id: int):
    db = connect_db()
    curr = db.cursor()
    uniswap_v2_exchanges, uniswap_v3_exchanges = load_exchanges()
    block_number, fa = load_candidate(w3, curr, uniswap_v2_exchanges, uniswap_v3_exchanges, candidate_id)
    shooter_address, receipt, (trace, txn) = shoot(w3, fa, block_number, do_trace = TraceMode.ALWAYS)
    decoded = decode_trace_calls(trace, txn, receipt)
    print('----------------------trace---------------------------')
    pretty_print_trace(decoded, txn, receipt)
    print('------------------------------------------------------')



def load_candidate(w3: web3.Web3, curr: psycopg2.extensions.cursor, uniswap_v2_exchanges, uniswap_v3_exchanges, id_: int) -> typing.Tuple[int, find_circuit.FoundArbitrage]:
    curr.execute(
        """
        SELECT ca.block_number, ca.exchanges, ca.directions, ca.amount_in, ca.profit_no_fee
        FROM candidate_arbitrages ca
        WHERE id = %s
        """,
        (id_,)
    )
    block_number, exchanges, directions, amount_in, profit = curr.fetchone()

    assert len(exchanges) == len(directions)

    # reconstruct found arbitrage
    amount_in = int(amount_in)
    profit = int(profit)
    circuit: typing.List[pricers.BaseExchangePricer] = []
    for exc in exchanges:
        address = web3.Web3.toChecksumAddress(exc.tobytes())
        
        if address in uniswap_v2_exchanges:
            token0, token1 = uniswap_v2_exchanges[address]
            pricer = pricers.UniswapV2Pricer(w3, address, token0, token1)
        else:
            assert address in uniswap_v3_exchanges
            token0, token1, fee = uniswap_v3_exchanges[address]
            pricer = pricers.UniswapV3Pricer(w3, address, token0, token1, fee)
        circuit.append(pricer)

    if directions[0] == True:
        assert circuit[0].token0 == WETH_ADDRESS
    else:
        assert directions[0] == False
        assert circuit[0].token1 == WETH_ADDRESS

    fa = find_circuit.FoundArbitrage(
        amount_in = amount_in,
        circuit = circuit,
        directions = directions,
        pivot_token = WETH_ADDRESS,
        profit = profit
    )
    return block_number, fa

