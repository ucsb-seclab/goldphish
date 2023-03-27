import collections
import web3
import web3.types
import typing
import networkx as nx
import logging

from backtest.gather_samples.models import Arbitrage, ArbitrageCycle, ArbitrageCycleExchange, ArbitrageCycleExchangeItem

l = logging.getLogger(__name__)

KNOWN_ROUTERS = set([
    '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
    '0xE592427A0AEce92De3Edee1F18E0157C05861564',
    '0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45'
])

def get_arbitrage_if_exists(
        w3: web3.Web3,
        tx_hash: bytes,
        txns: typing.List,
    ):
    full_txn = w3.eth.get_transaction_receipt('0x' + tx_hash.hex())
    return get_arbitrage_from_receipt_if_exists(full_txn, txns)

ERC20TransactionArgs = typing.TypedDict('ERC20TransactionArgs', {
    'to':   str,
    'from': str,
    'value': int,
})

ERC20Transaction = typing.TypedDict(
    'ERC20Transaction',
    {
        'address': str,
        'transactionHash': bytes,
        'args': ERC20TransactionArgs,
    }
)
MovementDesc = typing.TypedDict(
    'MovementDesc',
    {
        'in': typing.List[ERC20Transaction],
        'out': typing.List[ERC20Transaction],
    }
)

def get_arbitrage_from_receipt_if_exists(
        full_txn: web3.types.TxReceipt,
        txns: typing.List[ERC20Transaction],
        least_profitable = False,
    ):
    addr_to_movements = get_addr_to_movements(txns)

    potential_exchanges = get_potential_exchanges(full_txn, addr_to_movements)

    if len(potential_exchanges) <= 1:
        # not enough exchanges to make a cycle
        return None

    l.debug(f'possible arbitrage 0x{full_txn["transactionHash"].hex()}')

    # Build the digraph of this exchange
    g = nx.DiGraph()
    for addr in potential_exchanges:
        addr: str
        ins  = addr_to_movements[addr]['in']
        outs = addr_to_movements[addr]['out']
        in_coins  = set(x['address'] for x in ins)
        out_coins = set(x['address'] for x in outs)
        assert len(in_coins) == 1 and len(out_coins) == 1
        
        coin_in = next(in_coins.__iter__())
        coin_out = next(out_coins.__iter__())

        # in_log = next(ins.__iter__())
        # out_log = next(outs.__iter__())

        coin_in_amt = sum(x['args']['value'] for x in ins)
        coin_out_amt = sum(x['args']['value'] for x in outs)

        item = (
            ArbitrageCycleExchangeItem(
                address = addr,
                amount_in=coin_in_amt,
                amount_out=coin_out_amt,
            ),
            ins,
            outs,
        )

        if not g.has_edge(coin_in, coin_out):
            g.add_edge(
                coin_in,
                coin_out,
                exchange=ArbitrageCycleExchange(
                    token_in=coin_in,
                    token_out=coin_out,
                    items=[item],
                ),
            )
        else:
            g[coin_in][coin_out]['exchange'].items.append(item)
    
    def is_arb_cycle(cycle: typing.List[str]) -> bool:
        sold_tokens = set()
        bought_tokens = set()
        # every token must be both sold and bought
        for u, v in zip(cycle, cycle[1:] + [cycle[0]]):
            exchange = g.get_edge_data(u, v)['exchange']
            assert isinstance(exchange, ArbitrageCycleExchange)
            sold_tokens.add(exchange.token_in)
            bought_tokens.add(exchange.token_out)

        if sold_tokens == bought_tokens:
            # found an arbitrage cycle, probably
            return True
        return False

    cycles_gen = filter(is_arb_cycle, nx.simple_cycles(g))
    try:
        first_cycle = next(cycles_gen)
    except StopIteration:
        # there are no cycles
        l.debug('no cycles')
        return None

    # count how many cycles there are
    n_cycles = 1
    try:
        while True:
            next(cycles_gen)
            n_cycles += 1
    except StopIteration:
        pass


    # the default Arbitrage to use, optionally we fill in only_cycle below
    arb = Arbitrage(
        txn_hash=full_txn['transactionHash'],
        block_number=full_txn['blockNumber'],
        gas_used=full_txn['gasUsed'],
        gas_price=full_txn['effectiveGasPrice'],
        shooter=full_txn['to'],
        n_cycles=n_cycles,
        only_cycle=None,
    )

    if n_cycles == 1:
        # fill in the cycle
        parsed_cycle = []
        for u, v in zip(first_cycle, first_cycle[1:] + [first_cycle[0]]):
            dat = g[u][v]
            exc: ArbitrageCycleExchange = dat['exchange']
            exc = exc._replace(items=[i for i, _, _ in exc.items])
            parsed_cycle.append(exc)

        # investigate the cycle and try to fill in the only_cycle

        # Loosely defined: an arbitrage exists where there is a cycle containing either
        # (a) an address that both sent and received by the same token, or
        # (b) a token that is sent by an address that did not (net) receive any tokens,
        #     and received by an address that did not (net) send any tokens. 
        
        # filter only to movements (in/out) done by exchanges within the cycle
        addr_to_movements_in_cycle = collections.defaultdict(lambda: {'ins': set(), 'outs': set()})
        for u, v in zip(first_cycle, first_cycle[1:] + [first_cycle[0]]):
            exchange = g[u][v]['exchange']
            assert isinstance(exchange, ArbitrageCycleExchange)
            for _, in_logs, out_logs in exchange.items:
                for in_log in in_logs:
                    addr_to_movements_in_cycle[in_log['args']['from']]['outs'].add(in_log)
                    addr_to_movements_in_cycle[in_log['args']['to']]['ins'].add(in_log)
                for out_log in out_logs:
                    addr_to_movements_in_cycle[out_log['args']['from']]['outs'].add(out_log)
                    addr_to_movements_in_cycle[out_log['args']['to']]['ins'].add(out_log)

        # maps (account address) -> (token address) |-> (net movement, int)
        token_movement_sums = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        for addr, movements in addr_to_movements_in_cycle.items():
            for in_xfer in movements['ins']:
                token_movement_sums[addr][in_xfer['address']] += in_xfer['args']['value']
            for out_xfer in movements['outs']:
                token_movement_sums[addr][out_xfer['address']] -= out_xfer['args']['value']


        if least_profitable == False:
            optimizing_test = lambda x, y: x > y
        else:
            assert least_profitable == True
            optimizing_test = lambda x, y: x < y
        #
        # Find the addresses which hold condition (a)
        #
        addrs_condition_a: typing.List[typing.Tuple[str, str, int]] = []
        addrs_condition_a_most_optimal = []
        for account_addr, token_movements in token_movement_sums.items():
            tokens_in = set(x['address'] for x in addr_to_movements_in_cycle[account_addr]['ins'])
            tokens_out = set(x['address'] for x in addr_to_movements_in_cycle[account_addr]['outs'])

            both_in_and_out = tokens_in.intersection(tokens_out)
            if len(both_in_and_out) > 0:
                # we're comparing apples to oranges here.... that's fine, it's an
                # arbitrary tiebreaker
                
                optimizing_token = None
                if least_profitable == False:
                    most_optimal_amt = float('-inf')
                else:
                    assert least_profitable == True
                    most_optimal_amt = float('inf')

                for token_addr in both_in_and_out:
                    net_movement = token_movements[token_addr]
                    if optimizing_test(net_movement, most_optimal_amt):
                        optimizing_token = token_addr
                        most_optimal_amt = net_movement

                report = (account_addr, optimizing_token, most_optimal_amt)
                if optimizing_test(most_optimal_amt, 0):
                    addrs_condition_a_most_optimal.append(report)
                if most_optimal_amt != 0:
                    addrs_condition_a.append(report)

        if len(addrs_condition_a_most_optimal) > 0:
            if least_profitable == False:
                most_optimal_cond_a = max(addrs_condition_a_most_optimal, key=lambda x: x[2])
            else:
                most_optimal_cond_a = min(addrs_condition_a_most_optimal, key=lambda x: x[2])
            # easy, we found the pivot account
            profiter, token, amount = most_optimal_cond_a
            only_cycle = ArbitrageCycle(
                cycle = parsed_cycle,
                profit_token = token,
                profit_taker = profiter,
                profit_amount = amount,
            )
        # elif len(addrs_condition_a) > 0:
        #     print('addrs_condition_a', addrs_condition_a)
        #     raise Exception(f'not sure about this one {full_txn["transactionHash"].hex()}')
        else:
            #
            # Find pairs of addresses that meet condition (b) as a tuple of (spender, receiver)
            #
            token_addr_to_only_sent_addresses = collections.defaultdict(lambda: [])
            token_addr_to_only_received_addresses = collections.defaultdict(lambda: [])
            # compute net token flows for each address
            for addr, movements in token_movement_sums.items():
                tokens_net_sent = set()
                tokens_net_received = set()
                for token_addr, net_movement in movements.items():
                    if net_movement > 0:
                        tokens_net_received.add(token_addr)
                    if net_movement < 0:
                        tokens_net_sent.add(token_addr)

                if len(tokens_net_sent) == 1 and len(tokens_net_received) == 0:
                    # if we only net sent ONE token (and one only), record
                    token_addr_to_only_sent_addresses[next(tokens_net_sent.__iter__())].append(addr)
                if len(tokens_net_sent) == 0 and len(tokens_net_received) == 1:
                    # same as above        ^                                 ^
                    # but reversed         |                                 |
                    #                      +---------------------------------+
                    token_addr_to_only_received_addresses[next(tokens_net_received.__iter__())].append(addr)

            # find out which token(s) are potential pivots
            type_b_tokens = set(token_addr_to_only_received_addresses.keys()).intersection(token_addr_to_only_sent_addresses.keys())

            if len(type_b_tokens) == 0:
                l.debug(f'no token pivoting')
                return None

            assert len(type_b_tokens) > 0

            optimizing_token = None

            if least_profitable == False:
                most_optimal_amt = float('-inf')
            else:
                assert least_profitable == True
                most_optimal_amt = float('inf')
            for tok in type_b_tokens:
                senders = token_addr_to_only_sent_addresses[tok]
                receivers = token_addr_to_only_received_addresses[tok]

                amount_sent = 0
                for sender in senders:
                    amount_sent += -(token_movement_sums[sender][tok]) # negative bc it's net sent
                amount_received = 0
                for receiver in receivers:
                    amount_received += token_movement_sums[receiver][tok]

                profit = amount_received - amount_sent
                if optimizing_test(profit, most_optimal_amt):
                    most_optimal_amt = profit
                    optimizing_token = tok

            if optimizing_test(most_optimal_amt, 0):
                # print(f'hmmmmmmmmm {full_txn["transactionHash"].hex()}')
                # import pdb; pdb.set_trace()
                # found a token that made profit, but not sure who the profiter is
                only_cycle = ArbitrageCycle(
                    cycle = parsed_cycle,
                    profit_token = optimizing_token,
                    profit_taker = None,
                    profit_amount = most_optimal_amt,
                )
            # elif len(addrs_condition_a) > 0:
            #     profiter, token, amount = addrs_condition_a[0]
            #     only_cycle = ArbitrageCycle(
            #         cycle = parsed_cycle,
            #         profit_token = optimizing_token,
            #         profit_taker = None,
            #         profit_amount = most_optimal_amt,
            #     )
            #     l.debug(f'negative proit cycle!!!!')
            else:
                l.debug('did not have profit in the right direction')
                return None
                # l.info(f'addrs condition a {addrs_condition_a}')
                # l.info(f'type b tokens {type_b_tokens}')
                # raise Exception(f'Processing arbitrage {full_txn["transactionHash"].hex()}')

        arb = arb._replace(only_cycle=only_cycle)

    return arb

def get_addr_to_movements(
        txns: typing.List[ERC20Transaction],
    ) -> typing.Dict[str, MovementDesc]:
    addr_to_movements: typing.Dict[str, MovementDesc] = collections.defaultdict(lambda: {'in': [], 'out': []})
    for txn in txns:
        to_addr = txn['args']['to']
        from_addr = txn['args']['from']
        addr_to_movements[to_addr]['in'].append(txn)
        addr_to_movements[from_addr]['out'].append(txn)

    return addr_to_movements

def get_potential_exchanges(
        full_txn: web3.types.TxReceipt,
        addr_to_movements: typing.Dict[str, MovementDesc],
    ) -> typing.Set[str]:
    potential_exchanges = set()
    for addr in addr_to_movements:
        if addr.startswith('0x' + '00' * 17):
            # l.warning(f'Ignoring zero address as exchange')
            continue
        if addr in KNOWN_ROUTERS:
            # l.warning(f'Ignoring router as exchange')
            continue

        ins  = addr_to_movements[addr]['in']
        outs = addr_to_movements[addr]['out']
        if len(ins) == 0 or len(outs) == 0:
            # not an exchange, did not both send a token and receive a token
            continue

        # make sure that the sender 'from' address is not marked as an exchange
        if addr == full_txn['from']:
            continue

        in_coins  = set(x['address'] for x in ins)
        out_coins = set(x['address'] for x in outs)
        if len(in_coins) == 1 and len(out_coins) == 1 and in_coins != out_coins:
            # one token in, another token out
            potential_exchanges.add(addr)

    return potential_exchanges
