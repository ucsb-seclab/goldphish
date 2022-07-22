"""
Measures the approximate maximal extractable value in each block

Since this is generally an NP-complete problem, we just use a simple
naive (Greedy) algorithm.
"""

import itertools
import logging
import psycopg2.extensions
import argparse
import typing
import web3
import networkx as nx
import web3.types

from backtest.utils import connect_db
from utils import WETH_ADDRESS

l = logging.getLogger(__name__)

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'find-mev'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (run before mass scan)')

    return parser_name, find_mev


def find_mev(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        db.commit()
        return

    l.info('starting mev-finding')

    blocks_to_analyze = get_blocks_to_analyze(curr)
    l.info(f'Have {len(blocks_to_analyze):,} blocks to analyze')

    for block_number in blocks_to_analyze:
        analyze_block(w3, curr, block_number)


def setup_db(curr: psycopg2.extensions.cursor):
    l.info('setting up database')
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS candidate_arbitrages_mev_selected (
            candidate_arbitrage_id INTEGER NOT NULL REFERENCES candidate_arbitrages (id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_arbitrages_mev_selected_id ON candidate_arbitrages_mev_selected (candidate_arbitrage_id);
        '''
    )


def get_blocks_to_analyze(curr: psycopg2.extensions.cursor) -> typing.List[int]:
    curr.execute(
        '''
        SELECT completed_on IS NOT NULL, block_number_start, block_number_end, progress
        FROM candidate_arbitrage_reservations
        WHERE claimed_on IS NOT NULL
        '''
    )

    ret = []
    for completed, start_block, end_block, progress in curr:
        if completed:
            assert progress == end_block
        ret.extend(range(start_block, progress + 1))
    
    assert len(set(ret)) == len(ret)

    return sorted(ret)

ARBITRAGE_NODE = 0x1
EXCHANGE_NODE = 0x2
def analyze_block(w3: web3.Web3, curr: psycopg2.extensions.cursor, block_number: int):
    # get all arbitrages and form the conflict graph
    curr.execute(
        '''
        SELECT id, exchanges, directions, profit_no_fee
        FROM candidate_arbitrages WHERE block_number = %s
        ''',
        (block_number,)
    )
    l.debug(f'Have {curr.rowcount:,} arbitrages in block {block_number}')
    if curr.rowcount == 0:
        return

    g = nx.Graph()

    for id_, exchanges, directions, profit_no_fee in curr:
        exchanges = [e.tobytes() for e in exchanges]
        directions = [d.tobytes() for d in directions]
        assert directions[0] == bytes.fromhex(WETH_ADDRESS[2:])
        directions = list(zip(directions, directions[1:] + [directions[0]]))
        profit_no_fee = int(profit_no_fee)

        assert 2 <= len(exchanges) <= 3
        assert len(directions) == len(exchanges)

        g.add_node(id_, weight=profit_no_fee, type=ARBITRAGE_NODE)
        for exc, dir in zip(exchanges, directions):
            dir = tuple(sorted(dir))
            node = (exc, dir)
            if not g.has_node(node):
                g.add_node(node, type=EXCHANGE_NODE)
            g.add_edge(id_, node)

    # collapse exchange nodes
    for node, data in list(g.nodes(data=True)):
        if data['type'] == ARBITRAGE_NODE:
            continue

        for n1, n2 in itertools.combinations(g.neighbors(node), 2):
            assert g.nodes[n1]['type'] == ARBITRAGE_NODE
            assert g.nodes[n2]['type'] == ARBITRAGE_NODE
            assert n1 != n2

            g.add_edge(n1, n2)
        
        g.remove_node(node)

    l.debug(f'Have {len(list(nx.connected_components(g))):,} connected components in graph')

    selected_arbitrages = []
    selected_arbitrage_weights = []

    while len(g.nodes) > 0:
        largest_node_by_weight = max(g.nodes, key = lambda x: g.nodes[x]['weight'])

        selected_arbitrages.append(largest_node_by_weight)
        selected_arbitrage_weights.append(g.nodes[largest_node_by_weight]['weight'])
        l.debug(f'selected arbitrage {largest_node_by_weight}')

        # remove the node and all its neighbors from the component
        for neighbor in list(g.neighbors(largest_node_by_weight)):
            g.remove_node(neighbor)
        g.remove_node(largest_node_by_weight)

    total_weight = sum(selected_arbitrage_weights)
    l.debug(f'Selected {len(selected_arbitrages):,} arbitrages in block {block_number} totaling {total_weight / (10 ** 18):.8f} ETH')

