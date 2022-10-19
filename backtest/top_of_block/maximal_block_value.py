"""
Measures the approximate maximal extractable value in each block

Since this is generally an NP-complete problem, we just use a simple
naive (Greedy) algorithm.
"""

import collections
import itertools
import logging
import psycopg2.extensions
import psycopg2.extras
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
    parser.add_argument('--priority', type=int)


    return parser_name, find_mev


def find_mev(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        db.commit()
        return

    l.info('starting mev-finding')

    assert 0 <= args.priority <= 29

    curr.execute(f'SELECT start_block, end_block FROM block_samples WHERE priority = %s', (args.priority,))
    assert curr.rowcount == 1
    start_block, end_block = curr.fetchone()

    analyze_blocks(curr, start_block, end_block)


def setup_db(curr: psycopg2.extensions.cursor):
    l.info('setting up database')
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS candidate_arbitrages_mev (
            block_number INTEGER NOT NULL,
            mev          NUMERIC(78, 0) NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_arbitrages_mev_block_number ON candidate_arbitrages_mev (block_number);


        CREATE TABLE IF NOT EXISTS candidate_arbitrages_mev_selected (
            block_number INTEGER NOT NULL,
            candidate_arbitrage_id BIGINTEGER NOT NULL REFERENCES candidate_arbitrages (id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrages_mev_selected_block_number ON candidate_arbitrages_mev_selected (block_number);
        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrages_mev_selected_id ON candidate_arbitrages_mev_selected (candidate_arbitrage_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_arbitrages_mev_selected_block_number_id ON candidate_arbitrages_mev_selected (block_number, candidate_arbitrage_id);
        '''
    )


ARBITRAGE_NODE = 0x1
EXCHANGE_NODE = 0x2
def analyze_blocks(curr: psycopg2.extensions.cursor, start_block: int, end_block: int):
    # get all arbitrage campaigns and arbitrages
    curr.execute(
        '''
        SELECT id, block_number_start, block_number_end
        FROM candidate_arbitrage_campaigns
        WHERE gas_pricer = 'median' AND niche LIKE 'nfb|%%' AND %s <= block_number_start AND block_number_end <= %s
        ''',
        (start_block, end_block)
    )
    l.debug(f'Have {curr.rowcount:,} campaigns in this priority')
    
    candidates = list(curr)
    candidates_with_arbitrage = []
    for id_, block_number_start, block_number_end in candidates:
        curr.execute(
            '''
            SELECT ca.block_number, ca.exchanges, profit_after_fee_wei, ca.id
            FROM candidate_arbitrage_campaign_member
            JOIN candidate_arbitrages ca ON ca.id = candidate_arbitrage_campaign_member.candidate_arbitrage_id
            WHERE candidate_arbitrage_campaign = %s
            ORDER BY ca.block_number
            ''',
            (id_,)
        )
        assert curr.rowcount > 0, f'campaign {id_} had no candidates'
        candidates_with_arbitrage.append((id_, block_number_start, block_number_end, list(curr)))

    candidates_by_start_block = collections.defaultdict(lambda: [])
    candidates_by_end_block = collections.defaultdict(lambda: [])
    for c in candidates_with_arbitrage:
        block_number_start = c[1]
        block_number_end = c[2]
        candidates_by_start_block[block_number_start].append(c)
        candidates_by_end_block[block_number_end].append(c)

    running_campaigns = {}

    for block_number in range(start_block, end_block + 1):
        for c in candidates_by_start_block[block_number]:
            id_ = c[0]
            running_campaigns[id_] = c
        
        for c in candidates_by_end_block[block_number - 1]:
            id_ = c[0]
            del running_campaigns[id_]
        
        l.debug(f'Have {len(running_campaigns):,} campaigns running in block {block_number:,}')

        available_candidates = []
        for c in running_campaigns.values():
            these_candidates = c[3]
            candidate = these_candidates[0]
            for maybe_candidate in these_candidates[1:]:
                if maybe_candidate[0] <= block_number:
                    candidate = maybe_candidate
                else:
                    break
            
            available_candidates.append(candidate)
    
        analyze_block(curr, available_candidates, block_number)
        curr.connection.commit()

def analyze_block(curr: psycopg2.extensions.cursor, candidates: typing.List[typing.Tuple[int, typing.List[bytes], int, int]], block_number: int):
    l.debug(f'Have {len(candidates):,} arbitrages in block {block_number}')

    g = nx.Graph()

    for _, exchanges, profit, id_ in candidates:
        exchanges = [e.tobytes() for e in exchanges]
        profit = int(profit)

        assert 2 <= len(exchanges) <= 3

        g.add_node(id_, weight=profit, type=ARBITRAGE_NODE)
        for exc in exchanges:
            node = exc
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

    psycopg2.extras.execute_batch(
        curr,
        '''
        INSERT INTO candidate_arbitrages_mev_selected (block_number, candidate_arbitrage_id)
        VALUES (%s, %s)
        ''',
        zip(itertools.repeat(block_number), selected_arbitrages)
    )

    curr.execute(
        '''
        INSERT INTO candidate_arbitrages_mev (block_number, mev) VALUES (%s, %s)
        ''',
        (block_number, total_weight)
    )

    l.debug(f'Selected {len(selected_arbitrages):,} arbitrages in block {block_number} totaling {total_weight / (10 ** 18):.8f} ETH')

