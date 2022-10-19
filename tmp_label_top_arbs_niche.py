import collections
import datetime
import subprocess
import argparse
import psycopg2.extras
import time
from backtest.utils import connect_db

db = connect_db()
curr = db.cursor()

parser = argparse.ArgumentParser()
parser.add_argument('ID', type=int)

args = parser.parse_args()

if False:

    curr.execute('SELECT MIN(start_block), MAX(end_block) FROM block_samples')
    start_block, end_block = curr.fetchone()


    curr.execute(
        '''
        CREATE TEMP TABLE niches AS 
        SELECT DISTINCT niche FROM naive_gas_price_estimate;
        '''
    )
    print(f'Have {curr.rowcount:,} niches')

    # compute niche validity
    curr.execute(
        '''
        SELECT niche, block_number
        FROM naive_gas_price_estimate
        ORDER BY block_number ASC
        '''
    )

    niche_last_seen = {}
    niche_expirations = collections.defaultdict(lambda: [])
    for niche, block_number in curr:
        if niche not in niche_last_seen:
            niche_last_seen[niche]= block_number
        else:
            niche_expirations[niche].append((niche_last_seen[niche], block_number - 1))
            niche_last_seen[niche] = block_number

    # flush
    for niche in niche_last_seen:
        niche_expirations[niche].append((niche_last_seen[niche], end_block))


    print('computed niche expirations, inserting')
    curr.execute(
        '''
        CREATE TEMP TABLE niche_expirations (niche TEXT, start_block BIGINT, end_block BIGINT);
        CREATE INDEX idx_niche_start ON niche_expirations(start_block);
        CREATE INDEX idx_niche_end ON niche_expirations(end_block);
        CREATE INDEX idx_niche_niche ON niche_expirations(niche);
        '''
    )
    for niche, vals in niche_expirations.items():
        psycopg2.extras.execute_batch(
            curr,
            '''
            INSERT INTO niche_expirations (niche, start_block, end_block) VALUES (%s, %s, %s)
            ''',
            [(niche, *x) for x in vals]
        )

    print(f'n to insert: {sum(len(x) for x in niche_expirations.values())}')
    print('inserted niche expirations')

    curr.execute(
        '''
        UPDATE naive_gas_price_estimate gpo
        SET end_block_inclusive = (SELECT ne.end_block FROM niche_expirations ne WHERE ne.niche = gpo.niche AND ne.start_block = gpo.block_number)
        '''
    )
    print(f'Filled {curr.rowcount:,} rows')

    db.commit()


    exit()

    curr.execute(
        '''
        CREATE TABLE naive_gas_price_estimate_blocks AS
        SELECT 
        FROM naive_gas_price_estimate gpo
        JOIN niche_expirations ne ON gpo.block_number = ne.start_block AND gpo.niche = ne.niche
        '''
    )
    exit()

curr.execute('SELECT start_block, end_block FROM block_samples WHERE priority = %s', (args.ID,))
start_block, end_block = curr.fetchone()
print(f'Querying for id={args.ID} from {start_block} to {end_block}')

# find campaigns
curr.execute(
    '''
    CREATE TEMP TABLE selected_campaigns AS
    SELECT id FROM top_candidate_arbitrage_campaigns WHERE %s <= end_block AND end_block <= %s
    ''',
    (start_block, end_block)
)
print(f'Have {curr.rowcount:,} campaigns to label niches in {args.ID} priority')


