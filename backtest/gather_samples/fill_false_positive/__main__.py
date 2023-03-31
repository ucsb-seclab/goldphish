import argparse
import logging
import time
import psycopg2
import psycopg2.extensions

from backtest.utils import connect_db

from utils import setup_logging


l = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')

    args = parser.parse_args()

    setup_logging('fill_false_positive', stdout_level=logging.DEBUG if args.verbose else logging.INFO)

    conn = connect_db()
    curr = conn.cursor()

    gen_false_positives(curr)

    l.info('done')

def gen_false_positives(curr: psycopg2.extensions.cursor):
    """
    Build temp table 'sample_arbitrage_false_positives'
    """
    t_start = time.time()

    curr.execute(
        '''
        SELECT EXISTS (
            SELECT FROM 
                pg_tables
            WHERE 
                schemaname = 'public' AND 
                tablename  = 'sample_arbitrage_false_positives'
        );
        ''',
    )
    (already_generated,) = curr.fetchone()
    if already_generated:
        print('Already generated false-positive record')

        if True:
            answer = input('REALLY re-generate record? (yes/NO)')
            if answer.lower().strip() != 'yes':
                return
            
            curr.execute(
                '''
                DROP TABLE sample_arbitrage_false_positives;
                DROP TABLE sample_arbitrages_no_fp;
                DROP TABLE sample_arbitrage_cycles_no_fp;
                DROP TABLE sample_arbitrage_cycle_exchanges_no_fp;
                DROP TABLE sample_arbitrage_cycle_exchange_items_no_fp;
                DROP TABLE sample_arbitrage_exchanges_no_fp;
                '''
            )

    print('Generating false-positive record')
    curr.execute('BEGIN TRANSACTION')
    curr.execute('SET max_parallel_workers_per_gather = 0;')

    curr.execute(
        '''
        CREATE TABLE sample_arbitrage_false_positives (
            sample_arbitrage_id INTEGER NOT NULL,
            reason TEXT NOT NULL
        );

        CREATE INDEX idx_sample_arbitrage_false_positives_id ON sample_arbitrage_false_positives (sample_arbitrage_id);
        '''
    )

    # remove anything with weird token movements
    curr.execute(
        '''
        INSERT INTO sample_arbitrage_false_positives (sample_arbitrage_id, reason)
        SELECT distinct sample_arbitrage_id, 'odd token'
        FROM sample_arbitrages_odd_tokens
        '''
    )
    assert curr.rowcount > 0
    print(f'Labeled {curr.rowcount:,} samples as false-positive because of weird token transfers')

    # remove tokenlon
    curr.execute(
        '''
        INSERT INTO sample_arbitrage_false_positives (sample_arbitrage_id, reason)
        SELECT id, 'tokenlon'
        FROM sample_arbitrages
        WHERE encode(shooter, 'hex') = '03f34be1bf910116595db1b11e9d1b2ca5d59659'
        '''
    )
    assert curr.rowcount > 0
    print(f'Labeled {curr.rowcount:,} samples as false-positive due to tokenlon')

    # remove Dexible https://dexible.io/
    curr.execute(
        '''
        INSERT INTO sample_arbitrage_false_positives (sample_arbitrage_id, reason)
        SELECT id, 'dexible'
        FROM sample_arbitrages
        WHERE encode(shooter, 'hex') = 'ad84693a21e0a1db73ae6c6e5aceb041a6c8b6b3'
        '''
    )
    assert curr.rowcount > 0
    print(f'Labeled {curr.rowcount:,} samples as false-positive due to Dexible')

    # remove CoW Swap
    curr.execute(
        '''
        INSERT INTO sample_arbitrage_false_positives (sample_arbitrage_id, reason)
        SELECT id, 'CoW Swap'
        FROM sample_arbitrages
        WHERE encode(shooter, 'hex') = '9008d19f58aabd9ed0d60971565aa8510560ab41' OR encode(shooter, 'hex') = '3328f5f2cecaf00a2443082b657cedeaf70bfaef'
        '''
    )
    assert curr.rowcount > 0
    print(f'Labeled {curr.rowcount:,} samples as false-positive due to CoW Swap')

    # # remove 0x exchange proxy
    # curr.execute(
    #     '''
    #     INSERT INTO sample_arbitrage_false_positives (sample_arbitrage_id, reason)
    #     SELECT id, '0x exchange proxy'
    #     FROM sample_arbitrages
    #     WHERE encode(shooter, 'hex') = 'def1c0ded9bec7f1a1670819833240f027b25eff'
    #     '''
    # )
    # assert curr.rowcount > 0
    # print(f'Labeled {curr.rowcount:,} samples as false-positive due to 0x proxy')


    # remove null address

    # find null exchange id
    curr.execute('SELECT id FROM sample_arbitrage_exchanges WHERE address = %s', (b'\x00' * 20,))
    assert curr.rowcount <= 1
    if curr.rowcount == 1:
        (zero_addr_exchange_id,) = curr.fetchone()
        print(f'Removing null exchange (id={zero_addr_exchange_id})')

        curr.execute(
            '''
            INSERT INTO sample_arbitrage_false_positives (sample_arbitrage_id, reason)
            SELECT sa.id, 'null address'
            FROM sample_arbitrages sa
            WHERE EXISTS(
                SELECT 1
                FROM sample_arbitrage_cycles sac
                JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
                JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
                WHERE sac.sample_arbitrage_id = sa.id AND sacei.exchange_id = %s
            )
            ''',
            (zero_addr_exchange_id,)
        )
        print(f'Labeled {curr.rowcount:,} arbitrages as false-positive due to null address')
    else:
        print('did not find zero address in exchanges')

    # remove arbitrages with exchange address that is the relayer
    curr.execute(
        '''
        INSERT INTO sample_arbitrage_false_positives (sample_arbitrage_id, reason)
        SELECT id, 'relayer is in exchanges list'
        FROM sample_arbitrages sa
        WHERE EXISTS(
            SELECT
            FROM sample_arbitrage_cycles sac
            join sample_arbitrage_cycle_exchanges sace on sac.id = sace.cycle_id
            join sample_arbitrage_cycle_exchange_items sacei on sace.id = sacei.cycle_exchange_id
            join sample_arbitrage_exchanges sax on sax.id = sacei.exchange_id
            WHERE sa.id = sac.sample_arbitrage_id AND
                sax.address = sa.shooter
        )
        '''
    )
    print(f'Labeled {curr.rowcount:,} arbitrages as false-positive because the relayer cannot be an exchange')

    curr.execute('SELECT COUNT(distinct sample_arbitrage_id) FROM sample_arbitrage_false_positives')
    (n_fp,) = curr.fetchone()

    curr.execute('SELECT COUNT(*) FROM sample_arbitrages')
    (tot_arbs,) = curr.fetchone()

    print(f'Have {n_fp:,} sample arbitrages labeled as false-positive ({n_fp / tot_arbs * 100:.2f}%)')

    # generate tables with false-positives removed
    curr.execute(
        '''
        CREATE TABLE sample_arbitrages_no_fp
        AS SELECT *
        FROM sample_arbitrages sa
        WHERE NOT EXISTS(SELECT 1 FROM sample_arbitrage_false_positives WHERE sa.id = sample_arbitrage_id);
        '''
    )
    assert curr.rowcount == tot_arbs - n_fp
    curr.execute('CREATE INDEX idx_sample_arbitrages_no_fp_id ON sample_arbitrages_no_fp (id);')

    curr.execute(
        '''
        CREATE TABLE sample_arbitrage_cycles_no_fp
        AS SELECT *
        FROM sample_arbitrage_cycles sac
        WHERE EXISTS(SELECT 1 FROM sample_arbitrages_no_fp sa WHERE sa.id = sac.sample_arbitrage_id)
        '''
    )
    assert curr.rowcount > 0
    curr.execute('CREATE INDEX idx_sample_arbitrage_cycles_no_fp_id ON sample_arbitrage_cycles_no_fp (id);')

    curr.execute(
        '''
        CREATE TABLE sample_arbitrage_cycle_exchanges_no_fp
        AS SELECT *
        FROM sample_arbitrage_cycle_exchanges sace
        WHERE EXISTS(SELECT 1 FROM sample_arbitrage_cycles_no_fp sac WHERE sac.id = sace.cycle_id)
        '''
    )
    assert curr.rowcount > 0
    curr.execute('CREATE INDEX idx_sample_arbitrage_cycle_exchanges_no_fp_id ON sample_arbitrage_cycle_exchanges_no_fp (id);')

    curr.execute(
        '''
        CREATE TABLE sample_arbitrage_cycle_exchange_items_no_fp
        AS SELECT *
        FROM sample_arbitrage_cycle_exchange_items sacei
        WHERE EXISTS(SELECT 1 FROM sample_arbitrage_cycle_exchanges_no_fp sace WHERE sace.id = sacei.cycle_exchange_id)
        '''
    )
    assert curr.rowcount > 0

    curr.execute(
        '''
        CREATE TABLE sample_arbitrage_exchanges_no_fp
        AS SELECT *
        FROM sample_arbitrage_exchanges sae
        WHERE EXISTS(
            SELECT 1
            FROM sample_arbitrages_no_fp sa
            JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
            JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
            JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
            WHERE sacei.exchange_id = sae.id
        );
        '''
    )
    assert curr.rowcount > 0
    curr.execute('CREATE INDEX idx_sample_arbitrage_exchanges_no_fp_id ON sample_arbitrage_exchanges_no_fp (id);')

    curr.execute('SELECT COUNT(*) FROM sample_arbitrage_exchanges')
    (n_exchanges,) = curr.fetchone()

    curr.execute('SELECT COUNT(*) FROM sample_arbitrage_exchanges_no_fp')
    (n_exchanges_no_fp,) = curr.fetchone()

    # sanity check for agreement
    curr.execute('SELECT COUNT(distinct exchange_id) FROM sample_arbitrage_cycle_exchange_items_no_fp')
    (n_distinct_exchanges,) = curr.fetchone()
    assert n_distinct_exchanges == n_exchanges_no_fp

    assert n_exchanges_no_fp < n_exchanges
    n_fp_exchanges = n_exchanges - n_exchanges_no_fp
    print(f'Marked {n_fp_exchanges:,} exchanges as false-positive ({n_fp_exchanges / n_exchanges * 100:.2f}%)')
    print(f'Took {time.time() - t_start:.1f} seconds to generate false-positive report')
    curr.connection.commit()
    print()


if __name__ == '__main__':
    main()
