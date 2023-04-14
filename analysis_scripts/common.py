"""
Common helpers for analysis
"""

import time
import psycopg2.extensions


def label_zerox_exchanges(curr: psycopg2.extensions.cursor):
    """
    Build temp table 'tmp_zerox_exchanges'
    """
    curr.execute(
        '''
        CREATE TEMP TABLE tmp_zerox_exchanges (
            exchange_id INTEGER NOT NULL
        );

        CREATE INDEX idx_tmp_zerox_exchanges ON tmp_zerox_exchanges(exchange_id);
        '''
    )

    curr.execute(
        '''
        CREATE TEMP TABLE tmp_exchanges_with_zerox
        AS SELECT distinct sacei.exchange_id
        FROM sample_arbitrage_cycle_exchange_item_is_zerox is_zerox
        JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.id = is_zerox.sample_arbitrage_cycle_exchange_item_id
        WHERE is_zerox.is_zerox = true
        '''
    )
    n_with_zerox = curr.rowcount

    curr.execute('SELECT COUNT(*) FROM sample_arbitrage_exchanges')
    (n_exchanges,) = curr.fetchone()

    print(f'Have {n_with_zerox:,} exchanges with zerox ({n_with_zerox / n_exchanges * 100:.2f}%)')

    # use counting to find exchanges which were only used in zerox

    # count all exchange usages
    curr.execute(
        '''
        CREATE TEMP TABLE tmp_exchange_use_count (
            id       INTEGER NOT NULL,
            cnt      INTEGER NOT NULL
        );

        INSERT INTO tmp_exchange_use_count
        SELECT sacei.exchange_id, COUNT(*)
        FROM sample_arbitrage_cycle_exchange_items sacei
        GROUP BY sacei.exchange_id;
        '''
    )
    assert curr.rowcount == n_exchanges

    # count all exchanges used as zerox
    curr.execute(
        '''
        CREATE TEMP TABLE tmp_exchange_use_zerox (
            id       INTEGER NOT NULL,
            cnt      INTEGER NOT NULL
        );

        INSERT INTO tmp_exchange_use_zerox
        SELECT sacei.exchange_id, COUNT(*)
        FROM sample_arbitrage_cycle_exchange_items sacei
        JOIN sample_arbitrage_cycle_exchange_item_is_zerox is_zerox
            ON sacei.id = is_zerox.sample_arbitrage_cycle_exchange_item_id
        WHERE is_zerox.is_zerox = true
        GROUP BY sacei.exchange_id;
        '''
    )
    assert curr.rowcount == n_with_zerox

    # find out which exchanges have every use as zerox
    curr.execute(
        '''
        INSERT INTO tmp_zerox_exchanges (exchange_id)
        SELECT euc.id
        FROM tmp_exchange_use_count euc
        JOIN tmp_exchange_use_zerox eucz ON euc.id = eucz.id AND euc.cnt = eucz.cnt
        '''
    )
    n_zerox = curr.rowcount
    print(f'Found {n_zerox:,} exchanges which just used zerox')
    print(f'    {n_zerox / n_exchanges * 100:.2f}% of all exchanges')
    print(f'    {n_zerox / n_with_zerox * 100:.2f}% of all exchanges that had any zerox action')



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


_setup_backrun_tables = False
def setup_backrun_arb_tables(curr: psycopg2.extensions.cursor, only_new = False):
    global _setup_backrun_tables
    if _setup_backrun_tables:
        return
    _setup_backrun_tables = True
    curr.execute(
        f'''
        CREATE TEMP TABLE tmp_backrunners (
            sample_arbitrage_id INTEGER NOT NULL,
            txn_hash BYTEA NOT NULL
        );

        CREATE TEMP TABLE tmp_not_backrunners (
            sample_arbitrage_id INTEGER NOT NULL,
            txn_hash BYTEA NOT NULL
        );

        INSERT INTO tmp_not_backrunners (sample_arbitrage_id, txn_hash)
        SELECT sample_arbitrage_id, sa.txn_hash
        FROM sample_arbitrage_backrun_detections sabd
        JOIN sample_arbitrages_no_fp sa ON sa.id = sabd.sample_arbitrage_id
        WHERE sabd.rerun_exactly = true AND sa.n_cycles = 1 and {"block_number < 15628035" if not only_new else "15628035 <= block_number"};

        INSERT INTO tmp_backrunners (sample_arbitrage_id, txn_hash)
        SELECT sample_arbitrage_id, sa.txn_hash
        FROM sample_arbitrage_backrun_detections sabd
        JOIN sample_arbitrages_no_fp sa ON sa.id = sabd.sample_arbitrage_id
        WHERE (sabd.rerun_reverted = true OR sabd.rerun_no_arbitrage = true) AND sa.n_cycles = 1 and {"block_number < 15628035" if not only_new else "15628035 <= block_number"};
        '''
    )



_setup_weth_arb_tables = False
def setup_weth_arb_tables(curr: psycopg2.extensions.cursor, only_new = False):
    """
    Build temp table 'tmp_weth_arbs' of WETH-profiting arbitrages
    """
    global _setup_weth_arb_tables
    if _setup_weth_arb_tables:
        return
    _setup_weth_arb_tables = True
    weth_token_id = 57

    curr.execute(
        f'''
        CREATE TEMP TABLE tmp_weth_arbs (
            id INTEGER NOT NULL,
            block_number INTEGER NOT NULL,
            revenue NUMERIC(78, 0) NOT NULL,
            coinbase_xfer NUMERIC(78, 0),
            fee NUMERIC(78, 0) NOT NULL,
            net_profit NUMERIC(78, 0),
            txn_hash BYTEA
        );

        INSERT INTO tmp_weth_arbs (id, block_number, revenue, coinbase_xfer, fee, txn_hash)
        SELECT sa.id, sa.block_number, sac.profit_amount, sa.coinbase_xfer, sa.gas_used * sa.gas_price, txn_hash
        FROM {"(select * from sample_arbitrages_no_fp x where 15628035 > x.block_number)" if not only_new else "(select * from sample_arbitrages_no_fp x where 15628035 <= x.block_number)"} sa
        JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
        WHERE sac.profit_token = %s
        ''',
        (weth_token_id,)
    )
    n_weth = curr.rowcount
    print(f'have {n_weth:,} arbitrages with WETH profit')

    curr.execute(
        '''
        UPDATE tmp_weth_arbs
        SET net_profit = (revenue - fee) - coinbase_xfer
        WHERE coinbase_xfer IS NOT NULL
        '''
    )
    n_nonnull_coinbase = curr.rowcount
    print(f'Have coinbase transfer info for {n_nonnull_coinbase:,} arbitrages; {n_nonnull_coinbase / n_weth * 100:.2f}%')
    print()


def setup_only_uniswap_tables(curr: psycopg2.extensions.cursor):
    #
    # find out what arbitrages only used uniswap
    #
    print('[*] collecting non-uniswap exchanges')
    start = time.time()
    curr.execute(
        '''
        CREATE TEMP TABLE non_uniswap_exchanges (
            id INTEGER NOT NULL
        );

        INSERT INTO non_uniswap_exchanges (id)
        SELECT id
        FROM sample_arbitrage_exchanges sae
        WHERE NOT EXISTS(SELECT 1 FROM uniswap_v1_exchanges uv1 WHERE uv1.address = sae.address) AND
            NOT EXISTS(SELECT 1 FROM uniswap_v2_exchanges uv2 WHERE uv2.address = sae.address) AND
            NOT EXISTS(SELECT 1 FROM uniswap_v3_exchanges uv3 WHERE uv3.address = sae.address)
        '''
    )
    elapsed = time.time() - start
    print(f'[*] found {curr.rowcount} non-uniswap exchanges (took {elapsed:.2f} seconds)')
    start = time.time()
    curr.execute(
        '''
        CREATE TEMP TABLE non_uniswap_cycle_items (
            id INTEGER NOT NULL PRIMARY KEY
        );

        INSERT INTO non_uniswap_cycle_items (id)
        SELECT sacei.id
        FROM sample_arbitrage_cycle_exchange_items sacei
        WHERE sacei.exchange_id IN (SELECT id FROM non_uniswap_exchanges);
        '''
    )
    elapsed = time.time() - start
    print(f'[*] found {curr.rowcount} cycle exchage items that were not uniswap (took {elapsed:.2f} seconds)')
    start = time.time()
    curr.execute(
        '''
        CREATE TEMP TABLE non_uniswap_arbitrages (
            id INTEGER NOT NULL PRIMARY KEY
        );

        INSERT INTO non_uniswap_arbitrages (id)
        SELECT distinct sac.sample_arbitrage_id
        FROM sample_arbitrage_cycles sac
        JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
        JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
        WHERE EXISTS(SELECT 1 FROM non_uniswap_cycle_items ci WHERE ci.id = sacei.id)
        '''
    )
    elapsed = time.time() - start
    print(f'[*] found {curr.rowcount} single-cycle arbitrages used a non-uniswap exchange (took {elapsed:.2f} seconds)')

    start = time.time()
    curr.execute(
        '''
        CREATE TEMP TABLE only_uniswap_arbitrages (
            id INTEGER NOT NULL PRIMARY KEY
        );

        INSERT INTO only_uniswap_arbitrages (id)
        SELECT id
        FROM sample_arbitrages
        WHERE NOT EXISTS(
            SELECT 1 FROM non_uniswap_arbitrages nua WHERE nua.id = sample_arbitrages.id
        );
        '''
    )
    elapsed = time.time() - start
    n_only_uniswap = curr.rowcount
    print(f'[*] found {n_only_uniswap:,} single-cycle arbitrages that only use uniswap (took {elapsed:.2f} seconds)')

