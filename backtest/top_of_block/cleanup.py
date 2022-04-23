"""
Some various database cleanup tasks and consistency-checking.
"""
import logging

from backtest.utils import connect_db


l = logging.getLogger(__name__)


def do_cleanup():
    db = connect_db()
    db.autocommit = False

    curr = db.cursor()

    #
    # TASK: look for incomplete verifications
    #
    curr.execute(
        '''
        CREATE TEMP TABLE tmp_abandoned_blocks_to_verify (block_number INTEGER NOT NULL);
        INSERT INTO tmp_abandoned_blocks_to_verify (block_number)
            SELECT block_number
            FROM candidate_arbitrage_blocks_to_verify
            WHERE verify_started IS NOT NULL AND
                verify_finished IS NULL AND
                verify_started + (interval '10 hours') < now()::timestamp;
        '''
    )
    
    curr.execute('SELECT COUNT(distinct block_number) FROM tmp_abandoned_blocks_to_verify')
    (n_abandoned,) = curr.fetchone()

    if n_abandoned > 0:
        ans = input(f'Have {n_abandoned} blocks abandoned during verification, requeue? (y/N) ')
        if ans.strip().lower() == 'y':
            l.info(f'cleaning up {n_abandoned} blocks abandoned')

            curr.execute(
                '''
                CREATE TEMP VIEW tmp_candidates_in_abandoned_block AS
                SELECT id
                    FROM candidate_arbitrages ca
                    WHERE EXISTS(SELECT 1 FROM tmp_abandoned_blocks_to_verify abv WHERE abv.block_number = ca.block_number)
                '''
            )

            n_verified_deleted = 0
            n_failed_deleted = 0

            curr.execute(
                '''
                DELETE
                    FROM verified_arbitrages va
                    WHERE EXISTS(
                            SELECT 1
                            FROM tmp_candidates_in_abandoned_block cab
                            WHERE cab.id = va.candidate_id
                    )
                ''',
            )
            n_verified_deleted += curr.rowcount

            curr.execute(
                '''
                DELETE
                    FROM failed_arbitrages fa
                    WHERE EXISTS(
                            SELECT 1
                            FROM tmp_candidates_in_abandoned_block cab
                            WHERE cab.id = fa.candidate_id
                    )
                ''',
            )
            n_failed_deleted += curr.rowcount

            curr.execute(
                '''
                UPDATE candidate_arbitrage_blocks_to_verify
                    SET verify_started = NULL
                    WHERE block_number IN (SELECT block_number FROM tmp_abandoned_blocks_to_verify)
                '''
            )
            assert curr.rowcount == n_abandoned

            l.info(f'Deleted {n_verified_deleted:,} verified arbitrages')
            l.info(f'Deleted {n_failed_deleted:,} failed arbitrages')
            l.info(f'Requeued {n_abandoned:,} abandoned blocks')

            ans = input('Commit? (y/N) ')
            if ans.strip().lower() == 'y':
                print('committing')
                db.commit()
            else:
                print('rolling back')
                db.rollback()

    #
    # TASK: requeue blocks where zero gas price was estimated
    #
    curr.execute(
        '''
        CREATE TEMP TABLE tmp_blocks_with_zero_gas (block_number INTEGER NOT NULL);
        INSERT INTO tmp_blocks_with_zero_gas (block_number)
            SELECT distinct block_number
            FROM verified_arbitrages va
            JOIN candidate_arbitrages ca ON va.candidate_id = ca.id
            WHERE va.gas_price <= 1
        '''
    )
    curr.execute(
        '''
        SELECT COUNT(*) FROM tmp_blocks_with_zero_gas
        '''
    )
    (n_blocks_zero_gas,) = curr.fetchone()
    if n_blocks_zero_gas > 0:
        ans = input(f'Found {n_blocks_zero_gas} blocks with gasprice set to zero, requeue? (y/N) ')
        if ans.lower().strip() == 'y':
            curr.execute(
                '''
                CREATE TEMP TABLE tmp_candidates_with_zero_gas (id INTEGER NOT NULL);
                INSERT INTO tmp_candidates_with_zero_gas (id)
                    SELECT id
                    FROM candidate_arbitrages ca
                    WHERE EXISTS(SELECT 1 FROM tmp_blocks_with_zero_gas abv WHERE abv.block_number = ca.block_number)
                '''
            )

            n_verified_deleted = 0
            n_failed_deleted = 0

            curr.execute(
                '''
                DELETE
                    FROM verified_arbitrages va
                    WHERE EXISTS(
                            SELECT 1
                            FROM tmp_candidates_with_zero_gas cab
                            WHERE cab.id = va.candidate_id
                    )
                ''',
            )
            n_verified_deleted += curr.rowcount

            curr.execute(
                '''
                DELETE
                    FROM failed_arbitrages fa
                    WHERE EXISTS(
                            SELECT 1
                            FROM tmp_candidates_with_zero_gas cab
                            WHERE cab.id = fa.candidate_id
                    )
                ''',
            )
            n_failed_deleted += curr.rowcount

            curr.execute(
                '''
                UPDATE candidate_arbitrage_blocks_to_verify
                    SET verify_started = NULL
                    WHERE block_number IN (SELECT block_number FROM tmp_blocks_with_zero_gas)
                '''
            )
            assert curr.rowcount == n_blocks_zero_gas

            l.info(f'Deleted {n_verified_deleted:,} verified arbitrages')
            l.info(f'Deleted {n_failed_deleted:,} failed arbitrages')
            l.info(f'Requeued {n_blocks_zero_gas:,} blocks with 0 gas')

            ans = input('Continue? (y/N) ')
            if ans.strip().lower() == 'y':
                db.commit()
            else:
                db.rollback()
