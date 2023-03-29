"""
Scrapes KyberSwap exchange addresses
"""


import typing
import logging
import psycopg2.extensions
import web3
import web3.logs
import web3.contract
import web3._utils.events
from eth_utils import event_abi_to_log_topic

from utils import get_abi
from backtest.gather_samples.tokens import get_token
from .base_log_scraper import BaseLogScraper, ScrapeResult, PrimeResult


l = logging.getLogger(__name__)

class KyberSwapScraper(BaseLogScraper):
    exchange_abi: typing.Dict
    factory: web3.contract.Contract

    def __init__(self) -> None:
        super().__init__()
        self.factory = web3.Web3().eth.contract(
            address = '0x833e4083B7ae46CeA85695c4f7ed25CDAd8886dE',
            abi = get_abi('kyberswap/factory.abi.json'),
        )
        self.pool_created_topic = event_abi_to_log_topic(self.factory.events.PoolCreated().abi)

    def prime(self, curr: psycopg2.extensions.cursor):
        curr.execute(
            """
            CREATE TABLE IF NOT EXISTS kyberswap_exchanges (
                id SERIAL PRIMARY KEY NOT NULL,
                token0_id INTEGER NOT NULL,
                token1_id INTEGER NOT NULL,
                amp_bps INTEGER NOT NULL,
                pool_index INTEGER NOT NULL CHECK(pool_index >= 1),
                index INTEGER NOT NULL,
                address BYTEA NOT NULL,
                origin_txn BYTEA NOT NULL,
                origin_block INTEGER NOT NULL,
                FOREIGN KEY (token0_id) REFERENCES tokens (id),
                FOREIGN KEY (token1_id) REFERENCES tokens (id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kyber_tokens_pool_idx ON kyberswap_exchanges (pool_index);
            CREATE INDEX IF NOT EXISTS idx_kyber_token0 ON kyberswap_exchanges (token0_id);
            CREATE INDEX IF NOT EXISTS idx_kyber_token1 ON kyberswap_exchanges (token1_id);
            CREATE INDEX IF NOT EXISTS idx_kyber_ex_addr ON kyberswap_exchanges USING hash (address);
            """
        )

        return PrimeResult(['0x833e4083B7ae46CeA85695c4f7ed25CDAd8886dE'])


    def scrape(
                self,
                curr: psycopg2.extensions.cursor,
                w3: web3.Web3,
                logs: typing.List[typing.Dict]
            ) -> ScrapeResult:
        relevant_logs = []
        for log in logs:
            # filter out irrelevant logs
            if log['address'] == self.factory.address and len(log['topics']) > 0 and log['topics'][0] == self.pool_created_topic:
                relevant_logs.append(log)

        l.debug(f'Have {len(relevant_logs)} relevant logs')

        for log in relevant_logs:
            self.process_factory_event(
                w3,
                curr,
                log
            )
        return ScrapeResult(set())

    def process_factory_event(
                self,
                w3: web3.Web3,
                curr: psycopg2.extensions.cursor,
                log
            ) -> typing.Optional[str]:
        receipt = self.factory.events.PoolCreated().processLog(log)

        # if we already know about this exchange then skip
        curr.execute(
            "SELECT id FROM kyberswap_exchanges WHERE address = %s",
            (bytes.fromhex(receipt['args']['pool'][2:]),)
        )
        if curr.rowcount > 0:
            id_ = curr.fetchone()[0]
            l.debug(f'Already know about this exchange, id={id_}')
            return

        block_number = receipt['blockNumber']

        token0_addr_sz: str = receipt['args']['token0']
        token0_addr = bytes.fromhex(token0_addr_sz[2:])
        token1_addr_sz: str = receipt['args']['token1']
        token1_addr = bytes.fromhex(token1_addr_sz[2:])
        
        token0_id = get_token(w3, curr, token0_addr, block_identifier=block_number + 1).id
        token1_id = get_token(w3, curr, token1_addr, block_identifier=block_number + 1).id

        # sanity check
        assert isinstance(token0_id, int)
        assert isinstance(token1_id, int)

        amp_bps = receipt['args']['ampBps']


        # record the exchange info
        curr.execute(
            """
            INSERT INTO kyberswap_exchanges (
                token0_id, token1_id, amp_bps, pool_index, index, address, origin_txn, origin_block
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING (id)
            """,
            (
                token0_id,
                token1_id,
                amp_bps,
                receipt['args']['totalPool'],
                0,
                bytes.fromhex(receipt['args']['pool'][2:]),
                receipt['transactionHash'],
                receipt['blockNumber'],
            )
        )
        l.info(f'Registered kyberswap_exchanges exchange id={curr.fetchone()[0]}')
        return receipt['args']['pool']