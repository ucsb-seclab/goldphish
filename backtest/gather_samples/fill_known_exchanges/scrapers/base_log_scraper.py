"""
base_log_scraper.py

Contains the base class interface for scrapers that operate on
a collection of log statements.
"""
import typing
import os
import json
import web3
import psycopg2.extensions


class ScrapeResult:
    new_watch_addrs: typing.Set[str]

    def __init__(self, new_watch_addrs: typing.Set[str]) -> None:
        self.new_watch_addrs = new_watch_addrs


class PrimeResult:
    watch_addrs: typing.Set[str]

    def __init__(self, watch_addrs: typing.Set[str]) -> None:
        self.watch_addrs = watch_addrs


class BaseLogScraper:
    _abi_cache: typing.ClassVar[typing.Dict[str, typing.Any]] = {}

    def __init__(self) -> None:
        pass

    def prime(self, curr: psycopg2.extensions.cursor) -> PrimeResult:
        """
        Prepare the scraper
        """
        raise NotImplementedError('Method not implemented')

    def scrape(
            self,
            curr: psycopg2.extensions.cursor,
            w3: web3.Web3,
            logs: typing.List[typing.Dict]
        ) -> ScrapeResult:
        raise NotImplementedError('Method not implemented')
