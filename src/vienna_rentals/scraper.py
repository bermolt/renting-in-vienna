"""Client for the willhaben search website."""

import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from tqdm import tqdm

from vienna_rentals import config
from vienna_rentals.models import Listing

logger = logging.getLogger(__name__)


def fetch_search_page(
    session: requests.Session, page: int
) -> dict[str, Any] | None:
    """Fetch one Willhaben search page and extract its search result."""
    params = {**config.SEARCH_PARAMS, "page": str(page)}

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            response = session.get(
                config.WILLHABEN_SEARCH_PAGE_URL,
                headers=config.WILLHABEN_HEADERS,
                params=params,
                timeout=config.REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                logger.warning(
                    "Rate limited on page %d; waiting 5 seconds.",
                    page,
                )
                time.sleep(5)
                continue

            response.raise_for_status()

            match = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                response.text,
            )

            if not match:
                logger.error(
                    "Could not find __NEXT_DATA__ on Willhaben page %d.",
                    page,
                )
                return None

            data = json.loads(match.group(1))
            return data["props"]["pageProps"]["searchResult"]

        except requests.Timeout:
            logger.warning(
                "Timeout on page %d (attempt %d/%d).",
                page,
                attempt + 1,
                config.MAX_RETRIES + 1,
            )

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "Error fetching page %d (attempt %d/%d): %s",
                page,
                attempt + 1,
                config.MAX_RETRIES + 1,
                e,
            )
            time.sleep(1)

    logger.error(
        "Failed to fetch page %d after %d retries.",
        page,
        config.MAX_RETRIES + 1,
    )
    return None


def get_total_pages(session: requests.Session) -> int:
    """Determine how many result pages the current search yields."""
    data = fetch_search_page(session, 1)

    if not data:
        return 0

    rows_found = data.get("rowsFound", 0)
    rows_returned = data.get("rowsReturned", 0)

    if rows_returned == 0:
        logger.error("Willhaben returned rowsReturned=0.")
        return 0

    total_pages = -(-rows_found // rows_returned)

    logger.info(
        "Total rows found: %d, rows per page: %d, total pages: %d",
        rows_found,
        rows_returned,
        total_pages,
    )

    return total_pages


def parse_page(data: dict[str, Any]) -> list[Listing]:
    """Extract listings from a Willhaben search result."""
    adverts = data.get("advertSummaryList", {}).get("advertSummary", [])
    return [Listing.from_advert(advert) for advert in adverts]


def fetch_page(
    session: requests.Session, page: int
) -> dict[str, Any] | None:
    """Fetch one result page."""
    return fetch_search_page(session, page)


def scrape_all_listings() -> list[Listing]:
    """Scrape every Willhaben result page."""
    listings: list[Listing] = []

    with requests.Session() as session:
        total_pages = get_total_pages(session)

        if total_pages == 0:
            logger.warning("No pages to scrape.")
            return listings

        for batch_start in range(
            1, total_pages + 1, config.PAGE_BATCH_SIZE
        ):
            batch_end = min(
                batch_start + config.PAGE_BATCH_SIZE - 1,
                total_pages,
            )

            with ThreadPoolExecutor(
                max_workers=config.MAX_WORKERS
            ) as executor:
                futures = [
                    executor.submit(fetch_page, session, page)
                    for page in range(batch_start, batch_end + 1)
                ]

                progress = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Scraping pages {batch_start}-{batch_end}",
                )

                for future in progress:
                    if data := future.result():
                        listings.extend(parse_page(data))

            if batch_end < total_pages:
                wait_time = random.randint(10, 20)
                logger.info(
                    "Scraped pages %d-%d; pausing %d seconds.",
                    batch_start,
                    batch_end,
                    wait_time,
                )
                time.sleep(wait_time)

    logger.info("Scraped %d listings in total.", len(listings))
    return listings