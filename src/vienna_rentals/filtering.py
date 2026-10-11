"""Filter listings down to the apartments worth a Telegram notification."""

import logging
import re

import pandas as pd

from vienna_rentals import config

logger = logging.getLogger(__name__)


def _is_excluded_listing(df: pd.DataFrame) -> pd.Series:
    """Identify listings matching excluded keywords or social-housing patterns."""
    text_columns = [
        column
        for column in df.columns
        if (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]))
    ]

    if not text_columns:
        return pd.Series(False, index=df.index)

    listing_text = df[text_columns].fillna("").astype(str).agg(" ".join, axis=1)

    keyword_patterns = [
        rf"(?<!\w){re.escape(keyword)}(?!\w)" for keyword in config.EXCLUDE_KEYWORDS
    ]

    patterns = keyword_patterns + list(config.SOCIAL_HOUSING_PATTERNS)
    combined_pattern = "|".join(f"(?:{pattern})" for pattern in patterns)

    return listing_text.str.contains(
        combined_pattern,
        case=False,
        regex=True,
        na=False,
    )


def _build_link(row: pd.Series) -> str:
    """Reconstruct the willhaben detail-page URL for a listing."""
    district = (
        str(row["District"]).replace(" ", "-").lower()
        if pd.notna(row["District"])
        else "unknown-district"
    )
    postcode = int(row["Postcode"]) if pd.notna(row["Postcode"]) else "unknown-postcode"
    location = (
        str(row["Location"]).split(",")[-1].strip().replace(" ", "-").lower()
        if pd.notna(row["Location"])
        else "unknown-location"
    )
    description = (
        str(row["Description"]).replace(" ", "-").replace(",", "").lower()
        if pd.notna(row["Description"])
        else "no-description"
    )
    return (
        f"{config.LISTING_BASE_URL}{district}-{postcode}-{location}/{description}-{row['Ad ID']}/"
    )


def filter_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the personal preferences from config and attach detail-page links."""
    df = df.copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Rooms"] = pd.to_numeric(df["Rooms"], errors="coerce")
    df["Numeric District"] = df["Location"].str.extract(r"(\d{1,2})\. Bezirk").astype(float)
    df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce", utc=True)

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=config.MAX_LISTING_AGE_DAYS)

    excluded = _is_excluded_listing(df)

    filtered = df[
        (df["State"] == "Wien")
        & (df["Price"] > config.MIN_PRICE)
        & (df["Price"] < config.MAX_PRICE)
        & (df["Rooms"] >= config.MIN_ROOMS)
        & (df["Property Type"] == config.PROPERTY_TYPE)
        & (df["Published Date"] >= cutoff)
        & (df["Numeric District"].isin(config.ALLOWED_DISTRICTS))
        & ~excluded
    ].copy()

    filtered["Link"] = filtered.apply(_build_link, axis=1)
    logger.info("%d of %d listings match the configured preferences.", len(filtered), len(df))
    return filtered
