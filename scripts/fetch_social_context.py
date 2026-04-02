#!/usr/bin/env python3
"""Fetch social context for extreme traffic events.

This script is designed for Prompt-STDiff case-study analysis.
Given an event-point table (dataset/node/time-window/location/keywords),
it fetches:
1) Reddit posts (public JSON endpoint, no API key required),
2) Optional GDELT article context (news proxy for public discourse),
3) Search URLs for X/Weibo for manual verification.

ASSUMPTION:
- X/Weibo direct API crawling is not used by default due API/ToS limits.
- Reddit + GDELT are used as practical public signals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests


REDDIT_URL = "https://www.reddit.com/search.json"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class SocialRow:
    """One fetched social-context row."""

    dataset: str
    node_index: int
    event_id: str
    source: str
    timestamp: str
    title: str
    text: str
    url: str
    query: str
    x_search_url: str
    weibo_search_url: str


def _to_utc_ts(s: str) -> pd.Timestamp:
    """Parse time string to UTC timestamp."""
    ts = pd.to_datetime(s, utc=True, errors="raise")
    return pd.Timestamp(ts)


def _build_query(row: pd.Series) -> str:
    """Build robust search query from event row."""
    def _norm(v: object) -> str:
        if pd.isna(v):
            return ""
        s = str(v).strip()
        if s.lower() in {"nan", "none", "null"}:
            return ""
        return s

    location = _norm(row.get("location", ""))
    keywords = _norm(row.get("keywords", ""))
    base_terms = "traffic accident OR crash OR lane closure OR congestion"
    if location and keywords:
        return f"({location}) AND ({keywords}) AND ({base_terms})"
    if location:
        return f"({location}) AND ({base_terms})"
    if keywords:
        return f"({keywords}) AND ({base_terms})"
    return base_terms


def _x_search_url(query: str) -> str:
    """Build X search URL for manual verification."""
    return f"https://x.com/search?q={quote_plus(query)}&src=typed_query&f=live"


def _weibo_search_url(query: str) -> str:
    """Build Weibo search URL for manual verification."""
    return f"https://s.weibo.com/weibo?q={quote_plus(query)}"


def fetch_reddit(
    query: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    limit: int = 200,
    timeout_sec: int = 20,
) -> List[Dict[str, str]]:
    """Fetch Reddit posts and filter by UTC time window."""
    headers = {"User-Agent": "prompt-stdiff-social-fetch/1.0"}
    params = {
        "q": query,
        "sort": "new",
        "limit": min(max(int(limit), 1), 250),
        "type": "link",
        "t": "all",
    }
    resp = requests.get(REDDIT_URL, params=params, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()

    out: List[Dict[str, str]] = []
    children = data.get("data", {}).get("children", [])
    for c in children:
        d = c.get("data", {})
        created = d.get("created_utc", None)
        if created is None:
            continue
        ts = pd.Timestamp(datetime.fromtimestamp(float(created), tz=timezone.utc))
        if ts < start_utc or ts > end_utc:
            continue

        title = str(d.get("title", "")).strip()
        text = str(d.get("selftext", "")).strip()
        permalink = str(d.get("permalink", "")).strip()
        url = f"https://www.reddit.com{permalink}" if permalink else str(d.get("url", ""))
        out.append(
            {
                "timestamp": ts.isoformat(),
                "title": title,
                "text": text,
                "url": url,
            }
        )
    return out


def _to_gdelt_dt(ts: pd.Timestamp) -> str:
    """Convert UTC timestamp to GDELT datetime format."""
    return ts.tz_convert("UTC").strftime("%Y%m%d%H%M%S")


def fetch_gdelt(
    query: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    max_records: int = 50,
    timeout_sec: int = 20,
) -> List[Dict[str, str]]:
    """Fetch GDELT article context as public-discourse proxy."""
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_records),
        "startdatetime": _to_gdelt_dt(start_utc),
        "enddatetime": _to_gdelt_dt(end_utc),
    }
    resp = requests.get(GDELT_URL, params=params, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()

    out: List[Dict[str, str]] = []
    for it in data.get("articles", []):
        title = str(it.get("title", "")).strip()
        text = str(it.get("seendate", "")).strip()
        url = str(it.get("url", "")).strip()
        ts_raw = str(it.get("seendate", "")).strip()
        try:
            # GDELT seendate: YYYYMMDDTHHMMSSZ
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        except Exception:
            ts = pd.NaT
        out.append(
            {
                "timestamp": ts.isoformat() if pd.notna(ts) else "",
                "title": title,
                "text": text,
                "url": url,
            }
        )
    return out


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description="Fetch social context for extreme event points.")
    parser.add_argument("--events_csv", type=Path, required=True, help="CSV with event points/time windows.")
    parser.add_argument("--out_csv", type=Path, required=True, help="Output merged social context CSV.")
    parser.add_argument("--reddit_limit", type=int, default=200)
    parser.add_argument("--with_gdelt", action="store_true")
    parser.add_argument("--timeout_sec", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.events_csv)
    required_cols = ["dataset", "node_index", "event_id", "start_time", "end_time"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"events_csv missing required columns: {missing}")

    rows: List[SocialRow] = []

    for _, r in df.iterrows():
        dataset = str(r["dataset"])
        node_index = int(r["node_index"])
        event_id = str(r["event_id"])
        start_utc = _to_utc_ts(str(r["start_time"]))
        end_utc = _to_utc_ts(str(r["end_time"]))
        query = _build_query(r)
        x_url = _x_search_url(query)
        wb_url = _weibo_search_url(query)

        # Reddit
        try:
            reddit_items = fetch_reddit(
                query=query,
                start_utc=start_utc,
                end_utc=end_utc,
                limit=int(args.reddit_limit),
                timeout_sec=int(args.timeout_sec),
            )
        except Exception as exc:
            reddit_items = [
                {
                    "timestamp": "",
                    "title": "REDDIT_FETCH_ERROR",
                    "text": str(exc),
                    "url": "",
                }
            ]

        for it in reddit_items:
            rows.append(
                SocialRow(
                    dataset=dataset,
                    node_index=node_index,
                    event_id=event_id,
                    source="reddit",
                    timestamp=it["timestamp"],
                    title=it["title"],
                    text=it["text"],
                    url=it["url"],
                    query=query,
                    x_search_url=x_url,
                    weibo_search_url=wb_url,
                )
            )

        # Optional GDELT
        if args.with_gdelt:
            try:
                gdelt_items = fetch_gdelt(
                    query=query,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    max_records=50,
                    timeout_sec=int(args.timeout_sec),
                )
            except Exception as exc:
                gdelt_items = [
                    {
                        "timestamp": "",
                        "title": "GDELT_FETCH_ERROR",
                        "text": str(exc),
                        "url": "",
                    }
                ]
            for it in gdelt_items:
                rows.append(
                    SocialRow(
                        dataset=dataset,
                        node_index=node_index,
                        event_id=event_id,
                        source="gdelt",
                        timestamp=it["timestamp"],
                        title=it["title"],
                        text=it["text"],
                        url=it["url"],
                        query=query,
                        x_search_url=x_url,
                        weibo_search_url=wb_url,
                    )
                )

        # Always add one manual-search row as fallback anchor.
        rows.append(
            SocialRow(
                dataset=dataset,
                node_index=node_index,
                event_id=event_id,
                source="manual_search",
                timestamp="",
                title="Manual X/Weibo search links",
                text="Use provided links to verify event context around the point/time window.",
                url="",
                query=query,
                x_search_url=x_url,
                weibo_search_url=wb_url,
            )
        )

    out_df = pd.DataFrame([r.__dict__ for r in rows])
    out_df = out_df.sort_values(["dataset", "event_id", "source", "timestamp"], kind="stable")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved social context: {args.out_csv} rows={len(out_df)}")


if __name__ == "__main__":
    main()
