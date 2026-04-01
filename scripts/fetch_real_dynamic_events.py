"""Fetch real dynamic events (weather + holiday + optional incidents) for Prompt-STDiff.

Output CSV columns are compatible with scripts/build_dynamic_semantic_bank.py:
- timestamp
- node_index
- text
- weather
- incident
- holiday
- time_context
- district
- event_type
- description
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.exceptions import RequestException

DEFAULT_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]


@dataclass
class EventRow:
    """One dynamic event record."""

    timestamp: pd.Timestamp
    node_index: int
    text: str
    weather: str
    incident: str
    holiday: str
    time_context: str
    district: str
    event_type: str
    description: str


def _time_context_label(ts: pd.Timestamp) -> str:
    """Map timestamp to coarse traffic time context."""
    hm = ts.hour + ts.minute / 60.0
    if 7.0 <= hm <= 9.5:
        return "morning_rush_hour"
    if 17.0 <= hm <= 19.5:
        return "evening_rush_hour"
    if 0.0 <= hm < 5.0:
        return "night_low_demand"
    return "off_peak"


def _safe_get_json(url: str, params: Dict[str, object], timeout_sec: int) -> Dict:
    """GET JSON with basic error checking."""
    resp = requests.get(url, params=params, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()


def fetch_openmeteo_weather(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timezone: str,
    timeout_sec: int = 30,
) -> List[EventRow]:
    """Fetch historical hourly weather from Open-Meteo archive API.

    API docs: https://open-meteo.com/
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,weather_code,wind_speed_10m",
        "timezone": timezone,
    }
    data = _safe_get_json(url=url, params=params, timeout_sec=timeout_sec)

    hourly = data.get("hourly", {})
    ts_list = hourly.get("time", [])
    temp_list = hourly.get("temperature_2m", [])
    prec_list = hourly.get("precipitation", [])
    code_list = hourly.get("weather_code", [])
    wind_list = hourly.get("wind_speed_10m", [])

    rows: List[EventRow] = []
    for i, t_str in enumerate(ts_list):
        ts = pd.Timestamp(t_str)
        temp = temp_list[i] if i < len(temp_list) else None
        prec = prec_list[i] if i < len(prec_list) else None
        code = code_list[i] if i < len(code_list) else None
        wind = wind_list[i] if i < len(wind_list) else None

        weather_desc = f"temp={temp}, precipitation={prec}, weather_code={code}, wind_speed={wind}"
        time_ctx = _time_context_label(ts)
        text = f"weather: {weather_desc}; time_context: {time_ctx}"

        rows.append(
            EventRow(
                timestamp=ts,
                node_index=-1,
                text=text,
                weather=weather_desc,
                incident="none",
                holiday="no",
                time_context=time_ctx,
                district="global",
                event_type="weather_context",
                description="open-meteo hourly weather event",
            )
        )
    return rows


def fetch_public_holidays(
    start_date: str,
    end_date: str,
    country_code: str,
    timezone: str,
    timeout_sec: int = 30,
) -> List[EventRow]:
    """Fetch public holidays from Nager.Date API."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    years = list(range(start_ts.year, end_ts.year + 1))
    rows: List[EventRow] = []

    for y in years:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{y}/{country_code}"
        resp = requests.get(url, timeout=timeout_sec)
        if resp.status_code != 200:
            continue

        items = resp.json()
        if not isinstance(items, list):
            continue

        for it in items:
            date_str = str(it.get("date", ""))
            if not date_str:
                continue
            day = pd.Timestamp(date_str)
            if day < start_ts.normalize() or day > end_ts.normalize():
                continue

            # Put holiday event at 00:00 local date.
            ts = pd.Timestamp(f"{day.date()} 00:00:00")
            local_name = str(it.get("localName", "holiday"))
            name = str(it.get("name", local_name))
            text = f"holiday: {name}; local_name: {local_name}"

            rows.append(
                EventRow(
                    timestamp=ts,
                    node_index=-1,
                    text=text,
                    weather="unknown",
                    incident="none",
                    holiday="yes",
                    time_context="holiday",
                    district="global",
                    event_type="holiday_context",
                    description="public holiday event",
                )
            )
    return rows


def _extract_center_latlon(el: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Extract element center latitude/longitude from Overpass element."""
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])
    center = el.get("center", {})
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None, None


def _classify_poi(tags: Dict[str, str]) -> Tuple[str, str]:
    """Classify OSM tags into coarse POI category and short descriptor."""
    amenity = str(tags.get("amenity", "")).lower()
    shop = str(tags.get("shop", "")).lower()
    leisure = str(tags.get("leisure", "")).lower()
    tourism = str(tags.get("tourism", "")).lower()
    public_transport = str(tags.get("public_transport", "")).lower()
    railway = str(tags.get("railway", "")).lower()

    if public_transport or railway in {"station", "halt", "tram_stop"}:
        return "transport_hub", f"public_transport={public_transport or railway}"
    if amenity in {"bus_station", "ferry_terminal", "parking"}:
        return "transport_hub", f"amenity={amenity}"
    if amenity in {"school", "college", "university", "kindergarten"}:
        return "school", f"amenity={amenity}"
    if amenity in {"hospital", "clinic", "doctors", "pharmacy"}:
        return "healthcare", f"amenity={amenity}"
    if amenity in {"restaurant", "cafe", "bar", "fast_food", "food_court"}:
        return "dining", f"amenity={amenity}"
    if leisure in {"stadium", "sports_centre", "pitch", "track"}:
        return "stadium", f"leisure={leisure}"
    if tourism in {"attraction", "museum", "gallery", "theme_park"}:
        return "tourism", f"tourism={tourism}"
    if shop:
        return "shopping", f"shop={shop}"
    if amenity:
        return "other_poi", f"amenity={amenity}"
    if leisure:
        return "other_poi", f"leisure={leisure}"
    if tourism:
        return "other_poi", f"tourism={tourism}"
    return "other_poi", "unknown"


def _build_overpass_poi_query(
    latitude: float,
    longitude: float,
    radius_m: int,
    timeout_sec: int,
) -> str:
    """Build Overpass query for POI retrieval around a coordinate."""
    return f"""
[out:json][timeout:{max(30, timeout_sec)}];
(
  node(around:{radius_m},{latitude},{longitude})[amenity];
  way(around:{radius_m},{latitude},{longitude})[amenity];
  relation(around:{radius_m},{latitude},{longitude})[amenity];
  node(around:{radius_m},{latitude},{longitude})[shop];
  way(around:{radius_m},{latitude},{longitude})[shop];
  relation(around:{radius_m},{latitude},{longitude})[shop];
  node(around:{radius_m},{latitude},{longitude})[leisure];
  way(around:{radius_m},{latitude},{longitude})[leisure];
  relation(around:{radius_m},{latitude},{longitude})[leisure];
  node(around:{radius_m},{latitude},{longitude})[tourism];
  way(around:{radius_m},{latitude},{longitude})[tourism];
  relation(around:{radius_m},{latitude},{longitude})[tourism];
  node(around:{radius_m},{latitude},{longitude})[public_transport];
  way(around:{radius_m},{latitude},{longitude})[public_transport];
  relation(around:{radius_m},{latitude},{longitude})[public_transport];
);
out center tags;
"""


def _compose_overpass_endpoints(primary_url: str, extra_urls: str) -> List[str]:
    """Compose unique endpoint list with primary URL first."""
    urls: List[str] = []
    if primary_url.strip():
        urls.append(primary_url.strip())
    for u in DEFAULT_OVERPASS_ENDPOINTS:
        if u not in urls:
            urls.append(u)
    if extra_urls.strip():
        for u in extra_urls.split(","):
            v = u.strip()
            if v and v not in urls:
                urls.append(v)
    return urls


def fetch_osm_poi_catalog(
    latitude: float,
    longitude: float,
    radius_m: int,
    overpass_url: str,
    overpass_extra_urls: str = "",
    max_retries: int = 2,
    timeout_sec: int = 60,
) -> pd.DataFrame:
    """Fetch POI catalog from Overpass API with endpoint fallback and retries."""
    endpoints = _compose_overpass_endpoints(
        primary_url=overpass_url,
        extra_urls=overpass_extra_urls,
    )
    radius_candidates = [
        int(radius_m),
        max(1000, int(radius_m * 0.7)),
        max(1000, int(radius_m * 0.5)),
    ]
    errors: List[str] = []
    payload: Optional[Dict] = None

    for radius in radius_candidates:
        query = _build_overpass_poi_query(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius,
            timeout_sec=timeout_sec,
        )
        for endpoint in endpoints:
            for attempt in range(1, max(1, int(max_retries)) + 1):
                try:
                    resp = requests.post(
                        endpoint,
                        data=query.encode("utf-8"),
                        timeout=timeout_sec,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    print(
                        f"[POI] Overpass success endpoint={endpoint} "
                        f"radius_m={radius} attempt={attempt}"
                    )
                    break
                except (RequestException, ValueError) as exc:
                    errors.append(
                        f"endpoint={endpoint}, radius={radius}, attempt={attempt}, err={exc}"
                    )
            if payload is not None:
                break
        if payload is not None:
            break

    if payload is None:
        msg = (
            "Failed to fetch OSM POI catalog from all endpoints. "
            f"tried_endpoints={len(endpoints)} tried_radii={radius_candidates}. "
        )
        if errors:
            msg += f"last_error={errors[-1]}"
        raise RuntimeError(msg)

    elements = payload.get("elements", [])
    rows: List[Dict[str, object]] = []
    for el in elements:
        tags = el.get("tags", {})
        if not isinstance(tags, dict):
            continue
        category, tag_desc = _classify_poi(tags)
        lat, lon = _extract_center_latlon(el)
        name = str(tags.get("name", ""))
        rows.append(
            {
                "osm_id": int(el.get("id", -1)),
                "osm_type": str(el.get("type", "")),
                "name": name,
                "category": category,
                "tag_desc": tag_desc,
                "lat": lat,
                "lon": lon,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["osm_id", "osm_type", "name", "category", "tag_desc", "lat", "lon"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["osm_id", "osm_type"], keep="first")
    return df.reset_index(drop=True)


def _is_poi_category_active(category: str, ts: pd.Timestamp) -> bool:
    """Heuristic POI activity schedule by category."""
    hour = ts.hour + ts.minute / 60.0
    weekday = ts.dayofweek  # Mon=0
    weekend = weekday >= 5

    # ASSUMPTION: category-specific active windows approximate human mobility patterns.
    if category == "transport_hub":
        return (7.0 <= hour <= 10.0) or (16.0 <= hour <= 20.0)
    if category == "school":
        return (weekday < 5) and ((7.0 <= hour <= 9.5) or (14.0 <= hour <= 17.0))
    if category == "shopping":
        return (weekend and (10.0 <= hour <= 21.0)) or ((not weekend) and (17.0 <= hour <= 21.0))
    if category == "stadium":
        return (weekday >= 4) and (18.0 <= hour <= 23.0)
    if category == "dining":
        return (11.0 <= hour <= 14.0) or (17.0 <= hour <= 22.0)
    if category == "tourism":
        return (weekend and (9.0 <= hour <= 19.0)) or ((not weekend) and (10.0 <= hour <= 18.0))
    if category == "healthcare":
        return True
    return (8.0 <= hour <= 20.0)


def generate_poi_context_events(
    poi_catalog: pd.DataFrame,
    start_date: str,
    end_date: str,
    timezone: str,
    top_k: int = 4,
    min_count: int = 1,
) -> List[EventRow]:
    """Generate hourly global POI context events from POI catalog."""
    if poi_catalog.empty:
        return []

    cat_counts = (
        poi_catalog["category"]
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )
    cat_counts = cat_counts[cat_counts >= int(min_count)]
    if cat_counts.empty:
        return []

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
    timeline = pd.date_range(start=start_ts, end=end_ts, freq="1h")

    rows: List[EventRow] = []
    for ts in timeline:
        active_items = []
        for cat, cnt in cat_counts.items():
            if _is_poi_category_active(category=str(cat), ts=ts):
                active_items.append((str(cat), int(cnt)))

        if not active_items:
            continue
        active_items = sorted(active_items, key=lambda x: x[1], reverse=True)[: max(1, int(top_k))]

        cat_text = ", ".join([f"{c}({n})" for c, n in active_items])
        time_ctx = _time_context_label(ts)
        text = f"poi_context: {cat_text}; time_context: {time_ctx}"

        rows.append(
            EventRow(
                timestamp=pd.Timestamp(ts),
                node_index=-1,
                text=text,
                weather="unknown",
                incident="none",
                holiday="no",
                time_context=time_ctx,
                district="global",
                event_type="poi_context",
                description="derived from OSM POI distribution and activity schedule",
            )
        )
    return rows


def load_optional_incidents_csv(
    incidents_csv: Optional[Path],
    timezone: str,
) -> List[EventRow]:
    """Load optional local incidents CSV.

    Expected columns:
    - timestamp (required)
    - description or text (required one)
    - node_index (optional, default -1)
    - district (optional)
    """
    if incidents_csv is None:
        return []
    if not incidents_csv.exists():
        raise FileNotFoundError(f"incidents_csv not found: {incidents_csv}")

    df = pd.read_csv(incidents_csv)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError("incidents_csv must contain 'timestamp' column")
    if "description" not in df.columns and "text" not in df.columns:
        raise ValueError("incidents_csv must contain 'description' or 'text' column")

    rows: List[EventRow] = []
    for _, r in df.iterrows():
        ts = pd.to_datetime(r["timestamp"], errors="coerce")
        if pd.isna(ts):
            continue

        desc = str(r.get("description", r.get("text", "incident")))
        node_idx = int(r.get("node_index", -1)) if pd.notna(r.get("node_index", -1)) else -1
        district = str(r.get("district", "global"))
        time_ctx = _time_context_label(pd.Timestamp(ts))
        text = f"incident: {desc}; time_context: {time_ctx}"

        rows.append(
            EventRow(
                timestamp=pd.Timestamp(ts),
                node_index=node_idx,
                text=text,
                weather="unknown",
                incident=desc,
                holiday="no",
                time_context=time_ctx,
                district=district,
                event_type="incident_context",
                description=desc,
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Fetch real dynamic events")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--start_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--timezone", type=str, default="Asia/Shanghai")

    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--country_code", type=str, default="US")
    parser.add_argument("--include_poi_context", action="store_true")
    parser.add_argument("--poi_radius_m", type=int, default=20000)
    parser.add_argument("--poi_overpass_url", type=str, default="https://overpass-api.de/api/interpreter")
    parser.add_argument(
        "--poi_overpass_extra_urls",
        type=str,
        default="",
        help="Comma-separated extra Overpass endpoints.",
    )
    parser.add_argument("--poi_max_retries", type=int, default=2)
    parser.add_argument(
        "--poi_fail_hard",
        action="store_true",
        help="If set, fail entire script when POI fetch fails.",
    )
    parser.add_argument("--poi_top_k", type=int, default=4)
    parser.add_argument("--poi_min_count", type=int, default=1)
    parser.add_argument(
        "--poi_catalog_csv",
        type=Path,
        default=None,
        help="Optional path to save fetched POI catalog from OSM.",
    )

    parser.add_argument(
        "--incidents_csv",
        type=Path,
        default=None,
        help="Optional local incidents CSV to merge",
    )
    parser.add_argument("--timeout_sec", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    """Main entry."""
    args = parse_args()

    weather_rows = fetch_openmeteo_weather(
        start_date=args.start_date,
        end_date=args.end_date,
        latitude=float(args.latitude),
        longitude=float(args.longitude),
        timezone=args.timezone,
        timeout_sec=int(args.timeout_sec),
    )

    holiday_rows = fetch_public_holidays(
        start_date=args.start_date,
        end_date=args.end_date,
        country_code=args.country_code,
        timezone=args.timezone,
        timeout_sec=int(args.timeout_sec),
    )

    incident_rows = load_optional_incidents_csv(
        incidents_csv=args.incidents_csv,
        timezone=args.timezone,
    )

    poi_rows: List[EventRow] = []
    if bool(args.include_poi_context):
        try:
            poi_catalog = fetch_osm_poi_catalog(
                latitude=float(args.latitude),
                longitude=float(args.longitude),
                radius_m=int(args.poi_radius_m),
                overpass_url=str(args.poi_overpass_url),
                overpass_extra_urls=str(args.poi_overpass_extra_urls),
                max_retries=int(args.poi_max_retries),
                timeout_sec=int(args.timeout_sec),
            )
            if args.poi_catalog_csv is not None:
                args.poi_catalog_csv.parent.mkdir(parents=True, exist_ok=True)
                poi_catalog.to_csv(args.poi_catalog_csv, index=False)
                print(f"Saved POI catalog: {args.poi_catalog_csv} rows={len(poi_catalog)}")

            poi_rows = generate_poi_context_events(
                poi_catalog=poi_catalog,
                start_date=args.start_date,
                end_date=args.end_date,
                timezone=args.timezone,
                top_k=int(args.poi_top_k),
                min_count=int(args.poi_min_count),
            )
        except Exception as exc:  # noqa: BLE001
            if bool(args.poi_fail_hard):
                raise
            print(f"[WARN] POI context fetch failed, continue without POI. err={exc}")

    all_rows = weather_rows + holiday_rows + incident_rows + poi_rows
    if not all_rows:
        raise RuntimeError("No events fetched. Check date range/network/API availability.")

    records = [r.__dict__ for r in all_rows]
    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"Saved real dynamic events: {args.out_csv}")
    print(f"rows={len(out_df)}")
    print("event counts:")
    print(out_df["event_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
