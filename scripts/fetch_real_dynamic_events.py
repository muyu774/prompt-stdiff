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
- source
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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
    source: str


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
                source="open_meteo",
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
                    source="nager_date",
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
                source="osm_overpass",
            )
        )
    return rows


def _load_sensor_points(
    sensor_metadata_csv: Path,
    idx_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    """Load sensor geolocation table for incident-to-sensor mapping."""
    df = pd.read_csv(sensor_metadata_csv)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    i_col = idx_col.strip().lower()
    la_col = lat_col.strip().lower()
    lo_col = lon_col.strip().lower()
    if not {i_col, la_col, lo_col}.issubset(df.columns):
        raise ValueError(
            f"sensor_metadata_csv must contain {i_col},{la_col},{lo_col}. "
            f"available={list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "node_index": pd.to_numeric(df[i_col], errors="coerce"),
            "lat": pd.to_numeric(df[la_col], errors="coerce"),
            "lon": pd.to_numeric(df[lo_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["node_index", "lat", "lon"]).copy()
    out["node_index"] = out["node_index"].astype(int)
    out = out.drop_duplicates(subset=["node_index"], keep="first").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid sensor coordinates found in sensor_metadata_csv.")
    return out


def _load_graph_neighbors(adjacency_csv: Path, num_nodes: int) -> Dict[int, List[int]]:
    """Load undirected graph neighbors from adjacency csv."""
    if not adjacency_csv.exists():
        return {}
    edges = pd.read_csv(adjacency_csv)
    edges.columns = [str(c).strip().lower().replace(" ", "_") for c in edges.columns]
    src_col = "src" if "src" in edges.columns else ("from" if "from" in edges.columns else None)
    dst_col = "dst" if "dst" in edges.columns else ("to" if "to" in edges.columns else None)
    if src_col is None or dst_col is None:
        return {}

    neighbors: Dict[int, List[int]] = {i: [] for i in range(int(num_nodes))}
    for _, row in edges.iterrows():
        s = pd.to_numeric(row[src_col], errors="coerce")
        d = pd.to_numeric(row[dst_col], errors="coerce")
        if pd.isna(s) or pd.isna(d):
            continue
        si = int(s)
        di = int(d)
        if 0 <= si < num_nodes and 0 <= di < num_nodes:
            neighbors[si].append(di)
            neighbors[di].append(si)
    return neighbors


def _bfs_within_hops(neighbors: Dict[int, List[int]], start: int, max_hops: int) -> set[int]:
    """Get reachable node set within hop budget."""
    if max_hops < 0:
        return set(neighbors.keys())
    seen = {int(start)}
    frontier = {int(start)}
    for _ in range(int(max_hops)):
        nxt: set[int] = set()
        for u in frontier:
            for v in neighbors.get(u, []):
                if v not in seen:
                    nxt.add(v)
                    seen.add(v)
        if not nxt:
            break
        frontier = nxt
    return seen


def _haversine_m(lat: float, lon: float, lat_arr: np.ndarray, lon_arr: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in meters."""
    r = 6371000.0
    phi1 = np.deg2rad(lat)
    phi2 = np.deg2rad(lat_arr)
    dphi = np.deg2rad(lat_arr - lat)
    dlambda = np.deg2rad(lon_arr - lon)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 1e-12)))
    return r * c


def _map_incident_nodes(
    lat: float,
    lon: float,
    sensor_points: pd.DataFrame,
    radius_m: float,
    neighbors: Optional[Dict[int, List[int]]] = None,
    topology_hops: int = 3,
) -> List[int]:
    """Map one incident location to sensor indices via geo + optional topology filter."""
    lat_arr = sensor_points["lat"].to_numpy(dtype=np.float64)
    lon_arr = sensor_points["lon"].to_numpy(dtype=np.float64)
    node_arr = sensor_points["node_index"].to_numpy(dtype=np.int64)
    dist = _haversine_m(lat=lat, lon=lon, lat_arr=lat_arr, lon_arr=lon_arr)

    in_radius_mask = dist <= float(radius_m)
    if not bool(np.any(in_radius_mask)):
        return []

    in_radius_nodes = node_arr[in_radius_mask]
    # Use nearest sensor as corridor anchor for topology pruning.
    nearest_node = int(node_arr[int(np.argmin(dist))])

    if neighbors:
        allowed = _bfs_within_hops(neighbors=neighbors, start=nearest_node, max_hops=topology_hops)
        topo_nodes = [int(x) for x in in_radius_nodes.tolist() if int(x) in allowed]
        if topo_nodes:
            return sorted(set(topo_nodes))

    return sorted(set([int(x) for x in in_radius_nodes.tolist()]))


def load_optional_incidents_csv(
    incidents_csv: Optional[Path],
    timezone: str,
    sensor_metadata_csv: Optional[Path] = None,
    sensor_idx_col: str = "node_index",
    sensor_lat_col: str = "latitude",
    sensor_lon_col: str = "longitude",
    incident_lat_col: str = "latitude",
    incident_lon_col: str = "longitude",
    incident_radius_m: float = 2000.0,
    adjacency_csv: Optional[Path] = None,
    topology_hops: int = 3,
) -> List[EventRow]:
    """Load optional local incidents CSV.

    Expected columns:
    - timestamp (required)
    - description or text (required one)
    - node_index (optional)
    - latitude/longitude (optional; used for geo mapping when node_index is absent)
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

    sensor_points: Optional[pd.DataFrame] = None
    neighbors: Optional[Dict[int, List[int]]] = None
    if sensor_metadata_csv is not None:
        if not sensor_metadata_csv.exists():
            raise FileNotFoundError(f"sensor_metadata_csv not found: {sensor_metadata_csv}")
        sensor_points = _load_sensor_points(
            sensor_metadata_csv=sensor_metadata_csv,
            idx_col=sensor_idx_col,
            lat_col=sensor_lat_col,
            lon_col=sensor_lon_col,
        )
        if adjacency_csv is not None:
            neighbors = _load_graph_neighbors(
                adjacency_csv=adjacency_csv,
                num_nodes=int(sensor_points["node_index"].max()) + 1,
            )

    rows: List[EventRow] = []
    idx_col = "node_index"
    lat_col = incident_lat_col.strip().lower()
    lon_col = incident_lon_col.strip().lower()
    for _, r in df.iterrows():
        ts = pd.to_datetime(r["timestamp"], errors="coerce")
        if pd.isna(ts):
            continue

        desc = str(r.get("description", r.get("text", "incident")))
        district = str(r.get("district", "global"))
        time_ctx = _time_context_label(pd.Timestamp(ts))
        text = f"incident: {desc}; time_context: {time_ctx}"

        mapped_nodes: List[int] = []
        node_raw = r.get(idx_col, None)
        if pd.notna(node_raw):
            try:
                mapped_nodes = [int(float(node_raw))]
            except (TypeError, ValueError):
                mapped_nodes = []

        if (not mapped_nodes) and (sensor_points is not None) and (lat_col in df.columns) and (lon_col in df.columns):
            lat_raw = pd.to_numeric(r.get(lat_col), errors="coerce")
            lon_raw = pd.to_numeric(r.get(lon_col), errors="coerce")
            if pd.notna(lat_raw) and pd.notna(lon_raw):
                mapped_nodes = _map_incident_nodes(
                    lat=float(lat_raw),
                    lon=float(lon_raw),
                    sensor_points=sensor_points,
                    radius_m=float(incident_radius_m),
                    neighbors=neighbors,
                    topology_hops=int(topology_hops),
                )
        if not mapped_nodes:
            mapped_nodes = [-1]

        for node_idx in mapped_nodes:
            rows.append(
                EventRow(
                    timestamp=pd.Timestamp(ts),
                    node_index=int(node_idx),
                    text=text,
                    weather="unknown",
                    incident=desc,
                    holiday="no",
                    time_context=time_ctx,
                    district=district,
                    event_type="incident_context",
                    description=desc,
                    source="incident_csv",
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
    parser.add_argument(
        "--sensor_metadata_csv",
        type=Path,
        default=None,
        help="Optional sensor metadata CSV with node_index+latitude+longitude for incident mapping.",
    )
    parser.add_argument("--sensor_idx_col", type=str, default="node_index")
    parser.add_argument("--sensor_lat_col", type=str, default="latitude")
    parser.add_argument("--sensor_lon_col", type=str, default="longitude")
    parser.add_argument("--incident_lat_col", type=str, default="latitude")
    parser.add_argument("--incident_lon_col", type=str, default="longitude")
    parser.add_argument("--incident_radius_m", type=float, default=2000.0)
    parser.add_argument(
        "--adjacency_csv",
        type=Path,
        default=None,
        help="Optional adjacency.csv for topology-aware incident filtering.",
    )
    parser.add_argument("--topology_hops", type=int, default=3)
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
        sensor_metadata_csv=args.sensor_metadata_csv,
        sensor_idx_col=str(args.sensor_idx_col),
        sensor_lat_col=str(args.sensor_lat_col),
        sensor_lon_col=str(args.sensor_lon_col),
        incident_lat_col=str(args.incident_lat_col),
        incident_lon_col=str(args.incident_lon_col),
        incident_radius_m=float(args.incident_radius_m),
        adjacency_csv=args.adjacency_csv,
        topology_hops=int(args.topology_hops),
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
