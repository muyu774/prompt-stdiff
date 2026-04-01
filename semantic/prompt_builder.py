"""Prompt templates for node-level semantic descriptions."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

MISSING_TOKENS = {"", "na", "n/a", "none", "null", "nan", "unknown", "unk"}

NODE_ID_KEYS = ("node_id", "sensor_id", "station_id", "detector_id", "id")
NODE_INDEX_KEYS = ("node_index", "index")
ROAD_NAME_KEYS = ("road_name", "road", "street", "highway_name")
ROAD_TYPE_KEYS = ("road_type", "road_class")
DIRECTION_KEYS = ("direction", "travel_direction")
DISTRICT_KEYS = ("district", "area", "zone", "region")
FUNCTIONAL_REGION_KEYS = ("functional_region", "function_region", "land_use")
POI_KEYS = ("poi_category", "poi", "nearby_poi_category")
TRAFFIC_HINT_KEYS = ("traffic_pattern_hint", "traffic_hint")
TIME_CONTEXT_KEYS = ("time_context",)
WEATHER_KEYS = ("weather", "weather_condition")
INCIDENT_KEYS = ("incident", "incident_status")
CALENDAR_KEYS = ("calendar_event", "holiday", "weekday_type")


def _clean_text(value: object) -> Optional[str]:
    """Convert value to clean text and drop missing placeholders."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return None
    return text


def _first_valid(meta: Dict[str, object], keys: Sequence[str]) -> Optional[str]:
    """Get first non-empty value from alias keys."""
    for key in keys:
        if key in meta:
            val = _clean_text(meta.get(key))
            if val is not None:
                return val
    return None


def build_node_prompt(meta: Dict[str, object]) -> str:
    """Build one node prompt from metadata.

    Args:
        meta: Metadata dict for a traffic node.

    Returns:
        Prompt string for semantic encoder.
    """
    node_id = _first_valid(meta, NODE_ID_KEYS) or "unknown_sensor"
    node_index = _first_valid(meta, NODE_INDEX_KEYS)

    road_name = _first_valid(meta, ROAD_NAME_KEYS)
    road_type = _first_valid(meta, ROAD_TYPE_KEYS)
    direction = _first_valid(meta, DIRECTION_KEYS)
    district = _first_valid(meta, DISTRICT_KEYS)
    functional_region = _first_valid(meta, FUNCTIONAL_REGION_KEYS)
    poi_category = _first_valid(meta, POI_KEYS)
    traffic_hint = _first_valid(meta, TRAFFIC_HINT_KEYS)

    time_context = _first_valid(meta, TIME_CONTEXT_KEYS)
    weather = _first_valid(meta, WEATHER_KEYS)
    incident = _first_valid(meta, INCIDENT_KEYS)
    calendar_event = _first_valid(meta, CALENDAR_KEYS)

    parts: List[str] = []
    if node_index is not None:
        parts.append(f"Sensor {node_id} is indexed as node {node_index}.")
    else:
        parts.append(f"Sensor {node_id}.")

    if road_name and road_type:
        parts.append(f"It is located on {road_name}, a {road_type}.")
    elif road_name:
        parts.append(f"It is located on {road_name}.")
    elif road_type:
        parts.append(f"It is located on a {road_type} road.")

    if district:
        parts.append(f"It is in the {district} area.")
    if direction:
        parts.append(f"Traffic mainly moves {direction}.")
    if functional_region:
        parts.append(f"This region is characterized as {functional_region}.")
    if poi_category:
        parts.append(f"Nearby points of interest include {poi_category}.")
    if traffic_hint:
        parts.append(f"Typical traffic behavior: {traffic_hint}.")

    if time_context:
        parts.append(f"Time context: {time_context}.")
    if weather:
        parts.append(f"Weather condition: {weather}.")
    if incident:
        parts.append(f"Incident status: {incident}.")
    if calendar_event:
        parts.append(f"Calendar context: {calendar_event}.")

    # ASSUMPTION: keep one short fallback sentence when metadata is sparse to avoid empty prompts.
    if len(parts) == 1:
        parts.append("Traffic context metadata is limited for this sensor.")

    return " ".join(parts)


def build_prompts(metas: Iterable[Dict[str, object]]) -> List[str]:
    """Build prompt list for all nodes."""
    return [build_node_prompt(m) for m in metas]
