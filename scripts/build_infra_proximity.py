import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

INFRA_FILES = [
    DATA / "critical_infrastructure.json",
    DATA / "infra_digital.json",
    DATA / "infra_hazardous.json",
]

EVENT_FILES = [
    DATA / "gdelt.geojson",
    DATA / "gdelt_linked.geojson",
    DATA / "gdelt_crossborder.geojson",
    DATA / "usgs.geojson",
    DATA / "gdacs.geojson",
    DATA / "local_events.geojson",
]

OUTPUT = DATA / "infra_proximity.json"
MAX_DISTANCE_KM = 50


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_infrastructure():
    items = []
    for path in INFRA_FILES:
        payload = load_json(path)
        if not payload:
            continue
        for item in payload.get("items", []):
            try:
                item["lat"] = float(item["lat"])
                item["lon"] = float(item["lon"])
            except Exception:
                continue
            item["_source_file"] = path.name
            items.append(item)
    return items


def load_events():
    events = []
    for path in EVENT_FILES:
        payload = load_json(path)
        if not payload:
            continue
        for feature in payload.get("features", []):
            geom = feature.get("geometry", {})
            props = feature.get("properties", {})
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue
            try:
                lon = float(coords[0])
                lat = float(coords[1])
            except Exception:
                continue
            events.append({
                "title": props.get("title") or props.get("name") or "Unnamed",
                "category": props.get("category") or props.get("gdelt_bucket") or props.get("type") or "unknown",
                "source": props.get("source") or props.get("domain") or props.get("feed_name") or path.stem,
                "time": props.get("time") or props.get("datetime") or props.get("date"),
                "url": props.get("url") or props.get("search_url"),
                "lat": lat,
                "lon": lon,
                "_source_file": path.name,
            })
    return events


def calculate_level(distance, criticality):
    try:
        criticality = float(criticality)
    except Exception:
        criticality = 5
    if distance <= 5 and criticality >= 8:
        return "critical"
    if distance <= 15 and criticality >= 8:
        return "high"
    if distance <= 30:
        return "medium"
    return "watch"


def calculate_score(distance, criticality):
    try:
        criticality = float(criticality)
    except Exception:
        criticality = 5
    distance_factor = max(0, 1 - (distance / MAX_DISTANCE_KM))
    return round((criticality * 10) * distance_factor, 2)


def build():
    infrastructure = load_infrastructure()
    events = load_events()
    matches = []

    for event in events:
        for infra in infrastructure:
            distance = haversine_km(event["lat"], event["lon"], infra["lat"], infra["lon"])
            if distance > MAX_DISTANCE_KM:
                continue
            level = calculate_level(distance, infra.get("criticality", 5))
            score = calculate_score(distance, infra.get("criticality", 5))
            matches.append({
                "level": level,
                "score": score,
                "distance_km": round(distance, 2),
                "event": {
                    "title": event["title"],
                    "category": event["category"],
                    "source": event["source"],
                    "time": event["time"],
                    "url": event["url"],
                    "lat": event["lat"],
                    "lon": event["lon"],
                    "source_file": event["_source_file"],
                },
                "infrastructure": {
                    "name": infra.get("name"),
                    "country": infra.get("country"),
                    "city": infra.get("city"),
                    "category": infra.get("category"),
                    "subtype": infra.get("subtype"),
                    "criticality": infra.get("criticality"),
                    "operator": infra.get("operator"),
                    "lat": infra.get("lat"),
                    "lon": infra.get("lon"),
                    "source_file": infra.get("_source_file"),
                },
            })

    severity_order = {"critical": 4, "high": 3, "medium": 2, "watch": 1}
    matches.sort(key=lambda x: (severity_order.get(x["level"], 0), x["score"]), reverse=True)

    report = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "infrastructure_count": len(infrastructure),
            "event_count": len(events),
            "match_count": len(matches),
            "max_distance_km": MAX_DISTANCE_KM,
        },
        "top_matches": matches[:50],
        "matches": matches,
    }

    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT}")
    print(f"Infrastructure: {len(infrastructure)}")
    print(f"Events: {len(events)}")
    print(f"Matches: {len(matches)}")


if __name__ == "__main__":
    build()

