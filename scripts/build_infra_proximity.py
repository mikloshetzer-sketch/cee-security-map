import json
import math
import hashlib
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

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def stable_id(*parts):
    raw = "|".join(str(p or "") for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


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
                item["criticality"] = float(item.get("criticality", 5))
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

            title = props.get("title") or props.get("name") or props.get("type") or "Unnamed"
            url = props.get("url") or props.get("search_url")
            source = props.get("source") or props.get("domain") or path.stem
            category = props.get("category") or props.get("gdelt_bucket") or props.get("type") or "unknown"
            time = props.get("time") or props.get("datetime") or props.get("date")

            event_id = props.get("id") or stable_id(title, url, source, time, lat, lon)

            events.append({
                "id": event_id,
                "title": title,
                "category": category,
                "source": source,
                "time": time,
                "url": url,
                "lat": lat,
                "lon": lon,
                "_source_file": path.name
            })

    return deduplicate_events(events)


def deduplicate_events(events):
    seen = {}
    for event in events:
        key = stable_id(
            event.get("title"),
            event.get("url"),
            event.get("source"),
            event.get("time")
        )

        if key not in seen:
            seen[key] = event

    return list(seen.values())


def calculate_level(distance, criticality):
    if distance <= 5 and criticality >= 8:
        return "critical"

    if distance <= 15 and criticality >= 8:
        return "high"

    if distance <= 30:
        return "medium"

    return "watch"


def calculate_score(distance, criticality):
    distance_factor = max(0, 1 - (distance / MAX_DISTANCE_KM))
    return round((criticality * 10) * distance_factor, 2)


def level_weight(level):
    return {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "watch": 1
    }.get(level, 0)


def is_better_match(candidate, current):
    if current is None:
        return True

    if level_weight(candidate["level"]) != level_weight(current["level"]):
        return level_weight(candidate["level"]) > level_weight(current["level"])

    if candidate["score"] != current["score"]:
        return candidate["score"] > current["score"]

    return candidate["distance_km"] < current["distance_km"]


def build_matches(infrastructure, events):
    all_matches = []

    best_by_infrastructure = {}
    best_by_event_infra = {}

    for event in events:
        for infra in infrastructure:
            distance = haversine_km(
                event["lat"],
                event["lon"],
                infra["lat"],
                infra["lon"]
            )

            if distance > MAX_DISTANCE_KM:
                continue

            level = calculate_level(distance, infra.get("criticality", 5))
            score = calculate_score(distance, infra.get("criticality", 5))

            match = {
                "id": stable_id(event["id"], infra.get("id"), distance),
                "level": level,
                "score": score,
                "distance_km": round(distance, 2),
                "event": {
                    "id": event["id"],
                    "title": event["title"],
                    "category": event["category"],
                    "source": event["source"],
                    "time": event["time"],
                    "url": event["url"],
                    "lat": event["lat"],
                    "lon": event["lon"],
                    "source_file": event["_source_file"]
                },
                "infrastructure": {
                    "id": infra.get("id"),
                    "name": infra.get("name"),
                    "country": infra.get("country"),
                    "city": infra.get("city"),
                    "category": infra.get("category"),
                    "subtype": infra.get("subtype"),
                    "criticality": infra.get("criticality"),
                    "operator": infra.get("operator"),
                    "lat": infra.get("lat"),
                    "lon": infra.get("lon"),
                    "source_file": infra.get("_source_file")
                }
            }

            all_matches.append(match)

            infra_id = infra.get("id") or infra.get("name")
            event_infra_key = stable_id(event["id"], infra_id)

            if event_infra_key not in best_by_event_infra:
                best_by_event_infra[event_infra_key] = match
            elif is_better_match(match, best_by_event_infra[event_infra_key]):
                best_by_event_infra[event_infra_key] = match

            if infra_id not in best_by_infrastructure:
                best_by_infrastructure[infra_id] = match
            elif is_better_match(match, best_by_infrastructure[infra_id]):
                best_by_infrastructure[infra_id] = match

    unique_event_infra_matches = list(best_by_event_infra.values())
    top_by_infrastructure = list(best_by_infrastructure.values())

    unique_event_infra_matches.sort(
        key=lambda x: (
            level_weight(x["level"]),
            x["score"],
            -x["distance_km"]
        ),
        reverse=True
    )

    top_by_infrastructure.sort(
        key=lambda x: (
            level_weight(x["level"]),
            x["score"],
            -x["distance_km"]
        ),
        reverse=True
    )

    return all_matches, unique_event_infra_matches, top_by_infrastructure


def build():
    infrastructure = load_infrastructure()
    events = load_events()

    all_matches, unique_matches, top_by_infra = build_matches(
        infrastructure,
        events
    )

    report = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "infrastructure_count": len(infrastructure),
            "event_count": len(events),
            "match_count": len(unique_matches),
            "raw_match_count": len(all_matches),
            "top_infrastructure_count": len(top_by_infra),
            "max_distance_km": MAX_DISTANCE_KM,
            "deduplication": {
                "events": "title + url + source + time",
                "matches": "best event-infrastructure pair",
                "top_matches": "one best match per infrastructure asset"
            }
        },
        "top_matches": top_by_infra[:50],
        "matches": unique_matches,
        "raw_matches": all_matches[:500]
    }

    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved: {OUTPUT}")
    print(f"Infrastructure: {len(infrastructure)}")
    print(f"Events: {len(events)}")
    print(f"Matches: {len(unique_matches)}")
    print(f"Top infrastructure matches: {len(top_by_infra)}")


if __name__ == "__main__":
    build()

