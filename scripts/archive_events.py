import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY_DIR = DATA / "history"

LOCAL_EVENTS = DATA / "local_events.geojson"
INFRA_PROX = DATA / "infra_proximity.json"

LOCAL_HISTORY = HISTORY_DIR / "local_events_history.geojson"
PROX_HISTORY = HISTORY_DIR / "infra_proximity_history.json"

KEEP_CURRENT_HOURS = 24
KEEP_HISTORY_DAYS = 30

MAX_LOCAL_HISTORY = 3000
MAX_PROX_HISTORY = 3000


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def is_current(value):
    dt = parse_time(value)
    if not dt:
        return True
    return datetime.now(timezone.utc) - dt <= timedelta(hours=KEEP_CURRENT_HOURS)


def is_in_history_window(value):
    dt = parse_time(value)
    if not dt:
        return False
    return datetime.now(timezone.utc) - dt <= timedelta(days=KEEP_HISTORY_DAYS)


def event_key(feature):
    p = feature.get("properties", {})
    return (
        p.get("title"),
        p.get("url"),
        p.get("source"),
        p.get("time")
    )


def match_key(match):
    e = match.get("event", {})
    i = match.get("infrastructure", {})
    return (
        e.get("title"),
        e.get("url"),
        e.get("time"),
        i.get("name"),
        i.get("country")
    )


def compact_match(match):
    e = match.get("event", {})
    i = match.get("infrastructure", {})

    return {
        "level": match.get("level"),
        "score": match.get("score"),
        "distance_km": match.get("distance_km"),
        "event": {
            "title": e.get("title"),
            "source": e.get("source"),
            "time": e.get("time"),
            "url": e.get("url"),
            "category": e.get("category"),
            "lat": e.get("lat"),
            "lon": e.get("lon"),
            "source_file": e.get("source_file")
        },
        "infrastructure": {
            "name": i.get("name"),
            "country": i.get("country"),
            "city": i.get("city"),
            "category": i.get("category"),
            "subtype": i.get("subtype"),
            "criticality": i.get("criticality"),
            "lat": i.get("lat"),
            "lon": i.get("lon")
        }
    }


def dedup(items, key_func):
    seen = set()
    out = []

    for item in items:
        key = key_func(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)

    return out


def sort_features_by_time(features):
    return sorted(
        features,
        key=lambda f: parse_time((f.get("properties", {}) or {}).get("time")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True
    )


def sort_matches_by_time(matches):
    return sorted(
        matches,
        key=lambda m: parse_time((m.get("event", {}) or {}).get("time")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True
    )


def archive_local_events():
    payload = load_json(LOCAL_EVENTS, {"type": "FeatureCollection", "features": []})
    history = load_json(LOCAL_HISTORY, {"type": "FeatureCollection", "features": []})

    fresh = []
    archive_add = []

    for f in payload.get("features", []):
        p = f.get("properties", {})
        t = p.get("time") or p.get("published") or p.get("updated")

        if is_current(t):
            fresh.append(f)
        elif is_in_history_window(t):
            archive_add.append(f)

    history_features = history.get("features", []) + archive_add
    history_features = [
        f for f in history_features
        if is_in_history_window((f.get("properties", {}) or {}).get("time"))
    ]

    history_features = dedup(history_features, event_key)
    history_features = sort_features_by_time(history_features)[:MAX_LOCAL_HISTORY]

    payload["features"] = dedup(fresh, event_key)
    history["features"] = history_features

    save_json(LOCAL_EVENTS, payload)
    save_json(LOCAL_HISTORY, history)

    print(f"Fresh local events: {len(payload['features'])}")
    print(f"Local history events: {len(history['features'])}")


def archive_proximity():
    payload = load_json(INFRA_PROX, {"matches": [], "top_matches": []})
    history = load_json(PROX_HISTORY, {"matches": []})

    fresh = []
    archive_add = []

    for m in payload.get("matches", []):
        e = m.get("event", {})
        t = e.get("time")

        if is_current(t):
            fresh.append(m)
        elif is_in_history_window(t):
            archive_add.append(compact_match(m))

    old_history = [
        compact_match(m) for m in history.get("matches", [])
        if is_in_history_window((m.get("event", {}) or {}).get("time"))
    ]

    history_matches = old_history + archive_add
    history_matches = dedup(history_matches, match_key)
    history_matches = sort_matches_by_time(history_matches)[:MAX_PROX_HISTORY]

    payload["matches"] = dedup(fresh, match_key)
    payload["top_matches"] = payload.get("top_matches", [])[:100]

    history = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "retention_days": KEEP_HISTORY_DAYS,
        "max_records": MAX_PROX_HISTORY,
        "matches": history_matches
    }

    save_json(INFRA_PROX, payload)
    save_json(PROX_HISTORY, history)

    print(f"Fresh proximity matches: {len(payload['matches'])}")
    print(f"Proximity history matches: {len(history_matches)}")


def main():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    archive_local_events()
    archive_proximity()
    print("Archive complete with size limits.")


if __name__ == "__main__":
    main()
