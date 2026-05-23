import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LOCAL_EVENTS = DATA / "local_events.geojson"
INFRA_PROX = DATA / "infra_proximity.json"

HISTORY_DIR = DATA / "history"

LOCAL_HISTORY = HISTORY_DIR / "local_events_history.geojson"
PROX_HISTORY = HISTORY_DIR / "infra_proximity_history.json"

KEEP_HOURS = 24


def ensure_history_dir():
    HISTORY_DIR.mkdir(exist_ok=True)


def load_json(path, default):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def parse_time(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except Exception:
        return None


def is_old(value):
    dt = parse_time(value)

    if not dt:
        return False

    now = datetime.now(timezone.utc)

    return (now - dt) > timedelta(hours=KEEP_HOURS)


def archive_local_events():

    payload = load_json(
        LOCAL_EVENTS,
        {
            "type": "FeatureCollection",
            "features": []
        }
    )

    history = load_json(
        LOCAL_HISTORY,
        {
            "type": "FeatureCollection",
            "features": []
        }
    )

    fresh = []
    archived = history.get("features", [])

    for feature in payload.get("features", []):

        props = feature.get("properties", {})

        t = (
            props.get("time")
            or props.get("published")
            or props.get("updated")
        )

        if is_old(t):
            archived.append(feature)
        else:
            fresh.append(feature)

    payload["features"] = fresh
    history["features"] = archived

    save_json(LOCAL_EVENTS, payload)
    save_json(LOCAL_HISTORY, history)

    print(f"Fresh local events: {len(fresh)}")
    print(f"Archived local events: {len(archived)}")


def archive_proximity():

    payload = load_json(
        INFRA_PROX,
        {
            "matches": [],
            "top_matches": []
        }
    )

    history = load_json(
        PROX_HISTORY,
        {
            "matches": [],
            "top_matches": []
        }
    )

    fresh_matches = []
    archived_matches = history.get("matches", [])

    for match in payload.get("matches", []):

        event = match.get("event", {})

        t = event.get("time")

        if is_old(t):
            archived_matches.append(match)
        else:
            fresh_matches.append(match)

    payload["matches"] = fresh_matches
    history["matches"] = archived_matches

    save_json(INFRA_PROX, payload)
    save_json(PROX_HISTORY, history)

    print(f"Fresh proximity matches: {len(fresh_matches)}")
    print(f"Archived proximity matches: {len(archived_matches)}")


def main():

    ensure_history_dir()

    archive_local_events()
    archive_proximity()

    print("Archive complete")


if __name__ == "__main__":
    main()
