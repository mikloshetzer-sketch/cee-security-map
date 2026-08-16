import json
import hashlib
import re
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


# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(path, default):
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[archive] Could not read {path}: {exc}")
        return default


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def now_utc():
    return datetime.now(timezone.utc)


def parse_time(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except Exception:
        return None


def feature_time(feature):
    p = feature.get("properties", {}) or {}

    return (
        p.get("time")
        or p.get("published")
        or p.get("published_utc")
        or p.get("updated")
    )


def is_current(value):
    dt = parse_time(value)

    if not dt:
        return True

    return now_utc() - dt <= timedelta(hours=KEEP_CURRENT_HOURS)


def is_in_history_window(value):
    dt = parse_time(value)

    if not dt:
        return False

    age = now_utc() - dt

    return (
        age >= timedelta(0)
        and age <= timedelta(days=KEEP_HISTORY_DAYS)
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    value = str(value or "").lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def round_coord(value, digits=2):
    try:
        return round(float(value), digits)
    except Exception:
        return None


def feature_coordinates(feature):
    geometry = feature.get("geometry") or {}

    if geometry.get("type") == "Point":
        coords = geometry.get("coordinates") or []

        if len(coords) >= 2:
            try:
                return float(coords[1]), float(coords[0])
            except Exception:
                pass

    p = feature.get("properties", {}) or {}

    lat = p.get("lat") or p.get("latitude")
    lon = p.get("lon") or p.get("longitude")

    try:
        return float(lat), float(lon)
    except Exception:
        return None, None


# ============================================================
# LOCAL EVENT IDENTITY
# ============================================================

def classifier_fingerprint(feature):
    """
    Prefer the classifier's own event fingerprint.

    Supported forms:
      fingerprint: "abc123"
      fingerprint: {"hash": "abc123", ...}
      fingerprint_hash: "abc123"
    """

    p = feature.get("properties", {}) or {}

    fp = p.get("fingerprint")

    if isinstance(fp, dict):
        value = fp.get("hash")

        if value:
            return str(value)

    elif fp:
        return str(fp)

    value = p.get("fingerprint_hash")

    if value:
        return str(value)

    return None


def fallback_event_signature(feature):
    """
    Fallback incident identity when no classifier fingerprint exists.

    This deliberately excludes:
      source
      URL
      article title

    because those identify an article, not the real-world incident.
    """

    p = feature.get("properties", {}) or {}

    dt = parse_time(feature_time(feature))

    if dt:
        # Day-level bucket is more suitable for multi-source incidents.
        day = dt.date().isoformat()
    else:
        day = ""

    lat, lon = feature_coordinates(feature)

    country = normalize_text(p.get("country"))

    family = normalize_text(
        p.get("incident_family")
        or p.get("category")
    )

    subcategory = normalize_text(
        p.get("incident_subcategory")
    )

    subtype = normalize_text(
        p.get("incident_subtype")
    )

    action = normalize_text(
        p.get("incident_action")
    )

    place = normalize_text(
        p.get("place")
        or p.get("city")
    )

    raw = "|".join([
        country,
        family,
        subcategory,
        subtype,
        action,
        day,
        place,
        str(round_coord(lat)),
        str(round_coord(lon)),
    ])

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:24]


def event_key(feature):
    """
    Event-level identity.

    First choice:
      classifier fingerprint

    Fallback:
      normalized incident signature
    """

    return (
        classifier_fingerprint(feature)
        or fallback_event_signature(feature)
    )


# ============================================================
# PROXIMITY IDENTITY
# ============================================================

def match_key(match):
    e = match.get("event", {}) or {}
    i = match.get("infrastructure", {}) or {}

    event_fp = e.get("fingerprint")

    if isinstance(event_fp, dict):
        event_fp = event_fp.get("hash")

    event_fp = (
        event_fp
        or e.get("fingerprint_hash")
        or e.get("url")
        or e.get("title")
    )

    return (
        event_fp,
        e.get("time"),
        i.get("name"),
        i.get("country")
    )


# ============================================================
# COMPACT PROXIMITY HISTORY
# ============================================================

def compact_match(match):
    e = match.get("event", {}) or {}
    i = match.get("infrastructure", {}) or {}

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

            "incident_family": e.get("incident_family"),
            "incident_subcategory": e.get("incident_subcategory"),
            "incident_subtype": e.get("incident_subtype"),
            "incident_action": e.get("incident_action"),

            "fingerprint": e.get("fingerprint"),
            "fingerprint_hash": e.get("fingerprint_hash"),

            "lat": e.get("lat"),
            "lon": e.get("lon"),
            "source_file": e.get("source_file"),
        },

        "infrastructure": {
            "name": i.get("name"),
            "country": i.get("country"),
            "city": i.get("city"),
            "category": i.get("category"),
            "subtype": i.get("subtype"),
            "criticality": i.get("criticality"),
            "lat": i.get("lat"),
            "lon": i.get("lon"),
        }
    }


# ============================================================
# DEDUP
# ============================================================

def dedup(items, key_func):
    """
    Preserve the first occurrence of each identity.
    """

    seen = set()
    out = []

    for item in items:
        key = key_func(item)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


# ============================================================
# SORTING
# ============================================================

def sort_features_by_time(features):
    minimum = datetime.min.replace(tzinfo=timezone.utc)

    return sorted(
        features,
        key=lambda f: parse_time(feature_time(f)) or minimum,
        reverse=True
    )


def sort_matches_by_time(matches):
    minimum = datetime.min.replace(tzinfo=timezone.utc)

    return sorted(
        matches,
        key=lambda m: (
            parse_time(
                (m.get("event", {}) or {}).get("time")
            )
            or minimum
        ),
        reverse=True
    )


# ============================================================
# LOCAL EVENTS ARCHIVE
# ============================================================

def archive_local_events():
    current_payload = load_json(
        LOCAL_EVENTS,
        {
            "type": "FeatureCollection",
            "features": []
        }
    )

    history_payload = load_json(
        LOCAL_HISTORY,
        {
            "type": "FeatureCollection",
            "features": []
        }
    )

    current_input = current_payload.get("features", []) or []
    old_history = history_payload.get("features", []) or []

    # --------------------------------------------------------
    # 1. Current operational layer
    #
    # Only <=24h remains in local_events.geojson.
    # --------------------------------------------------------

    fresh = []

    for feature in current_input:
        t = feature_time(feature)

        if is_current(t):
            fresh.append(feature)

    fresh = sort_features_by_time(
        dedup(fresh, event_key)
    )

    # --------------------------------------------------------
    # 2. History
    #
    # IMPORTANT CHANGE:
    #
    # Every observed event is immediately copied to history,
    # including today's events.
    #
    # This prevents loss when fetch_local_sources.py later
    # replaces local_events.geojson.
    # --------------------------------------------------------

    history_candidates = []

    # Previously saved history
    for feature in old_history:
        if is_in_history_window(feature_time(feature)):
            history_candidates.append(feature)

    # Every current input record is also persisted immediately
    for feature in current_input:
        if is_in_history_window(feature_time(feature)):
            history_candidates.append(feature)

    history_features = dedup(
        history_candidates,
        event_key
    )

    history_features = sort_features_by_time(
        history_features
    )[:MAX_LOCAL_HISTORY]

    # --------------------------------------------------------
    # Save current operational layer
    # --------------------------------------------------------

    current_payload["type"] = "FeatureCollection"
    current_payload["features"] = fresh

    # --------------------------------------------------------
    # Save rolling history
    # --------------------------------------------------------

    history_payload = {
        "type": "FeatureCollection",

        "meta": {
            "generated_utc": now_utc().isoformat(),
            "retention_days": KEEP_HISTORY_DAYS,
            "max_records": MAX_LOCAL_HISTORY,
            "event_identity": "classifier_fingerprint_then_fallback",
            "current_hours": KEEP_CURRENT_HOURS,
            "current_count": len(fresh),
            "history_count": len(history_features),
        },

        "features": history_features,
    }

    save_json(
        LOCAL_EVENTS,
        current_payload
    )

    save_json(
        LOCAL_HISTORY,
        history_payload
    )

    print(
        f"Fresh local events: {len(fresh)}"
    )

    print(
        f"Local history events: {len(history_features)}"
    )


# ============================================================
# INFRASTRUCTURE PROXIMITY ARCHIVE
# ============================================================

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
            "matches": []
        }
    )

    current_matches = payload.get("matches", []) or []

    fresh = []

    for match in current_matches:
        event = match.get("event", {}) or {}
        t = event.get("time")

        if is_current(t):
            fresh.append(match)

    # --------------------------------------------------------
    # Keep existing history
    # --------------------------------------------------------

    history_candidates = []

    for match in history.get("matches", []) or []:
        event = match.get("event", {}) or {}

        if is_in_history_window(
            event.get("time")
        ):
            history_candidates.append(
                compact_match(match)
            )

    # --------------------------------------------------------
    # IMPORTANT:
    # Current proximity observations are immediately archived.
    # --------------------------------------------------------

    for match in current_matches:
        event = match.get("event", {}) or {}

        if is_in_history_window(
            event.get("time")
        ):
            history_candidates.append(
                compact_match(match)
            )

    history_matches = dedup(
        history_candidates,
        match_key
    )

    history_matches = sort_matches_by_time(
        history_matches
    )[:MAX_PROX_HISTORY]

    payload["matches"] = dedup(
        fresh,
        match_key
    )

    # Rebuild top_matches from current records only
    payload["top_matches"] = (
        payload["matches"][:100]
    )

    history_payload = {
        "generated_utc": now_utc().isoformat(),
        "retention_days": KEEP_HISTORY_DAYS,
        "max_records": MAX_PROX_HISTORY,
        "matches": history_matches,
    }

    save_json(
        INFRA_PROX,
        payload
    )

    save_json(
        PROX_HISTORY,
        history_payload
    )

    print(
        f"Fresh proximity matches: "
        f"{len(payload['matches'])}"
    )

    print(
        f"Proximity history matches: "
        f"{len(history_matches)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "=== CEE event archive ==="
    )

    print(
        f"Current window: {KEEP_CURRENT_HOURS}h"
    )

    print(
        f"History retention: {KEEP_HISTORY_DAYS} days"
    )

    archive_local_events()
    archive_proximity()

    print(
        "Archive complete with rolling history."
    )


if __name__ == "__main__":
    main()
