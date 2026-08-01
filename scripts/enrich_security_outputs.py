import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from email.utils import parsedate_to_datetime
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "history"

SUMMARY = DATA / "summary.json"
WEEKLY = DATA / "weekly.json"
META = DATA / "meta.json"
RISK_DAILY = DATA / "risk_daily.json"
RISK_DEBUG = DATA / "risk_incidents_debug.json"

LOCAL_EVENTS = DATA / "local_events.geojson"
LOCAL_HISTORY = HISTORY / "local_events_history.geojson"

INFRA_PROX = DATA / "infra_proximity.json"
INFRA_PROX_HISTORY = HISTORY / "infra_proximity_history.json"

GDELT_CROSSBORDER = DATA / "gdelt_crossborder.geojson"

# Risk input is intentionally narrow. GDELT GEO, GDELT Linked and Trusted RSS
# remain dashboard/context layers and must never directly affect country risk.
SECURITY_EVENT_FILES = [
    LOCAL_EVENTS,
    LOCAL_HISTORY,
    GDELT_CROSSBORDER,
]

LOCAL_RISK_FILES = {LOCAL_EVENTS.name, LOCAL_HISTORY.name}
CROSSBORDER_RISK_FILE = GDELT_CROSSBORDER.name

COUNTRY_NAME_MAP = {
    "Czechia": "Czech Republic",
    "Czech Republic": "Czech Republic",
    "Hungary": "Hungary",
    "Romania": "Romania",
    "Slovakia": "Slovakia",
    "Poland": "Poland",
    "Lithuania": "Lithuania",
    "Latvia": "Latvia",
    "Estonia": "Estonia",
}

MONITORED_COUNTRIES = [
    "Czech Republic",
    "Hungary",
    "Romania",
    "Slovakia",
    "Poland",
    "Lithuania",
    "Latvia",
    "Estonia",
]

RISK_CATEGORY_GROUPS = {
    "military_drone": {"military", "drone", "drone_airspace", "kinetic_attack", "military_accident"},
    "cyber": {"cyber", "cyber_incident"},
    "infrastructure": {
        "explosion", "fire", "hazardous", "energy", "transport",
        "infrastructure_disruption", "sabotage", "major_fire", "hazardous_incident"
    },
}

RISK_WEIGHTS = {
    "incident_pressure": 0.30,
    "military_drone": 0.20,
    "cyber": 0.15,
    "infrastructure": 0.15,
    "crossborder": 0.10,
    "trend": 0.10,
}



def now_utc():
    return datetime.now(timezone.utc)


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

    raw = str(value).strip()

    try:
        return datetime.fromisoformat(
            raw.replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except Exception:
        pass

    try:
        dt = parsedate_to_datetime(raw)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def age_hours(value):
    dt = parse_time(value)
    if not dt:
        return None
    return (now_utc() - dt).total_seconds() / 3600


def norm_country(country):
    return COUNTRY_NAME_MAP.get(country, country or "Unknown")



TARGET_COUNTRY_TERMS = {
    "Czech Republic": ["czech republic", "czechia", "czech", "česko", "praha", "prague"],
    "Hungary": ["hungary", "hungarian", "magyarország", "magyar", "budapest", "paks", "kecskemét"],
    "Romania": [
        "romania", "românia", "romanian", "român", "románia",
        "bucharest", "bucurești", "bucuresti", "fetești", "fetesti",
        "buzău", "buzau", "brăila", "braila", "constanța", "constanta",
        "cernavodă", "cernavoda", "sfântu gheorghe", "sfantu gheorghe"
    ],
    "Slovakia": ["slovakia", "slovak", "slovensko", "bratislava", "košice", "kosice"],
    "Poland": ["poland", "polish", "polska", "warsaw", "warszawa", "rzeszów", "rzeszow"],
    "Lithuania": ["lithuania", "lithuanian", "lietuva", "vilnius", "kaunas", "klaipėda", "klaipeda"],
    "Latvia": ["latvia", "latvian", "latvija", "riga", "ventspils"],
    "Estonia": ["estonia", "estonian", "eesti", "tallinn", "tartu", "narva"],
}

SECURITY_INCIDENT_PATTERNS = {
    "drone": [
        "drone", "drón", "uav", "shahed", "dronă", "drona",
        "dronei", "dronele", "dronelor"
    ],
    "cyber": [
        "cyberattack", "cyber attack", "ransomware attack",
        "ddos attack", "data breach", "kibertámadás"
    ],
    "military": [
        "military helicopter crash", "military aircraft crash",
        "fighter jet crash", "airspace violation", "airspace breach",
        "shot down", "intercepted", "missile attack", "airstrike",
        "rakétatámadás", "légtérsértés", "lelőtt", "doborât"
    ],
    "explosion": ["explosion", "blast", "robbanás", "explozie"],
    "hazardous": [
        "chemical leak", "gas leak", "industrial accident",
        "vegyi szivárgás", "gázszivárgás"
    ],
    "infrastructure": [
        "blackout", "power outage", "grid failure",
        "airport closure", "port closure", "rail disruption",
        "áramszünet"
    ],
}

NOISE_TERMS = [
    "formula 1", "formula one", "f1", "grand prix", "motorsport",
    "archaeology", "archaeological", "roman military camp",
    "street racing", "road safety", "traffic accident",
    "trade talks", "investment", "procurement", "construction",
    "appointment", "ceo", "director appointed",
]


def normalize_text(value):
    return re.sub(r"\\s+", " ", str(value or "").lower()).strip()


def contains_term(text, term):
    t = normalize_text(text)
    k = normalize_text(term)
    if not k:
        return False

    if re.fullmatch(r"[\\wÀ-ž-]+", k, flags=re.UNICODE):
        return re.search(
            rf"(?<!\\w){re.escape(k)}(?!\\w)",
            t,
            flags=re.UNICODE,
        ) is not None

    return k in t


def contains_any(text, terms):
    return any(contains_term(text, term) for term in terms)


def stable_event_key(*parts):
    raw = "|".join(str(p or "") for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def detect_security_country(props, fallback_path=None):
    raw_country = norm_country(
        props.get("country")
        or props.get("event_country")
        or props.get("location_country")
    )

    if raw_country in MONITORED_COUNTRIES:
        return raw_country

    text = " ".join([
        str(props.get("title") or props.get("name") or ""),
        str(props.get("summary") or props.get("description") or ""),
        str(props.get("url") or props.get("search_url") or ""),
    ])

    hits = []
    for country, terms in TARGET_COUNTRY_TERMS.items():
        if any(contains_term(text, term) for term in terms):
            hits.append(country)

    hits = list(dict.fromkeys(hits))
    return hits[0] if len(hits) == 1 else None


def canonical_risk_category(props):
    """Return the classifier-backed category used by the risk model.

    This function never promotes a background or legacy record using free-text
    keyword guessing. It only translates already validated classifier fields.
    """
    family = str(props.get("incident_family") or props.get("family") or "").lower()
    subcategory = str(
        props.get("incident_subcategory")
        or props.get("subcategory")
        or ""
    ).lower()
    subtype = str(props.get("incident_subtype") or props.get("subtype") or "").lower()
    existing = str(props.get("category") or "").lower()

    if subcategory == "drone" or subtype.startswith("drone_"):
        return "drone"
    if subcategory == "missile" or subtype.startswith("missile_"):
        return "kinetic_attack"
    if family == "cyber" or subcategory == "cyber":
        return "cyber"
    if family in {"air", "military"}:
        return "military"
    if family in {"infrastructure", "energy"}:
        if subcategory in {"fire", "explosion", "hazardous", "sabotage"}:
            return subcategory
        return "infrastructure_disruption"
    if family in {"public_safety", "hazard"}:
        if subcategory in {"fire", "explosion", "hazardous"}:
            return subcategory

    allowed_existing = {
        "drone", "drone_airspace", "cyber", "cyber_incident",
        "military", "military_accident", "kinetic_attack",
        "explosion", "fire", "hazardous", "hazardous_incident",
        "sabotage", "infrastructure_disruption", "major_fire",
        "energy", "transport",
    }
    return existing if existing in allowed_existing else None


def feature_passes_risk_gate(props, source_file):
    """Apply source-specific admission rules for risk evidence.

    Local-event files are generated directly by ``SecurityClassifier`` but older
    compatible records do not necessarily contain ``classification_source`` or
    ``risk_eligible``. Their classifier provenance is proven by the combination
    of ``actual_incident``, ``matched_rule_id``, ``article_role`` and classifier
    version/taxonomy fields.

    GDELT cross-border records use the strict explicit gate. Legacy/fallback
    records are never admitted.
    """
    if props.get("actual_incident") is not True:
        return False
    if not props.get("matched_rule_id"):
        return False

    article_role = str(props.get("article_role") or "incident").lower()
    if article_role != "incident":
        return False

    if source_file in LOCAL_RISK_FILES:
        # Local records may legitimately predate these two explicit fields.
        # Reject an explicit false flag, but allow a missing one.
        if props.get("risk_eligible") is False:
            return False

        classification_source = props.get("classification_source")
        if classification_source not in {None, "", "security_classifier"}:
            return False

        # Require additional classifier evidence when classification_source is
        # absent, preventing old keyword-only local records from entering risk.
        if not classification_source and not (
            props.get("classifier_version")
            and props.get("taxonomy_version")
            and (
                props.get("incident_family")
                or props.get("family")
            )
        ):
            return False

        return True

    if source_file == CROSSBORDER_RISK_FILE:
        return (
            props.get("classification_source") == "security_classifier"
            and props.get("risk_eligible") is True
        )

    return False


def normalize_security_feature(feature, source_file):
    props = dict(feature.get("properties") or {})

    if not feature_passes_risk_gate(props, source_file):
        return None

    country = detect_security_country(props, source_file)
    if country not in MONITORED_COUNTRIES:
        return None

    category = canonical_risk_category(props)
    if not category:
        return None

    dt = parse_time(
        props.get("time")
        or props.get("datetime")
        or props.get("date")
    )
    if not dt:
        return None

    props["country"] = country
    props["category"] = category
    props["incident_type"] = category
    props["time"] = dt.isoformat()
    props["_risk_source_file"] = source_file

    if source_file == CROSSBORDER_RISK_FILE:
        props["crossborder"] = True

    if not props.get("severity"):
        if category in {"drone", "military", "kinetic_attack", "explosion"}:
            props["severity"] = "high"
        elif category in {"cyber", "hazardous", "sabotage"}:
            props["severity"] = "medium"
        else:
            props["severity"] = "info"

    return {
        "type": "Feature",
        "geometry": feature.get("geometry"),
        "properties": props,
    }


def load_validated_security_events(days=7):
    """Load only validated incident evidence used by the risk model.

    Included:
      * local_events.geojson and its history, classifier-validated;
      * gdelt_crossborder.geojson, under the stricter risk gate.

    Excluded by design:
      * gdelt.geojson;
      * gdelt_linked.geojson;
      * trusted_rss.json;
      * every legacy_fallback/background/reaction record.
    """
    cutoff = now_utc() - timedelta(days=days)
    candidates = []

    for path in SECURITY_EVENT_FILES:
        payload = load_json(path, {"features": []})

        for feature in payload.get("features", []):
            normalized = normalize_security_feature(feature, path.name)
            if not normalized:
                continue

            props = normalized["properties"]
            dt = parse_time(props.get("time"))
            if not dt or dt < cutoff:
                continue

            candidates.append(normalized)

    # Prefer classifier fingerprint. Fall back conservatively to source URL/title.
    seen = set()
    result = []

    for feature in candidates:
        props = feature.get("properties") or {}
        fingerprint = props.get("fingerprint") or {}
        if isinstance(fingerprint, dict):
            fingerprint_hash = fingerprint.get("hash")
        else:
            fingerprint_hash = str(fingerprint or "")

        dt = parse_time(props.get("time"))
        day = dt.date().isoformat() if dt else ""
        title = normalize_text(props.get("title") or props.get("name") or "")
        title_tokens = [
            token for token in re.findall(r"[\wÀ-ž-]{4,}", title, flags=re.UNICODE)
            if token not in {
                "romania", "hungary", "poland", "latvia", "estonia",
                "lithuania", "slovakia", "czechia",
            }
        ]
        title_sig = " ".join(title_tokens[:10])

        key = fingerprint_hash or stable_event_key(
            props.get("country"),
            props.get("incident_subtype") or props.get("category"),
            day,
            props.get("url") or title_sig,
        )

        if key in seen:
            continue
        seen.add(key)
        result.append(feature)

    return result


def local_event_score(props):
    severity = str(props.get("severity") or "info").lower()
    category = str(props.get("category") or "local_media").lower()
    geocode_quality = str(props.get("geocode_quality") or "").lower()

    score = 0.4

    if severity == "high":
        score += 1.2
    elif severity == "medium":
        score += 0.8
    else:
        score += 0.3

    if category in {"explosion", "fire", "hazardous", "energy"}:
        score += 1.0
    elif category in {"cyber", "military", "drone"}:
        score += 0.9
    elif category in {"transport"}:
        score += 0.6

    if geocode_quality == "city":
        score += 0.5

    h = age_hours(props.get("time"))
    if h is not None:
        if h <= 6:
            score *= 1.25
        elif h <= 24:
            score *= 1.10
        elif h > 72:
            score *= 0.55

    return round(score, 3)


def proximity_score(match):
    level = str(match.get("level") or "watch").lower()
    infra = match.get("infrastructure") or {}
    event = match.get("event") or {}

    score = {
        "critical": 4.0,
        "high": 2.8,
        "medium": 1.6,
        "watch": 0.8
    }.get(level, 0.6)

    try:
        score += float(infra.get("criticality") or 5) / 10
    except Exception:
        pass

    source_file = event.get("source_file") or ""
    if source_file == "local_events.geojson":
        score *= 1.25

    h = age_hours(event.get("time"))
    if h is not None:
        if h <= 6:
            score *= 1.25
        elif h <= 24:
            score *= 1.10
        elif h > 72:
            score *= 0.6

    return round(score, 3)


def load_local_events(days=7):
    features = []

    for path in [LOCAL_EVENTS, LOCAL_HISTORY]:
        payload = load_json(path, {"features": []})
        for f in payload.get("features", []):
            props = f.get("properties") or {}
            dt = parse_time(props.get("time"))
            if not dt:
                continue
            if dt >= now_utc() - timedelta(days=days):
                features.append(f)

    return features


def proximity_event_passes_risk_gate(event):
    source_file = str(event.get("source_file") or event.get("_risk_source_file") or "")

    # Context-only sources can never contribute through infrastructure proximity.
    if source_file in {"gdelt.geojson", "gdelt_linked.geojson", "trusted_rss.json"}:
        return False

    if source_file == CROSSBORDER_RISK_FILE:
        return bool(
            event.get("classification_source") == "security_classifier"
            and event.get("actual_incident") is True
            and event.get("risk_eligible") is True
            and str(event.get("article_role") or "").lower() == "incident"
            and event.get("matched_rule_id")
        )

    if source_file in LOCAL_RISK_FILES or not source_file:
        return bool(
            event.get("classification_source") == "security_classifier"
            and event.get("actual_incident") is True
            and event.get("risk_eligible") is not False
            and str(event.get("article_role") or "incident").lower() == "incident"
            and event.get("matched_rule_id")
        )

    return False


def load_proximity(days=7):
    matches = []
    cutoff = now_utc() - timedelta(days=days)
    seen = set()

    for path in [INFRA_PROX, INFRA_PROX_HISTORY]:
        payload = load_json(path, {"matches": []})

        for m in payload.get("matches", []):
            event = m.get("event") or {}
            infra = m.get("infrastructure") or {}

            dt = parse_time(event.get("time"))
            if not dt or dt < cutoff:
                continue

            if not proximity_event_passes_risk_gate(event):
                continue

            location_quality = str(
                event.get("location_quality")
                or event.get("geocode_quality")
                or ""
            ).lower()
            if location_quality not in {
                "city", "specific_place", "precise"
            }:
                continue

            if not (event.get("geolocation_method") or event.get("geocode_quality")):
                continue

            key = (
                event.get("id")
                or event.get("fingerprint_hash")
                or event.get("url")
                or stable_event_key(event.get("title"), event.get("time")),
                infra.get("id") or infra.get("name"),
            )
            if key in seen:
                continue

            seen.add(key)
            matches.append(m)

    return matches


def build_local_stats(local_events):
    by_country = defaultdict(lambda: {
        "count": 0,
        "score": 0.0,
        "categories": Counter(),
        "high_events": []
    })

    for f in local_events:
        props = f.get("properties") or {}
        country = norm_country(props.get("country"))
        score = local_event_score(props)

        by_country[country]["count"] += 1
        by_country[country]["score"] += score
        by_country[country]["categories"][props.get("category") or "local_media"] += 1

        if props.get("severity") in {"high", "medium"}:
            by_country[country]["high_events"].append({
                "title": props.get("title"),
                "source": props.get("source"),
                "url": props.get("url"),
                "time": props.get("time"),
                "category": props.get("category"),
                "severity": props.get("severity"),
                "place": props.get("place")
            })

    return by_country


def build_proximity_stats(matches):
    by_country = defaultdict(lambda: {
        "count": 0,
        "score": 0.0,
        "levels": Counter(),
        "top": []
    })

    for m in matches:
        infra = m.get("infrastructure") or {}
        country = norm_country(infra.get("country"))
        score = proximity_score(m)
        level = m.get("level") or "watch"

        by_country[country]["count"] += 1
        by_country[country]["score"] += score
        by_country[country]["levels"][level] += 1
        by_country[country]["top"].append(m)

    for country in by_country:
        by_country[country]["top"].sort(
            key=lambda x: proximity_score(x),
            reverse=True
        )
        by_country[country]["top"] = by_country[country]["top"][:5]

    return by_country


def enrich_summary(local_events, prox_matches):
    summary = load_json(SUMMARY, {
        "generated_utc": now_utc().isoformat(),
        "headline": "Napi kivonat",
        "bullets": []
    })

    local_24h = [
        f for f in local_events
        if (age_hours((f.get("properties") or {}).get("time")) or 999) <= 24
    ]

    prox_24h = [
        m for m in prox_matches
        if (age_hours((m.get("event") or {}).get("time")) or 999) <= 24
    ]

    bullets = summary.get("bullets") or []

    added = []

    if local_24h:
        countries = Counter(
            norm_country((f.get("properties") or {}).get("country"))
            for f in local_24h
        )
        top = ", ".join([f"{c}: {n}" for c, n in countries.most_common(4)])
        added.append(
            f"Helyi források alapján az elmúlt 24 órában {len(local_24h)} infrastruktúra- vagy biztonsági relevanciájú lokális esemény jelent meg. Fő érintett országok: {top}."
        )

    if prox_24h:
        critical = [m for m in prox_24h if m.get("level") in {"critical", "high"}]
        added.append(
            f"Az infrastruktúra-közelségi modul {len(prox_24h)} friss kapcsolatot azonosított események és kritikus objektumok között; ebből {len(critical)} magas vagy kritikus szintű."
        )

    if prox_24h:
        top = sorted(prox_24h, key=lambda x: proximity_score(x), reverse=True)[:3]
        for m in top:
            ev = m.get("event") or {}
            infra = m.get("infrastructure") or {}
            added.append(
                f"{m.get('level', 'watch').upper()} infrastruktúra-jelzés: {infra.get('name')} ({infra.get('country')}) – {round(m.get('distance_km', 0), 1)} km-re ettől: {ev.get('title')}."
            )

    summary["bullets"] = added + bullets
    summary["local_infrastructure"] = {
        "last_24h_local_events": len(local_24h),
        "last_24h_proximity_matches": len(prox_24h),
        "top_proximity": prox_24h[:10]
    }

    save_json(SUMMARY, summary)


def enrich_weekly(local_events, prox_matches):
    weekly = load_json(WEEKLY, {
        "generated_utc": now_utc().isoformat(),
        "headline": "Közép–Kelet-Európa heti biztonsági brief",
        "bullets": [],
        "examples": []
    })

    local_stats = build_local_stats(local_events)
    prox_stats = build_proximity_stats(prox_matches)

    bullets = weekly.get("bullets") or []
    added = []

    if local_events:
        top_local = sorted(
            local_stats.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:4]
        txt = ", ".join([f"{c}: {round(v['score'], 1)}" for c, v in top_local])
        added.append(
            f"A helyi forrásokból érkező infrastruktúra- és biztonsági jelzések alapján a heti lokális nyomás legerősebben itt jelent meg: {txt}."
        )

    if prox_matches:
        top_prox = sorted(
            prox_stats.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:4]
        txt = ", ".join([f"{c}: {round(v['score'], 1)}" for c, v in top_prox])
        added.append(
            f"A kritikus infrastruktúra-közelségi mutató alapján a heti kitettség fő országai: {txt}."
        )

    weekly["bullets"] = added + bullets
    weekly["local_infrastructure_weekly"] = {
        "local_event_count_7d": len(local_events),
        "proximity_match_count_7d": len(prox_matches),
        "local_by_country": {
            c: {
                "count": v["count"],
                "score": round(v["score"], 3),
                "categories": dict(v["categories"]),
                "high_events": v["high_events"][:5]
            }
            for c, v in local_stats.items()
        },
        "proximity_by_country": {
            c: {
                "count": v["count"],
                "score": round(v["score"], 3),
                "levels": dict(v["levels"]),
                "top": v["top"][:3]
            }
            for c, v in prox_stats.items()
        }
    }

    examples = weekly.get("examples") or []
    for f in local_events[:10]:
        p = f.get("properties") or {}
        examples.append({
            "title": p.get("title"),
            "url": p.get("url"),
            "domain": p.get("source"),
            "time_utc": p.get("time"),
            "type": "local_event"
        })

    weekly["examples"] = examples[:40]

    if "weekly_assessment_plain" in weekly:
        weekly["weekly_assessment_plain"] = added + weekly.get("weekly_assessment_plain", [])

    save_json(WEEKLY, weekly)



def clamp(value, low=0.0, high=10.0):
    return max(low, min(high, float(value)))


def saturating_score(value, scale):
    """
    Convert an unbounded activity value to 0..10 without allowing one large
    raw count to force every country to the ceiling.
    """
    value = max(0.0, float(value or 0.0))
    scale = max(0.001, float(scale))
    return round(10.0 * value / (value + scale), 3)


def event_category(props):
    return str(
        props.get("incident_type")
        or props.get("category")
        or "unknown"
    ).lower()


def country_event_components(local_events):
    result = defaultdict(lambda: {
        "all_score": 0.0,
        "all_count": 0,
        "military_drone_score": 0.0,
        "military_drone_count": 0,
        "cyber_score": 0.0,
        "cyber_count": 0,
        "infrastructure_score": 0.0,
        "infrastructure_count": 0,
        "crossborder_score": 0.0,
        "crossborder_count": 0,
        "recent_24h_score": 0.0,
        "previous_24_72h_score": 0.0,
    })

    seen = set()

    for feature in local_events:
        props = feature.get("properties") or {}
        country = norm_country(props.get("country"))
        if country not in MONITORED_COUNTRIES:
            continue

        event_key = (
            props.get("id")
            or props.get("url")
            or (
                props.get("title"),
                props.get("time"),
                props.get("source"),
            )
        )
        dedup_key = (country, str(event_key))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        score = local_event_score(props)
        category = event_category(props)
        age = age_hours(props.get("time"))

        row = result[country]
        row["all_score"] += score
        row["all_count"] += 1

        if category in RISK_CATEGORY_GROUPS["military_drone"]:
            row["military_drone_score"] += score
            row["military_drone_count"] += 1

        if category in RISK_CATEGORY_GROUPS["cyber"]:
            row["cyber_score"] += score
            row["cyber_count"] += 1

        if category in RISK_CATEGORY_GROUPS["infrastructure"]:
            row["infrastructure_score"] += score
            row["infrastructure_count"] += 1

        crossborder_flag = any([
            props.get("crossborder") is True,
            props.get("cross_border") is True,
            str(props.get("scope") or "").lower() == "crossborder",
            str(props.get("category") or "").lower() == "crossborder",
        ])
        if crossborder_flag:
            row["crossborder_score"] += score
            row["crossborder_count"] += 1

        if age is not None:
            if age <= 24:
                row["recent_24h_score"] += score
            elif age <= 72:
                row["previous_24_72h_score"] += score

    return result


def country_proximity_components(prox_matches):
    result = defaultdict(lambda: {
        "score": 0.0,
        "count": 0,
        "high_critical": 0,
    })

    seen = set()

    for match in prox_matches:
        infra = match.get("infrastructure") or {}
        event = match.get("event") or {}
        country = norm_country(infra.get("country"))

        if country not in MONITORED_COUNTRIES:
            continue

        dedup_key = (
            country,
            event.get("id"),
            infra.get("id") or infra.get("name"),
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        score = proximity_score(match)
        result[country]["score"] += score
        result[country]["count"] += 1

        if str(match.get("level") or "").lower() in {"high", "critical"}:
            result[country]["high_critical"] += 1

    return result


def score_country_risk(country, event_row, prox_row):
    incident_pressure = saturating_score(event_row["all_score"], 10.0)
    military_drone = saturating_score(event_row["military_drone_score"], 5.0)
    cyber = saturating_score(event_row["cyber_score"], 4.0)

    infra_raw = event_row["infrastructure_score"] + prox_row["score"] * 1.35
    infrastructure = saturating_score(infra_raw, 6.0)

    crossborder = saturating_score(event_row["crossborder_score"], 4.0)

    recent = event_row["recent_24h_score"]
    previous_daily = event_row["previous_24_72h_score"] / 2.0

    if recent <= 0 and previous_daily <= 0:
        trend = 0.0
    else:
        ratio = (recent + 0.5) / (previous_daily + 0.5)
        trend = clamp(5.0 + 3.2 * (ratio - 1.0), 0.0, 10.0)

    dimensions = {
        "incident_pressure": incident_pressure,
        "military_drone": military_drone,
        "cyber": cyber,
        "infrastructure": infrastructure,
        "crossborder": crossborder,
        "trend": round(trend, 3),
    }

    normalized = round(sum(
        dimensions[name] * weight
        for name, weight in RISK_WEIGHTS.items()
    ), 3)

    evidence_count = event_row["all_count"] + prox_row["count"]
    active_dimensions = sum(
        1 for name in [
            "incident_pressure", "military_drone", "cyber",
            "infrastructure", "crossborder"
        ]
        if dimensions[name] > 0
    )

    confidence_value = clamp(
        0.20
        + min(0.45, evidence_count * 0.055)
        + min(0.25, active_dimensions * 0.05),
        0.0,
        1.0,
    )

    if confidence_value >= 0.72:
        confidence = "high"
    elif confidence_value >= 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    drivers = []
    driver_labels = {
        "incident_pressure": "biztonsági incidensnyomás",
        "military_drone": "katonai és drónaktivitás",
        "cyber": "kiberincidensek",
        "infrastructure": "kritikus infrastruktúra-kitettség",
        "crossborder": "határon átnyúló nyomás",
        "trend": "friss aktivitási trend",
    }

    for name, value in sorted(
        dimensions.items(),
        key=lambda item: item[1] * RISK_WEIGHTS[item[0]],
        reverse=True,
    ):
        if value >= 2.0:
            drivers.append(driver_labels[name])

    return {
        "normalized": normalized,
        "dimensions": dimensions,
        "confidence": confidence,
        "confidence_value": round(confidence_value, 3),
        "drivers": drivers[:4],
        "local_event_count": event_row["all_count"],
        "infra_proximity_count": prox_row["count"],
        "high_critical_infra_count": prox_row["high_critical"],
    }


def risk_level(score):
    if score >= 8.0:
        return "critical"
    if score >= 5.0:
        return "tense"
    if score >= 2.0:
        return "elevated"
    return "normal"


def enrich_risk(local_events, prox_matches):
    """
    Rebuild country risk from current validated evidence.

    Important: the previous normalized value is NOT used as a new baseline.
    This prevents the enrichment workflow from adding the same bonus again
    on every run and gradually pushing all countries toward 10.
    """
    risk = load_json(RISK_DAILY, {
        "generated_utc": now_utc().isoformat(),
        "country_scores": {},
        "countries": [],
        "region": {}
    })

    event_components = country_event_components(local_events)
    proximity_components = country_proximity_components(prox_matches)

    country_scores = {}
    countries = []

    for country in MONITORED_COUNTRIES:
        event_row = event_components[country]
        prox_row = proximity_components[country]

        scored = score_country_risk(country, event_row, prox_row)
        normalized = scored["normalized"]

        row = {
            "normalized": normalized,
            "overall": risk_level(normalized),
            "confidence": scored["confidence"],
            "confidence_value": scored["confidence_value"],
            "dimensions": scored["dimensions"],
            "drivers": scored["drivers"],
            "local_event_count": scored["local_event_count"],
            "infra_proximity_count": scored["infra_proximity_count"],
            "high_critical_infra_count": scored["high_critical_infra_count"],
            "model": "cee_country_risk_v2",
            "weights": RISK_WEIGHTS,
        }

        country_scores[country] = row

        countries.append({
            "country": country,
            "overall": row["overall"],
            "overall_score": normalized,
            "normalized": normalized,
            "confidence": row["confidence"],
            "confidence_value": row["confidence_value"],
            "drivers": row["drivers"],
            "dimensions": row["dimensions"],
        })

    countries.sort(key=lambda x: x["normalized"], reverse=True)

    # Regional score: activity-sensitive mean. It remains comparable over time
    # but does not become identical to the highest-risk country.
    region_scores = [x["normalized"] for x in countries]
    region_score = round(sum(region_scores) / len(MONITORED_COUNTRIES), 3)

    evidence_total = sum(
        x["local_event_count"] + x["infra_proximity_count"]
        for x in country_scores.values()
    )
    confidence_values = [
        x["confidence_value"] for x in country_scores.values()
    ]
    region_confidence_value = round(
        sum(confidence_values) / len(confidence_values),
        3,
    )

    dimension_scores = {}
    for dimension in RISK_WEIGHTS:
        dimension_scores[dimension] = round(
            sum(
                country_scores[c]["dimensions"][dimension]
                for c in MONITORED_COUNTRIES
            ) / len(MONITORED_COUNTRIES),
            3,
        )

    risk["country_scores"] = country_scores
    risk["countries"] = countries
    risk["region"] = {
        "overall": risk_level(region_score),
        "overall_score": region_score,
        "confidence": (
            "high" if region_confidence_value >= 0.72
            else "medium" if region_confidence_value >= 0.45
            else "low"
        ),
        "confidence_value": region_confidence_value,
        "dimension_scores": dimension_scores,
        "evidence_count": evidence_total,
    }
    risk["model"] = {
        "name": "cee_country_risk_v2",
        "scale": "0-10",
        "countries": MONITORED_COUNTRIES,
        "weights": RISK_WEIGHTS,
        "method": "weighted saturating dimensions rebuilt from current 7-day validated evidence",
        "important": "previous normalized scores are not cumulatively re-added",
    }
    risk["generated_utc"] = now_utc().isoformat()

    save_json(RISK_DAILY, risk)


def enrich_meta(local_events, prox_matches):
    meta = load_json(META, {"generated_utc": now_utc().isoformat(), "counts": {}})
    counts = meta.get("counts") or {}

    counts["local_events"] = sum(
        1 for f in local_events
        if (f.get("properties") or {}).get("_risk_source_file") in LOCAL_RISK_FILES
    )
    counts["validated_security_events"] = len(local_events)
    counts["validated_crossborder_events"] = sum(
        1 for f in local_events
        if (f.get("properties") or {}).get("_risk_source_file") == CROSSBORDER_RISK_FILE
    )
    counts["infra_proximity_matches"] = len(prox_matches)

    meta["counts"] = counts
    meta["local_infrastructure"] = {
        "enabled": True,
        "local_events": "local_events.geojson",
        "infra_proximity": "infra_proximity.json",
        "history": {
            "local_events": "history/local_events_history.geojson",
            "infra_proximity": "history/infra_proximity_history.json"
        }
    }

    save_json(META, meta)


def save_risk_debug(security_events, days=7):
    """Write an auditable list of the exact event records admitted to risk."""
    country_summary = {
        country: {
            "incident_count": 0,
            "category_counts": {},
            "source_files": {},
        }
        for country in MONITORED_COUNTRIES
    }
    countries = {country: [] for country in MONITORED_COUNTRIES}

    category_counters = defaultdict(Counter)
    source_counters = defaultdict(Counter)

    for feature in security_events:
        props = feature.get("properties") or {}
        country = norm_country(props.get("country"))
        if country not in countries:
            continue

        category = event_category(props)
        source_file = props.get("_risk_source_file") or "unknown"
        category_counters[country][category] += 1
        source_counters[country][source_file] += 1

        fingerprint = props.get("fingerprint") or {}
        fingerprint_hash = (
            fingerprint.get("hash")
            if isinstance(fingerprint, dict)
            else str(fingerprint or "") or None
        )

        countries[country].append({
            "id": props.get("id"),
            "date": (parse_time(props.get("time")).date().isoformat()
                     if parse_time(props.get("time")) else None),
            "time": props.get("time"),
            "category": category,
            "severity": props.get("severity"),
            "title": props.get("title") or props.get("name"),
            "source": props.get("source"),
            "url": props.get("url"),
            "source_file": source_file,
            "classification_source": (
                props.get("classification_source") or "security_classifier_compatible_local"
            ),
            "actual_incident": props.get("actual_incident"),
            "risk_eligible": props.get("risk_eligible"),
            "article_role": props.get("article_role"),
            "incident_family": props.get("incident_family") or props.get("family"),
            "incident_subcategory": props.get("incident_subcategory") or props.get("subcategory"),
            "incident_subtype": props.get("incident_subtype") or props.get("subtype"),
            "classification_confidence": (
                props.get("classification_confidence") or props.get("confidence")
            ),
            "matched_rule_id": props.get("matched_rule_id"),
            "fingerprint_hash": fingerprint_hash,
        })

    for country in MONITORED_COUNTRIES:
        countries[country].sort(key=lambda row: row.get("time") or "", reverse=True)
        country_summary[country] = {
            "incident_count": len(countries[country]),
            "category_counts": dict(category_counters[country]),
            "source_files": dict(source_counters[country]),
        }

    payload = {
        "generated_utc": now_utc().isoformat(),
        "purpose": "Exact validated incident records admitted to cee_country_risk_v2.",
        "model": "cee_country_risk_v2",
        "window_days": days,
        "totals": {
            "incident_count": len(security_events),
            "local_event_count": sum(
                1 for f in security_events
                if (f.get("properties") or {}).get("_risk_source_file") in LOCAL_RISK_FILES
            ),
            "crossborder_event_count": sum(
                1 for f in security_events
                if (f.get("properties") or {}).get("_risk_source_file") == CROSSBORDER_RISK_FILE
            ),
        },
        "country_summary": country_summary,
        "countries": countries,
    }
    save_json(RISK_DEBUG, payload)


def main():
    security_events = load_validated_security_events(days=7)
    prox_matches = load_proximity(days=7)

    enrich_summary(security_events, prox_matches)
    enrich_weekly(security_events, prox_matches)
    enrich_risk(security_events, prox_matches)
    enrich_meta(security_events, prox_matches)
    save_risk_debug(security_events, days=7)

    print("Security outputs enriched.")
    print(f"Validated security events 7d: {len(security_events)}")
    print(f"Validated infrastructure proximity matches 7d: {len(prox_matches)}")


if __name__ == "__main__":
    main()
