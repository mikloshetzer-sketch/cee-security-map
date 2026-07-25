import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from email.utils import parsedate_to_datetime
import hashlib
import re
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "history"

SUMMARY = DATA / "summary.json"
WEEKLY = DATA / "weekly.json"
META = DATA / "meta.json"
RISK_DAILY = DATA / "risk_daily.json"
RISK_INCIDENTS_DEBUG = DATA / "risk_incidents_debug.json"

LOCAL_EVENTS = DATA / "local_events.geojson"
LOCAL_HISTORY = HISTORY / "local_events_history.geojson"

INFRA_PROX = DATA / "infra_proximity.json"
INFRA_PROX_HISTORY = HISTORY / "infra_proximity_history.json"

GDELT = DATA / "gdelt.geojson"
GDELT_LINKED = DATA / "gdelt_linked.geojson"
GDELT_CROSSBORDER = DATA / "gdelt_crossborder.geojson"

SECURITY_EVENT_FILES = [
    LOCAL_EVENTS,
    LOCAL_HISTORY,
    GDELT,
    GDELT_LINKED,
    GDELT_CROSSBORDER,
]

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


def infer_security_category(props):
    explicit = str(
        props.get("incident_type")
        or props.get("category")
        or props.get("gdelt_bucket")
        or ""
    ).lower()

    text = " ".join([
        str(props.get("title") or props.get("name") or ""),
        str(props.get("summary") or props.get("description") or ""),
        str(props.get("url") or props.get("search_url") or ""),
    ])

    if contains_any(text, NOISE_TERMS):
        return None

    # Strong text evidence overrides an upstream "other" label.
    if contains_any(text, SECURITY_INCIDENT_PATTERNS["drone"]):
        if contains_any(text, [
            "shot down", "downed", "intercepted", "airspace",
            "fired at", "lelőtt", "légtér", "doborât",
            "spațiul aerian", "spatiul aerian"
        ]):
            return "drone"

    if contains_any(text, SECURITY_INCIDENT_PATTERNS["cyber"]):
        return "cyber"

    if contains_any(text, SECURITY_INCIDENT_PATTERNS["military"]):
        return "military"

    if contains_any(text, SECURITY_INCIDENT_PATTERNS["explosion"]):
        return "explosion"

    if contains_any(text, SECURITY_INCIDENT_PATTERNS["hazardous"]):
        return "hazardous"

    if contains_any(text, SECURITY_INCIDENT_PATTERNS["infrastructure"]):
        return "infrastructure_disruption"

    if explicit in {
        "drone", "drone_airspace", "cyber", "cyber_incident",
        "military", "military_accident", "kinetic_attack",
        "explosion", "hazardous", "hazardous_incident",
        "sabotage", "infrastructure_disruption", "major_fire"
    }:
        return explicit

    return None


def normalize_security_feature(feature, source_file):
    props = dict(feature.get("properties") or {})
    country = detect_security_country(props, source_file)
    if country not in MONITORED_COUNTRIES:
        return None

    category = infer_security_category(props)
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
    props["incident_type"] = props.get("incident_type") or category
    props["time"] = dt.isoformat()

    if not props.get("severity"):
        if category in {"drone", "military", "kinetic_attack", "explosion"}:
            props["severity"] = "high"
        elif category in {"cyber", "hazardous", "sabotage"}:
            props["severity"] = "medium"
        else:
            props["severity"] = "info"

    props["_risk_source_file"] = source_file

    return {
        "type": "Feature",
        "geometry": feature.get("geometry"),
        "properties": props,
    }



INCIDENT_CLUSTER_MAX_HOURS = 30.0
INCIDENT_TEXT_SIMILARITY = 0.28

INCIDENT_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "near",
    "romania", "romanian", "românia", "român", "hungary", "hungarian",
    "poland", "polish", "latvia", "latvian", "lithuania", "lithuanian",
    "estonia", "estonian", "slovakia", "slovak", "czech", "czechia",
    "republic", "drone", "drón", "dronă", "drona", "uav",
    "today", "yesterday", "after", "over", "into", "space", "airspace",
    "doua", "două", "noua", "nouă", "care", "este", "sunt", "din",
    "pentru", "după", "dupa", "asupra", "româniei", "romaniei",
}

DRONE_SHOOTDOWN_TERMS = [
    "shot down", "shoot down", "downed", "intercepted", "destroyed",
    "lelőtt", "lelőttek", "hatástalanítás", "hatástalanított",
    "doborât", "doborâtă", "doborârea", "doborata",
]

DRONE_AIRSPACE_TERMS = [
    "airspace", "légtér", "spațiul aerian", "spatiul aerian",
    "air space", "incursion", "incursiune", "violated", "violation",
]

CYBER_ATTACK_TERMS = [
    "ransomware", "ddos", "data breach", "cyberattack", "cyber attack",
    "kibertámadás",
]


def semantic_tokens(text):
    words = re.findall(r"[\wÀ-ž-]{3,}", normalize_text(text), flags=re.UNICODE)
    return {word for word in words if word not in INCIDENT_STOPWORDS}


def semantic_similarity(text_a, text_b):
    a = normalize_text(text_a)
    b = normalize_text(text_b)
    if not a or not b:
        return 0.0

    ta = semantic_tokens(a)
    tb = semantic_tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    return max(jaccard, sequence * 0.60)


def risk_event_text(feature):
    props = feature.get("properties") or {}
    return " ".join([
        str(props.get("title") or props.get("name") or ""),
        str(props.get("summary") or props.get("description") or ""),
        str(props.get("url") or ""),
    ])


def risk_incident_signature(feature):
    props = feature.get("properties") or {}
    category = str(props.get("incident_type") or props.get("category") or "").lower()
    text = risk_event_text(feature)
    signature = set()

    if category in {"drone", "drone_airspace"} or contains_any(
        text, SECURITY_INCIDENT_PATTERNS["drone"]
    ):
        signature.add("drone")
        if contains_any(text, DRONE_SHOOTDOWN_TERMS):
            signature.add("shootdown")
        if contains_any(text, DRONE_AIRSPACE_TERMS):
            signature.add("airspace")
        if contains_any(text, ["f-16", "f16"]):
            signature.add("f16")
        if contains_term(text, "shahed"):
            signature.add("shahed")

    if category in {"cyber", "cyber_incident"}:
        signature.add("cyber")
        if contains_any(text, CYBER_ATTACK_TERMS):
            signature.add("cyber_attack")

    if category in {"military", "military_accident"}:
        signature.add("military")
        if contains_any(text, ["crash", "crashed", "lezuhant", "prăbușit", "prabusit"]):
            signature.add("accident")

    if category in {"explosion", "kinetic_attack"}:
        signature.add(category)

    if category in {
        "hazardous", "hazardous_incident",
        "infrastructure_disruption", "sabotage"
    }:
        signature.add(category)

    return signature


def risk_event_time(feature):
    return parse_time((feature.get("properties") or {}).get("time"))


def same_risk_incident(feature_a, feature_b):
    pa = feature_a.get("properties") or {}
    pb = feature_b.get("properties") or {}

    if norm_country(pa.get("country")) != norm_country(pb.get("country")):
        return False

    category_a = str(pa.get("incident_type") or pa.get("category") or "").lower()
    category_b = str(pb.get("incident_type") or pb.get("category") or "").lower()

    if category_a != category_b:
        drone_family = {"drone", "drone_airspace"}
        cyber_family = {"cyber", "cyber_incident"}
        military_family = {"military", "military_accident"}

        if not (
            {category_a, category_b}.issubset(drone_family)
            or {category_a, category_b}.issubset(cyber_family)
            or {category_a, category_b}.issubset(military_family)
        ):
            return False

    ta = risk_event_time(feature_a)
    tb = risk_event_time(feature_b)

    if ta and tb:
        hours = abs((ta - tb).total_seconds()) / 3600.0
        if hours > INCIDENT_CLUSTER_MAX_HOURS:
            return False

        # Different days are normally different incidents.
        if ta.date() != tb.date() and hours > 6:
            return False

    sig_a = risk_incident_signature(feature_a)
    sig_b = risk_incident_signature(feature_b)
    shared = sig_a & sig_b

    if {"drone", "shootdown"}.issubset(shared):
        return True

    if {"drone", "airspace"}.issubset(shared):
        return semantic_similarity(
            risk_event_text(feature_a), risk_event_text(feature_b)
        ) >= 0.12

    if "cyber" in shared:
        return semantic_similarity(
            risk_event_text(feature_a), risk_event_text(feature_b)
        ) >= 0.30

    return semantic_similarity(
        risk_event_text(feature_a), risk_event_text(feature_b)
    ) >= INCIDENT_TEXT_SIMILARITY


def merge_risk_incident(primary, incoming):
    pp = primary.setdefault("properties", {})
    ip = incoming.get("properties") or {}

    sources = list(
        pp.get("risk_sources")
        or pp.get("sources")
        or ([pp.get("source")] if pp.get("source") else [])
    )
    for source in (
        ip.get("risk_sources")
        or ip.get("sources")
        or ([ip.get("source")] if ip.get("source") else [])
    ):
        if source and source not in sources:
            sources.append(source)

    urls = list(
        pp.get("risk_urls")
        or pp.get("urls")
        or ([pp.get("url")] if pp.get("url") else [])
    )
    for url in (
        ip.get("risk_urls")
        or ip.get("urls")
        or ([ip.get("url")] if ip.get("url") else [])
    ):
        if url and url not in urls:
            urls.append(url)

    titles = list(
        pp.get("risk_titles")
        or ([pp.get("title")] if pp.get("title") else [])
    )
    incoming_title = ip.get("title")
    if incoming_title and incoming_title not in titles:
        titles.append(incoming_title)

    pp["risk_sources"] = sources
    pp["risk_urls"] = urls
    pp["risk_titles"] = titles
    pp["source_count"] = len(sources)
    pp["article_count"] = int(pp.get("article_count") or 1) + int(
        ip.get("article_count") or 1
    )

    severity_rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    if severity_rank.get(str(ip.get("severity") or "").lower(), 0) > \
       severity_rank.get(str(pp.get("severity") or "").lower(), 0):
        pp["severity"] = ip.get("severity")

    t_primary = parse_time(pp.get("time"))
    t_incoming = parse_time(ip.get("time"))
    if t_incoming and (not t_primary or t_incoming < t_primary):
        pp["time"] = ip.get("time")

    return primary


def cluster_risk_incidents(features):
    clusters = []

    ordered = sorted(
        features,
        key=lambda f: risk_event_time(f) or datetime.min.replace(tzinfo=timezone.utc),
    )

    for feature in ordered:
        props = feature.setdefault("properties", {})
        props["article_count"] = int(props.get("article_count") or 1)
        props["risk_sources"] = list(
            props.get("sources")
            or ([props.get("source")] if props.get("source") else [])
        )
        props["risk_urls"] = list(
            props.get("urls")
            or ([props.get("url")] if props.get("url") else [])
        )
        props["risk_titles"] = [props.get("title")] if props.get("title") else []

        merged = False
        for index, existing in enumerate(clusters):
            if same_risk_incident(existing, feature):
                clusters[index] = merge_risk_incident(existing, feature)
                merged = True
                break

        if not merged:
            clusters.append(feature)

    return clusters


def load_validated_security_events(days=7):
    """
    Unified risk input:
    local + GDELT + linked + cross-border.

    First normalize/filter articles, then collapse multiple reports of the
    same real-world event into one risk incident.
    """
    cutoff = now_utc() - timedelta(days=days)
    candidates = []

    for path in SECURITY_EVENT_FILES:
        payload = load_json(path, {"features": []})

        for feature in payload.get("features", []):
            normalized = normalize_security_feature(
                feature,
                path.name,
            )
            if not normalized:
                continue

            props = normalized["properties"]
            dt = parse_time(props.get("time"))

            if not dt or dt < cutoff:
                continue

            candidates.append(normalized)

    # Exact article dedup first.
    exact_seen = set()
    article_unique = []

    for feature in candidates:
        props = feature.get("properties") or {}

        exact_key = stable_event_key(
            props.get("url"),
            props.get("title"),
            props.get("source"),
            props.get("time"),
        )

        if exact_key in exact_seen:
            continue

        exact_seen.add(exact_key)
        article_unique.append(feature)

    # Then real-world incident clustering.
    return cluster_risk_incidents(article_unique)


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

            # Reject legacy proximity rows created before the validated
            # incident/geolocation model existed.
            if not event.get("incident_type"):
                continue

            location_quality = str(
                event.get("location_quality") or ""
            ).lower()

            if location_quality not in {
                "city", "specific_place", "precise"
            }:
                continue

            if not event.get("geolocation_method"):
                continue

            key = (
                event.get("id"),
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



def write_risk_incidents_debug(security_events):
    """
    Write an audit file showing exactly which clustered incidents are used
    by the country-risk model. This file does not change risk calculations.
    """
    countries = {
        country: []
        for country in MONITORED_COUNTRIES
    }

    total_articles = 0
    total_sources = 0

    for index, feature in enumerate(security_events, start=1):
        props = feature.get("properties") or {}
        country = norm_country(props.get("country"))

        if country not in MONITORED_COUNTRIES:
            continue

        dt = parse_time(props.get("time"))
        category = str(
            props.get("incident_type")
            or props.get("category")
            or "unknown"
        ).lower()

        titles = list(
            props.get("risk_titles")
            or ([props.get("title")] if props.get("title") else [])
        )
        urls = list(
            props.get("risk_urls")
            or props.get("urls")
            or ([props.get("url")] if props.get("url") else [])
        )
        sources = list(
            props.get("risk_sources")
            or props.get("sources")
            or ([props.get("source")] if props.get("source") else [])
        )

        article_count = int(props.get("article_count") or max(1, len(titles)))
        source_count = int(props.get("source_count") or max(1, len(sources)))

        total_articles += article_count
        total_sources += source_count

        representative_title = (
            props.get("title")
            or (titles[0] if titles else "")
            or props.get("name")
            or "Untitled incident"
        )

        cluster_id = stable_event_key(
            country,
            category,
            dt.isoformat() if dt else props.get("time"),
            representative_title,
        )

        countries[country].append({
            "cluster_id": cluster_id,
            "date": dt.date().isoformat() if dt else None,
            "time": dt.isoformat() if dt else props.get("time"),
            "category": category,
            "severity": props.get("severity"),
            "representative_title": representative_title,
            "article_count": article_count,
            "source_count": source_count,
            "sources": sources,
            "titles": titles,
            "urls": urls,
            "risk_score_input": local_event_score(props),
            "source_file": props.get("_risk_source_file"),
        })

    country_summary = {}

    for country in MONITORED_COUNTRIES:
        incidents = countries[country]
        incidents.sort(
            key=lambda row: (
                row.get("time") or "",
                row.get("category") or "",
                row.get("representative_title") or "",
            ),
            reverse=True,
        )

        category_counts = Counter(
            row["category"]
            for row in incidents
        )

        country_summary[country] = {
            "incident_count": len(incidents),
            "article_count": sum(
                row["article_count"]
                for row in incidents
            ),
            "source_count": sum(
                row["source_count"]
                for row in incidents
            ),
            "category_counts": dict(
                sorted(category_counts.items())
            ),
        }

    payload = {
        "generated_utc": now_utc().isoformat(),
        "purpose": (
            "Audit of incident-level records actually used by "
            "cee_country_risk_v2. This file is diagnostic only."
        ),
        "model": "cee_country_risk_v2",
        "window_days": 7,
        "totals": {
            "incident_count": len(security_events),
            "article_count": total_articles,
            "source_count": total_sources,
        },
        "country_summary": country_summary,
        "countries": countries,
    }

    save_json(RISK_INCIDENTS_DEBUG, payload)


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
        "article_count": 0,
        "source_count": 0,
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

        # By this stage each feature is already one clustered incident.
        event_key = stable_event_key(
            country,
            props.get("category"),
            props.get("time"),
            "|".join(props.get("risk_titles") or [str(props.get("title") or "")]),
        )

        if event_key in seen:
            continue

        seen.add(event_key)

        score = local_event_score(props)
        category = event_category(props)
        age = age_hours(props.get("time"))

        row = result[country]
        row["all_score"] += score
        row["all_count"] += 1
        row["article_count"] += int(props.get("article_count") or 1)
        row["source_count"] += max(
            1,
            int(props.get("source_count") or 1),
        )

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
    corroboration = max(
        0,
        event_row.get("source_count", 0) - event_row["all_count"],
    )
    active_dimensions = sum(
        1 for name in [
            "incident_pressure", "military_drone", "cyber",
            "infrastructure", "crossborder"
        ]
        if dimensions[name] > 0
    )

    confidence_value = clamp(
        0.20
        + min(0.40, evidence_count * 0.055)
        + min(0.20, active_dimensions * 0.05)
        + min(0.15, corroboration * 0.015),
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
        "article_count": event_row.get("article_count", event_row["all_count"]),
        "source_count": event_row.get("source_count", event_row["all_count"]),
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
            "article_count": scored.get("article_count", scored["local_event_count"]),
            "source_count": scored.get("source_count", scored["local_event_count"]),
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
        "method": "weighted saturating dimensions from 7-day incident-clustered validated evidence",
        "important": "previous normalized scores are not cumulatively re-added",
    }
    risk["generated_utc"] = now_utc().isoformat()

    save_json(RISK_DAILY, risk)


def enrich_meta(local_events, prox_matches):
    meta = load_json(META, {"generated_utc": now_utc().isoformat(), "counts": {}})
    counts = meta.get("counts") or {}

    counts["local_events"] = len(local_events)
    counts["validated_security_events"] = len(local_events)
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


def main():
    security_events = load_validated_security_events(days=7)
    prox_matches = load_proximity(days=7)

    write_risk_incidents_debug(security_events)
    enrich_summary(security_events, prox_matches)
    enrich_weekly(security_events, prox_matches)
    enrich_risk(security_events, prox_matches)
    enrich_meta(security_events, prox_matches)

    print("Security outputs enriched.")
    print(f"Risk incident audit saved: {RISK_INCIDENTS_DEBUG}")
    print(f"Validated security events 7d: {len(security_events)}")
    print(f"Validated infrastructure proximity matches 7d: {len(prox_matches)}")


if __name__ == "__main__":
    main()
