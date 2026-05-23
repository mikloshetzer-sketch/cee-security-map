import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "history"

SUMMARY = DATA / "summary.json"
WEEKLY = DATA / "weekly.json"
META = DATA / "meta.json"
RISK_DAILY = DATA / "risk_daily.json"

LOCAL_EVENTS = DATA / "local_events.geojson"
LOCAL_HISTORY = HISTORY / "local_events_history.geojson"

INFRA_PROX = DATA / "infra_proximity.json"
INFRA_PROX_HISTORY = HISTORY / "infra_proximity_history.json"

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
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def age_hours(value):
    dt = parse_time(value)
    if not dt:
        return None
    return (now_utc() - dt).total_seconds() / 3600


def norm_country(country):
    return COUNTRY_NAME_MAP.get(country, country or "Unknown")


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

    for path in [INFRA_PROX, INFRA_PROX_HISTORY]:
        payload = load_json(path, {"matches": []})
        for m in payload.get("matches", []):
            event = m.get("event") or {}
            dt = parse_time(event.get("time"))
            if not dt:
                continue
            if dt >= now_utc() - timedelta(days=days):
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


def enrich_risk(local_events, prox_matches):
    risk = load_json(RISK_DAILY, {
        "generated_utc": now_utc().isoformat(),
        "country_scores": {},
        "countries": [],
        "region": {
            "overall": "normal",
            "overall_score": 0,
            "confidence": "low",
            "confidence_value": 0,
            "dimensions": {},
            "dimension_scores": {}
        }
    })

    local_stats = build_local_stats(local_events)
    prox_stats = build_proximity_stats(prox_matches)

    country_scores = risk.get("country_scores") or {}

    all_countries = set(country_scores.keys()) | set(local_stats.keys()) | set(prox_stats.keys())

    for country in all_countries:
        row = country_scores.get(country, {})
        base_norm = float(row.get("normalized", 0.0) or 0.0)

        local_score = local_stats.get(country, {}).get("score", 0.0)
        prox_score = prox_stats.get(country, {}).get("score", 0.0)

        infra_bonus = min(2.5, local_score * 0.18 + prox_score * 0.22)
        new_norm = min(10.0, round(base_norm + infra_bonus, 3))

        row["base_normalized_before_local_infra"] = round(base_norm, 3)
        row["normalized"] = new_norm
        row["local_event_score"] = round(local_score, 3)
        row["infra_proximity_score"] = round(prox_score, 3)
        row["local_event_count"] = local_stats.get(country, {}).get("count", 0)
        row["infra_proximity_count"] = prox_stats.get(country, {}).get("count", 0)

        country_scores[country] = row

    risk["country_scores"] = country_scores

    countries = []
    for country, row in sorted(country_scores.items(), key=lambda x: x[1].get("normalized", 0), reverse=True):
        normalized = float(row.get("normalized", 0.0) or 0.0)
        overall = "critical" if normalized >= 8 else "tense" if normalized >= 5 else "elevated" if normalized >= 2 else "normal"

        countries.append({
            "country": country,
            "overall": overall,
            "overall_score": round(normalized, 3),
            "normalized": round(normalized, 3),
            "confidence": "medium" if row.get("local_event_count", 0) or row.get("infra_proximity_count", 0) else "derived",
            "drivers": [
                x for x in [
                    "lokális infrastruktúra-események" if row.get("local_event_count", 0) else None,
                    "kritikus infrastruktúra-közeli incidensek" if row.get("infra_proximity_count", 0) else None
                ] if x
            ]
        })

    risk["countries"] = countries

    region_scores = [float(x.get("normalized", 0.0) or 0.0) for x in country_scores.values()]
    region_score = round(sum(region_scores) / max(1, len(region_scores)), 3)

    region = risk.get("region") or {}
    region["overall_score"] = region_score
    region["overall"] = "critical" if region_score >= 8 else "tense" if region_score >= 5 else "elevated" if region_score >= 2 else "normal"
    region["confidence"] = "medium"
    region["confidence_value"] = min(1.0, round(0.45 + 0.02 * len(local_events) + 0.03 * len(prox_matches), 3))

    dims = region.get("dimension_scores") or {}
    dims["infrastructure"] = round(float(dims.get("infrastructure", 0.0) or 0.0) + len(prox_matches) * 0.15 + len(local_events) * 0.08, 3)
    dims["cyber"] = round(float(dims.get("cyber", 0.0) or 0.0) + sum(1 for f in local_events if (f.get("properties") or {}).get("category") == "cyber") * 0.25, 3)
    region["dimension_scores"] = dims

    risk["region"] = region
    risk["generated_utc"] = now_utc().isoformat()

    save_json(RISK_DAILY, risk)


def enrich_meta(local_events, prox_matches):
    meta = load_json(META, {"generated_utc": now_utc().isoformat(), "counts": {}})
    counts = meta.get("counts") or {}

    counts["local_events"] = len(local_events)
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
    local_events = load_local_events(days=7)
    prox_matches = load_proximity(days=7)

    enrich_summary(local_events, prox_matches)
    enrich_weekly(local_events, prox_matches)
    enrich_risk(local_events, prox_matches)
    enrich_meta(local_events, prox_matches)

    print("Security outputs enriched.")
    print(f"Local events 7d: {len(local_events)}")
    print(f"Infrastructure proximity matches 7d: {len(prox_matches)}")


if __name__ == "__main__":
    main()
