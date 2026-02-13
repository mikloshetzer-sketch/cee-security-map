#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Tuple, Optional

import requests
from dateutil import parser as dateparser


# =========================
# PATHS
# =========================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

USER_AGENT = "cee-security-map/3.1 (github actions)"
TIMEOUT = 30

# =========================
# CEE COUNTRIES (STRICT)
# =========================
CEE_COUNTRIES = [
    "Hungary",
    "Poland",
    "Czech Republic",
    "Slovakia",
    "Romania",
    "Latvia",
    "Lithuania",
    "Estonia",
]

# rough bbox (final is country name from GDELT location.country)
CEE_BBOX = (11.0, 42.5, 30.5, 61.5)


# =========================
# BASICS
# =========================
def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def to_utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time_iso(t: Optional[str]) -> Optional[datetime]:
    if not t:
        return None
    try:
        dt = dateparser.parse(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def in_bbox(lon: float, lat: float) -> bool:
    lon_min, lat_min, lon_max, lat_max = CEE_BBOX
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def in_cee_country(country_name: Optional[str]) -> bool:
    return bool(country_name) and (country_name in CEE_COUNTRIES)


def to_feature(lon: float, lat: float, props: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


def save_geojson(path: str, features: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2, ensure_ascii=False)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================
# HTTP helper (retry + backoff)
# =========================
def http_get(url: str, params: Optional[dict] = None) -> requests.Response:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/plain,*/*",
    }

    backoff = 2
    last_exc: Optional[Exception] = None

    for attempt in range(1, 5):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)

            # Retry on typical transient errors
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"[http_get] retry {attempt}/4 status={r.status_code}")
                time.sleep(backoff)
                backoff *= 2
                continue

            r.raise_for_status()
            return r

        except Exception as e:
            last_exc = e
            print(f"[http_get] error retry {attempt}/4: {e}")
            time.sleep(backoff)
            backoff *= 2

    raise last_exc if last_exc else RuntimeError("http_get failed")


# =========================
# GDELT FETCH (robust JSON handling)
# =========================
def fetch_gdelt(days: int = 7, max_records: int = 250) -> List[Dict[str, Any]]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # bővített kulcsszavak
    keywords = [
        "protest", "riot", "violence", "clash", "border", "checkpoint",
        "military", "army", "troops", "mobilization",
        "cyber", "ransomware", "ddos", "attack", "explosion",
        "security", "police", "terror", "arrest",
        "sanctions", "energy", "gas", "pipeline", "infrastructure",
        "sabotage", "disinformation", "spy", "espionage"
    ]

    # ország-keresési tokenek (query-hez)
    countries_q = ["Hungary", "Poland", "Czech", "Slovakia", "Romania", "Latvia", "Lithuania", "Estonia"]

    query = "(" + " OR ".join(keywords) + ") AND (" + " OR ".join(countries_q) + ")"

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "sort": "HybridRel",
    }

    resp = http_get(url, params=params)

    # --- Robust JSON parse ---
    try:
        data = resp.json()
    except Exception:
        head = (resp.text or "")[:400].replace("\n", " ")
        print(f"[GDELT] Non-JSON response. status={resp.status_code} head={head!r}")
        return []

    out: List[Dict[str, Any]] = []
    for a in data.get("articles", []) or []:
        loc = a.get("location") or {}
        geo = (loc.get("geo") or {})

        lat = geo.get("latitude")
        lon = geo.get("longitude")
        country = loc.get("country")  # IMPORTANT: this is what we filter on

        if lat is None or lon is None:
            continue

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            continue

        # quick bbox filter + strict country filter
        if not in_bbox(lon_f, lat_f):
            continue
        if not in_cee_country(country):
            continue

        dt = parse_time_iso(a.get("seendate"))

        out.append(
            to_feature(
                lon_f,
                lat_f,
                {
                    "source": "GDELT",
                    "kind": "news_event",
                    "title": a.get("title"),
                    "time": to_utc_z(dt) if dt else None,
                    "url": a.get("url"),
                    "domain": a.get("domain"),
                    "language": a.get("language"),
                    "country": country,
                    "type": "News",
                },
            )
        )

    print(f"[GDELT] parsed articles={len(data.get('articles', []) or [])}, kept={len(out)}")
    return out


# =========================
# MAIN
# =========================
def main() -> int:
    ensure_dirs()

    print("Fetching GDELT…")
    try:
        gdelt = fetch_gdelt(days=7, max_records=250)
    except Exception as e:
        # IMPORTANT: never fail the job on GDELT instability
        print(f"[GDELT] fetch failed, continuing with empty list: {e}")
        gdelt = []

    save_geojson(os.path.join(DATA_DIR, "gdelt.geojson"), gdelt)

    summary = {
        "generated_utc": to_utc_z(datetime.now(timezone.utc)),
        "headline": "Közép–Kelet Európa biztonsági helyzet – napi kivonat",
        "bullets": [
            f"GDELT események száma: {len(gdelt)} (csak 8 CEE ország, bbox+country szűréssel).",
            "Megjegyzés: automatikus OSINT-kivonat; a linkelt források kézi ellenőrzése javasolt.",
        ],
    }
    save_json(os.path.join(DATA_DIR, "summary.json"), summary)

    meta = {
        "generated_utc": to_utc_z(datetime.now(timezone.utc)),
        "counts": {"gdelt": len(gdelt)},
        "countries": CEE_COUNTRIES,
        "bbox": {"lon_min": CEE_BBOX[0], "lat_min": CEE_BBOX[1], "lon_max": CEE_BBOX[2], "lat_max": CEE_BBOX[3]},
    }
    save_json(os.path.join(DATA_DIR, "meta.json"), meta)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
