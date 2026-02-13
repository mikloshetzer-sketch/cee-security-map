#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Tuple

import requests
from dateutil import parser as dateparser


# =========================
# PATHS
# =========================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

USER_AGENT = "cee-security-map/3.0"
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

CEE_BBOX = (11.0, 42.5, 30.5, 61.5)

ROLLING_DAYS = 7


# =========================
# BASICS
# =========================
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def http_get(url: str, params=None):
    headers = {"User-Agent": USER_AGENT}
    return requests.get(url, params=params, headers=headers, timeout=TIMEOUT)


def to_utc_z(dt: datetime):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time_iso(t: str | None):
    if not t:
        return None
    try:
        return dateparser.parse(t).astimezone(timezone.utc)
    except:
        return None


def in_bbox(lon: float, lat: float):
    lon_min, lat_min, lon_max, lat_max = CEE_BBOX
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def to_feature(lon: float, lat: float, props: Dict[str, Any]):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


def save_geojson(path: str, features: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2, ensure_ascii=False)


# =========================
# COUNTRY FILTER
# =========================
def in_cee_country(country_name: str | None):
    if not country_name:
        return False
    return country_name in CEE_COUNTRIES


# =========================
# GDELT FETCH (BŐVÍTETT)
# =========================
def fetch_gdelt():
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    keywords = [
        "protest","riot","violence","border","military","army","troops",
        "cyber","attack","explosion","security","police","sanctions",
        "energy","gas","infrastructure","conflict","demonstration"
    ]

    countries = ["Hungary","Poland","Czech","Slovakia","Romania","Latvia","Lithuania","Estonia"]

    query = "(" + " OR ".join(keywords) + ") AND (" + " OR ".join(countries) + ")"

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": "250",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }

    resp = http_get(url, params)
    data = resp.json()

    out = []

    for a in data.get("articles", []):
        loc = a.get("location") or {}
        geo = loc.get("geo") or {}

        lat = geo.get("latitude")
        lon = geo.get("longitude")
        country = loc.get("country")

        if lat is None or lon is None:
            continue

        lat = float(lat)
        lon = float(lon)

        if not in_bbox(lon, lat):
            continue

        if not in_cee_country(country):
            continue

        dt = parse_time_iso(a.get("seendate"))

        out.append(
            to_feature(
                lon,
                lat,
                {
                    "source": "GDELT",
                    "title": a.get("title"),
                    "time": to_utc_z(dt) if dt else None,
                    "url": a.get("url"),
                },
            )
        )

    return out


# =========================
# MAIN
# =========================
def main():

    ensure_dirs()

    print("Fetching GDELT…")
    gdelt = fetch_gdelt()

    save_geojson(os.path.join(DATA_DIR, "gdelt.geojson"), gdelt)

    print("Saving summary…")

    summary = {
        "generated_utc": to_utc_z(datetime.now(timezone.utc)),
        "headline": "Közép–Kelet Európa biztonsági helyzet – napi kivonat",
        "bullets": [
            f"GDELT események száma: {len(gdelt)}",
            "Automatikus OSINT kivonat.",
        ],
    }

    with open(os.path.join(DATA_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
