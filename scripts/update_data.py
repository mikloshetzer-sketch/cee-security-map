#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dateutil import parser as dateparser

# =========================
# PATHS
# =========================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

GDELT_GEOJSON_PATH = os.path.join(DATA_DIR, "gdelt.geojson")
GDELT_DEBUG_PATH = os.path.join(DATA_DIR, "gdelt_debug.json")
USGS_GEOJSON_PATH = os.path.join(DATA_DIR, "usgs.geojson")
GDACS_GEOJSON_PATH = os.path.join(DATA_DIR, "gdacs.geojson")

HOTSPOTS_GEOJSON_PATH = os.path.join(DATA_DIR, "hotspots.geojson")
HOTSPOTS_JSON_PATH = os.path.join(DATA_DIR, "hotspots.json")
EARLY_GEOJSON_PATH = os.path.join(DATA_DIR, "early.geojson")
EARLY_JSON_PATH = os.path.join(DATA_DIR, "early.json")

SUMMARY_PATH = os.path.join(DATA_DIR, "summary.json")
WEEKLY_PATH = os.path.join(DATA_DIR, "weekly.json")
META_PATH = os.path.join(DATA_DIR, "meta.json")

CACHE_PATH = os.path.join(DATA_DIR, "geocode_cache.json")
COUNTRIES_CACHE_PATH = os.path.join(DATA_DIR, "cee_countries.geojson")
BORDERS_PATH = os.path.join(DATA_DIR, "cee_borders.geojson")

USER_AGENT = "cee-security-map/2.6 (github actions)"
TIMEOUT = 30

# =========================
# REGION: 8 countries (CEE)
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

# rough bbox prefilter (final filter is point-in-polygon)
CEE_BBOX = (11.0, 42.5, 30.5, 61.5)

# rolling windows
ROLLING_DAYS = 7
USGS_DAYS = 7
GDACS_DAYS = 14
GDELT_DAYS = 7

# GDELT GEO2.1: points per query (cap)
GDELT_MAXPOINTS = 250

# Rate limits / politeness
SLEEP_BETWEEN_GDELT_CALLS = 0.25

# -------------------------
# EARLY WARNING zones (optional)
# -------------------------
SENSITIVE_ZONES = [
    {"name": "PL–BY / Suwałki környék (tág)", "bbox": (22.0, 53.5, 24.5, 55.8), "mult": 1.20},
    {"name": "RO–MD határ (tág)", "bbox": (26.0, 45.0, 29.5, 48.2), "mult": 1.15},
    {"name": "HU–UA perem (tág)", "bbox": (21.0, 47.5, 23.5, 49.3), "mult": 1.12},
]

# =========================
# Basics
# =========================
def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
    h = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        h.update(headers)

    backoff = 2
    last_exc: Optional[Exception] = None

    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, headers=h, timeout=TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(backoff)
            backoff *= 2

    raise last_exc if last_exc else RuntimeError("http_get failed")

def in_bbox(lon: float, lat: float, bbox: Tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lon_max) and (lat_min <= lat <= lat_max)

def to_feature(lon: float, lat: float, props: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }

def save_geojson(path: str, features: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

def load_geojson_features(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        feats = data.get("features") or []
        return feats if isinstance(feats, list) else []
    except Exception:
        return []

# -------------------------
# Time helpers
# -------------------------
def to_utc_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")

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

def clamp_times(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for f in features:
        p = f.get("properties") or {}
        dt = parse_time_iso(p.get("time"))
        if dt is not None:
            p["time"] = to_utc_z(dt)
            f["properties"] = p
        out.append(f)
    return out

def trim_by_days(features: List[Dict[str, Any]], keep_days: int) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=keep_days)
    kept = []
    for f in features:
        dt = parse_time_iso((f.get("properties") or {}).get("time"))
        if dt is None:
            continue
        if dt >= cutoff:
            kept.append(f)
    return kept

# -------------------------
# Dedup
# -------------------------
def dedup_key(feature: Dict[str, Any]) -> Optional[str]:
    p = feature.get("properties") or {}
    src = p.get("source") or ""
    url = p.get("url")
    title = p.get("title")
    tm = p.get("time")
    kind = p.get("kind") or ""
    if url:
        return f"{src}|{url}"
    if title and tm:
        return f"{src}|{kind}|{tm}|{title}"
    return None

def merge_dedup(old_feats: List[Dict[str, Any]], new_feats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for f in (new_feats + old_feats):
        k = dedup_key(f)
        if not k:
            merged.append(f)
            continue
        if k in seen:
            continue
        seen.add(k)
        merged.append(f)

    def sort_key(feat: Dict[str, Any]) -> float:
        dt = parse_time_iso((feat.get("properties") or {}).get("time"))
        return dt.timestamp() if dt else 0.0

    merged.sort(key=sort_key, reverse=True)
    return merged

# =========================
# Point-in-polygon (no shapely)
# =========================
def point_in_ring(lon: float, lat: float, ring: List[List[float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 4:
        return False
    x, y = lon, lat
    for i in range(n - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if x < xinters:
                inside = not inside
    return inside

def point_in_polygon(lon: float, lat: float, poly_coords: List[List[List[float]]]) -> bool:
    if not poly_coords:
        return False
    outer = poly_coords[0]
    if not point_in_ring(lon, lat, outer):
        return False
    for hole in poly_coords[1:]:
        if point_in_ring(lon, lat, hole):
            return False
    return True

def point_in_feature(lon: float, lat: float, geom: Dict[str, Any]) -> bool:
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return False
    if gtype == "Polygon":
        return point_in_polygon(lon, lat, coords)
    if gtype == "MultiPolygon":
        for poly in coords:
            if point_in_polygon(lon, lat, poly):
                return True
        return False
    return False

def load_or_build_country_geoms() -> Dict[str, Dict[str, Any]]:
    need_refresh = True
    if os.path.exists(COUNTRIES_CACHE_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(COUNTRIES_CACHE_PATH), tz=timezone.utc)
        if datetime.now(timezone.utc) - mtime < timedelta(days=7):
            need_refresh = False

    if need_refresh:
        url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        data = http_get(url).json()
        keep = set(CEE_COUNTRIES)
        feats = []
        for f in (data.get("features") or []):
            props = f.get("properties") or {}
            name = props.get("name")
            if name in keep:
                feats.append(f)
        with open(COUNTRIES_CACHE_PATH, "w", encoding="utf-8") as fp:
            json.dump({"type": "FeatureCollection", "features": feats}, fp, ensure_ascii=False, indent=2)

    with open(COUNTRIES_CACHE_PATH, "r", encoding="utf-8") as fp:
        cached = json.load(fp) or {}

    geoms: Dict[str, Dict[str, Any]] = {}
    for f in (cached.get("features") or []):
        props = f.get("properties") or {}
        name = props.get("name")
        geom = f.get("geometry") or {}
        if name and geom:
            geoms[name] = geom
    return geoms

def in_cee_countries(lon: float, lat: float, geoms: Dict[str, Dict[str, Any]]) -> bool:
    if not (CEE_BBOX[0] <= lon <= CEE_BBOX[2] and CEE_BBOX[1] <= lat <= CEE_BBOX[3]):
        return False
    for name in CEE_COUNTRIES:
        geom = geoms.get(name)
        if geom and point_in_feature(lon, lat, geom):
            return True
    return False

def ensure_cee_borders(geoms: Dict[str, Dict[str, Any]]) -> None:
    feats = []
    for name in CEE_COUNTRIES:
        if name in geoms:
            feats.append({"type": "Feature", "properties": {"name": name}, "geometry": geoms[name]})
    with open(BORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False, indent=2)

# =========================
# Categorization for bubble
# =========================
CATEGORY_RULES = [
    ("Cyber", re.compile(r"\b(cyber|ransomware|ddos|hack|hacker|malware|phishing|disinformation)\b", re.I)),
    ("Határ", re.compile(r"\b(border|checkpoint|incursion|smuggling|migrant|crossing)\b", re.I)),
    ("Tüntetés", re.compile(r"\b(protest|demonstration|strike|rally|riot)\b", re.I)),
    ("Erőszak", re.compile(r"\b(attack|explosion|arson|sabotage|shooting|stabbing)\b", re.I)),
    ("Rendészet/Jog", re.compile(r"\b(police|arrest|detained|court|trial|sentence)\b", re.I)),
    ("Katonai", re.compile(r"\b(military|troops|deployment|exercise|drill|nato)\b", re.I)),
    ("Energia/Infrastruktúra", re.compile(r"\b(energy|pipeline|power outage|blackout|infrastructure)\b", re.I)),
]

def categorize(title: str) -> str:
    t = title or ""
    for name, rx in CATEGORY_RULES:
        if rx.search(t):
            return name
    return "Egyéb"

# =========================
# SOURCES
# =========================
def fetch_usgs(geoms: Dict[str, Dict[str, Any]], days: int = 7, min_magnitude: float = 2.5) -> List[Dict[str, Any]]:
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": end.strftime("%Y-%m-%d"),
        "minmagnitude": str(min_magnitude),
    }
    data = http_get(url, params=params).json()

    out: List[Dict[str, Any]] = []
    for f in data.get("features", []) or []:
        coords = (f.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        if not in_cee_countries(lon, lat, geoms):
            continue
        p = f.get("properties") or {}
        t_ms = p.get("time")
        dt = None
        if isinstance(t_ms, (int, float)):
            dt = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)

        out.append(
            to_feature(
                lon, lat,
                {
                    "source": "USGS",
                    "kind": "earthquake",
                    "title": p.get("title"),
                    "category": "Földrengés",
                    "mag": p.get("mag"),
                    "place": p.get("place"),
                    "time": to_utc_z(dt) if dt else None,
                    "url": p.get("url"),
                    "type": "Earthquake",
                },
            )
        )
    return out

def fetch_gdacs(geoms: Dict[str, Dict[str, Any]], days: int = 14) -> List[Dict[str, Any]]:
    url = "https://www.gdacs.org/xml/rss.xml"
    xml = http_get(url).text
    items = xml.split("<item>")[1:]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    def get_tag(chunk: str, tag: str) -> Optional[str]:
        open_t = f"<{tag}>"
        close_t = f"</{tag}>"
        if open_t in chunk and close_t in chunk:
            return chunk.split(open_t, 1)[1].split(close_t, 1)[0].strip()
        return None

    out: List[Dict[str, Any]] = []
    for raw in items:
        chunk = raw.split("</item>")[0]
        title = get_tag(chunk, "title") or ""
        link = get_tag(chunk, "link")
        pub = get_tag(chunk, "pubDate")
        point = get_tag(chunk, "georss:point") or get_tag(chunk, "point")
        if not pub or not point:
            continue
        try:
            pub_dt = dateparser.parse(pub).astimezone(timezone.utc)
        except Exception:
            continue
        if pub_dt < cutoff:
            continue
        try:
            lat_s, lon_s = point.split()
            lat, lon = float(lat_s), float(lon_s)
        except Exception:
            continue
        if not in_cee_countries(lon, lat, geoms):
            continue

        out.append(
            to_feature(
                lon, lat,
                {
                    "source": "GDACS",
                    "kind": "disaster_alert",
                    "title": title,
                    "category": "Riasztás",
                    "time": to_utc_z(pub_dt),
                    "url": link,
                    "type": "Alert",
                },
            )
        )
    return out

def gdelt_geo_query(query: str, start: datetime, end: datetime, maxpoints: int) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Calls GDELT GEO 2.1 API, returns:
      ok, status_code, parsed(json or debug)
    """
    url = "https://api.gdeltproject.org/api/v2/geo/geo"
    params = {
        "query": query,
        "format": "json",
        "mode": "pointdata",
        "formatgeo": "json",
        "maxpoints": str(maxpoints),
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "sort": "datedesc",
    }
    resp = http_get(url, params=params)
    status = resp.status_code
    try:
        data = resp.json()
        return True, status, data
    except Exception:
        head = (resp.text or "")[:200].replace("\n", " ")
        return False, status, {"non_json": True, "status": status, "head": head}

def fetch_gdelt_geo(geoms: Dict[str, Dict[str, Any]], days: int = 7, per_query_points: int = 250) -> List[Dict[str, Any]]:
    """
    Robust GDELT collection:
    - short keyword buckets
    - per country
    - GEO 2.1 pointdata
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    keyword_buckets = [
        ["protest", "demonstration", "strike", "riot", "clash", "violence"],
        ["attack", "explosion", "arson", "sabotage", "shooting", "stabbing"],
        ["border", "checkpoint", "incursion", "smuggling", "police", "arrest"],
        ["detained", "court", "military", "troops", "deployment", "exercise"],
        ["cyber", "ransomware", "ddos", "hack", "disinformation", "energy"],
        ["pipeline", "power outage", "infrastructure"],
    ]

    all_features: List[Dict[str, Any]] = []
    debug_runs: List[Dict[str, Any]] = []

    for country in CEE_COUNTRIES:
        for kw in keyword_buckets:
            # Keep query short and safe
            # Example: (Hungary) AND ("protest" OR "riot" OR ...)
            q = f'("{country}") AND (' + " OR ".join([f'"{k}"' for k in kw]) + ")"
            ok, status, data = gdelt_geo_query(q, start, end, per_query_points)

            returned = 0
            if ok and isinstance(data, dict):
                # GEO API returns "features" when formatgeo=json
                feats = data.get("features") or []
                if isinstance(feats, list):
                    returned = len(feats)
                    for f in feats:
                        geom = f.get("geometry") or {}
                        coords = geom.get("coordinates") or []
                        if len(coords) < 2:
                            continue
                        lon, lat = float(coords[0]), float(coords[1])
                        if not in_cee_countries(lon, lat, geoms):
                            continue

                        p = f.get("properties") or {}
                        title = (p.get("name") or p.get("title") or p.get("url") or "GDELT event").strip()
                        url = p.get("url") or p.get("url2") or p.get("sourceCountry")  # url usually present
                        domain = p.get("domain")
                        # GDELT geo pointdata often has "date" like 20260216000000 or similar
                        dt_iso = None
                        d = p.get("date") or p.get("seendate")
                        if d:
                            try:
                                # handle YYYYMMDDHHMMSS numeric strings
                                if isinstance(d, str) and len(d) >= 14 and d[:14].isdigit():
                                    dt = datetime.strptime(d[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                                else:
                                    dt = dateparser.parse(str(d)).astimezone(timezone.utc)
                                dt_iso = to_utc_z(dt)
                            except Exception:
                                dt_iso = None

                        all_features.append(
                            to_feature(
                                lon, lat,
                                {
                                    "source": "GDELT",
                                    "kind": "news_event",
                                    "type": "News",
                                    "title": title,
                                    "category": categorize(title),
                                    "time": dt_iso,
                                    "url": url,
                                    "domain": domain,
                                    "country_hint": country,
                                },
                            )
                        )

            debug_runs.append(
                {
                    "ok": bool(ok),
                    "status": status,
                    "query_len": len(q),
                    "returned": returned,
                    "country": country,
                    "kw": kw,
                    # keep debug small:
                    "head": None if ok else (data.get("head") if isinstance(data, dict) else None),
                }
            )

            time.sleep(SLEEP_BETWEEN_GDELT_CALLS)

    # write debug
    dbg = {
        "generated_utc": to_utc_z(datetime.now(timezone.utc)),
        "api": "geo2.1",
        "days": days,
        "per_query_points": per_query_points,
        "runs": debug_runs,
        "total_features_pre_dedup": len(all_features),
    }
    with open(GDELT_DEBUG_PATH, "w", encoding="utf-8") as f:
        json.dump(dbg, f, ensure_ascii=False, indent=2)

    return all_features

# =========================
# Scoring + hotspots + early
# =========================
def score_feature(props: Dict[str, Any]) -> float:
    src = props.get("source")
    kind = props.get("kind")
    if src == "GDELT" and kind == "news_event":
        return 1.0
    if src == "GDACS":
        return 0.5
    if src == "USGS":
        try:
            m = float(props.get("mag"))
        except Exception:
            m = 0.0
        return 0.2 + min(0.6, max(0.0, (m - 3.0) * 0.15))
    return 0.1

def time_decay(dt: Optional[datetime], now: datetime) -> float:
    if dt is None:
        return 0.6
    age_hours = (now - dt).total_seconds() / 3600.0
    return 0.5 ** (age_hours / 72.0)

def grid_key(lon: float, lat: float, cell_deg: float) -> Tuple[int, int]:
    return (int(math.floor(lon / cell_deg)), int(math.floor(lat / cell_deg)))

def cell_center(ix: int, iy: int, cell_deg: float) -> Tuple[float, float]:
    return ((ix + 0.5) * cell_deg, (iy + 0.5) * cell_deg)

def trend_from(last7: float, prev7: float) -> Tuple[str, Optional[float], str]:
    if last7 <= 0 and prev7 <= 0:
        return "na", 0.0, "·"
    if prev7 <= 0 and last7 > 0:
        return "new", None, "🆕"
    change = (last7 - prev7) / prev7 * 100.0
    if change >= 12:
        return "up", change, "🔺"
    if change <= -12:
        return "down", change, "🔻"
    return "flat", change, "▬"

def build_hotspots_with_trend(all_features: List[Dict[str, Any]], cell_deg: float = 0.5, top_n: int = 10):
    now = datetime.now(timezone.utc)
    cutoff_7 = now - timedelta(days=7)
    cutoff_14 = now - timedelta(days=14)

    acc: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for f in all_features:
        coords = (f.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        props = f.get("properties") or {}
        dt = parse_time_iso(props.get("time"))
        s = score_feature(props) * time_decay(dt, now)

        k = grid_key(lon, lat, cell_deg)
        bucket = acc.get(k)
        if bucket is None:
            acc[k] = {
                "score": 0.0,
                "count": 0,
                "sources": {"GDELT": 0, "USGS": 0, "GDACS": 0},
                "last7_score": 0.0,
                "prev7_score": 0.0,
            }
            bucket = acc[k]

        bucket["score"] += s
        bucket["count"] += 1
        src = props.get("source")
        if src in bucket["sources"]:
            bucket["sources"][src] += 1

        if dt is not None:
            if dt >= cutoff_7:
                bucket["last7_score"] += s
            elif cutoff_14 <= dt < cutoff_7:
                bucket["prev7_score"] += s

    hotspot_features: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for (ix, iy), v in acc.items():
        lon_c, lat_c = cell_center(ix, iy, cell_deg)
        last7 = float(v["last7_score"])
        prev7 = float(v["prev7_score"])
        trend_code, change_pct, arrow = trend_from(last7, prev7)

        props = {
            "type": "hotspot_cell",
            "score": round(float(v["score"]), 3),
            "count": int(v["count"]),
            "cell_deg": cell_deg,
            "sources": v["sources"],
            "last7_score": round(last7, 3),
            "prev7_score": round(prev7, 3),
            "trend": trend_code,
            "trend_arrow": arrow,
            "change_pct": None if change_pct is None else round(change_pct, 1),
        }

        hotspot_features.append(to_feature(lon_c, lat_c, props))
        rows.append({"lon": lon_c, "lat": lat_c, **props})

    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
    return hotspot_features, rows_sorted[:top_n]

# =========================
# Reverse geocode cache (top hotspots only)
# =========================
def load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def save_cache(cache: Dict[str, Any]) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def cache_key(lat: float, lon: float) -> str:
    return f"{lat:.2f},{lon:.2f}"

def reverse_geocode_osm(lat: float, lon: float, cache: Dict[str, Any]) -> str:
    k = cache_key(lat, lon)
    if k in cache:
        return cache[k]

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "jsonv2", "lat": str(lat), "lon": str(lon), "zoom": "10", "addressdetails": "1"}

    try:
        resp = http_get(url, params=params, headers={"Accept-Language": "en"})
        data = resp.json()
        addr = data.get("address") or {}

        name = (
            addr.get("county")
            or addr.get("state")
            or addr.get("municipality")
            or addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or ""
        )
        country = addr.get("country") or ""
        place = f"{name}, {country}" if name and country and country not in name else (name or country or "unknown")

        cache[k] = place
        time.sleep(1.0)
        return place
    except Exception:
        cache[k] = "unknown"
        return "unknown"

# =========================
# Weekly summary
# =========================
STOP = {"the","a","an","and","or","to","of","in","on","for","with","as","at","by","from",
        "is","are","was","were","be","been","it","this","that","these","those",
        "over","after","before","into","about","amid","during","near",
        "says","say","new","up","down"}
WORD_RE = re.compile(r"[a-zA-Z]{3,}")

def extract_topics(titles: List[str], top_k: int = 6) -> List[str]:
    freq: Dict[str, int] = {}
    for t in titles:
        for w in WORD_RE.findall((t or "").lower()):
            if w in STOP:
                continue
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def build_weekly(all_features: List[Dict[str, Any]]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff_7 = now - timedelta(days=7)

    week: List[Tuple[datetime, Dict[str, Any]]] = []
    for f in all_features:
        p = f.get("properties") or {}
        dt = parse_time_iso(p.get("time"))
        if dt is None:
            continue
        if dt >= cutoff_7:
            week.append((dt, f))

    counts = {"GDELT": 0, "USGS": 0, "GDACS": 0}
    gdelt_items: List[Tuple[datetime, Dict[str, Any]]] = []

    for dt, f in week:
        src = (f.get("properties") or {}).get("source")
        if src in counts:
            counts[src] += 1
        if src == "GDELT":
            gdelt_items.append((dt, f))

    gdelt_items.sort(key=lambda x: x[0], reverse=True)

    examples = []
    for dt, f in gdelt_items[:5]:
        p = f.get("properties") or {}
        examples.append({
            "time_utc": to_utc_z(dt),
            "title": p.get("title"),
            "url": p.get("url"),
            "domain": p.get("domain"),
            "category": p.get("category"),
        })

    topics = extract_topics([((x[1].get("properties") or {}).get("title") or "") for x in gdelt_items[:50]])

    bullets = [
        f"Híralapú jelzések (GDELT): {counts['GDELT']} db az elmúlt 7 napban.",
        f"Természeti/ellátási stresszorok: USGS {counts['USGS']} esemény, GDACS {counts['GDACS']} riasztás (CEE országok területén).",
    ]
    if topics:
        bullets.append("Gyakori témák a hírcímekben: " + ", ".join(topics) + ".")
    bullets.append("Megjegyzés: automatikus OSINT-kivonat; a linkelt források kézi ellenőrzése javasolt.")

    return {
        "generated_utc": to_utc_z(now),
        "headline": "Heti kivonat – elmúlt 7 nap",
        "bullets": bullets,
        "counts": counts,
        "examples": examples,
    }

# =========================
# Daily summary
# =========================
def pct_change(curr: float, prev: float) -> Optional[float]:
    if prev <= 0 and curr <= 0:
        return 0.0
    if prev <= 0:
        return None
    return (curr - prev) / prev * 100.0

def compute_total_score(features: List[Dict[str, Any]], now: datetime) -> float:
    total = 0.0
    for f in features:
        props = f.get("properties") or {}
        dt = parse_time_iso(props.get("time"))
        total += score_feature(props) * time_decay(dt, now)
    return total

def make_summary(all_features: List[Dict[str, Any]], counts: Dict[str, int]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff_7 = now - timedelta(days=7)
    cutoff_14 = now - timedelta(days=14)

    last7 = []
    prev7 = []
    for f in all_features:
        dt = parse_time_iso((f.get("properties") or {}).get("time"))
        if dt is None:
            continue
        if dt >= cutoff_7:
            last7.append(f)
        elif cutoff_14 <= dt < cutoff_7:
            prev7.append(f)

    score_last7 = compute_total_score(last7, now)
    score_prev7 = compute_total_score(prev7, now)
    change = pct_change(score_last7, score_prev7)

    if change is None:
        trend_text = "Trend: nincs elég bázisadat az összehasonlításhoz."
    else:
        if change > 12:
            trend_text = f"Trend: emelkedő (+{change:.0f}%) az előző 7 naphoz képest."
        elif change < -12:
            trend_text = f"Trend: csökkenő ({change:.0f}%) az előző 7 naphoz képest."
        else:
            trend_text = f"Trend: nagyjából stagnáló ({change:+.0f}%) az előző 7 naphoz képest."

    bullets = [
        trend_text,
        f"Forráskép: GDELT {counts.get('gdelt',0)}, USGS {counts.get('usgs',0)}, GDACS {counts.get('gdacs',0)}.",
        "Megjegyzés: automatikus OSINT-kivonat; a linkelt források kézi ellenőrzése javasolt.",
    ]

    return {
        "generated_utc": to_utc_z(now),
        "headline": "Közép–Kelet Európa biztonsági helyzet – napi kivonat",
        "bullets": bullets,
        "stats": {
            "score_last7": round(score_last7, 3),
            "score_prev7": round(score_prev7, 3),
            "change_pct": None if change is None else round(change, 2),
        },
    }

# =========================
# MAIN
# =========================
def main() -> int:
    ensure_dirs()

    geoms = load_or_build_country_geoms()
    ensure_cee_borders(geoms)

    # Load previous rolling layers
    prev_usgs = load_geojson_features(USGS_GEOJSON_PATH)
    prev_gdacs = load_geojson_features(GDACS_GEOJSON_PATH)
    prev_gdelt = load_geojson_features(GDELT_GEOJSON_PATH)

    # Fetch new
    try:
        usgs_new = fetch_usgs(geoms, days=USGS_DAYS, min_magnitude=2.5)
    except Exception as e:
        print(f"[USGS] fetch failed: {e}")
        usgs_new = []

    try:
        gdacs_new = fetch_gdacs(geoms, days=GDACS_DAYS)
    except Exception as e:
        print(f"[GDACS] fetch failed: {e}")
        gdacs_new = []

    try:
        gdelt_new = fetch_gdelt_geo(geoms, days=GDELT_DAYS, per_query_points=GDELT_MAXPOINTS)
    except Exception as e:
        print(f"[GDELT] fetch failed: {e}")
        gdelt_new = []

    # Merge rolling + trim
    usgs_merged = merge_dedup(clamp_times(prev_usgs), clamp_times(usgs_new))
    gdacs_merged = merge_dedup(clamp_times(prev_gdacs), clamp_times(gdacs_new))
    gdelt_merged = merge_dedup(clamp_times(prev_gdelt), clamp_times(gdelt_new))

    usgs = trim_by_days(usgs_merged, keep_days=ROLLING_DAYS)
    gdacs = trim_by_days(gdacs_merged, keep_days=ROLLING_DAYS)
    gdelt = trim_by_days(gdelt_merged, keep_days=ROLLING_DAYS)

    # Save layers
    save_geojson(USGS_GEOJSON_PATH, usgs)
    save_geojson(GDACS_GEOJSON_PATH, gdacs)
    save_geojson(GDELT_GEOJSON_PATH, gdelt)

    all_feats = gdelt + gdacs + usgs

    # Hotspots
    hotspot_geo, top_hotspots = build_hotspots_with_trend(all_feats, cell_deg=0.5, top_n=10)

    # reverse geocode top hotspots (optional)
    cache = load_cache()
    for h in top_hotspots:
        h["place"] = reverse_geocode_osm(float(h["lat"]), float(h["lon"]), cache)
    save_cache(cache)

    save_geojson(HOTSPOTS_GEOJSON_PATH, hotspot_geo)
    with open(HOTSPOTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"generated_utc": to_utc_z(datetime.now(timezone.utc)), "top": top_hotspots}, f, ensure_ascii=False, indent=2)

    # Summaries + meta
    counts = {"usgs": len(usgs), "gdacs": len(gdacs), "gdelt": len(gdelt), "hotspot_cells": len(hotspot_geo)}
    summary = make_summary(all_feats, counts)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    weekly = build_weekly(all_feats)
    with open(WEEKLY_PATH, "w", encoding="utf-8") as f:
        json.dump(weekly, f, ensure_ascii=False, indent=2)

    meta = {
        "generated_utc": to_utc_z(datetime.now(timezone.utc)),
        "counts": counts,
        "rolling_days": ROLLING_DAYS,
        "countries": CEE_COUNTRIES,
        "gdelt": {"api": "geo2.1", "days": GDELT_DAYS, "maxpoints": GDELT_MAXPOINTS},
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
