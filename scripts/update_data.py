#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

import requests
from dateutil import parser as dateparser

# ============================================================
# PATHS
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_PATH = os.path.join(DATA_DIR, "geocode_cache.json")
COUNTRIES_CACHE_PATH = os.path.join(DATA_DIR, "cee_countries.geojson")

USER_AGENT = "cee-security-map/3.0 (github actions)"
TIMEOUT = 30

# ============================================================
# REGION: 8 countries (CEE)
# ============================================================
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

# rough pre-filter (final filter is point-in-polygon)
CEE_BBOX = (11.0, 42.5, 30.5, 61.5)

ROLLING_DAYS = 7
GDELT_GEO_DAYS = 7     # GEO API window
GDELT_EXPORT_DAYS = 14 # linkes események (export) window
USGS_DAYS = 7
GDACS_DAYS = 14  # RSS window; trim to 7

# ============================================================
# GDELT EXPORT SETTINGS (linkes események)
# ============================================================
MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
MAX_SOURCES_PER_EVENT = 8

# ============================================================
# CATEGORY MODEL (amit kértél)
# ============================================================
# A kategóriát két forrásból tudjuk:
#  - GEO API: a lekérdezés "keyword bucket" alapján adjuk (stabil, de nem mindig cikklinkes)
#  - EXPORT CSV: a CAMEO root/event alapján + fallback kulcsszó (stabil, és van SOURCEURL)
CATEGORY_BUCKETS = [
    ("protest", ["protest", "demonstration", "strike", "riot", "clash", "violence"]),
    ("border", ["border", "checkpoint", "incursion", "smuggling", "migration", "asylum"]),
    ("police", ["police", "arrest", "detained", "court", "raid", "prosecutor"]),
    ("military", ["military", "troops", "deployment", "exercise", "drill", "mobilization"]),
    ("drone", ["drone", "uav", "unmanned", "quadcop", "shahed"]),
    ("cyber", ["cyber", "ransomware", "ddos", "hack", "malware", "disinformation"]),
    ("energy", ["energy", "pipeline", "power outage", "electricity", "grid", "gas"]),
    ("infrastructure", ["infrastructure", "rail", "bridge", "port", "airport"]),
    ("security_politics", ["security", "intelligence", "sanctions", "terror", "extremism"]),
]

# CAMEO root (28. oszlop a GDELT exportban): durva megfeleltetés
# Megjegyzés: ez nem tökéletes – de stabilabb, mint a semmi.
CAMEO_ROOT_TO_CAT = {
    # erőszak/konfliktus
    "18": "violence",
    "19": "violence",
    "20": "violence",
    # jog/állam, rendvédelem (néhány root)
    "14": "police",
    "15": "police",
    # diplomácia/államközi
    "01": "security_politics",
    "02": "security_politics",
    # gazdaság/energia jellegűt nem mindig ad root, ezt kulcsszóval erősítjük
}

# ============================================================
# EARLY WARNING zones (optional)
# ============================================================
SENSITIVE_ZONES = [
    {"name": "PL–BY / Suwałki környék (tág)", "bbox": (22.0, 53.5, 24.5, 55.8), "mult": 1.20},
    {"name": "RO–MD határ (tág)", "bbox": (26.0, 45.0, 29.5, 48.2), "mult": 1.15},
    {"name": "HU–UA perem (tág)", "bbox": (21.0, 47.5, 23.5, 49.3), "mult": 1.12},
]

# ============================================================
# Basics
# ============================================================
def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
    h = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        h.update(headers)
    backoff = 2
    last_exc: Optional[Exception] = None
    for _ in range(1, 4):
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

def http_get_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")

def http_get_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as r:
        return r.read()

def in_bbox(lon: float, lat: float, bbox: Tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)

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

# ============================================================
# Time helpers
# ============================================================
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

# ============================================================
# Dedup
# ============================================================
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

# ============================================================
# Point-in-polygon (no shapely)
# ============================================================
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
    """
    country_name -> geometry (Polygon/MultiPolygon), cached weekly to data/cee_countries.geojson
    """
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
    if not in_bbox(lon, lat, CEE_BBOX):
        return False
    for name in CEE_COUNTRIES:
        geom = geoms.get(name)
        if geom and point_in_feature(lon, lat, geom):
            return True
    return False

# ============================================================
# Borders layer
# ============================================================
def ensure_cee_borders(geoms: Dict[str, Dict[str, Any]]) -> None:
    out_path = os.path.join(DATA_DIR, "cee_borders.geojson")
    feats = []
    for name in CEE_COUNTRIES:
        if name in geoms:
            feats.append({
                "type": "Feature",
                "properties": {"name": name},
                "geometry": geoms[name],
            })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False, indent=2)

# ============================================================
# Reverse geocode cache (top hotspots only)
# ============================================================
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
    params = {
        "format": "jsonv2",
        "lat": str(lat),
        "lon": str(lon),
        "zoom": "10",
        "addressdetails": "1",
    }

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

# ============================================================
# Sources: USGS / GDACS
# ============================================================
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
                    "category": "natural",
                    "mag": p.get("mag"),
                    "place": p.get("place"),
                    "time": to_utc_z(dt) if dt else None,
                    "url": p.get("url"),
                    "title": p.get("title"),
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
        title = get_tag(chunk, "title")
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
                    "category": "natural",
                    "title": title,
                    "time": to_utc_z(pub_dt),
                    "url": link,
                    "type": "Alert",
                },
            )
        )
    return out

# ============================================================
# GDELT GEO (sok pont, link nem mindig)
# ============================================================
def gdelt_geo_query(country: str, kw: List[str], days: int, maxpoints: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Uses GDELT 2.1 GEO API:
      https://api.gdeltproject.org/api/v2/geo/geo
    Returns (features, debug)
    """
    url = "https://api.gdeltproject.org/api/v2/geo/geo"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # Query: keyword OR keyword ... AND country
    # Keep it SHORT (GDELT errors if too long).
    kwq = " OR ".join([f'"{k}"' if " " in k else k for k in kw])
    q = f"({kwq}) AND sourceCountry:{country}"

    params = {
        "query": q,
        "format": "json",
        "mode": "GeoJSON",
        "formatgeojson": "1",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "maxpoints": str(maxpoints),
        "timelinesmooth": "0",
    }

    resp = http_get(url, params=params)
    txt = resp.text or ""
    # GEO API néha visszaad HTML-t hibánál, ezért védjük
    try:
        data = resp.json()
    except Exception:
        return [], {"ok": False, "status": resp.status_code, "non_json": True, "head": txt[:120]}

    feats = data.get("features") or []
    # készítünk egy "fallback" kereső-linket (legalább legyen hova kattintani)
    # Ez nem a konkrét cikk URL, hanem a GDELT DOC API keresés.
    search_url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        + "mode=ArtList&format=html&query="
        + requests.utils.quote(q)
    )

    out = []
    for f in feats:
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        props = f.get("properties") or {}

        # GEO API-nál a pontos idő sokszor nincs; ha nincs, adunk "now"-t (de ettől még dedup működik URL alapján)
        t = props.get("date") or props.get("datetime") or props.get("time") or None
        dt = parse_time_iso(t) if t else None
        if dt is None:
            dt = datetime.now(timezone.utc)

        # A GEO API nem garantál cikklinket, de néha ad; ha nem, adjunk "search_url"-t
        url_guess = props.get("url") or props.get("sourceUrl") or None

        out.append(
            to_feature(
                lon, lat,
                {
                    "source": "GDELT",
                    "kind": "news_geo",
                    "type": "News",
                    "time": to_utc_z(dt),
                    "title": props.get("name") or props.get("title") or f"{country}",
                    "url": url_guess,  # lehet None
                    "search_url": search_url,  # mindig van
                    "country_hint": country,
                    "category": None,  # később a bucket név lesz
                    "gdelt_bucket": None,
                },
            )
        )

    dbg = {
        "ok": True,
        "status": resp.status_code,
        "query_len": len(q),
        "returned": len(out),
        "country": country,
        "kw": kw,
    }
    return out, dbg

def fetch_gdelt_geo(geoms: Dict[str, Dict[str, Any]], days: int = 7, maxpoints_per_query: int = 250) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Many points, not always linkable.
    We'll run multiple short queries (country x bucket) to avoid "query too long".
    """
    all_out: List[Dict[str, Any]] = []
    dbg_runs: List[Dict[str, Any]] = []

    # GDELT GEO API expects sourceCountry as 2-letter? In practice it supports country names sometimes inconsistently.
    # We use "countryName" mapping: best working in many cases is "Hungary" etc, but if you want, later we can map to ISO2.
    # For now, keep as you already tested successfully.
    for country in CEE_COUNTRIES:
        for bucket_name, kw in CATEGORY_BUCKETS:
            feats, dbg = gdelt_geo_query(country=country, kw=kw, days=days, maxpoints=maxpoints_per_query)
            dbg["bucket"] = bucket_name
            dbg_runs.append(dbg)

            # filter to our polygons (geo api can leak)
            for f in feats:
                coords = (f.get("geometry") or {}).get("coordinates") or []
                if len(coords) < 2:
                    continue
                lon, lat = float(coords[0]), float(coords[1])
                if not in_cee_countries(lon, lat, geoms):
                    continue
                p = f.get("properties") or {}
                p["category"] = bucket_name
                p["gdelt_bucket"] = bucket_name
                f["properties"] = p
                all_out.append(f)

    # lightweight dedup by url/search_url + time+title
    all_out = merge_dedup([], clamp_times(all_out))
    all_out = trim_by_days(all_out, keep_days=ROLLING_DAYS)

    debug = {
        "generated_utc": to_utc_z(datetime.now(timezone.utc)),
        "api": "geo",
        "days": days,
        "per_query_points": maxpoints_per_query,
        "runs": dbg_runs,
    }
    return all_out, debug

# ============================================================
# GDELT EXPORT (linkes események)
# ============================================================
def parse_masterfilelist(master_text: str) -> List[str]:
    urls = []
    for line in master_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        url = parts[2].strip()
        # normalize https->http to avoid occasional SSL issues in GH Actions
        if url.startswith("https://data.gdeltproject.org/"):
            url = "http://data.gdeltproject.org/" + url[len("https://data.gdeltproject.org/"):]
        if url.endswith(".export.CSV.zip") and "/gdeltv2/" in url:
            urls.append(url)
    return urls

def extract_timestamp_from_url(url: str) -> Optional[datetime]:
    base = url.split("/")[-1]
    ts = base.split(".")[0]
    if len(ts) != 14 or not ts.isdigit():
        return None
    return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

def yyyymmdd_to_iso(s: str) -> str:
    if not s or len(s) != 8:
        return ""
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"

def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def norm_loc(s: str) -> str:
    if not s:
        return "unknown"
    return " ".join(s.strip().lower().split())

def add_unique(lst: List[str], url: str) -> None:
    if not url:
        return
    if url not in lst and len(lst) < MAX_SOURCES_PER_EVENT:
        lst.append(url)

def classify_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    for bucket_name, kw in CATEGORY_BUCKETS:
        for k in kw:
            if k.lower() in t:
                return bucket_name
    return None

def fetch_gdelt_export_linked(geoms: Dict[str, Dict[str, Any]], lookback_days: int = 14) -> List[Dict[str, Any]]:
    """
    Pull GDELT v2 export CSV ZIP files for the last N days.
    Filter to our 8 countries by point-in-polygon.
    Produce aggregated events with multiple source URLs.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    master = http_get_text(MASTERFILELIST_URL)
    urls = parse_masterfilelist(master)

    recent = []
    for u in urls:
        ts = extract_timestamp_from_url(u)
        if ts and ts >= cutoff:
            recent.append((ts, u))
    recent.sort(key=lambda x: x[0])

    if not recent:
        return []

    live_agg: Dict[str, Dict[str, Any]] = {}
    rows = 0

    for ts, url in recent:
        try:
            zbytes = http_get_bytes(url)
            zf = zipfile.ZipFile(io.BytesIO(zbytes))
            name = zf.namelist()[0]
            raw = zf.read(name).decode("utf-8", errors="replace")
        except Exception:
            continue

        reader = csv.reader(io.StringIO(raw), delimiter="\t")
        for row in reader:
            rows += 1
            if len(row) < 61:
                continue

            # GDELT 2.1 Events export columns:
            # 0 GlobalEventID
            # 1 SQLDATE (yyyymmdd)
            # 26 EventCode
            # 28 EventRootCode
            # 52 ActionGeo_FullName (location string)
            # 56 ActionGeo_Lat
            # 57 ActionGeo_Long
            # 60 SOURCEURL
            gid = str(row[0]).strip()
            day = str(row[1]).strip()
            event_code = str(row[26]).strip()
            root = str(row[28]).strip()
            fullname = str(row[52]).strip()
            lat = safe_float(str(row[56]).strip())
            lon = safe_float(str(row[57]).strip())
            sourceurl = str(row[60]).strip()

            if lat is None or lon is None:
                continue
            if not in_cee_countries(lon, lat, geoms):
                continue

            date_iso = yyyymmdd_to_iso(day)
            if not date_iso:
                continue

            # category: root->cat + keyword fallback
            category = CAMEO_ROOT_TO_CAT.get(root)
            if category is None:
                category = classify_from_text(fullname) or "other"

            loc_norm = norm_loc(fullname)
            key = f"{date_iso}|{category}|{loc_norm}"

            if key not in live_agg:
                live_agg[key] = {
                    "date": date_iso,
                    "time": f"{date_iso}T00:00:00Z",
                    "category": category,
                    "event_root_code": root,
                    "event_codes": set([event_code]) if event_code else set(),
                    "gdelt_ids": set([gid]) if gid else set(),
                    "location": fullname or "unknown",
                    "loc_norm": loc_norm,
                    "lat_sum": lat,
                    "lon_sum": lon,
                    "n": 1,
                    "sources": [sourceurl] if sourceurl else [],
                }
            else:
                ev = live_agg[key]
                ev["lat_sum"] += lat
                ev["lon_sum"] += lon
                ev["n"] += 1
                if fullname and ev["location"] == "unknown":
                    ev["location"] = fullname
                if event_code:
                    ev["event_codes"].add(event_code)
                if gid:
                    ev["gdelt_ids"].add(gid)
                add_unique(ev["sources"], sourceurl)

    live_features: List[Dict[str, Any]] = []
    for ev in live_agg.values():
        lat = ev["lat_sum"] / max(1, ev["n"])
        lon = ev["lon_sum"] / max(1, ev["n"])
        live_features.append(
            to_feature(
                lon, lat,
                {
                    "source": "GDELT",
                    "kind": "news_linked",
                    "type": "News",
                    "time": ev["time"],
                    "date": ev["date"],
                    "title": ev["location"],  # headline-like surrogate; for real headline, DOC api needed
                    "location": ev["location"],
                    "category": ev["category"],
                    "event_root_code": ev["event_root_code"],
                    "event_codes": sorted([c for c in ev["event_codes"] if c]),
                    "gdelt_ids_count": len(ev["gdelt_ids"]),
                    "sources_count": len(ev["sources"]),
                    "sources": ev["sources"],  # list of URLs
                    # convenience field for popup:
                    "url": ev["sources"][0] if ev["sources"] else None,
                },
            )
        )

    # newest first
    live_features.sort(key=lambda f: (f.get("properties", {}).get("date", ""), f.get("properties", {}).get("sources_count", 0)), reverse=True)
    return live_features

# ============================================================
# Scoring + hotspots + trend
# ============================================================
def score_feature(props: Dict[str, Any]) -> float:
    src = props.get("source")
    kind = props.get("kind")
    if src == "GDELT" and kind in ("news_linked",):
        return 1.3
    if src == "GDELT" and kind in ("news_geo", "news_event"):
        return 1.0
    if src == "GDACS":
        return 0.6
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

# ============================================================
# EARLY WARNING
# ============================================================
def zone_multiplier(lon: float, lat: float) -> Tuple[float, Optional[str]]:
    mult = 1.0
    zname: Optional[str] = None
    for z in SENSITIVE_ZONES:
        if in_bbox(lon, lat, z["bbox"]):
            if z["mult"] > mult:
                mult = float(z["mult"])
                zname = str(z["name"])
    return mult, zname

def neighbor_keys(k: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = k
    return [(x-1,y-1),(x,y-1),(x+1,y-1),(x-1,y),(x+1,y),(x-1,y+1),(x,y+1),(x+1,y+1)]

def build_early_warning(all_features: List[Dict[str, Any]], cell_deg: float = 0.5, lookback_days: int = 7, recent_hours: int = 48, top_n: int = 10):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    recent_cut = now - timedelta(hours=recent_hours)

    acc: Dict[Tuple[int,int], Dict[str, Any]] = {}

    def get_bucket(k: Tuple[int,int]) -> Dict[str, Any]:
        b = acc.get(k)
        if b is None:
            b = {
                "recent": 0.0,
                "baseline": 0.0,
                "src_recent": {"GDELT": 0, "USGS": 0, "GDACS": 0},
            }
            acc[k] = b
        return b

    for f in all_features:
        coords = (f.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        props = f.get("properties") or {}
        dt = parse_time_iso(props.get("time"))
        if dt is None or dt < cutoff:
            continue

        s = score_feature(props)
        k = grid_key(lon, lat, cell_deg)
        b = get_bucket(k)

        src = props.get("source")
        if dt >= recent_cut:
            b["recent"] += s
            if src in b["src_recent"]:
                b["src_recent"][src] += 1
        else:
            b["baseline"] += s

    raw: Dict[Tuple[int,int], float] = {}
    meta: Dict[Tuple[int,int], Dict[str, Any]] = {}

    for k, b in acc.items():
        lon_c, lat_c = cell_center(k[0], k[1], cell_deg)
        recent = float(b["recent"])
        base = float(b["baseline"])
        if recent <= 0.75:
            continue

        ratio = (recent + 0.5) / (base + 1.5)
        src_mix = sum(1 for v in b["src_recent"].values() if v > 0)
        mix_boost = 1.0 + 0.08 * (src_mix - 1) if src_mix >= 2 else 1.0
        z_mult, z_name = zone_multiplier(lon_c, lat_c)
        esc = (recent * 10.0) * math.log1p(ratio) * mix_boost * z_mult

        raw[k] = esc
        meta[k] = {
            "recent": round(recent, 3),
            "baseline": round(base, 3),
            "ratio": round(ratio, 3),
            "src_mix": int(src_mix),
            "zone": z_name,
            "zone_mult": round(z_mult, 2),
            "src_recent": b["src_recent"],
        }

    if not raw:
        return [], []

    max_raw = max(raw.values()) or 1.0

    signals: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for k, esc_raw in raw.items():
        neigh = neighbor_keys(k)
        neigh_active = sum(1 for nk in neigh if nk in raw and raw[nk] >= 0.35 * max_raw)
        spread_boost = 1.0 + 0.06 * neigh_active
        score0_100 = min(100.0, (esc_raw * spread_boost) / max_raw * 100.0)

        lon_c, lat_c = cell_center(k[0], k[1], cell_deg)
        props = {
            "type": "early_warning",
            "escalation": round(score0_100, 1),
            "cell_deg": cell_deg,
            "neighbor_active": int(neigh_active),
            **meta[k],
        }
        signals.append(to_feature(lon_c, lat_c, props))
        rows.append({"lon": lon_c, "lat": lat_c, **props})

    rows_sorted = sorted(rows, key=lambda x: x["escalation"], reverse=True)
    return signals, rows_sorted[:top_n]

# ============================================================
# Weekly topics (simple)
# ============================================================
STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","as","at","by","from",
    "is","are","was","were","be","been","it","this","that","these","those",
    "over","after","before","into","about","amid","during","near",
    "says","say","new","up","down",
    "hungary","poland","czech","slovak","romania","latvia","lithuania","estonia"
}
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

    week = []
    for f in all_features:
        p = f.get("properties") or {}
        dt = parse_time_iso(p.get("time"))
        if dt and dt >= cutoff_7:
            week.append((dt, f))

    counts = {"GDELT": 0, "USGS": 0, "GDACS": 0}
    titles = []

    for dt, f in week:
        src = (f.get("properties") or {}).get("source")
        if src in counts:
            counts[src] += 1
        if src == "GDELT":
            titles.append(((f.get("properties") or {}).get("title") or ""))

    topics = extract_topics(titles[:80])

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
    }

# ============================================================
# Daily summary + alert
# ============================================================
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

def alert_from_top(top: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not top:
        return None
    arrow = top.get("trend_arrow")
    ch = top.get("change_pct")
    place = top.get("place") or "ismeretlen térség"

    if arrow == "🆕":
        return {"level": "info", "title": "Új góc", "text": f"Új hotspot jelent meg: {place}. Érdemes követni 24–72 órában."}
    if arrow == "🔺":
        if ch is not None and ch >= 25:
            return {"level": "high", "title": "Emelkedő feszültség", "text": f"Erősödő hotspot: {place} (+{ch:.0f}%)."}
        return {"level": "medium", "title": "Emelkedő feszültség", "text": f"Felfutó jelzések: {place}."}
    return None

def make_summary(all_features: List[Dict[str, Any]], top_hotspots: List[Dict[str, Any]], counts: Dict[str, int]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff_7 = now - timedelta(days=7)
    cutoff_14 = now - timedelta(days=14)

    last7, prev7 = [], []
    for f in all_features:
        dt = parse_time_iso((f.get("properties") or {}).get("time"))
        if not dt:
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

    top = top_hotspots[0] if top_hotspots else None
    if top:
        place = top.get("place") or "ismeretlen térség"
        arrow = top.get("trend_arrow", "")
        chv = top.get("change_pct")
        ch_txt = "n/a" if chv is None else f"{chv:+.0f}%"
        top_text = (
            f"Legerősebb góc: {place} {arrow} "
            f"(rácspont {top['lat']:.2f}, {top['lon']:.2f}; score {float(top['score']):.2f}; 7 napos változás: {ch_txt})."
        )
    else:
        top_text = "Legerősebb góc: jelenleg nincs elég geokódolt jelzés a térképes kiemeléshez."

    bullets = [
        top_text,
        trend_text,
        f"Forráskép: GDELT {counts.get('gdelt',0)}, USGS {counts.get('usgs',0)}, GDACS {counts.get('gdacs',0)}.",
        "Megjegyzés: automatikus OSINT-kivonat; a linkelt források kézi ellenőrzése javasolt.",
    ]

    return {
        "generated_utc": to_utc_z(now),
        "headline": "Közép–Kelet Európa biztonsági helyzet – napi kivonat",
        "bullets": bullets,
        "alert": alert_from_top(top),
        "stats": {
            "score_last7": round(score_last7, 3),
            "score_prev7": round(score_prev7, 3),
            "change_pct": None if change is None else round(change, 2),
        },
    }

# ============================================================
# MAIN
# ============================================================
def main() -> int:
    ensure_dirs()

    # country geometries
    geoms = load_or_build_country_geoms()

    # borders
    ensure_cee_borders(geoms)

    # Load previous rolling layers
    prev_usgs = load_geojson_features(os.path.join(DATA_DIR, "usgs.geojson"))
    prev_gdacs = load_geojson_features(os.path.join(DATA_DIR, "gdacs.geojson"))
    prev_gdelt = load_geojson_features(os.path.join(DATA_DIR, "gdelt.geojson"))
    prev_gdelt_linked = load_geojson_features(os.path.join(DATA_DIR, "gdelt_linked.geojson"))

    # Fetch new: USGS / GDACS
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

    # Fetch new: GDELT GEO (many points) + debug
    try:
        gdelt_geo_new, gdelt_debug = fetch_gdelt_geo(geoms, days=GDELT_GEO_DAYS, maxpoints_per_query=250)
    except Exception as e:
        print(f"[GDELT GEO] fetch failed: {e}")
        gdelt_geo_new, gdelt_debug = [], {"ok": False, "error": str(e)}

    # Fetch new: GDELT EXPORT (linked events)
    try:
        gdelt_linked_new = fetch_gdelt_export_linked(geoms, lookback_days=GDELT_EXPORT_DAYS)
    except Exception as e:
        print(f"[GDELT EXPORT] fetch failed: {e}")
        gdelt_linked_new = []

    # Merge rolling + trim
    usgs_merged = merge_dedup(clamp_times(prev_usgs), clamp_times(usgs_new))
    gdacs_merged = merge_dedup(clamp_times(prev_gdacs), clamp_times(gdacs_new))
    gdelt_merged = merge_dedup(clamp_times(prev_gdelt), clamp_times(gdelt_geo_new))
    gdelt_linked_merged = merge_dedup(clamp_times(prev_gdelt_linked), clamp_times(gdelt_linked_new))

    usgs = trim_by_days(usgs_merged, keep_days=ROLLING_DAYS)
    gdacs = trim_by_days(gdacs_merged, keep_days=ROLLING_DAYS)
    gdelt = trim_by_days(gdelt_merged, keep_days=ROLLING_DAYS)
    gdelt_linked = trim_by_days(gdelt_linked_merged, keep_days=GDELT_EXPORT_DAYS)

    # Save source layers
    save_geojson(os.path.join(DATA_DIR, "usgs.geojson"), usgs)
    save_geojson(os.path.join(DATA_DIR, "gdacs.geojson"), gdacs)
    save_geojson(os.path.join(DATA_DIR, "gdelt.geojson"), gdelt)  # GEO points
    save_geojson(os.path.join(DATA_DIR, "gdelt_linked.geojson"), gdelt_linked)  # EXPORT linked events

    # Save debug
    with open(os.path.join(DATA_DIR, "gdelt_debug.json"), "w", encoding="utf-8") as f:
        json.dump(gdelt_debug, f, ensure_ascii=False, indent=2)

    # Hotspots: combine all
    all_feats = gdelt + gdelt_linked + gdacs + usgs

    hotspot_geo, top_hotspots = build_hotspots_with_trend(all_feats, cell_deg=0.5, top_n=10)

    cache = load_cache()
    for h in top_hotspots:
        h["place"] = reverse_geocode_osm(float(h["lat"]), float(h["lon"]), cache)
    save_cache(cache)

    save_geojson(os.path.join(DATA_DIR, "hotspots.geojson"), hotspot_geo)
    with open(os.path.join(DATA_DIR, "hotspots.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_utc": to_utc_z(datetime.now(timezone.utc)), "top": top_hotspots}, f, ensure_ascii=False, indent=2)

    # Early
    early_geo, early_top = build_early_warning(all_feats, cell_deg=0.5, lookback_days=7, recent_hours=48, top_n=10)
    cache = load_cache()
    for e in early_top:
        e["place"] = reverse_geocode_osm(float(e["lat"]), float(e["lon"]), cache)
    save_cache(cache)

    save_geojson(os.path.join(DATA_DIR, "early.geojson"), early_geo)
    with open(os.path.join(DATA_DIR, "early.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_utc": to_utc_z(datetime.now(timezone.utc)), "top": early_top}, f, ensure_ascii=False, indent=2)

    # Summaries + meta
    counts = {
        "usgs": len(usgs),
        "gdacs": len(gdacs),
        "gdelt": len(gdelt),
        "gdelt_linked": len(gdelt_linked),
        "hotspot_cells": len(hotspot_geo),
    }
    summary = make_summary(all_feats, top_hotspots, counts)
    with open(os.path.join(DATA_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    weekly = build_weekly(all_feats)
    with open(os.path.join(DATA_DIR, "weekly.json"), "w", encoding="utf-8") as f:
        json.dump(weekly, f, ensure_ascii=False, indent=2)

    meta = {
        "generated_utc": to_utc_z(datetime.now(timezone.utc)),
        "counts": counts,
        "rolling_days": ROLLING_DAYS,
        "countries": CEE_COUNTRIES,
        "gdelt": {
            "geo_days": GDELT_GEO_DAYS,
            "export_days": GDELT_EXPORT_DAYS,
        }
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
