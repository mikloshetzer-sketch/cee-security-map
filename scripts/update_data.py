#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
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
CACHE_PATH = os.path.join(DATA_DIR, "geocode_cache.json")
COUNTRIES_CACHE_PATH = os.path.join(DATA_DIR, "cee_countries.geojson")

USER_AGENT = "cee-security-map/2.4 (github actions)"
REQ_TIMEOUT = 40

# =========================
# CEE COUNTRIES
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

# ISO2 a GDELT query-hez (sourceCountry)
ISO2 = {
    "Hungary": "HU",
    "Poland": "PL",
    "Czech Republic": "CZ",
    "Slovakia": "SK",
    "Romania": "RO",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Estonia": "EE",
}

CEE_BBOX = (11.0, 42.5, 30.5, 61.5)

# =========================
# WINDOWS
# =========================
GDELT_DAYS_ON_MAP = 14       # térkép pontok
ROLLING_DAYS = 7             # hotspot/summary/early
USGS_DAYS = 7
GDACS_DAYS = 14

# =========================
# GDELT (API)
# =========================
# A GDELT Geo 2.1 endpoint (GeoJSON pontokat ad)
GDELT_GEO_URL = "https://api.gdeltproject.org/api/v2/geo/geo"

# Egy query-ből mennyi pontot kérünk
PER_QUERY_POINTS = 200

# Kategóriák (amit kértél)
CATEGORY_QUERIES = {
    # tüntetés / sztrájk / zavargás
    "protest": [
        "protest", "demonstration", "strike", "riot", "clash"
    ],
    # katonai
    "military": [
        "military", "troops", "deployment", "exercise", "mobilization"
    ],
    # határ / migráció / csempészet / incursion
    "border": [
        "border", "checkpoint", "incursion", "smuggling", "migrants"
    ],
    # kiber + dezinformáció
    "cyber": [
        "cyber", "ransomware", "ddos", "hack", "disinformation"
    ],
    # energia / infrastruktúra
    "energy": [
        "pipeline", "power outage", "substation", "grid", "infrastructure"
    ],
    # drón
    "drone": [
        "drone", "uav", "quadrocopter", "unmanned aerial"
    ],
    # security politics (tágabb: terror/attack/assault/plot/weapon stb.)
    "security_politics": [
        "attack", "explosion", "arson", "sabotage", "terror"
    ],
}

CATEGORY_HU = {
    "protest": "tüntetés/szociális",
    "military": "katonai",
    "border": "határ/biztonság",
    "cyber": "kiber/dezinformáció",
    "energy": "energia/infrastruktúra",
    "drone": "drón/UAV",
    "security_politics": "biztonságpolitika",
    "disaster": "katasztrófa/természeti",
    "other": "egyéb",
}

# =========================
# EARLY WARNING zones (optional)
# =========================
SENSITIVE_ZONES = [
    {"name": "PL–BY / Suwałki környék (tág)", "bbox": (22.0, 53.5, 24.5, 55.8), "mult": 1.20},
    {"name": "RO–MD határ (tág)", "bbox": (26.0, 45.0, 29.5, 48.2), "mult": 1.15},
    {"name": "HU–UA perem (tág)", "bbox": (21.0, 47.5, 23.5, 49.3), "mult": 1.12},
]

# =========================
# Utils
# =========================
def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

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

def in_bbox(lon: float, lat: float, bbox: Tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)

def to_feature(lon: float, lat: float, props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props}

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

def dedup_key(feature: Dict[str, Any]) -> Optional[str]:
    p = feature.get("properties") or {}
    src = p.get("source") or ""
    url = p.get("url")
    gid = p.get("gdelt_id")
    if gid:
        return f"{src}|{gid}"
    if url:
        return f"{src}|{url}"
    title = p.get("title")
    tm = p.get("time")
    if title and tm:
        return f"{src}|{tm}|{title}"
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

def trim_by_days(features: List[Dict[str, Any]], keep_days: int) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=keep_days)
    kept = []
    for f in features:
        dt = parse_time_iso((f.get("properties") or {}).get("time"))
        if dt and dt >= cutoff:
            kept.append(f)
    return kept

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
        return any(point_in_polygon(lon, lat, poly) for poly in coords)
    return False

def http_get(url: str, params: Optional[dict] = None) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    r = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r

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

def ensure_cee_borders(geoms: Dict[str, Dict[str, Any]]) -> None:
    out_path = os.path.join(DATA_DIR, "cee_borders.geojson")
    feats = []
    for name in CEE_COUNTRIES:
        if name in geoms:
            feats.append({"type": "Feature", "properties": {"name": name}, "geometry": geoms[name]})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False, indent=2)

def in_cee_countries(lon: float, lat: float, geoms: Dict[str, Dict[str, Any]]) -> bool:
    if not in_bbox(lon, lat, CEE_BBOX):
        return False
    for name in CEE_COUNTRIES:
        geom = geoms.get(name)
        if geom and point_in_feature(lon, lat, geom):
            return True
    return False

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
        resp = http_get(url, params=params)
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
# USGS / GDACS (unchanged style)
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
                    "category": "disaster",
                    "mag": p.get("mag"),
                    "place": p.get("place"),
                    "time": to_utc_z(dt) if dt else None,
                    "url": p.get("url"),
                    "title": p.get("title") or "Earthquake",
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
                    "category": "disaster",
                    "title": title or "GDACS Alert",
                    "time": to_utc_z(pub_dt),
                    "url": link,
                    "type": "Alert",
                },
            )
        )
    return out

# =========================
# GDELT GEO by category
# =========================
def gdelt_query(country_iso2: str, keywords: List[str]) -> str:
    # rövid query: sourceCountry + (kw OR kw OR ...)
    # (A “Your query too short or too long” hibát így elkerüljük.)
    kw = " OR ".join([f'"{k}"' if " " in k else k for k in keywords])
    return f"sourceCountry:{country_iso2} ({kw})"

def fetch_gdelt_geo(geoms: Dict[str, Dict[str, Any]], days: int, per_query_points: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    debug = {"generated_utc": to_utc_z(now), "api": "geo", "days": days, "per_query_points": per_query_points, "runs": []}
    out: List[Dict[str, Any]] = []
    seen = set()

    for country in CEE_COUNTRIES:
        iso2 = ISO2[country]
        for category, kws in CATEGORY_QUERIES.items():
            q = gdelt_query(iso2, kws)
            params = {
                "query": q,
                "format": "geojson",
                "mode": "timelinevolraw",
                "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                "enddatetime": now.strftime("%Y%m%d%H%M%S"),
                "maxpoints": str(per_query_points),
            }

            ok = True
            status = 0
            returned = 0
            try:
                r = http_get(GDELT_GEO_URL, params=params)
                status = r.status_code
                data = r.json()
                feats = (data.get("features") or [])
                returned = len(feats)
            except Exception as e:
                ok = False
                feats = []

            debug["runs"].append({"ok": ok, "status": status, "query_len": len(q), "returned": returned, "country": country, "kw": kws})

            # GeoJSON pontok → egységes properties
            for f in feats:
                geom = f.get("geometry") or {}
                coords = geom.get("coordinates") or []
                if len(coords) < 2:
                    continue
                lon, lat = float(coords[0]), float(coords[1])

                # szigorú 8 ország poligon szűrés
                if not in_cee_countries(lon, lat, geoms):
                    continue

                p = f.get("properties") or {}
                # GDELT geo API property nevei változhatnak – több mezőt próbálunk:
                url = p.get("url") or p.get("sourceUrl") or p.get("SOURCEURL") or p.get("SourceURL")
                title = p.get("title") or p.get("name") or p.get("NAME") or p.get("summary") or country
                seendate = p.get("seendate") or p.get("SEENDATE") or p.get("date") or p.get("DATE")
                domain = p.get("domain") or p.get("DOMAIN")

                dt = parse_time_iso(seendate)
                if dt is None:
                    # ha nincs, akkor most-ot írunk (különben kiszórja a trim)
                    dt = now

                # dedupe: url vagy (title+time+lonlat)
                key = url or f"{title}|{to_utc_z(dt)}|{lon:.3f},{lat:.3f}|{category}"
                if key in seen:
                    continue
                seen.add(key)

                props = {
                    "source": "GDELT",
                    "kind": "news_event",
                    "category": category,
                    "category_hu": CATEGORY_HU.get(category, category),
                    "title": str(title),
                    "time": to_utc_z(dt),
                    "url": url,
                    "domain": domain,
                    "country_hint": country,
                    "type": "News",
                }
                out.append(to_feature(lon, lat, props))

    # idő szerint rendezzük
    out.sort(key=lambda f: (parse_time_iso((f.get("properties") or {}).get("time")) or datetime(1970,1,1,tzinfo=timezone.utc)).timestamp(), reverse=True)
    return out, debug

# =========================
# Hotspots + Early + Summaries
# =========================
def score_feature(props: Dict[str, Any]) -> float:
    src = props.get("source")
    if src == "GDELT":
        cat = (props.get("category") or "other").lower()
        base = 1.0
        if cat in ("military", "security_politics"):
            base = 1.25
        elif cat in ("border", "drone"):
            base = 1.15
        elif cat in ("cyber", "energy"):
            base = 1.10
        elif cat in ("protest",):
            base = 1.0
        return base
    if src == "GDACS":
        return 0.55
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
            acc[k] = {"score": 0.0, "count": 0, "sources": {"GDELT": 0, "USGS": 0, "GDACS": 0}, "last7_score": 0.0, "prev7_score": 0.0}
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
            b = {"recent": 0.0, "baseline": 0.0, "src_recent": {"GDELT": 0, "USGS": 0, "GDACS": 0}}
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
        props = {"type": "early_warning", "escalation": round(score0_100, 1), "cell_deg": cell_deg, "neighbor_active": int(neigh_active), **meta[k]}
        signals.append(to_feature(lon_c, lat_c, props))
        rows.append({"lon": lon_c, "lat": lat_c, **props})

    rows_sorted = sorted(rows, key=lambda x: x["escalation"], reverse=True)
    return signals, rows_sorted[:top_n]

def compute_total_score(features: List[Dict[str, Any]], now: datetime) -> float:
    total = 0.0
    for f in features:
        props = f.get("properties") or {}
        dt = parse_time_iso(props.get("time"))
        total += score_feature(props) * time_decay(dt, now)
    return total

def pct_change(curr: float, prev: float) -> Optional[float]:
    if prev <= 0 and curr <= 0:
        return 0.0
    if prev <= 0:
        return None
    return (curr - prev) / prev * 100.0

def make_summary(all_features: List[Dict[str, Any]], top_hotspots: List[Dict[str, Any]], counts: Dict[str, int]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff_7 = now - timedelta(days=7)
    cutoff_14 = now - timedelta(days=14)

    last7, prev7 = [], []
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
        top_text = "Legerősebb góc: jelenleg nincs elég jelzés a térképes kiemeléshez."

    bullets = [
        top_text,
        trend_text,
        f"Forráskép (7 nap): GDELT {counts.get('gdelt_7d',0)}, USGS {counts.get('usgs',0)}, GDACS {counts.get('gdacs',0)}.",
        "Megjegyzés: automatikus OSINT-kivonat; a linkelt források kézi ellenőrzése javasolt.",
    ]
    return {"generated_utc": to_utc_z(now), "headline": "Közép–Kelet Európa biztonsági helyzet – napi kivonat", "bullets": bullets}

def build_weekly(all_features: List[Dict[str, Any]]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff_7 = now - timedelta(days=7)

    week = []
    for f in all_features:
        dt = parse_time_iso((f.get("properties") or {}).get("time"))
        if dt and dt >= cutoff_7:
            week.append((dt, f))

    counts = {"GDELT": 0, "USGS": 0, "GDACS": 0}
    for dt, f in week:
        src = (f.get("properties") or {}).get("source")
        if src in counts:
            counts[src] += 1

    bullets = [
        f"Híralapú jelzések (GDELT): {counts['GDELT']} db az elmúlt 7 napban.",
        f"Természeti/ellátási stresszorok: USGS {counts['USGS']} esemény, GDACS {counts['GDACS']} riasztás (CEE országok területén).",
        "Megjegyzés: automatikus OSINT-kivonat; a linkelt források kézi ellenőrzése javasolt.",
    ]
    return {"generated_utc": to_utc_z(now), "headline": "Heti kivonat – elmúlt 7 nap", "bullets": bullets, "counts": counts}

# =========================
# MAIN
# =========================
def main() -> int:
    ensure_dirs()
    geoms = load_or_build_country_geoms()
    ensure_cee_borders(geoms)

    prev_usgs = load_geojson_features(os.path.join(DATA_DIR, "usgs.geojson"))
    prev_gdacs = load_geojson_features(os.path.join(DATA_DIR, "gdacs.geojson"))
    prev_gdelt = load_geojson_features(os.path.join(DATA_DIR, "gdelt.geojson"))

    # USGS/GDACS
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

    # GDELT (API)
    try:
        gdelt_new, gdelt_dbg = fetch_gdelt_geo(geoms, days=GDELT_DAYS_ON_MAP, per_query_points=PER_QUERY_POINTS)
    except Exception as e:
        gdelt_new, gdelt_dbg = [], {"generated_utc": to_utc_z(datetime.now(timezone.utc)), "error": str(e)[:200]}

    with open(os.path.join(DATA_DIR, "gdelt_debug.json"), "w", encoding="utf-8") as f:
        json.dump(gdelt_dbg, f, ensure_ascii=False, indent=2)

    # Merge + trim
    usgs_merged = merge_dedup(prev_usgs, usgs_new)
    gdacs_merged = merge_dedup(prev_gdacs, gdacs_new)
    gdelt_merged = merge_dedup(prev_gdelt, gdelt_new)

    usgs = trim_by_days(usgs_merged, keep_days=ROLLING_DAYS)
    gdacs = trim_by_days(gdacs_merged, keep_days=ROLLING_DAYS)
    gdelt = trim_by_days(gdelt_merged, keep_days=GDELT_DAYS_ON_MAP)

    save_geojson(os.path.join(DATA_DIR, "usgs.geojson"), usgs)
    save_geojson(os.path.join(DATA_DIR, "gdacs.geojson"), gdacs)
    save_geojson(os.path.join(DATA_DIR, "gdelt.geojson"), gdelt)

    # Hotspots & summaries (7 nap)
    gdelt_7d = trim_by_days(gdelt, keep_days=ROLLING_DAYS)
    all_feats_7d = gdelt_7d + gdacs + usgs

    hotspot_geo, top_hotspots = build_hotspots_with_trend(all_feats_7d, cell_deg=0.5, top_n=10)

    cache = load_cache()
    for h in top_hotspots:
        h["place"] = reverse_geocode_osm(float(h["lat"]), float(h["lon"]), cache)
    save_cache(cache)

    save_geojson(os.path.join(DATA_DIR, "hotspots.geojson"), hotspot_geo)
    with open(os.path.join(DATA_DIR, "hotspots.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_utc": to_utc_z(datetime.now(timezone.utc)), "top": top_hotspots}, f, ensure_ascii=False, indent=2)

    early_geo, early_top = build_early_warning(all_feats_7d, cell_deg=0.5, lookback_days=7, recent_hours=48, top_n=10)
    cache = load_cache()
    for e in early_top:
        e["place"] = reverse_geocode_osm(float(e["lat"]), float(e["lon"]), cache)
    save_cache(cache)

    save_geojson(os.path.join(DATA_DIR, "early.geojson"), early_geo)
    with open(os.path.join(DATA_DIR, "early.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_utc": to_utc_z(datetime.now(timezone.utc)), "top": early_top}, f, ensure_ascii=False, indent=2)

    counts = {
        "usgs": len(usgs),
        "gdacs": len(gdacs),
        "gdelt_14d": len(gdelt),
        "gdelt_7d": len(gdelt_7d),
        "hotspot_cells": len(hotspot_geo),
    }

    summary = make_summary(all_feats_7d, top_hotspots, counts)
    with open(os.path.join(DATA_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    weekly = build_weekly(all_feats_7d)
    with open(os.path.join(DATA_DIR, "weekly.json"), "w", encoding="utf-8") as f:
        json.dump(weekly, f, ensure_ascii=False, indent=2)

    meta = {
        "generated_utc": to_utc_z(datetime.now(timezone.utc)),
        "counts": counts,
        "rolling_days": ROLLING_DAYS,
        "gdelt_days_on_map": GDELT_DAYS_ON_MAP,
        "countries": CEE_COUNTRIES,
        "gdelt": {
            "api": "geo",
            "days": GDELT_DAYS_ON_MAP,
            "per_query_points": PER_QUERY_POINTS,
            "categories": list(CATEGORY_QUERIES.keys()),
        },
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
