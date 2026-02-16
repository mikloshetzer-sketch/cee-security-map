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

# =========================
# PATHS
# =========================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_PATH = os.path.join(DATA_DIR, "geocode_cache.json")
COUNTRIES_CACHE_PATH = os.path.join(DATA_DIR, "cee_countries.geojson")

USER_AGENT = "cee-security-map/2.3 (github actions)"
TIMEOUT = 60

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

CEE_BBOX = (11.0, 42.5, 30.5, 61.5)

# GDELT: 14 nap vissza a pontokhoz
GDELT_LOOKBACK_DAYS = 14

# Summaries/hotspots: maradhat 7 nap (ahogy eddig)
ROLLING_DAYS = 7
USGS_DAYS = 7
GDACS_DAYS = 14  # RSS

# =========================
# GDELT exports (DAILY only)
# =========================
MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
# Napi exportok: pl. 20260216000000.export.CSV.zip (a masterfilelistben is benne van)
DAILY_EXPORT_RE = re.compile(r"/(\d{14})\.export\.CSV\.zip$")

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
        if dt and dt >= cutoff:
            kept.append(f)
    return kept

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
    r = requests.get(url, params=params, headers=headers, timeout=35)
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
# USGS / GDACS
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
# GDELT EVENTS EXPORT (DAILY)
# =========================
def http_get_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", errors="replace")

def http_get_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=TIMEOUT) as r:
        return r.read()

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
        if url.startswith("https://data.gdeltproject.org/"):
            url = "http://data.gdeltproject.org/" + url[len("https://data.gdeltproject.org/") :]
        # csak export CSV zip
        if url.endswith(".export.CSV.zip") and "/gdeltv2/" in url:
            urls.append(url)
    return urls

def extract_timestamp_from_url(url: str) -> Optional[datetime]:
    m = DAILY_EXPORT_RE.search(url)
    if not m:
        return None
    ts = m.group(1)
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def yyyymmdd_to_iso(s: str) -> str:
    if not s or len(s) != 8:
        return ""
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"

# CAMEO root codes -> kategória (a térképen ezek jelennek meg)
ROOT_TO_CATEGORY = {
    "14": "protest",
    "15": "military",
    "16": "police",
    "17": "police",
    "18": "security",
    "19": "security",
    "20": "security",
    "13": "security",
}

ROOT_LABEL = {
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit force posture",
    "16": "Reduce relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Mass violence",
}

def fetch_gdelt_events_export_daily(geoms: Dict[str, Dict[str, Any]], lookback_days: int = 14) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Letölti a masterfilelist alapján a LOOKBACK ablakba eső DAILY exportokat (1/nap),
    és a CSV sorokból olyan pontokat csinál, amiben VAN forráslink (sourceurl).
    Szűrés: csak a 8 ország poligonjában lévő pontok.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    master = http_get_text(MASTERFILELIST_URL)
    urls = parse_masterfilelist(master)

    recent: List[Tuple[datetime, str]] = []
    for u in urls:
        ts = extract_timestamp_from_url(u)
        if ts and ts >= cutoff:
            # csak napi fájlok: *000000.export.CSV.zip
            # (a masterfilelistben sok 15 perces is van, ezt így kiszűrjük)
            if ts.hour == 0 and ts.minute == 0:
                recent.append((ts, u))

    recent.sort(key=lambda x: x[0])

    debug = {
        "generated_utc": to_utc_z(now),
        "lookback_days": lookback_days,
        "daily_files": len(recent),
        "files": [{"ts": to_utc_z(ts), "url": u} for ts, u in recent[:20]],
        "kept_points": 0,
        "rows_seen": 0,
    }

    if not recent:
        return [], debug

    out: List[Dict[str, Any]] = []
    seen_gid = set()

    for ts, url in recent:
        try:
            zbytes = http_get_bytes(url)
            zf = zipfile.ZipFile(io.BytesIO(zbytes))
            name = zf.namelist()[0]
            raw = zf.read(name).decode("utf-8", errors="replace")
        except Exception as e:
            # ne álljon meg a teljes futás
            continue

        reader = csv.reader(io.StringIO(raw), delimiter="\t")
        for row in reader:
            debug["rows_seen"] += 1
            if len(row) < 61:
                continue

            gid = str(row[0]).strip()
            day = str(row[1]).strip()        # YYYYMMDD
            event_code = str(row[26]).strip()
            root = str(row[28]).strip()
            fullname = str(row[52]).strip()
            lat = safe_float(str(row[56]).strip())
            lon = safe_float(str(row[57]).strip())
            sourceurl = str(row[60]).strip()

            if not gid or gid in seen_gid:
                continue
            if lat is None or lon is None:
                continue
            if not sourceurl:
                continue

            # csak a “releváns” root-ok (különben túl sok pont)
            if root not in ROOT_TO_CATEGORY:
                continue

            # poligon szűrés a 8 országra
            if not in_cee_countries(lon, lat, geoms):
                continue

            date_iso = yyyymmdd_to_iso(day)
            if not date_iso:
                continue

            # idő: a napi dátumból 12:00Z (nincs perces pontosság az exportból)
            t = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(hours=12)

            cat = ROOT_TO_CATEGORY.get(root, "other")
            root_label = ROOT_LABEL.get(root, root)

            # “title” generálás (exportban nincs cikk-cím)
            location = fullname or "unknown"
            title = f"{root_label} – {location}"

            props = {
                "source": "GDELT",
                "kind": "event",
                "category": cat,
                "title": title,
                "location": location,
                "time": to_utc_z(t),
                "url": sourceurl,        # EZ a forráslink
                "gdelt_id": gid,
                "event_root_code": root,
                "event_code": event_code,
                "type": "News",
            }

            out.append(to_feature(lon, lat, props))
            seen_gid.add(gid)

    debug["kept_points"] = len(out)
    return out, debug

# =========================
# Hotspots + Early + Summaries (marad a korábbi logika)
# =========================
def score_feature(props: Dict[str, Any]) -> float:
    src = props.get("source")
    if src == "GDELT":
        cat = props.get("category") or "other"
        base = 1.0
        if cat in ("military", "security"):
            base = 1.2
        elif cat in ("border",):
            base = 1.1
        elif cat in ("protest", "police"):
            base = 1.0
        return base
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
        p = f.get("properties") or {}
        dt = parse_time_iso(p.get("time"))
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

    # Fetch USGS/GDACS
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

    # Fetch GDELT from export daily (WITH source url!)
    try:
        gdelt_new, gdelt_dbg = fetch_gdelt_events_export_daily(geoms, lookback_days=GDELT_LOOKBACK_DAYS)
    except Exception as e:
        gdelt_new, gdelt_dbg = [], {"generated_utc": to_utc_z(datetime.now(timezone.utc)), "error": str(e)[:200]}

    with open(os.path.join(DATA_DIR, "gdelt_debug.json"), "w", encoding="utf-8") as f:
        json.dump(gdelt_dbg, f, ensure_ascii=False, indent=2)

    # Merge + trim
    usgs_merged = merge_dedup(clamp_times(prev_usgs), clamp_times(usgs_new))
    gdacs_merged = merge_dedup(clamp_times(prev_gdacs), clamp_times(gdacs_new))
    gdelt_merged = merge_dedup(clamp_times(prev_gdelt), clamp_times(gdelt_new))

    # USGS/GDACS 7 nap
    usgs = trim_by_days(usgs_merged, keep_days=ROLLING_DAYS)
    gdacs = trim_by_days(gdacs_merged, keep_days=ROLLING_DAYS)

    # GDELT 14 nap (a térképre)
    gdelt = trim_by_days(gdelt_merged, keep_days=GDELT_LOOKBACK_DAYS)

    save_geojson(os.path.join(DATA_DIR, "usgs.geojson"), usgs)
    save_geojson(os.path.join(DATA_DIR, "gdacs.geojson"), gdacs)
    save_geojson(os.path.join(DATA_DIR, "gdelt.geojson"), gdelt)

    # Hotspots & summaries: csak 7 napból számolunk (ahogy eddig)
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

    counts = {"usgs": len(usgs), "gdacs": len(gdacs), "gdelt_14d": len(gdelt), "gdelt_7d": len(gdelt_7d), "hotspot_cells": len(hotspot_geo)}

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
        "gdelt_days_on_map": GDELT_LOOKBACK_DAYS,
        "countries": CEE_COUNTRIES,
        "gdelt": {"source": "events_export_daily", "lookback_days": GDELT_LOOKBACK_DAYS},
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
