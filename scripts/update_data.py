#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import math
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

USER_AGENT = "cee-security-map/1.0"
TIMEOUT = 30

CEE_COUNTRIES = [
    "Hungary","Poland","Czech Republic","Slovakia",
    "Romania","Latvia","Lithuania","Estonia"
]

CEE_BBOX = (11.5, 43.0, 29.8, 60.8)

ROLLING_DAYS = 7
GDELT_DAYS = 7
USGS_DAYS = 7
GDACS_DAYS = 14

# =========================
# IO
# =========================
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def http_get(url: str, params=None, headers=None):
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    return requests.get(url, params=params, headers=h, timeout=TIMEOUT)

def to_feature(lon, lat, props):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }

def save_geojson(path, features):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection","features": features}, f, ensure_ascii=False, indent=2)

# =========================
# TIME
# =========================
def to_utc_z(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

# =========================
# SOURCES
# =========================
def fetch_usgs():
    url="https://earthquake.usgs.gov/fdsnws/event/1/query"
    end=datetime.now(timezone.utc)
    start=end-timedelta(days=USGS_DAYS)
    params={
        "format":"geojson",
        "starttime":start.strftime("%Y-%m-%d"),
        "endtime":end.strftime("%Y-%m-%d"),
        "minmagnitude":"2.5"
    }
    data=http_get(url,params=params).json()
    out=[]
    for f in data.get("features",[]):
        coords=f["geometry"]["coordinates"]
        lon,lat=coords[0],coords[1]
        if not (CEE_BBOX[0]<=lon<=CEE_BBOX[2] and CEE_BBOX[1]<=lat<=CEE_BBOX[3]):
            continue
        p=f["properties"]
        dt=datetime.fromtimestamp(p["time"]/1000,tz=timezone.utc)
        out.append(to_feature(lon,lat,{
            "source":"USGS",
            "time":to_utc_z(dt),
            "title":p.get("title"),
            "mag":p.get("mag"),
            "url":p.get("url")
        }))
    return out

def fetch_gdacs():
    url="https://www.gdacs.org/xml/rss.xml"
    xml=http_get(url).text
    items=xml.split("<item>")[1:]
    out=[]
    for raw in items:
        chunk=raw.split("</item>")[0]
        if "<georss:point>" not in chunk:
            continue
        point=chunk.split("<georss:point>")[1].split("</georss:point>")[0]
        lat,lon=map(float,point.split())
        if not (CEE_BBOX[0]<=lon<=CEE_BBOX[2] and CEE_BBOX[1]<=lat<=CEE_BBOX[3]):
            continue
        title=chunk.split("<title>")[1].split("</title>")[0]
        out.append(to_feature(lon,lat,{
            "source":"GDACS",
            "title":title,
            "time":to_utc_z(datetime.now(timezone.utc))
        }))
    return out

def fetch_gdelt():
    url="https://api.gdeltproject.org/api/v2/doc/doc"
    end=datetime.now(timezone.utc)
    start=end-timedelta(days=GDELT_DAYS)
    query="(protest OR riot OR military OR cyber OR attack) AND (Hungary OR Poland OR Romania OR Latvia OR Lithuania OR Estonia OR Slovakia OR Czech)"
    params={
        "query":query,
        "mode":"ArtList",
        "format":"json",
        "maxrecords":"200",
        "startdatetime":start.strftime("%Y%m%d%H%M%S"),
        "enddatetime":end.strftime("%Y%m%d%H%M%S")
    }
    r=http_get(url,params=params)
    try:
        data=r.json()
    except:
        return []
    out=[]
    for a in data.get("articles",[]):
        geo=a.get("location",{}).get("geo",{})
        if not geo:
            continue
        lat=float(geo["latitude"])
        lon=float(geo["longitude"])
        if not (CEE_BBOX[0]<=lon<=CEE_BBOX[2] and CEE_BBOX[1]<=lat<=CEE_BBOX[3]):
            continue
        dt=dateparser.parse(a["seendate"]).astimezone(timezone.utc)
        out.append(to_feature(lon,lat,{
            "source":"GDELT",
            "time":to_utc_z(dt),
            "title":a.get("title"),
            "url":a.get("url")
        }))
    return out

# =========================
# HOTSPOT
# =========================
def score(props):
    if props["source"]=="GDELT": return 1
    if props["source"]=="GDACS": return 0.5
    if props["source"]=="USGS":
        m=props.get("mag",0)
        return 0.2+m*0.1
    return 0.1

def build_hotspots(all_feats):
    acc={}
    for f in all_feats:
        lon,lat=f["geometry"]["coordinates"]
        key=(round(lon,1),round(lat,1))
        acc.setdefault(key,0)
        acc[key]+=score(f["properties"])
    out=[]
    for (lon,lat),s in acc.items():
        out.append(to_feature(lon,lat,{"score":round(s,2)}))
    return out

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    usgs=fetch_usgs()
    gdacs=fetch_gdacs()
    gdelt=fetch_gdelt()

    all_feats=usgs+gdacs+gdelt

    save_geojson(os.path.join(DATA_DIR,"usgs.geojson"),usgs)
    save_geojson(os.path.join(DATA_DIR,"gdacs.geojson"),gdacs)
    save_geojson(os.path.join(DATA_DIR,"gdelt.geojson"),gdelt)

    hotspots=build_hotspots(all_feats)
    save_geojson(os.path.join(DATA_DIR,"hotspots.geojson"),hotspots)

    print("OK")

if __name__=="__main__":
    main()
