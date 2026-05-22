import json
import re
import feedparser
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LOCAL_SOURCES_FILE = DATA / "local_sources.json"
OUTPUT_FILE = DATA / "local_events.geojson"

MAX_ITEMS_PER_SOURCE = 10


COUNTRY_COORDS = {
    "Hungary": [47.1625, 19.5033],
    "Romania": [45.9432, 24.9668],
    "Slovakia": [48.6690, 19.6990],
    "Czechia": [49.8175, 15.4730],
    "Poland": [51.9194, 19.1451],
    "Lithuania": [55.1694, 23.8813],
    "Latvia": [56.8796, 24.6032],
    "Estonia": [58.5953, 25.0136]
}


RSS_FEEDS = {

    # HUNGARY
    "Telex": "https://telex.hu/rss",
    "HVG": "https://hvg.hu/rss",
    "Portfolio": "https://www.portfolio.hu/rss/all.xml",
    "24.hu": "https://24.hu/feed/",
    "hirado.hu": "https://hirado.hu/feed/",

    # ROMANIA
    "Digi24": "https://www.digi24.ro/rss",
    "HotNews": "https://hotnews.ro/feed",
    "G4Media": "https://www.g4media.ro/feed",
    "Agerpres": "https://agerpres.ro/rss",
    "Economica.net": "https://www.economica.net/feed",

    # SLOVAKIA
    "DennikN": "https://dennikn.sk/feed/",
    "SME": "https://www.sme.sk/rss-title",
    "Aktuality": "https://www.aktuality.sk/rss/",
    "TA3": "https://www.ta3.com/rss",
    "TASR": "https://www.tasr.sk/rss.xml",

    # CZECHIA
    "CT24": "https://ct24.ceskatelevize.cz/rss/hlavni-zpravy",
    "iROZHLAS": "https://www.irozhlas.cz/rss/irozhlas",
    "SeznamZpravy": "https://www.seznamzpravy.cz/rss",
    "Novinky": "https://www.novinky.cz/rss",
    "CTK": "https://www.ceskenoviny.cz/rss/",

    # POLAND
    "PAP": "https://www.pap.pl/rss",
    "TVN24": "https://tvn24.pl/najnowsze.xml",
    "Onet": "https://www.onet.pl/rss.xml",
    "Rzeczpospolita": "https://www.rp.pl/rss",
    "NotesFromPoland": "https://notesfrompoland.com/feed/",

    # LITHUANIA
    "LRT": "https://www.lrt.lt/rss",
    "DelfiLT": "https://www.delfi.lt/rss/",
    "15min": "https://www.15min.lt/rss",
    "BNSLithuania": "https://www.bns.lt/rss",
    "VersloZinios": "https://www.vz.lt/rss",

    # LATVIA
    "LSM": "https://www.lsm.lv/rss/",
    "DelfiLV": "https://www.delfi.lv/rss/",
    "TVNET": "https://www.tvnet.lv/rss",
    "LETA": "https://www.leta.lv/rss",
    "BNN": "https://bnn-news.com/feed",

    # ESTONIA
    "ERR": "https://news.err.ee/rss",
    "Postimees": "https://www.postimees.ee/rss",
    "DelfiEE": "https://www.delfi.ee/rss",
    "Aripaev": "https://www.aripaev.ee/rss",
    "BNSEstonia": "https://www.bns.ee/rss"
}


KEYWORDS = [
    "explosion",
    "fire",
    "blast",
    "drone",
    "cyber",
    "pipeline",
    "power",
    "blackout",
    "gas",
    "oil",
    "nuclear",
    "military",
    "airport",
    "port",
    "rail",
    "infrastructure",
    "industrial",
    "chemical",
    "electricity",
    "energy",
    "accident",
    "attack",
    "flood",
    "emergency",
    "security"
]


def load_sources():
    with LOCAL_SOURCES_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_keyword(text):
    t = normalize_text(text)
    return any(k in t for k in KEYWORDS)


def build_feature(entry, source_name, country):

    title = entry.get("title", "Untitled")
    summary = entry.get("summary", "")
    link = entry.get("link")

    combined = f"{title} {summary}"

    if not contains_keyword(combined):
        return None

    coords = COUNTRY_COORDS.get(country)

    if not coords:
        return None

    lat, lon = coords

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": {
            "title": title,
            "summary": summary[:500],
            "source": source_name,
            "country": country,
            "url": link,
            "time": entry.get("published"),
            "category": "local_media",
            "severity": "info"
        }
    }


def fetch_feed(source_name, url, country):

    features = []

    try:
        parsed = feedparser.parse(url)

        for entry in parsed.entries[:MAX_ITEMS_PER_SOURCE]:

            feature = build_feature(entry, source_name, country)

            if feature:
                features.append(feature)

    except Exception as e:
        print(f"ERROR {source_name}: {e}")

    return features


def main():

    payload = load_sources()

    features = []

    for country_block in payload.get("countries", []):

        country = country_block.get("country")

        for source_name in country_block.get("sources", []):

            rss_url = RSS_FEEDS.get(source_name)

            if not rss_url:
                print(f"Missing RSS mapping: {source_name}")
                continue

            print(f"Fetching: {source_name}")

            fetched = fetch_feed(source_name, rss_url, country)

            features.extend(fetched)

    geojson = {
        "type": "FeatureCollection",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "features": features
    }

    OUTPUT_FILE.write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Features: {len(features)}")


if __name__ == "__main__":
    main()
