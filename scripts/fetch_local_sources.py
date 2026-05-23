import json
import re
import feedparser
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LOCAL_SOURCES_FILE = DATA / "local_sources.json"
OUTPUT_FILE = DATA / "local_events.geojson"

MAX_ITEMS_PER_SOURCE = 15

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

CITY_COORDS = {
    "Tiszaújváros": [47.933, 21.050],
    "Százhalombatta": [47.316, 18.914],
    "Paks": [46.572, 18.854],
    "Budapest": [47.497, 19.040],
    "Constanța": [44.173, 28.638],
    "Cernavodă": [44.322, 28.057],
    "Bucharest": [44.426, 26.102],
    "Bratislava": [48.148, 17.107],
    "Košice": [48.716, 21.261],
    "Prague": [50.075, 14.438],
    "Ostrava": [49.820, 18.262],
    "Płock": [52.576, 19.701],
    "Gdańsk": [54.383, 18.670],
    "Warsaw": [52.229, 21.012],
    "Rzeszów": [50.041, 21.999],
    "Klaipėda": [55.706, 21.127],
    "Vilnius": [54.687, 25.279],
    "Riga": [56.949, 24.105],
    "Ventspils": [57.394, 21.560],
    "Tallinn": [59.437, 24.753],
    "Tartu": [58.378, 26.729],
    "Narva": [59.377, 27.420]
}

RSS_FEEDS = {
    "Telex": "https://telex.hu/rss",
    "HVG": "https://hvg.hu/rss",
    "Portfolio": "https://www.portfolio.hu/rss/all.xml",
    "24.hu": "https://24.hu/feed/",
    "hirado.hu": "https://hirado.hu/feed/",

    "Digi24": "https://www.digi24.ro/rss",
    "HotNews": "https://hotnews.ro/feed",
    "G4Media": "https://www.g4media.ro/feed",
    "Agerpres": "https://agerpres.ro/rss",
    "Economica.net": "https://www.economica.net/feed",

    "DennikN": "https://dennikn.sk/feed/",
    "SME": "https://www.sme.sk/rss-title",
    "Aktuality": "https://www.aktuality.sk/rss/",
    "TA3": "https://www.ta3.com/rss",
    "TASR": "https://www.tasr.sk/rss.xml",

    "CT24": "https://ct24.ceskatelevize.cz/rss/hlavni-zpravy",
    "iROZHLAS": "https://www.irozhlas.cz/rss/irozhlas",
    "SeznamZpravy": "https://www.seznamzpravy.cz/rss",
    "Novinky": "https://www.novinky.cz/rss",
    "CTK": "https://www.ceskenoviny.cz/rss/",

    "PAP": "https://www.pap.pl/rss",
    "TVN24": "https://tvn24.pl/najnowsze.xml",
    "Onet": "https://www.onet.pl/rss.xml",
    "Rzeczpospolita": "https://www.rp.pl/rss",
    "NotesFromPoland": "https://notesfrompoland.com/feed/",

    "LRT": "https://www.lrt.lt/rss",
    "DelfiLT": "https://www.delfi.lt/rss/",
    "15min": "https://www.15min.lt/rss",
    "BNSLithuania": "https://www.bns.lt/rss",
    "VersloZinios": "https://www.vz.lt/rss",

    "LSM": "https://www.lsm.lv/rss/",
    "DelfiLV": "https://www.delfi.lv/rss/",
    "TVNET": "https://www.tvnet.lv/rss",
    "LETA": "https://www.leta.lv/rss",
    "BNN": "https://bnn-news.com/feed",

    "ERR": "https://news.err.ee/rss",
    "Postimees": "https://www.postimees.ee/rss",
    "DelfiEE": "https://www.delfi.ee/rss",
    "Aripaev": "https://www.aripaev.ee/rss",
    "BNSEstonia": "https://www.bns.ee/rss"
}

GLOBAL_KEYWORDS = [
    "explosion", "fire", "blast", "drone", "cyber", "pipeline", "blackout",
    "gas", "oil", "nuclear", "military", "airport", "port", "rail",
    "infrastructure", "industrial", "chemical", "electricity", "energy",
    "accident", "attack", "emergency", "security", "refinery", "power plant"
]

COUNTRY_KEYWORDS = {
    "Hungary": [
        "robbanás", "tűz", "füst", "baleset", "ipari baleset", "üzemzavar",
        "finomító", "vegyipari", "petrolkémiai", "erőmű", "áramszünet",
        "gázvezeték", "olajvezeték", "MOL", "katasztrófavédelem",
        "repülőtér", "kikötő", "vasút", "katonai", "kibertámadás"
    ],
    "Romania": [
        "explozie", "incendiu", "fum", "accident", "accident industrial",
        "rafinărie", "petrochimic", "centrală", "pană de curent",
        "conductă", "gaz", "petrol", "militar", "aeroport", "port",
        "cale ferată", "atac cibernetic", "urgență"
    ],
    "Slovakia": [
        "výbuch", "požiar", "dym", "nehoda", "priemyselná nehoda",
        "rafinéria", "petrochémia", "elektráreň", "výpadok prúdu",
        "plynovod", "ropovod", "vojenský", "letisko", "prístav",
        "železnica", "kybernetický útok", "núdzový stav"
    ],
    "Czechia": [
        "výbuch", "požár", "kouř", "nehoda", "průmyslová nehoda",
        "rafinerie", "petrochemie", "elektrárna", "výpadek proudu",
        "plynovod", "ropovod", "vojenský", "letiště", "přístav",
        "železnice", "kybernetický útok", "mimořádná událost"
    ],
    "Poland": [
        "wybuch", "pożar", "dym", "awaria", "wypadek", "wypadek przemysłowy",
        "rafineria", "petrochemia", "elektrownia", "przerwa w dostawie prądu",
        "gazociąg", "ropociąg", "wojskowy", "lotnisko", "port",
        "kolej", "cyberatak", "zagrożenie", "służby"
    ],
    "Lithuania": [
        "sprogimas", "gaisras", "dūmai", "avarija", "pramoninė avarija",
        "naftos perdirbimo", "elektrinė", "elektros tiekimo sutrikimas",
        "dujotiekis", "naftotiekis", "karinis", "oro uostas", "uostas",
        "geležinkelis", "kibernetinė ataka", "ekstremali situacija"
    ],
    "Latvia": [
        "sprādziens", "ugunsgrēks", "dūmi", "avārija", "rūpnieciska avārija",
        "naftas pārstrāde", "elektrostacija", "elektrības pārrāvums",
        "gāzesvads", "naftas vads", "militārs", "lidosta", "osta",
        "dzelzceļš", "kiberuzbrukums", "ārkārtas situācija"
    ],
    "Estonia": [
        "plahvatus", "tulekahju", "suits", "avarii", "tööstusõnnetus",
        "rafineerimistehas", "elektrijaam", "elektrikatkestus",
        "gaasitoru", "naftajuhe", "sõjaline", "lennujaam", "sadam",
        "raudtee", "küberrünnak", "hädaolukord"
    ]
}


def load_sources():
    if not LOCAL_SOURCES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {LOCAL_SOURCES_FILE}")

    with LOCAL_SOURCES_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def get_keywords(country):
    return GLOBAL_KEYWORDS + COUNTRY_KEYWORDS.get(country, [])


def contains_keyword(text, country):
    t = normalize_text(text)
    return any(normalize_text(k) in t for k in get_keywords(country))


def detect_city(text):
    t = normalize_text(text)

    for city, coords in CITY_COORDS.items():
        if normalize_text(city) in t:
            return city, coords

    return None, None


def classify_category(text):
    t = normalize_text(text)

    if any(k in t for k in ["cyber", "kiber", "kibernet", "küber", "cyberatak"]):
        return "cyber"

    if any(k in t for k in ["military", "katonai", "wojsk", "vojensk", "sõjaline", "karinis"]):
        return "military"

    if any(k in t for k in ["explosion", "robbanás", "výbuch", "wybuch", "sprogimas", "sprādziens", "plahvatus"]):
        return "explosion"

    if any(k in t for k in ["fire", "tűz", "požiar", "požár", "pożar", "gaisras", "ugunsgrēks", "tulekahju"]):
        return "fire"

    if any(k in t for k in ["energy", "energia", "erőmű", "elektrárna", "elektrownia", "elektrinė", "elektrostacija"]):
        return "energy"

    if any(k in t for k in ["rail", "vasút", "kolej", "železnica", "geležinkelis", "dzelzceļš", "raudtee"]):
        return "transport"

    if any(k in t for k in ["port", "kikötő", "prístav", "přístav", "uostas", "osta", "sadam"]):
        return "transport"

    if any(k in t for k in ["airport", "repülőtér", "letisko", "letiště", "lotnisko", "oro uostas", "lidosta", "lennujaam"]):
        return "transport"

    if any(k in t for k in ["chemical", "vegyipari", "petro", "petrolkémiai", "rafin", "rafiner"]):
        return "hazardous"

    return "local_media"


def estimate_severity(text):
    t = normalize_text(text)

    critical_words = [
        "explosion", "robbanás", "výbuch", "wybuch", "sprogimas", "sprādziens", "plahvatus",
        "attack", "támadás", "atak", "kibertámadás", "cyberattack", "blackout", "áramszünet"
    ]

    high_words = [
        "fire", "tűz", "požiar", "požár", "pożar", "gaisras", "ugunsgrēks", "tulekahju",
        "accident", "baleset", "awaria", "avarija", "nehoda", "avārija"
    ]

    if any(w in t for w in critical_words):
        return "high"

    if any(w in t for w in high_words):
        return "medium"

    return "info"


def build_feature(entry, source_name, country):
    title = entry.get("title", "Untitled")
    summary = entry.get("summary", "")
    link = entry.get("link")
    published = entry.get("published") or entry.get("updated")

    combined = f"{title} {summary}"

    if not contains_keyword(combined, country):
        return None

    city, coords = detect_city(combined)

    if coords:
        lat, lon = coords
        place = city
    else:
        lat, lon = COUNTRY_COORDS.get(country)
        place = country

    category = classify_category(combined)
    severity = estimate_severity(combined)

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": {
            "title": title,
            "summary": str(summary)[:600],
            "source": source_name,
            "country": country,
            "place": place,
            "url": link,
            "time": published,
            "category": category,
            "severity": severity,
            "kind": "local_media"
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


def deduplicate_features(features):
    seen = set()
    clean = []

    for f in features:
        p = f.get("properties", {})
        key = (
            p.get("title", "").strip().lower(),
            p.get("source", "").strip().lower(),
            p.get("country", "").strip().lower()
        )

        if key in seen:
            continue

        seen.add(key)
        clean.append(f)

    return clean


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

            print(f"Fetching: {source_name} ({country})")
            fetched = fetch_feed(source_name, rss_url, country)
            features.extend(fetched)

    features = deduplicate_features(features)

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
