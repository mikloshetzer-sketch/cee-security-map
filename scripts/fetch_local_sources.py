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
    "Algyő": [46.335, 20.209],
    "Hajdúszoboszló": [47.443, 21.391],
    "Visonta": [47.784, 20.033],

    "Constanța": [44.173, 28.638],
    "Cernavodă": [44.322, 28.057],
    "Năvodari": [44.335, 28.642],
    "Brazi": [44.848, 26.029],
    "Bucharest": [44.426, 26.102],
    "Galați": [45.435, 28.008],

    "Bratislava": [48.148, 17.107],
    "Mochovce": [48.264, 18.455],
    "Jaslovské Bohunice": [48.494, 17.681],
    "Veľké Kapušany": [48.548, 22.079],
    "Košice": [48.716, 21.261],

    "Prague": [50.075, 14.438],
    "Praha": [50.075, 14.438],
    "Temelín": [49.181, 14.376],
    "Dukovany": [49.085, 16.148],
    "Litvínov": [50.604, 13.618],
    "Kralupy": [50.241, 14.312],
    "Ostrava": [49.820, 18.262],

    "Płock": [52.576, 19.701],
    "Gdańsk": [54.383, 18.670],
    "Gdynia": [54.533, 18.550],
    "Warsaw": [52.229, 21.012],
    "Warszawa": [52.229, 21.012],
    "Rzeszów": [50.041, 21.999],
    "Świnoujście": [53.910, 14.286],
    "Bełchatów": [51.267, 19.325],

    "Klaipėda": [55.706, 21.127],
    "Vilnius": [54.687, 25.279],
    "Kaunas": [54.898, 23.904],
    "Alytus": [54.396, 24.041],
    "Šiauliai": [55.893, 23.395],
    "Rukla": [55.000, 24.000],

    "Riga": [56.949, 24.105],
    "Ventspils": [57.394, 21.560],
    "Ādaži": [57.070, 24.337],
    "Lielvārde": [56.778, 24.853],
    "Inčukalns": [57.098, 24.686],

    "Tallinn": [59.437, 24.753],
    "Tartu": [58.378, 26.729],
    "Narva": [59.377, 27.420],
    "Paldiski": [59.350, 24.050],
    "Tapa": [59.260, 25.958],
    "Ämari": [59.260, 24.208],
    "Muuga": [59.500, 24.960]
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

POSITIVE_KEYWORDS = {
    "general": [
        "explosion", "blast", "fire", "industrial accident", "chemical leak",
        "pipeline leak", "blackout", "power outage", "cyberattack",
        "drone attack", "missile", "evacuation", "emergency services",
        "refinery fire", "airport closed", "port closed", "rail disruption"
    ],
    "Hungary": [
        "robbanás", "tűz", "füst", "ipari baleset", "üzemzavar",
        "vegyi szivárgás", "gázszivárgás", "áramszünet", "kibertámadás",
        "dróntámadás", "katasztrófavédelem", "kiürítés", "finomító",
        "petrolkémiai", "MOL", "Tiszaújváros", "Százhalombatta", "Paks"
    ],
    "Romania": [
        "explozie", "incendiu", "fum", "accident industrial", "scurgere chimică",
        "pană de curent", "atac cibernetic", "rafinărie", "centrală",
        "evacuare", "urgență", "Constanța", "Cernavodă", "Năvodari"
    ],
    "Slovakia": [
        "výbuch", "požiar", "dym", "priemyselná nehoda", "únik chemikálií",
        "výpadok prúdu", "kybernetický útok", "rafinéria", "elektráreň",
        "evakuácia", "Bratislava", "Mochovce", "Bohunice"
    ],
    "Czechia": [
        "výbuch", "požár", "kouř", "průmyslová nehoda", "únik chemikálií",
        "výpadek proudu", "kybernetický útok", "rafinerie", "elektrárna",
        "evakuace", "Temelín", "Dukovany", "Litvínov", "Kralupy"
    ],
    "Poland": [
        "wybuch", "pożar", "dym", "awaria", "wypadek przemysłowy",
        "wyciek chemikaliów", "przerwa w dostawie prądu", "cyberatak",
        "rafineria", "elektrownia", "ewakuacja", "Płock", "Gdańsk",
        "Rzeszów", "Świnoujście"
    ],
    "Lithuania": [
        "sprogimas", "gaisras", "dūmai", "pramoninė avarija",
        "cheminis nuotėkis", "elektros tiekimo sutrikimas", "kibernetinė ataka",
        "evakuacija", "Klaipėda", "Vilnius", "Šiauliai"
    ],
    "Latvia": [
        "sprādziens", "ugunsgrēks", "dūmi", "rūpnieciska avārija",
        "ķīmisko vielu noplūde", "elektrības pārrāvums", "kiberuzbrukums",
        "evakuācija", "Rīga", "Riga", "Ventspils", "Ādaži"
    ],
    "Estonia": [
        "plahvatus", "tulekahju", "suits", "tööstusõnnetus",
        "keemialeke", "elektrikatkestus", "küberrünnak",
        "evakuatsioon", "Tallinn", "Tartu", "Narva", "Paldiski", "Tapa"
    ]
}

NEGATIVE_KEYWORDS = [
    "housing", "real estate", "mortgage", "rent", "apartment",
    "lakhatás", "ingatlan", "albérlet", "bérleti díj", "lakás",
    "piano", "music", "concert", "festival", "culture", "sport",
    "zongora", "zongorázás", "koncert", "fesztivál", "kultúra", "sport",
    "tourism", "travel", "spa", "wellness", "hotel",
    "turizmus", "utazás", "fürdő", "wellness", "szálloda",
    "lázně", "uzdrowisko", "sanatorium", "spa",
    "health resort", "resort", "holiday", "vacation",
    "student", "university researchers", "ai algorithm", "friendly ai"
]

INFRA_HINTS = [
    "MOL", "refinery", "finomító", "rafinărie", "rafinéria", "rafinerie", "rafineria",
    "power plant", "erőmű", "centrală", "elektráreň", "elektrárna", "elektrownia",
    "nuclear", "atomerőmű", "Cernavodă", "Temelín", "Dukovany", "Paks",
    "airport", "repülőtér", "letisko", "letiště", "lotnisko", "lidosta", "lennujaam",
    "port", "kikötő", "prístav", "přístav", "uostas", "osta", "sadam",
    "pipeline", "gázvezeték", "olajvezeték", "gas storage", "LNG",
    "military base", "air base", "katonai bázis", "airbase",
    "data center", "internet exchange", "cyber"
]


def load_sources():
    if not LOCAL_SOURCES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {LOCAL_SOURCES_FILE}")

    with LOCAL_SOURCES_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def has_negative(text):
    t = normalize_text(text)
    return any(normalize_text(k) in t for k in NEGATIVE_KEYWORDS)


def has_positive(text, country):
    t = normalize_text(text)
    keywords = POSITIVE_KEYWORDS["general"] + POSITIVE_KEYWORDS.get(country, [])
    return any(normalize_text(k) in t for k in keywords)


def has_infra_hint(text):
    t = normalize_text(text)
    return any(normalize_text(k) in t for k in INFRA_HINTS)


def detect_city(text):
    t = normalize_text(text)

    for city, coords in CITY_COORDS.items():
        if normalize_text(city) in t:
            return city, coords

    return None, None


def classify_category(text):
    t = normalize_text(text)

    if any(k in t for k in ["cyberattack", "kibertámadás", "kybernetický útok", "cyberatak", "küberrünnak"]):
        return "cyber"

    if any(k in t for k in ["drone", "drón", "uav"]):
        return "drone"

    if any(k in t for k in ["military", "katonai", "wojsk", "vojensk", "sõjaline", "karinis"]):
        return "military"

    if any(k in t for k in ["explosion", "blast", "robbanás", "výbuch", "wybuch", "sprogimas", "sprādziens", "plahvatus"]):
        return "explosion"

    if any(k in t for k in ["fire", "tűz", "požiar", "požár", "pożar", "gaisras", "ugunsgrēks", "tulekahju"]):
        return "fire"

    if any(k in t for k in ["blackout", "power outage", "áramszünet", "výpadok prúdu", "výpadek proudu", "przerwa w dostawie prądu"]):
        return "energy"

    if any(k in t for k in ["chemical", "vegyi", "chem", "petro", "rafin", "finomító"]):
        return "hazardous"

    if any(k in t for k in ["airport", "port", "rail", "repülőtér", "kikötő", "vasút", "lotnisko", "kolej"]):
        return "transport"

    return "local_media"


def estimate_severity(text):
    t = normalize_text(text)

    if any(k in t for k in [
        "explosion", "blast", "robbanás", "výbuch", "wybuch",
        "sprogimas", "sprādziens", "plahvatus", "cyberattack",
        "kibertámadás", "drone attack", "dróntámadás"
    ]):
        return "high"

    if any(k in t for k in [
        "fire", "tűz", "požiar", "požár", "pożar",
        "gaisras", "ugunsgrēks", "tulekahju", "blackout",
        "power outage", "áramszünet"
    ]):
        return "medium"

    return "info"


def build_feature(entry, source_name, country):
    title = entry.get("title", "Untitled")
    summary = entry.get("summary", "")
    link = entry.get("link")
    published = entry.get("published") or entry.get("updated")

    combined = f"{title} {summary}"

    if has_negative(combined):
        return None

    city, coords = detect_city(combined)

    positive = has_positive(combined, country)
    infra_hint = has_infra_hint(combined)

    if not positive:
        return None

    if not city and not infra_hint:
        return None

    if coords:
        lat, lon = coords
        place = city
        geocode_quality = "city"
    else:
        lat, lon = COUNTRY_COORDS.get(country)
        place = country
        geocode_quality = "country_fallback"

    if lat is None or lon is None:
        return None

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
            "kind": "local_media",
            "geocode_quality": geocode_quality
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
