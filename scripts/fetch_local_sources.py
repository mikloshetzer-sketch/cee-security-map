import json
import re
import feedparser
import math
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LOCAL_SOURCES_FILE = DATA / "local_sources.json"
OUTPUT_FILE = DATA / "local_events.geojson"
DEBUG_FILE = DATA / "local_events_debug.json"

MAX_ITEMS_PER_SOURCE = 50

TARGET_COUNTRIES = {
    "Hungary",
    "Romania",
    "Slovakia",
    "Czechia",
    "Poland",
    "Lithuania",
    "Latvia",
    "Estonia",
}

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
    "București": [44.426, 26.102],
    "Galați": [45.435, 28.008],
    "Brăila": [45.269, 27.957],
    "Braila": [45.269, 27.957],
    "Buzău": [45.150, 26.824],
    "Buzau": [45.150, 26.824],
    "Ialomița": [44.603, 27.378],
    "Ialomita": [44.603, 27.378],
    "Sulina": [45.156, 29.653],
    "Padina": [44.833, 27.117],
    "Fetești": [44.366, 27.833],
    "Fetesti": [44.366, 27.833],

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

CITY_COUNTRY = {
    "Tiszaújváros": "Hungary",
    "Százhalombatta": "Hungary",
    "Paks": "Hungary",
    "Budapest": "Hungary",
    "Algyő": "Hungary",
    "Hajdúszoboszló": "Hungary",
    "Visonta": "Hungary",
    "Constanța": "Romania",
    "Cernavodă": "Romania",
    "Năvodari": "Romania",
    "Brazi": "Romania",
    "Bucharest": "Romania",
    "București": "Romania",
    "Galați": "Romania",
    "Brăila": "Romania",
    "Braila": "Romania",
    "Buzău": "Romania",
    "Buzau": "Romania",
    "Ialomița": "Romania",
    "Ialomita": "Romania",
    "Sulina": "Romania",
    "Padina": "Romania",
    "Fetești": "Romania",
    "Fetesti": "Romania",
    "Bratislava": "Slovakia",
    "Mochovce": "Slovakia",
    "Jaslovské Bohunice": "Slovakia",
    "Veľké Kapušany": "Slovakia",
    "Košice": "Slovakia",
    "Prague": "Czechia",
    "Praha": "Czechia",
    "Temelín": "Czechia",
    "Dukovany": "Czechia",
    "Litvínov": "Czechia",
    "Kralupy": "Czechia",
    "Ostrava": "Czechia",
    "Płock": "Poland",
    "Gdańsk": "Poland",
    "Gdynia": "Poland",
    "Warsaw": "Poland",
    "Warszawa": "Poland",
    "Rzeszów": "Poland",
    "Świnoujście": "Poland",
    "Bełchatów": "Poland",
    "Klaipėda": "Lithuania",
    "Vilnius": "Lithuania",
    "Kaunas": "Lithuania",
    "Alytus": "Lithuania",
    "Šiauliai": "Lithuania",
    "Rukla": "Lithuania",
    "Riga": "Latvia",
    "Ventspils": "Latvia",
    "Ādaži": "Latvia",
    "Lielvārde": "Latvia",
    "Inčukalns": "Latvia",
    "Tallinn": "Estonia",
    "Tartu": "Estonia",
    "Narva": "Estonia",
    "Paldiski": "Estonia",
    "Tapa": "Estonia",
    "Ämari": "Estonia",
    "Muuga": "Estonia",
}

COUNTRY_TERMS = {
    "Hungary": [
        "hungary", "hungarian", "magyarország", "magyarországi", "magyar"
    ],
    "Romania": [
        "romania", "românia", "romanian", "român", "română", "româniei",
        "románia", "romániai", "román"
    ],
    "Slovakia": [
        "slovakia", "slovak", "slovensko", "slovenský", "slovenská",
        "szlovákia", "szlovák"
    ],
    "Czechia": [
        "czechia", "czech republic", "czech", "česko", "česká", "český",
        "csehország", "cseh"
    ],
    "Poland": [
        "poland", "polish", "polska", "polski", "lengyelország", "lengyel"
    ],
    "Lithuania": [
        "lithuania", "lithuanian", "lietuva", "lietuvos", "lietuvoje",
        "litvánia", "litván"
    ],
    "Latvia": [
        "latvia", "latvian", "latvija", "latvijas", "latvijā",
        "lettország", "lett"
    ],
    "Estonia": [
        "estonia", "estonian", "eesti", "eestis",
        "észtország", "észt"
    ]
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
        "drone", "drón", "uav", "shahed", "drone attack", "missile", "strike", "attack", "air defence", "air defense", "evacuation", "emergency services",
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

SECURITY_EVENT_TERMS = [
    "explosion", "blast", "robbanás", "výbuch", "wybuch", "sprogimas",
    "sprādziens", "plahvatus",
    "missile", "rocket", "air strike", "airstrike", "strike",
    "drone attack", "dróntámadás", "uav attack", "shahed",
    "shot down", "shoot down", "downed", "intercepted",
    "lelőtt", "lelőttek", "doborât", "doborâtă",
    "airspace violation", "airspace breach", "légtérsértés",
    "military attack", "armed attack",
    "cyberattack", "cyber attack", "kibertámadás", "cyberatak",
    "küberrünnak", "sabotage", "szabotázs",
    "critical infrastructure", "kritikus infrastruktúra",
    "industrial accident", "ipari baleset", "chemical leak",
    "vegyi szivárgás", "gas leak", "gázszivárgás",
    "blackout", "power outage", "áramszünet",
    "refinery fire", "airport closed", "port closed",
    "rail disruption", "evacuation", "kiürítés"
]

SECURITY_CONTEXT_TERMS = [
    "military", "katonai", "armed forces", "air force", "légierő",
    "army", "navy", "nato", "defence", "defense", "védelmi",
    "border guard", "határőrség", "police", "rendőrség",
    "emergency services", "katasztrófavédelem",
    "airport", "repülőtér", "airbase", "air base", "katonai bázis",
    "refinery", "finomító", "power plant", "erőmű", "nuclear",
    "atomerőmű", "pipeline", "vezeték", "port", "kikötő"
]

DRONE_SECURITY_CONTEXT = [
    "attack", "strike", "missile", "military", "army", "air force",
    "légierő", "nato", "border", "határ", "airspace", "légtér",
    "shot down", "downed", "intercepted", "lelőtt", "doborât",
    "shahed", "explosive", "weapon", "fegyver"
]

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

FORCE_INCLUDE_TERMS = [
    "Tiszaújváros", "MOL", "robbanás", "tűz", "petrolkémiai",
    "Százhalombatta", "Paks", "katasztrófavédelem",
    "Cernavodă", "Constanța", "Năvodari", "Mihail Kogălniceanu",
    "Slovnaft", "Mochovce", "Bohunice",
    "Temelín", "Dukovany", "Litvínov", "Kralupy",
    "Płock", "Gdańsk", "Rzeszów", "Świnoujście",
    "Klaipėda", "Šiauliai", "Riga", "Ādaži",
    "Tallinn", "Ämari", "Tapa"
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


def has_force_include(text):
    t = normalize_text(text)
    return any(normalize_text(k) in t for k in FORCE_INCLUDE_TERMS)


def contains_term(text, term):
    """
    Unicode-safe whole-term matching.

    Prevents false positives such as:
      Riga -> "brigadă"
      Paks -> "paksuks"
    """
    t = normalize_text(text)
    k = normalize_text(term)

    if not k:
        return False

    pattern = rf"(?<!\w){re.escape(k)}(?!\w)"
    return re.search(pattern, t, flags=re.UNICODE) is not None


def detect_city(text, preferred_country=None):
    matches = []

    for city, coords in CITY_COORDS.items():
        if contains_term(text, city):
            city_country = CITY_COUNTRY.get(city)
            priority = 0 if preferred_country and city_country == preferred_country else 1
            matches.append((priority, -len(city), city, coords, city_country))

    if not matches:
        return None, None, None

    matches.sort()
    _, _, city, coords, city_country = matches[0]
    return city, coords, city_country


def detect_event_country(text, source_country):
    """
    Strict whitelist geolocation for the eight monitored CEE countries.

    The RSS source country is metadata only. It is never used automatically
    as the event location.

    Accepted evidence:
      1. a recognized city belonging to TARGET_COUNTRIES;
      2. exactly one explicit monitored-country/demonym mention.

    If the article cannot be tied reliably to one monitored country,
    return None and exclude it.
    """
    city, _, city_country = detect_city(text)

    if city and city_country in TARGET_COUNTRIES:
        return city_country

    country_hits = []

    for country, terms in COUNTRY_TERMS.items():
        if country not in TARGET_COUNTRIES:
            continue

        for term in terms:
            if contains_term(text, term):
                country_hits.append(country)
                break

    unique_hits = list(dict.fromkeys(country_hits))

    if len(unique_hits) == 1:
        return unique_hits[0]

    return None


def classify_category(text):
    t = normalize_text(text)

    if any(k in t for k in ["cyberattack", "kibertámadás", "kybernetický útok", "cyberatak", "küberrünnak"]):
        return "cyber"

    if any(k in t for k in ["drone", "drón", "uav", "dróntámadás"]):
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
        "kibertámadás", "drone attack", "dróntámadás",
        "shot down", "shoot down", "downed", "intercepted",
        "doborât", "doborata", "doborâtă", "lelőtt", "lelőttek",
        "shahed"
    ]):
        return "high"

    if any(k in t for k in [
        "fire", "tűz", "požiar", "požár", "pożar",
        "gaisras", "ugunsgrēks", "tulekahju", "blackout",
        "power outage", "áramszünet"
    ]):
        return "medium"

    return "info"


def has_security_event_signal(text):
    t = normalize_text(text)
    return any(normalize_text(k) in t for k in SECURITY_EVENT_TERMS)


def has_security_context(text):
    t = normalize_text(text)
    return any(normalize_text(k) in t for k in SECURITY_CONTEXT_TERMS)


def is_security_relevant(text):
    t = normalize_text(text)

    if has_security_event_signal(text):
        return True

    if any(k in t for k in ["drone", "drón", "uav"]):
        return any(normalize_text(k) in t for k in DRONE_SECURITY_CONTEXT)

    fire_terms = [
        "fire", "tűz", "požiar", "požár", "pożar",
        "gaisras", "ugunsgrēks", "tulekahju"
    ]
    if any(normalize_text(k) in t for k in fire_terms):
        return has_security_context(text)

    return False


def rejection_reason(text, country):
    if country not in TARGET_COUNTRIES:
        return "outside_target_countries"

    if has_negative(text):
        return "negative_keyword"

    if not is_security_relevant(text):
        return "not_security_relevant"

    if not has_positive(text, country):
        return "no_positive_keyword"

    return None


def build_feature(entry, source_name, country):
    title = entry.get("title", "Untitled")
    summary = entry.get("summary", "")
    link = entry.get("link")
    published = entry.get("published") or entry.get("updated")

    title_text = str(title or "")
    summary_text = str(summary or "")
    combined = f"{title_text} {summary_text}"

    title_country = detect_event_country(title_text, country)
    combined_country = detect_event_country(combined, country)

    if title_country in TARGET_COUNTRIES:
        event_country = title_country
    else:
        event_country = combined_country

    if event_country is None:
        return None, "no_target_country_evidence"

    reason = rejection_reason(combined, event_country)

    if reason:
        return None, reason

    title_city, title_coords, title_city_country = detect_city(
        title_text,
        preferred_country=event_country
    )

    if title_coords and title_city_country == event_country:
        city = title_city
        coords = title_coords
        city_country = title_city_country
    else:
        city, coords, city_country = detect_city(
            combined,
            preferred_country=event_country
        )

    if coords and city_country == event_country:
        lat, lon = coords
        place = city
        geocode_quality = "city"
    else:
        fallback_coords = COUNTRY_COORDS.get(event_country)

        if not fallback_coords:
            return None, "missing_coordinates"

        lat, lon = fallback_coords
        place = event_country
        geocode_quality = "country_fallback"

    category = classify_category(combined)
    severity = estimate_severity(combined)

    if has_force_include(combined):
        severity = "high"

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": {
            "title": title,
            "summary": summary_text[:800],
            "source": source_name,
            "country": event_country,
            "place": place,
            "url": link,
            "time": published,
            "category": category,
            "severity": severity,
            "kind": "local_media",
            "geocode_quality": geocode_quality,
            "force_included": has_force_include(combined),
            "source_count": 1,
            "sources": [source_name],
            "urls": [link] if link else [],
            "merged_titles": [title_text]
        }
    }, None


def fetch_feed(source_name, url, country):
    features = []
    debug_rows = []

    try:
        parsed = feedparser.parse(url)

        for entry in parsed.entries[:MAX_ITEMS_PER_SOURCE]:
            feature, reason = build_feature(entry, source_name, country)

            title = entry.get("title", "Untitled")
            link = entry.get("link")

            if feature:
                features.append(feature)
                debug_rows.append({
                    "source": source_name,
                    "country": country,
                    "title": title,
                    "url": link,
                    "status": "included",
                    "reason": None
                })
            else:
                debug_rows.append({
                    "source": source_name,
                    "country": country,
                    "title": title,
                    "url": link,
                    "status": "excluded",
                    "reason": reason
                })

    except Exception as e:
        debug_rows.append({
            "source": source_name,
            "country": country,
            "status": "error",
            "reason": str(e)
        })

        print(f"ERROR {source_name}: {e}")

    return features, debug_rows


DEDUP_MAX_DISTANCE_KM = 140.0
DEDUP_MAX_TIME_HOURS = 18.0
DEDUP_TEXT_SIMILARITY = 0.24

DEDUP_STOPWORDS = {
    "a", "az", "egy", "és", "hogy", "is", "nem", "meg", "már", "ma",
    "the", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "from", "by", "at", "as", "after", "near", "over",
    "un", "o", "și", "si", "de", "la", "în", "din", "pe", "cu",
    "este", "fost", "care",
}

EVENT_SIGNATURE_TERMS = {
    "shahed", "f-16", "f16", "eurofighter", "missile", "rocket",
    "drone", "drón", "uav", "lelőtt", "downed", "intercepted",
    "doborât", "doborâtă", "airspace", "légtér",
    "explosion", "robbanás", "blast",
    "cyberattack", "kibertámadás",
    "blackout", "áramszünet",
    "evacuation", "kiürítés",
}


def parse_event_time(value):
    if not value:
        return None

    raw = str(value).strip()

    try:
        dt = parsedate_to_datetime(raw)
        if dt:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )

    return 2 * radius_km * math.asin(math.sqrt(a))


def event_tokens(text):
    t = normalize_text(text)
    words = re.findall(r"[\wÀ-ž-]+", t, flags=re.UNICODE)

    return {
        word
        for word in words
        if len(word) >= 3 and word not in DEDUP_STOPWORDS
    }


def text_similarity(text_a, text_b):
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if not a or not b:
        return 0.0

    tokens_a = event_tokens(a)
    tokens_b = event_tokens(b)

    if tokens_a and tokens_b:
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    else:
        jaccard = 0.0

    sequence = SequenceMatcher(None, a, b).ratio()

    return max(jaccard, sequence * 0.65)


def shared_signature_terms(text_a, text_b):
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    return {
        term
        for term in EVENT_SIGNATURE_TERMS
        if normalize_text(term) in a and normalize_text(term) in b
    }


def feature_text(feature):
    p = feature.get("properties", {})
    return f"{p.get('title', '')} {p.get('summary', '')}"


def feature_coordinates(feature):
    coords = feature.get("geometry", {}).get("coordinates", [])

    if not isinstance(coords, list) or len(coords) < 2:
        return None

    try:
        lon = float(coords[0])
        lat = float(coords[1])
        return lat, lon
    except (TypeError, ValueError):
        return None


def same_event(feature_a, feature_b):
    pa = feature_a.get("properties", {})
    pb = feature_b.get("properties", {})

    if pa.get("country") != pb.get("country"):
        return False

    if pa.get("category") != pb.get("category"):
        return False

    time_a = parse_event_time(pa.get("time"))
    time_b = parse_event_time(pb.get("time"))

    if time_a and time_b:
        hours = abs((time_a - time_b).total_seconds()) / 3600.0
        if hours > DEDUP_MAX_TIME_HOURS:
            return False

    coord_a = feature_coordinates(feature_a)
    coord_b = feature_coordinates(feature_b)

    distance = None
    if coord_a and coord_b:
        distance = haversine_km(
            coord_a[0], coord_a[1],
            coord_b[0], coord_b[1]
        )

        if distance > DEDUP_MAX_DISTANCE_KM:
            return False

    text_a = feature_text(feature_a)
    text_b = feature_text(feature_b)

    similarity = text_similarity(text_a, text_b)
    signatures = shared_signature_terms(text_a, text_b)

    if similarity >= DEDUP_TEXT_SIMILARITY:
        return True

    strong_signatures = {
        "shahed", "f-16", "f16", "eurofighter",
        "lelőtt", "downed", "intercepted", "doborât", "doborâtă"
    }

    if signatures & strong_signatures:
        if distance is None or distance <= DEDUP_MAX_DISTANCE_KM:
            return True

    return False


def severity_rank(value):
    ranks = {
        "info": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }
    return ranks.get(str(value or "").lower(), 0)


def location_rank(feature):
    quality = feature.get("properties", {}).get("geocode_quality")

    if quality == "city":
        return 2

    if quality == "country_fallback":
        return 1

    return 0


def merge_event_features(primary, incoming):
    pp = primary.setdefault("properties", {})
    ip = incoming.get("properties", {})

    sources = list(pp.get("sources") or ([pp.get("source")] if pp.get("source") else []))
    incoming_sources = list(ip.get("sources") or ([ip.get("source")] if ip.get("source") else []))

    for source in incoming_sources:
        if source and source not in sources:
            sources.append(source)

    urls = list(pp.get("urls") or ([pp.get("url")] if pp.get("url") else []))
    incoming_urls = list(ip.get("urls") or ([ip.get("url")] if ip.get("url") else []))

    for url in incoming_urls:
        if url and url not in urls:
            urls.append(url)

    titles = list(pp.get("merged_titles") or ([pp.get("title")] if pp.get("title") else []))
    incoming_titles = list(ip.get("merged_titles") or ([ip.get("title")] if ip.get("title") else []))

    for title in incoming_titles:
        if title and title not in titles:
            titles.append(title)

    pp["sources"] = sources
    pp["source_count"] = len(sources)
    pp["urls"] = urls
    pp["merged_titles"] = titles

    if severity_rank(ip.get("severity")) > severity_rank(pp.get("severity")):
        pp["severity"] = ip.get("severity")

    if location_rank(incoming) > location_rank(primary):
        primary["geometry"] = incoming.get("geometry", primary.get("geometry"))
        pp["place"] = ip.get("place", pp.get("place"))
        pp["geocode_quality"] = ip.get("geocode_quality", pp.get("geocode_quality"))

    if len(str(ip.get("summary") or "")) > len(str(pp.get("summary") or "")):
        pp["summary"] = ip.get("summary")

    primary_time = parse_event_time(pp.get("time"))
    incoming_time = parse_event_time(ip.get("time"))

    if incoming_time and (not primary_time or incoming_time < primary_time):
        pp["time"] = ip.get("time")

    return primary


def deduplicate_features(features):
    clusters = []

    for feature in features:
        merged = False

        for index, existing in enumerate(clusters):
            if same_event(existing, feature):
                clusters[index] = merge_event_features(existing, feature)
                merged = True
                break

        if not merged:
            p = feature.setdefault("properties", {})

            if "sources" not in p:
                p["sources"] = [p.get("source")] if p.get("source") else []

            if "urls" not in p:
                p["urls"] = [p.get("url")] if p.get("url") else []

            if "merged_titles" not in p:
                p["merged_titles"] = [p.get("title")] if p.get("title") else []

            p["source_count"] = len(p["sources"])
            clusters.append(feature)

    return clusters


def main():
    payload = load_sources()
    features = []
    debug = []

    for country_block in payload.get("countries", []):
        country = country_block.get("country")

        for source_name in country_block.get("sources", []):
            rss_url = RSS_FEEDS.get(source_name)

            if not rss_url:
                debug.append({
                    "source": source_name,
                    "country": country,
                    "status": "error",
                    "reason": "missing_rss_mapping"
                })
                print(f"Missing RSS mapping: {source_name}")
                continue

            print(f"Fetching: {source_name} ({country})")
            fetched, debug_rows = fetch_feed(source_name, rss_url, country)
            features.extend(fetched)
            debug.extend(debug_rows)

    features = deduplicate_features(features)

    geojson = {
        "type": "FeatureCollection",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "features": features
    }

    debug_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "max_items_per_source": MAX_ITEMS_PER_SOURCE,
        "included_count": len(features),
        "debug_count": len(debug),
        "rows": debug[:2000]
    }

    OUTPUT_FILE.write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    DEBUG_FILE.write_text(
        json.dumps(debug_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved debug: {DEBUG_FILE}")
    print(f"Features: {len(features)}")
    print(f"Debug rows: {len(debug)}")


if __name__ == "__main__":
    main()

