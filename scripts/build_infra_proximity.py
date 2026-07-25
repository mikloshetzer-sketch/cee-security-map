import json
import math
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

INFRA_FILES = [
    DATA / "critical_infrastructure.json",
    DATA / "infra_digital.json",
    DATA / "infra_hazardous.json",
]

EVENT_FILES = [
    DATA / "gdelt.geojson",
    DATA / "gdelt_linked.geojson",
    DATA / "gdelt_crossborder.geojson",
    DATA / "usgs.geojson",
    DATA / "gdacs.geojson",
    DATA / "local_events.geojson",
]

OUTPUT = DATA / "infra_proximity.json"

MAX_DISTANCE_KM = 50

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

COUNTRY_ALIASES = {
    "Czech Republic": "Czechia",
    "Czechia": "Czechia",
    "Hungary": "Hungary",
    "Romania": "Romania",
    "Slovakia": "Slovakia",
    "Poland": "Poland",
    "Lithuania": "Lithuania",
    "Latvia": "Latvia",
    "Estonia": "Estonia",
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
    ],
}

CITY_COUNTRY = {
    # Hungary
    "budapest": "Hungary",
    "paks": "Hungary",
    "kecskemét": "Hungary",
    "kecskemet": "Hungary",
    "szolnok": "Hungary",
    "százhalombatta": "Hungary",
    "szazhalombatta": "Hungary",
    "tiszaújváros": "Hungary",
    "tiszaujvaros": "Hungary",
    "algyő": "Hungary",
    "algyo": "Hungary",
    "hajdúszoboszló": "Hungary",
    "hajduszoboszlo": "Hungary",

    # Romania
    "bucharest": "Romania",
    "bucurești": "Romania",
    "bucuresti": "Romania",
    "constanța": "Romania",
    "constanta": "Romania",
    "cernavodă": "Romania",
    "cernavoda": "Romania",
    "năvodari": "Romania",
    "navodari": "Romania",
    "fetești": "Romania",
    "fetesti": "Romania",
    "buzău": "Romania",
    "buzau": "Romania",
    "brăila": "Romania",
    "braila": "Romania",
    "galați": "Romania",
    "galati": "Romania",
    "sulina": "Romania",
    "padina": "Romania",
    "mihail kogălniceanu": "Romania",
    "mihail kogalniceanu": "Romania",
    "cincu": "Romania",

    # Slovakia
    "bratislava": "Slovakia",
    "košice": "Slovakia",
    "kosice": "Slovakia",
    "mochovce": "Slovakia",
    "jaslovské bohunice": "Slovakia",
    "jaslovske bohunice": "Slovakia",
    "veľké kapušany": "Slovakia",
    "velke kapusany": "Slovakia",
    "sliač": "Slovakia",
    "sliac": "Slovakia",

    # Czechia
    "prague": "Czechia",
    "praha": "Czechia",
    "temelín": "Czechia",
    "temelin": "Czechia",
    "dukovany": "Czechia",
    "litvínov": "Czechia",
    "litvinov": "Czechia",
    "kralupy": "Czechia",
    "ostrava": "Czechia",

    # Poland
    "warsaw": "Poland",
    "warszawa": "Poland",
    "płock": "Poland",
    "plock": "Poland",
    "gdańsk": "Poland",
    "gdansk": "Poland",
    "gdynia": "Poland",
    "rzeszów": "Poland",
    "rzeszow": "Poland",
    "świnoujście": "Poland",
    "swinoujscie": "Poland",
    "bełchatów": "Poland",
    "belchatow": "Poland",

    # Lithuania
    "vilnius": "Lithuania",
    "kaunas": "Lithuania",
    "klaipėda": "Lithuania",
    "klaipeda": "Lithuania",
    "alytus": "Lithuania",
    "šiauliai": "Lithuania",
    "siauliai": "Lithuania",
    "rukla": "Lithuania",

    # Latvia
    "riga": "Latvia",
    "ventspils": "Latvia",
    "ādaži": "Latvia",
    "adazi": "Latvia",
    "lielvārde": "Latvia",
    "lielvarde": "Latvia",
    "inčukalns": "Latvia",
    "incukalns": "Latvia",

    # Estonia
    "tallinn": "Estonia",
    "tartu": "Estonia",
    "narva": "Estonia",
    "paldiski": "Estonia",
    "tapa": "Estonia",
    "ämari": "Estonia",
    "amari": "Estonia",
    "muuga": "Estonia",
}


COUNTRY_BOUNDS = {
    "Hungary": (45.7, 48.7, 16.0, 23.0),
    "Romania": (43.5, 48.4, 20.0, 30.2),
    "Slovakia": (47.7, 49.7, 16.7, 22.7),
    "Czechia": (48.4, 51.2, 12.0, 19.0),
    "Poland": (49.0, 55.1, 14.0, 24.3),
    "Lithuania": (53.8, 56.5, 20.8, 26.9),
    "Latvia": (55.6, 58.2, 20.7, 28.3),
    "Estonia": (57.4, 59.9, 21.5, 28.3),
}

CITY_COORDS_VALIDATION = {
    "budapest": (47.497, 19.040), "paks": (46.622, 18.855),
    "kecskemét": (46.906, 19.691), "kecskemet": (46.906, 19.691),
    "szolnok": (47.174, 20.176), "százhalombatta": (47.317, 18.910),
    "szazhalombatta": (47.317, 18.910), "tiszaújváros": (47.922, 21.052),
    "tiszaujvaros": (47.922, 21.052),

    "bucharest": (44.426, 26.102), "bucurești": (44.426, 26.102),
    "bucuresti": (44.426, 26.102), "constanța": (44.173, 28.638),
    "constanta": (44.173, 28.638), "cernavodă": (44.322, 28.057),
    "cernavoda": (44.322, 28.057), "năvodari": (44.335, 28.642),
    "navodari": (44.335, 28.642), "fetești": (44.366, 27.833),
    "fetesti": (44.366, 27.833), "buzău": (45.150, 26.824),
    "buzau": (45.150, 26.824), "brăila": (45.269, 27.957),
    "braila": (45.269, 27.957), "galați": (45.435, 28.008),
    "galati": (45.435, 28.008), "sulina": (45.156, 29.653),
    "padina": (44.833, 27.117), "mihail kogălniceanu": (44.362, 28.488),
    "mihail kogalniceanu": (44.362, 28.488), "cincu": (45.917, 24.783),

    "bratislava": (48.148, 17.107), "košice": (48.716, 21.261),
    "kosice": (48.716, 21.261), "mochovce": (48.264, 18.455),
    "sliač": (48.637, 19.134), "sliac": (48.637, 19.134),

    "prague": (50.075, 14.438), "praha": (50.075, 14.438),
    "temelín": (49.181, 14.376), "temelin": (49.181, 14.376),
    "dukovany": (49.085, 16.148), "litvínov": (50.604, 13.618),
    "litvinov": (50.604, 13.618), "kralupy": (50.241, 14.312),
    "ostrava": (49.820, 18.262),

    "warsaw": (52.229, 21.012), "warszawa": (52.229, 21.012),
    "płock": (52.576, 19.701), "plock": (52.576, 19.701),
    "gdańsk": (54.383, 18.670), "gdansk": (54.383, 18.670),
    "gdynia": (54.533, 18.550), "rzeszów": (50.041, 21.999),
    "rzeszow": (50.041, 21.999),

    "vilnius": (54.687, 25.279), "kaunas": (54.898, 23.904),
    "klaipėda": (55.706, 21.127), "klaipeda": (55.706, 21.127),
    "alytus": (54.396, 24.041), "šiauliai": (55.893, 23.395),
    "siauliai": (55.893, 23.395), "rukla": (55.000, 24.000),

    "riga": (56.949, 24.105), "ventspils": (57.394, 21.560),
    "ādaži": (57.070, 24.337), "adazi": (57.070, 24.337),
    "lielvārde": (56.778, 24.853), "lielvarde": (56.778, 24.853),
    "inčukalns": (57.098, 24.686), "incukalns": (57.098, 24.686),

    "tallinn": (59.437, 24.753), "tartu": (58.378, 26.729),
    "narva": (59.377, 27.420), "paldiski": (59.350, 24.050),
    "tapa": (59.260, 25.958), "ämari": (59.260, 24.208),
    "amari": (59.260, 24.208), "muuga": (59.500, 24.960),
}

OUTSIDE_AREA_TERMS = [
    "russia", "russian", "moscow", "moskva", "st. petersburg",
    "saint petersburg", "kaliningrad", "kursk", "belgorod", "bryansk",
    "ukraine", "ukrainian", "kyiv", "kiev", "odesa", "odessa",
    "kharkiv", "dnipro", "lviv",
    "belarus", "belarusian", "minsk",
    "moldova", "moldovan", "chișinău", "chisinau",
    "iran", "iranian", "tehran",
    "israel", "israeli", "gaza",
    "france", "french", "spain", "spanish",
    "bulgaria", "bulgarian",
    "united states", "u.s.", "usa", "american",
]


# ---------------------------------------------------------------------
# SECURITY RELEVANCE FILTER
# ---------------------------------------------------------------------
#
# Infrastructure proximity must measure security-relevant incidents near
# critical infrastructure, not ordinary news that happens to be geocoded
# to the same city.
#
# USGS and GDACS are treated as inherently event-based sources.
# Local events have already passed the local-source security filter, but
# are checked here again as a second defensive layer.
# GDELT-derived events require a security category or explicit incident
# wording in their title/summary.
# ---------------------------------------------------------------------

ALWAYS_RELEVANT_SOURCE_FILES = {
    "usgs.geojson",
    "gdacs.geojson",
}

STRONG_SECURITY_CATEGORIES = {
    "cyber",
    "drone",
    "military",
    "explosion",
    "hazardous",
    "conflict",
    "security",
    "terrorism",
    "attack",
    "missile",
    "airstrike",
    "sabotage",
}

CONTEXTUAL_SECURITY_CATEGORIES = {
    "fire",
    "energy",
    "transport",
    "emergency",
    "disaster",
    "local_media",
    "unknown",
}

SECURITY_EVENT_TERMS = [
    "attacked", "attack on", "under attack", "strike hit", "struck",
    "airstrike", "air strike", "missile strike", "missile attack",
    "rocket attack", "shelling", "bombing", "explosion", "blast",
    "shot down", "shoot down", "downed", "intercepted",
    "fired at", "opened fire", "airspace violation", "airspace breach",
    "breached airspace", "crashed", "crash", "detonated",
    "megtámadt", "támadás érte", "csapás érte", "rakétatámadás",
    "robbanás", "lelőtt", "lelőttek", "elfogták", "elfogás",
    "légtérsértés", "lezuhant", "becsapódott",
    "atacat", "lovit", "lovitură", "lovitura", "explozie",
    "doborât", "doborâtă", "doborârea", "interceptat", "interceptată",
    "a tras asupra", "spațiul aerian", "spatiul aerian",
    "prăbușit", "prabusit",
    "cyberattack", "cyber attack", "ransomware attack",
    "ransomware incident", "ddos attack", "data breach",
    "systems compromised", "network compromised",
    "kibertámadás", "adatlopás", "rendszereket feltörték",
    "sabotage", "szabotázs", "industrial accident", "chemical leak",
    "gas leak", "pipeline leak", "blackout", "power outage",
    "grid failure", "refinery fire", "airport closed",
    "airport closure", "port closed", "port closure",
    "rail disruption", "evacuation", "emergency shutdown",
    "ipari baleset", "vegyi szivárgás", "gázszivárgás",
    "áramszünet", "finomítótűz", "kiürítés",
    "major fire", "large fire", "incendiu major",
]

SECURITY_CONTEXT_TERMS = [
    # Infrastructure
    "critical infrastructure",
    "power plant",
    "nuclear",
    "nuclear plant",
    "refinery",
    "pipeline",
    "gas storage",
    "lng",
    "airport",
    "airbase",
    "air base",
    "port",
    "railway",
    "rail",
    "data center",
    "internet exchange",
    "substation",
    "power grid",

    # Security / military
    "military",
    "armed forces",
    "air force",
    "army",
    "navy",
    "nato",
    "border guard",
    "defence",
    "defense",
    "emergency services",

    # Hungarian
    "kritikus infrastruktúra",
    "erőmű",
    "atomerőmű",
    "finomító",
    "vezeték",
    "gáztároló",
    "repülőtér",
    "katonai bázis",
    "kikötő",
    "vasút",
    "adatközpont",
    "villamos hálózat",
    "katonai",
    "légierő",
    "honvédség",
    "nato",
    "határőrség",
    "katasztrófavédelem",

    # Romanian
    "infrastructură critică",
    "infrastructura critica",
    "centrală",
    "centrala",
    "rafinărie",
    "rafinarie",
    "aeroport",
    "bază militară",
    "baza militara",
    "port",
    "cale ferată",
    "cale ferata",
    "forțele aeriene",
    "fortele aeriene",
    "armată",
    "armata",
]

NEGATIVE_CONTEXT_TERMS = [
    # Management / corporate / ordinary economic news
    "appoint",
    "appointed",
    "appointment",
    "resign",
    "resigned",
    "ceo",
    "chief executive",
    "director appointed",
    "management",
    "board member",
    "shareholder",
    "earnings",
    "profit",
    "revenue",
    "investment plan",
    "trade talks",
    "trade agreement",
    "commercial cooperation",

    # Construction / routine infrastructure development
    "construction begins",
    "construction works",
    "construction project",
    "renovation",
    "modernisation",
    "modernization",
    "tender",
    "procurement",
    "formula 1",
    "formula one",
    "f1",
    "grand prix",
    "mercedes-amg",
    "defense autonomy",
    "defence autonomy",
    "defence manufacturing collaboration",
    "defense manufacturing collaboration",

    # Hungarian
    "vezérigazgató",
    "igazgató",
    "vezetőváltás",
    "kinevezték",
    "felmentette",
    "kirúgta",
    "beruházás",
    "kereskedelmi",
    "együttműködés",
    "építkezés",
    "építési munkák",
    "felújítás",
    "közbeszerzés",

    # Romanian
    "director general",
    "numit",
    "demis",
    "investiție",
    "investitie",
    "schimburi comerciale",
    "cooperare comercială",
    "cooperare comerciala",
    "construcție",
    "constructie",
]


# ---------------------------------------------------------------------
# EVENT MODEL FOR INFRASTRUCTURE PROXIMITY
# ---------------------------------------------------------------------

BROAD_CITY_NAMES = {
    "budapest", "bucharest", "bucurești", "bucuresti",
    "bratislava", "prague", "praha", "warsaw", "warszawa",
    "vilnius", "riga", "tallinn",
}

SPORTS_TERMS = [
    "formula 1", "formula one", "f1", "grand prix", "motorsport",
    "speedcafe", "race", "racing", "driver", "mercedes-amg",
]

ARCHAEOLOGY_TERMS = [
    "archaeology", "archaeological", "roman military camp",
    "marcus aurelius", "ancient", "excavation", "buried remains",
]

ORDINARY_CRIME_TERMS = [
    "murder suspect", "attacked victim", "stabbing", "bar fight",
    "domestic violence", "street fight",
]

ROAD_ACCIDENT_TERMS = [
    "road safety", "street racing", "car crash", "traffic accident",
    "road accident", "vehicle collision", "autóbaleset", "közúti baleset",
]

POLICY_ONLY_TERMS = [
    "strategy", "cooperation", "collaboration", "ties", "autonomy",
    "manufacturing", "may deploy", "planned deployment", "procurement",
    "investment", "exercise announced", "evacuation plan",
]

INCIDENT_TYPE_PRIORITY = {
    "drone_airspace": 7,
    "cyber_incident": 6,
    "sabotage": 6,
    "kinetic_attack": 6,
    "explosion": 5,
    "military_accident": 5,
    "hazardous_incident": 5,
    "infrastructure_disruption": 5,
    "bomb_threat": 4,
    "major_fire": 4,
}

# Infrastructure categories that can logically be affected by an event type.
COMPATIBLE_INFRA_CATEGORIES = {
    "cyber_incident": {"digital"},
    "drone_airspace": {"military", "transport", "energy", "hazardous"},
    "military_accident": {"military", "transport"},
    "kinetic_attack": {"military", "transport", "energy", "digital", "hazardous"},
    "explosion": {"military", "transport", "energy", "digital", "hazardous"},
    "sabotage": {"military", "transport", "energy", "digital", "hazardous"},
    "hazardous_incident": {"hazardous", "energy", "transport"},
    "infrastructure_disruption": {"transport", "energy", "digital", "hazardous"},
    "bomb_threat": {"transport", "military"},
    "major_fire": {"hazardous", "energy", "transport", "military"},
}

# Coarse location may be useful for mapping, but not for saying that an
# incident occurred "near" a specific infrastructure asset.
LOCATION_QUALITY_RANK = {
    "country_fallback": 0,
    "country_approx": 0,
    "city_approx": 1,
    "city": 2,
    "specific_place": 3,
    "precise": 4,
}


MIN_INFRA_LOCATION_QUALITY = "city"

LOCATION_ACTION_TERMS = [
    "shot down", "downed", "intercepted", "fired at", "airspace violation",
    "airspace breach", "entered airspace", "crashed", "explosion", "blast",
    "attack", "strike", "cyberattack", "ransomware", "ddos", "data breach",
    "sabotage", "blackout", "power outage", "chemical leak", "gas leak",
    "evacuation", "bomb threat", "lelőtt", "lelőttek", "elfogták",
    "légtérsértés", "lezuhant", "robbanás", "támadás", "kibertámadás",
    "áramszünet", "szabotázs", "kiürítés", "doborât", "doborâtă",
    "interceptat", "interceptată", "spațiul aerian", "spatiul aerian",
    "prăbușit", "prabusit", "explozie", "atac",
]

NAMED_PLACE_COORDS = {
    "mihail kogălniceanu air base": ("Romania", 44.362, 28.488),
    "mihail kogalniceanu air base": ("Romania", 44.362, 28.488),
    "mihail kogălniceanu base": ("Romania", 44.362, 28.488),
    "mihail kogalniceanu base": ("Romania", 44.362, 28.488),
    "57th air base": ("Romania", 44.362, 28.488),
    "fetești air base": ("Romania", 44.393, 27.727),
    "fetesti air base": ("Romania", 44.393, 27.727),
    "86th air base": ("Romania", 44.393, 27.727),
    "cernavodă nuclear power plant": ("Romania", 44.322, 28.057),
    "cernavoda nuclear power plant": ("Romania", 44.322, 28.057),
    "kecskemét air base": ("Hungary", 46.917, 19.749),
    "kecskemet air base": ("Hungary", 46.917, 19.749),
    "paks nuclear power plant": ("Hungary", 46.573, 18.854),
    "sliač air base": ("Slovakia", 48.637, 19.134),
    "sliac air base": ("Slovakia", 48.637, 19.134),
    "mochovce nuclear power plant": ("Slovakia", 48.263, 18.455),
    "čáslav air base": ("Czechia", 49.939, 15.382),
    "caslav air base": ("Czechia", 49.939, 15.382),
    "temelín nuclear power plant": ("Czechia", 49.181, 14.376),
    "temelin nuclear power plant": ("Czechia", 49.181, 14.376),
    "dukovany nuclear power plant": ("Czechia", 49.085, 16.149),
    "rzeszów-jasionka airport": ("Poland", 50.110, 22.019),
    "rzeszow-jasionka airport": ("Poland", 50.110, 22.019),
    "jasionka airport": ("Poland", 50.110, 22.019),
    "šiauliai air base": ("Lithuania", 55.894, 23.395),
    "siauliai air base": ("Lithuania", 55.894, 23.395),
    "rukla military base": ("Lithuania", 55.000, 24.000),
    "lielvārde air base": ("Latvia", 56.778, 24.853),
    "lielvarde air base": ("Latvia", 56.778, 24.853),
    "ādaži military base": ("Latvia", 57.070, 24.337),
    "adazi military base": ("Latvia", 57.070, 24.337),
    "ämari air base": ("Estonia", 59.260, 24.208),
    "amari air base": ("Estonia", 59.260, 24.208),
    "tapa military base": ("Estonia", 59.260, 25.958),
}

DEDUP_MAX_HOURS = 36.0
DEDUP_MAX_DISTANCE_KM = 140.0
DEDUP_TEXT_THRESHOLD = 0.24

def load_json(path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def contains_term(text, term):
    t = normalize_text(text)
    k = normalize_text(term)

    if not k:
        return False

    if re.fullmatch(r"[\wÀ-ž-]+", k, flags=re.UNICODE):
        return re.search(
            rf"(?<!\w){re.escape(k)}(?!\w)",
            t,
            flags=re.UNICODE,
        ) is not None

    return k in t


def contains_phrase(text, phrase):
    return contains_term(text, phrase)


def contains_any(text, terms):
    return any(contains_term(text, term) for term in terms)


def url_evidence_text(url):
    if not url:
        return ""

    try:
        parsed = urlparse(str(url))
        raw = unquote(f"{parsed.netloc} {parsed.path} {parsed.query}")
    except Exception:
        raw = str(url)

    raw = re.sub(r"[-_/?.=&+]+", " ", raw)
    return normalize_text(raw)


def event_evidence_text(event):
    return " ".join(
        part for part in [
            str(event.get("title") or ""),
            str(event.get("summary") or ""),
            url_evidence_text(event.get("url")),
        ]
        if part
    )


def coordinate_in_country(lat, lon, country):
    bounds = COUNTRY_BOUNDS.get(country)
    if not bounds:
        return False
    min_lat, max_lat, min_lon, max_lon = bounds
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def city_coordinate_supported(event, city_hits, max_distance_km=90.0):
    if not city_hits:
        return True

    checked = False
    for city, _country in city_hits:
        coords = CITY_COORDS_VALIDATION.get(city)
        if not coords:
            continue
        checked = True
        if haversine_km(event["lat"], event["lon"], coords[0], coords[1]) <= max_distance_km:
            return True

    return not checked


def canonical_country(value):
    if not value:
        return None

    raw = str(value).strip()

    if raw in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[raw]

    low = normalize_text(raw)

    for alias, country in COUNTRY_ALIASES.items():
        if normalize_text(alias) == low:
            return country

    return None


def explicit_target_countries(text):
    hits = []

    for country, terms in COUNTRY_TERMS.items():
        if any(contains_term(text, term) for term in terms):
            hits.append(country)

    for city, country in CITY_COUNTRY.items():
        if contains_term(text, city) and country not in hits:
            hits.append(country)

    return hits


def explicit_target_cities(text):
    hits = []

    for city, country in CITY_COUNTRY.items():
        if contains_term(text, city):
            hits.append((city, country))

    return hits


def has_outside_area_focus(text):
    return contains_any(text, OUTSIDE_AREA_TERMS)


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


def source_domain_country(url):
    if not url:
        return None

    try:
        host = urlparse(str(url)).netloc.lower().split(":")[0]
    except Exception:
        return None

    tld_map = {
        ".hu": "Hungary",
        ".ro": "Romania",
        ".sk": "Slovakia",
        ".cz": "Czechia",
        ".pl": "Poland",
        ".lt": "Lithuania",
        ".lv": "Latvia",
        ".ee": "Estonia",
    }

    for tld, country in tld_map.items():
        if host.endswith(tld):
            return country

    return None


def actual_location_evidence(event):
    """
    Evidence that comes from the underlying article rather than a synthetic
    GDELT location label.

    gdelt_linked titles are frequently generated place labels such as
    "Budapest, Budapest, Hungary", so they are deliberately excluded here.
    """
    source_file = str(event.get("_source_file") or "").lower()

    parts = [
        str(event.get("summary") or ""),
        url_evidence_text(event.get("url")),
    ]

    if source_file != "gdelt_linked.geojson":
        parts.insert(0, str(event.get("title") or ""))

    return " ".join(part for part in parts if part)


def classify_incident_type(event):
    evidence = event_evidence_text(event)
    actual = actual_location_evidence(event)

    # Hard exclusions first.
    if contains_any(actual, SPORTS_TERMS):
        return None

    if contains_any(actual, ARCHAEOLOGY_TERMS):
        return None

    if contains_any(actual, ORDINARY_CRIME_TERMS):
        return None

    if contains_any(actual, ROAD_ACCIDENT_TERMS):
        return None

    drone_terms = [
        "drone", "drón", "uav", "shahed", "dronă", "drona",
        "dronei", "dronele", "dronelor",
    ]
    drone_actions = [
        "shot down", "shoot down", "downed", "intercepted", "fired at",
        "airspace violation", "airspace breach", "breached airspace",
        "entered airspace", "airspace intercept",
        "lelőtt", "lelőttek", "légtérsértés", "elfogás",
        "doborât", "doborâtă", "doborârea", "interceptat",
        "interceptată", "spațiul aerian", "spatiul aerian",
    ]

    if contains_any(evidence, drone_terms) and contains_any(evidence, drone_actions):
        return "drone_airspace"

    if contains_any(
        evidence,
        [
            "cyberattack", "cyber attack", "ransomware attack",
            "ransomware incident", "ddos attack", "data breach",
            "systems compromised", "network compromised",
            "kibertámadás", "adatlopás",
        ],
    ):
        return "cyber_incident"

    if contains_any(evidence, ["sabotage", "szabotázs"]):
        return "sabotage"

    # A military aircraft/vehicle crash is relevant. A generic road/sports
    # crash is not.
    military_vehicle = contains_any(
        evidence,
        [
            "military helicopter", "army helicopter", "air force helicopter",
            "military aircraft", "army aircraft", "air force aircraft",
            "military jet", "fighter jet", "army vehicle",
            "katonai helikopter", "katonai repülő", "légierő",
        ],
    )
    if military_vehicle and contains_any(
        evidence,
        ["crash", "crashed", "lezuhant", "prăbușit", "prabusit"],
    ):
        return "military_accident"

    if contains_any(
        evidence,
        [
            "missile attack", "missile strike", "rocket attack",
            "airstrike", "air strike", "shelling", "bombing",
            "armed attack", "támadás érte", "rakétatámadás",
        ],
    ):
        return "kinetic_attack"

    if contains_any(evidence, ["explosion", "blast", "robbanás", "explozie"]):
        return "explosion"

    if contains_any(
        evidence,
        [
            "chemical leak", "gas leak", "pipeline leak",
            "industrial accident", "vegyi szivárgás", "gázszivárgás",
        ],
    ):
        return "hazardous_incident"

    if contains_any(
        evidence,
        [
            "blackout", "power outage", "grid failure",
            "airport closed", "airport closure", "port closed",
            "port closure", "rail disruption", "emergency shutdown",
            "áramszünet", "repülőtér lezár", "kikötő lezár",
        ],
    ):
        return "infrastructure_disruption"

    if contains_any(
        evidence,
        ["bomb threat", "bomb scare", "bomba fenyegetés", "bombariadó"],
    ) and contains_any(
        evidence,
        ["evacuation", "evacuated", "kiürítés"],
    ):
        return "bomb_threat"

    if contains_any(
        evidence,
        ["major fire", "large fire", "refinery fire", "finomítótűz"],
    ):
        return "major_fire"

    return None


def infer_event_country(event):
    actual = actual_location_evidence(event)
    hits = list(dict.fromkeys(explicit_target_countries(actual)))

    if len(hits) == 1:
        return hits[0]

    domain_country = source_domain_country(event.get("url"))
    if domain_country in TARGET_COUNTRIES:
        # Domain evidence is only a fallback when no contradictory monitored
        # country is present in the actual article evidence.
        if not hits:
            return domain_country

    return None


def infer_location_quality(event, detected_country):
    source_file = str(event.get("_source_file") or "").lower()
    props_quality = normalize_text(event.get("geocode_quality"))
    actual = actual_location_evidence(event)
    actual_city_hits = [
        (city, country)
        for city, country in explicit_target_cities(actual)
        if country == detected_country
    ]

    if source_file == "local_events.geojson":
        if props_quality == "city":
            return "city"
        return "country_fallback"

    # For GDELT, only a city/place independently present in the underlying
    # article evidence can make the coordinate usable for proximity.
    if actual_city_hits:
        broad_only = all(city in BROAD_CITY_NAMES for city, _ in actual_city_hits)

        if not city_coordinate_supported(event, actual_city_hits, max_distance_km=70.0):
            return "country_approx"

        return "city_approx" if broad_only else "specific_place"

    return "country_approx"


def text_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    words_a = set(re.findall(r"[\wÀ-ž-]{3,}", a, flags=re.UNICODE))
    words_b = set(re.findall(r"[\wÀ-ž-]{3,}", b, flags=re.UNICODE))

    if words_a and words_b:
        jaccard = len(words_a & words_b) / len(words_a | words_b)
    else:
        jaccard = 0.0

    sequence = SequenceMatcher(None, a, b).ratio()
    return max(jaccard, sequence * 0.65)


def incident_signature(event):
    evidence = event_evidence_text(event)
    signature = set()

    for token, terms in {
        "drone": ["drone", "drón", "uav", "shahed", "dronă", "drona"],
        "shootdown": ["shot down", "downed", "intercepted", "lelőtt", "doborât"],
        "airspace": ["airspace", "légtér", "spațiul aerian", "spatiul aerian"],
        "f16": ["f-16", "f16"],
        "cyber": ["cyberattack", "cyber attack", "ransomware", "ddos"],
        "explosion": ["explosion", "blast", "robbanás", "explozie"],
        "bomb_threat": ["bomb threat", "bomb scare", "bombariadó"],
        "military_crash": ["military helicopter", "military aircraft", "fighter jet"],
    }.items():
        if contains_any(evidence, terms):
            signature.add(token)

    return signature



def extract_named_place_candidates(event):
    evidence = actual_location_evidence(event)
    candidates = []
    for place_name, (country, lat, lon) in NAMED_PLACE_COORDS.items():
        if contains_term(evidence, place_name):
            candidates.append({
                "country": country, "lat": lat, "lon": lon,
                "place": place_name, "quality": "precise",
                "method": "named_place", "score": 100,
            })
    return candidates


def extract_city_candidates(event):
    evidence = actual_location_evidence(event)
    if not contains_any(evidence, LOCATION_ACTION_TERMS):
        return []

    candidates = []
    for city, country in explicit_target_cities(evidence):
        coords = CITY_COORDS_VALIDATION.get(city)
        if coords:
            candidates.append({
                "country": country, "lat": coords[0], "lon": coords[1],
                "place": city, "quality": "city",
                "method": "article_city", "score": 70,
            })
    return candidates


def extract_original_coordinate_candidate(event):
    country = canonical_country(event.get("country")) or infer_event_country(event)
    if country not in TARGET_COUNTRIES:
        return None

    quality = event.get("location_quality") or infer_location_quality(event, country)
    rank = LOCATION_QUALITY_RANK.get(quality, 0)
    if rank < LOCATION_QUALITY_RANK["city"]:
        return None

    return {
        "country": country, "lat": event["lat"], "lon": event["lon"],
        "place": event.get("place"), "quality": quality,
        "method": "source_coordinate", "score": 60 + rank,
    }


def geolocation_candidates(event):
    candidates = extract_named_place_candidates(event) + extract_city_candidates(event)
    original = extract_original_coordinate_candidate(event)
    if original:
        candidates.append(original)
    return candidates


def choose_cluster_geolocation(cluster_events):
    candidates = []
    for event in cluster_events:
        for candidate in geolocation_candidates(event):
            c = dict(candidate)
            c["source"] = event.get("source")
            candidates.append(c)

    if not candidates:
        return None

    groups = []
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        found = None
        for g in groups:
            if g["country"] == c["country"] and haversine_km(
                g["lat"], g["lon"], c["lat"], c["lon"]
            ) <= 35:
                found = g
                break

        if found:
            found["members"].append(c)
            if c.get("source"):
                found["sources"].add(c["source"])
            found["score"] += c["score"]
            if LOCATION_QUALITY_RANK.get(c["quality"], 0) > LOCATION_QUALITY_RANK.get(found["quality"], 0):
                found.update({
                    "lat": c["lat"], "lon": c["lon"], "place": c["place"],
                    "quality": c["quality"], "method": c["method"],
                })
        else:
            groups.append({
                "country": c["country"], "lat": c["lat"], "lon": c["lon"],
                "place": c["place"], "quality": c["quality"],
                "method": c["method"], "score": c["score"],
                "sources": {c.get("source")} if c.get("source") else set(),
                "members": [c],
            })

    groups.sort(
        key=lambda g: (
            LOCATION_QUALITY_RANK.get(g["quality"], 0),
            len(g["sources"]), g["score"]
        ),
        reverse=True,
    )
    best = groups[0]

    if len(groups) > 1:
        second = groups[1]
        if (
            second["country"] != best["country"]
            and len(second["sources"]) >= len(best["sources"])
            and second["score"] >= best["score"] * 0.85
        ):
            return None

    return {
        "country": best["country"], "lat": round(best["lat"], 6),
        "lon": round(best["lon"], 6), "place": best["place"],
        "location_quality": best["quality"],
        "geolocation_method": best["method"],
        "geolocation_source_count": len(best["sources"]),
        "geolocation_evidence_count": len(best["members"]),
    }


def enrich_incident_geolocation(event):
    geo = choose_cluster_geolocation(event.get("_cluster_members") or [event])
    if not geo:
        event["geolocation_method"] = "unresolved"
        event["geolocation_source_count"] = 0
        event["geolocation_evidence_count"] = 0
        return event
    event.update(geo)
    return event


def same_incident(a, b):
    if a.get("country") != b.get("country"):
        return False

    if a.get("incident_type") != b.get("incident_type"):
        return False

    time_a = parse_event_time(a.get("time"))
    time_b = parse_event_time(b.get("time"))

    if time_a and time_b:
        if abs((time_a - time_b).total_seconds()) / 3600.0 > DEDUP_MAX_HOURS:
            return False

    sig_a = incident_signature(a)
    sig_b = incident_signature(b)

    # Strong multilingual semantic identity.
    if {"drone", "shootdown"}.issubset(sig_a & sig_b):
        return True

    if "cyber" in (sig_a & sig_b):
        return text_similarity(event_evidence_text(a), event_evidence_text(b)) >= 0.18

    # Geographic + textual identity.
    distance = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    if distance <= DEDUP_MAX_DISTANCE_KM:
        return text_similarity(event_evidence_text(a), event_evidence_text(b)) >= DEDUP_TEXT_THRESHOLD

    return False


def location_quality_rank(event):
    return LOCATION_QUALITY_RANK.get(event.get("location_quality"), 0)


def merge_incidents(primary, incoming):
    sources = list(primary.get("sources") or [primary.get("source")])
    for source in incoming.get("sources") or [incoming.get("source")]:
        if source and source not in sources:
            sources.append(source)

    urls = list(primary.get("urls") or ([primary.get("url")] if primary.get("url") else []))
    for url in incoming.get("urls") or ([incoming.get("url")] if incoming.get("url") else []):
        if url and url not in urls:
            urls.append(url)

    primary["sources"] = sources
    primary["urls"] = urls
    primary["source_count"] = len(sources)

    members = list(primary.get("_cluster_members") or [dict(primary)])
    members.extend(incoming.get("_cluster_members") or [dict(incoming)])
    primary["_cluster_members"] = members

    if len(str(incoming.get("title") or "")) > len(str(primary.get("title") or "")):
        primary["title"] = incoming.get("title")

    t1 = parse_event_time(primary.get("time"))
    t2 = parse_event_time(incoming.get("time"))
    if t2 and (not t1 or t2 < t1):
        primary["time"] = incoming.get("time")

    return primary


def deduplicate_incidents(events):
    clusters = []
    ordered = sorted(
        events,
        key=lambda e: INCIDENT_TYPE_PRIORITY.get(e.get("incident_type"), 0),
        reverse=True,
    )

    for event in ordered:
        event["_cluster_members"] = [dict(event)]
        merged = False

        for i, existing in enumerate(clusters):
            if same_incident(existing, event):
                clusters[i] = merge_incidents(existing, event)
                merged = True
                break

        if not merged:
            event["sources"] = [event.get("source")] if event.get("source") else []
            event["urls"] = [event.get("url")] if event.get("url") else []
            event["source_count"] = len(event["sources"])
            clusters.append(event)

    enriched = []
    for event in clusters:
        event = enrich_incident_geolocation(event)
        event.pop("_cluster_members", None)
        enriched.append(event)

    return enriched


def event_infra_compatible(event, infra):
    incident_type = event.get("incident_type")
    infra_category = normalize_text(infra.get("category"))
    allowed = COMPATIBLE_INFRA_CATEGORIES.get(incident_type, set())

    if infra_category not in allowed:
        return False

    quality = event.get("location_quality")
    if LOCATION_QUALITY_RANK.get(quality, 0) < LOCATION_QUALITY_RANK[MIN_INFRA_LOCATION_QUALITY]:
        return False

    if quality == "city":
        return event.get("geolocation_method") in {"article_city", "source_coordinate"}

    return True


def event_location_supported(event):
    """
    Country validation only. Precise geolocation is deferred until after
    incident clustering, so all reports can contribute location evidence.
    """
    source_file = str(event.get("_source_file") or "").lower()

    if source_file in ALWAYS_RELEVANT_SOURCE_FILES:
        return True

    detected_country = infer_event_country(event)
    if source_file == "local_events.geojson":
        detected_country = canonical_country(event.get("country"))

    if detected_country not in TARGET_COUNTRIES:
        return False

    event["country"] = detected_country
    event["location_quality"] = infer_location_quality(event, detected_country)
    return True


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def stable_id(*parts):
    raw = "|".join(str(p or "") for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_infrastructure():
    items = []

    for path in INFRA_FILES:
        payload = load_json(path)

        if not payload:
            continue

        for item in payload.get("items", []):
            country = canonical_country(item.get("country"))

            if country not in TARGET_COUNTRIES:
                continue

            try:
                item["lat"] = float(item["lat"])
                item["lon"] = float(item["lon"])
                item["criticality"] = float(item.get("criticality", 5))
            except Exception:
                continue

            item["country"] = country
            item["_source_file"] = path.name
            items.append(item)

    return items


def event_security_relevant(event):
    incident_type = classify_incident_type(event)

    if not incident_type:
        return False

    event["incident_type"] = incident_type
    return True


def load_events():
    events = []

    for path in EVENT_FILES:
        payload = load_json(path)

        if not payload:
            continue

        for feature in payload.get("features", []):
            geom = feature.get("geometry", {})
            props = feature.get("properties", {})

            if geom.get("type") != "Point":
                continue

            coords = geom.get("coordinates", [])

            if len(coords) < 2:
                continue

            try:
                lon = float(coords[0])
                lat = float(coords[1])
            except Exception:
                continue

            title = (
                props.get("title")
                or props.get("name")
                or props.get("type")
                or "Unnamed"
            )

            summary = (
                props.get("summary")
                or props.get("description")
                or props.get("snippet")
                or ""
            )

            url = props.get("url") or props.get("search_url")
            source = props.get("source") or props.get("domain") or path.stem
            category = (
                props.get("category")
                or props.get("gdelt_bucket")
                or props.get("type")
                or "unknown"
            )
            time = (
                props.get("time")
                or props.get("datetime")
                or props.get("date")
            )
            country = (
                props.get("country")
                or props.get("event_country")
                or props.get("location_country")
            )

            event_id = (
                props.get("id")
                or stable_id(title, url, source, time, lat, lon)
            )

            event = {
                "id": event_id,
                "title": title,
                "summary": summary,
                "category": category,
                "source": source,
                "time": time,
                "url": url,
                "country": country,
                "place": props.get("place"),
                "geocode_quality": props.get("geocode_quality"),
                "lat": lat,
                "lon": lon,
                "_source_file": path.name,
            }

            # Gate 1: must genuinely belong to one of the eight monitored
            # countries. GDELT geocode alone is not enough.
            if not event_location_supported(event):
                continue

            # Gate 2: must be a real security/disruption event.
            if not event_security_relevant(event):
                continue

            events.append(event)

    return deduplicate_incidents(deduplicate_events(events))


def deduplicate_events(events):
    seen = {}

    for event in events:
        key = stable_id(
            event.get("title"),
            event.get("url"),
            event.get("source"),
            event.get("time"),
        )

        if key not in seen:
            seen[key] = event

    return list(seen.values())


def calculate_level(distance, criticality):
    if distance <= 5 and criticality >= 8:
        return "critical"

    if distance <= 15 and criticality >= 8:
        return "high"

    if distance <= 30:
        return "medium"

    return "watch"


def calculate_score(distance, criticality):
    distance_factor = max(0, 1 - (distance / MAX_DISTANCE_KM))
    return round((criticality * 10) * distance_factor, 2)


def level_weight(level):
    return {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "watch": 1,
    }.get(level, 0)


def is_better_match(candidate, current):
    if current is None:
        return True

    if level_weight(candidate["level"]) != level_weight(current["level"]):
        return (
            level_weight(candidate["level"])
            > level_weight(current["level"])
        )

    if candidate["score"] != current["score"]:
        return candidate["score"] > current["score"]

    return candidate["distance_km"] < current["distance_km"]


def build_matches(infrastructure, events):
    all_matches = []

    best_by_infrastructure = {}
    best_by_event_infra = {}

    for event in events:
        for infra in infrastructure:
            if not event_infra_compatible(event, infra):
                continue

            distance = haversine_km(
                event["lat"],
                event["lon"],
                infra["lat"],
                infra["lon"],
            )

            if distance > MAX_DISTANCE_KM:
                continue

            level = calculate_level(
                distance,
                infra.get("criticality", 5),
            )

            score = calculate_score(
                distance,
                infra.get("criticality", 5),
            )

            match = {
                "id": stable_id(
                    event["id"],
                    infra.get("id"),
                    distance,
                ),
                "level": level,
                "score": score,
                "distance_km": round(distance, 2),
                "event": {
                    "id": event["id"],
                    "title": event["title"],
                    "category": event["category"],
                    "incident_type": event.get("incident_type"),
                    "source": event["source"],
                    "source_count": event.get("source_count", 1),
                    "sources": event.get("sources", [event["source"]]),
                    "time": event["time"],
                    "url": event["url"],
                    "urls": event.get("urls", [event["url"]] if event["url"] else []),
                    "lat": event["lat"],
                    "lon": event["lon"],
                    "location_quality": event.get("location_quality"),
                    "place": event.get("place"),
                    "geolocation_method": event.get("geolocation_method"),
                    "geolocation_source_count": event.get("geolocation_source_count", 0),
                    "geolocation_evidence_count": event.get("geolocation_evidence_count", 0),
                    "source_file": event["_source_file"],
                },
                "infrastructure": {
                    "id": infra.get("id"),
                    "name": infra.get("name"),
                    "country": infra.get("country"),
                    "city": infra.get("city"),
                    "category": infra.get("category"),
                    "subtype": infra.get("subtype"),
                    "criticality": infra.get("criticality"),
                    "operator": infra.get("operator"),
                    "lat": infra.get("lat"),
                    "lon": infra.get("lon"),
                    "source_file": infra.get("_source_file"),
                },
            }

            all_matches.append(match)

            infra_id = infra.get("id") or infra.get("name")
            event_infra_key = stable_id(event["id"], infra_id)

            if event_infra_key not in best_by_event_infra:
                best_by_event_infra[event_infra_key] = match
            elif is_better_match(
                match,
                best_by_event_infra[event_infra_key],
            ):
                best_by_event_infra[event_infra_key] = match

            if infra_id not in best_by_infrastructure:
                best_by_infrastructure[infra_id] = match
            elif is_better_match(
                match,
                best_by_infrastructure[infra_id],
            ):
                best_by_infrastructure[infra_id] = match

    unique_event_infra_matches = list(
        best_by_event_infra.values()
    )
    top_by_infrastructure = list(
        best_by_infrastructure.values()
    )

    unique_event_infra_matches.sort(
        key=lambda x: (
            level_weight(x["level"]),
            x["score"],
            -x["distance_km"],
        ),
        reverse=True,
    )

    top_by_infrastructure.sort(
        key=lambda x: (
            level_weight(x["level"]),
            x["score"],
            -x["distance_km"],
        ),
        reverse=True,
    )

    return (
        all_matches,
        unique_event_infra_matches,
        top_by_infrastructure,
    )


def build():
    infrastructure = load_infrastructure()
    events = load_events()

    all_matches, unique_matches, top_by_infra = build_matches(
        infrastructure,
        events,
    )

    report = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "infrastructure_count": len(infrastructure),
            "event_count": len(events),
            "match_count": len(unique_matches),
            "raw_match_count": len(all_matches),
            "top_infrastructure_count": len(top_by_infra),
            "max_distance_km": MAX_DISTANCE_KM,
            "target_countries": sorted(TARGET_COUNTRIES),
            "filters": {
                "location": "strict 8-country text-supported whitelist",
                "security": "typed actual incidents only; sports/crime/road-accident/policy noise rejected",
                "gdelt_geocode": "synthetic GDELT place labels not trusted; article evidence + quality tiers",
            },
            "event_model": {
                "deduplication": "country + incident_type + time + semantic similarity",
                "location_policy": "multi-source geolocation after incident clustering; unresolved/coarse locations excluded",
                "compatibility": "incident type must match infrastructure category",
                "geolocation": "named place > article city > trusted source coordinate; GDELT fallback never upgraded automatically"
            },
            "deduplication": {
                "events": "exact article dedup then incident-level clustering",
                "matches": "best event-infrastructure pair",
                "top_matches": "one best match per infrastructure asset",
            },
        },
        "top_matches": top_by_infra[:50],
        "matches": unique_matches,
        "raw_matches": all_matches[:500],
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {OUTPUT}")
    print(f"Infrastructure: {len(infrastructure)}")
    print(f"Validated CEE security events: {len(events)}")
    print(f"Matches: {len(unique_matches)}")
    print(
        f"Top infrastructure matches: "
        f"{len(top_by_infra)}"
    )


if __name__ == "__main__":
    build()
