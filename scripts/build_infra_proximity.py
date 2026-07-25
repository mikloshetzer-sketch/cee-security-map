import json
import math
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote

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


def event_location_supported(event):
    """
    Strict 8-country CEE location validation.
    Text/URL evidence and coordinates must agree.
    """
    source_file = str(event.get("_source_file") or "").lower()
    props_country = canonical_country(event.get("country"))

    if source_file == "local_events.geojson":
        return props_country in TARGET_COUNTRIES

    if source_file in ALWAYS_RELEVANT_SOURCE_FILES:
        return True

    evidence = event_evidence_text(event)
    summary_url = " ".join([
        str(event.get("summary") or ""),
        url_evidence_text(event.get("url")),
    ])

    city_hits = explicit_target_cities(evidence)
    country_hits = list(dict.fromkeys(explicit_target_countries(evidence)))

    if city_hits:
        city_countries = {country for _, country in city_hits}
        if len(city_countries) != 1:
            return False
        detected_country = next(iter(city_countries))

        if props_country and props_country != detected_country:
            return False
        if not coordinate_in_country(event["lat"], event["lon"], detected_country):
            return False
        if not city_coordinate_supported(event, city_hits):
            return False

        non_title_target_hits = explicit_target_countries(summary_url)
        if has_outside_area_focus(summary_url) and not non_title_target_hits:
            return False

        return True

    if len(country_hits) != 1:
        return False

    detected_country = country_hits[0]
    if props_country and props_country != detected_country:
        return False
    if not coordinate_in_country(event["lat"], event["lon"], detected_country):
        return False

    if has_outside_area_focus(summary_url):
        non_title_target_hits = explicit_target_countries(summary_url)
        if not non_title_target_hits:
            return False

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
    """
    Infrastructure proximity requires an actual incident.
    GDELT military/cyber/drone labels are context only, never sufficient alone.
    """
    source_file = str(event.get("_source_file") or "").lower()
    evidence = event_evidence_text(event)

    if source_file in ALWAYS_RELEVANT_SOURCE_FILES:
        return True

    negative_signal = contains_any(evidence, NEGATIVE_CONTEXT_TERMS)
    event_signal = contains_any(evidence, SECURITY_EVENT_TERMS)
    context_signal = contains_any(evidence, SECURITY_CONTEXT_TERMS)

    if negative_signal and not event_signal:
        return False

    if not event_signal:
        drone_mention = contains_any(
            evidence,
            ["drone", "drón", "uav", "dronă", "drona",
             "dronei", "dronele", "dronelor", "shahed"],
        )
        drone_action = contains_any(
            evidence,
            ["shot down", "downed", "intercepted", "fired at",
             "airspace violation", "airspace breach", "breached airspace",
             "crashed", "explosion",
             "lelőtt", "lelőttek", "légtérsértés", "lezuhant",
             "doborât", "doborâtă", "interceptat", "interceptată",
             "spațiul aerian", "spatiul aerian"],
        )
        return drone_mention and drone_action

    if contains_any(evidence, ["major fire", "large fire", "incendiu major"]) and not context_signal:
        return False

    policy_terms = [
        "cooperation hub", "collaboration", "ties", "strategy",
        "autonomy", "manufacturing", "may deploy", "deployment planned",
        "archaeology", "archaeological", "roman military camp",
        "training exercise announced", "procurement", "investment",
        "defence cooperation", "defense cooperation",
    ]
    if contains_any(evidence, policy_terms):
        disruptive_terms = [
            "explosion", "blast", "shot down", "downed", "intercepted",
            "airspace violation", "cyberattack", "ransomware attack",
            "ddos attack", "data breach", "blackout", "power outage",
            "sabotage", "evacuation", "chemical leak", "gas leak",
            "robbanás", "lelőtt", "légtérsértés", "kibertámadás",
            "áramszünet", "szabotázs", "kiürítés", "doborât", "explozie",
        ]
        if not contains_any(evidence, disruptive_terms):
            return False

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

    return deduplicate_events(events)


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
                    "source": event["source"],
                    "time": event["time"],
                    "url": event["url"],
                    "lat": event["lat"],
                    "lon": event["lon"],
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
                "security": "actual incident evidence required; GDELT category alone rejected",
                "gdelt_geocode": "text + URL + country bounds + city-distance validation",
            },
            "deduplication": {
                "events": "title + url + source + time",
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
