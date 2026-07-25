import json
import math
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone

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
    # Kinetic / military
    "attack",
    "attacked",
    "strike",
    "airstrike",
    "air strike",
    "missile",
    "rocket",
    "shelling",
    "bombing",
    "explosion",
    "blast",
    "drone attack",
    "uav attack",
    "shahed",
    "shot down",
    "shoot down",
    "downed",
    "intercepted",
    "airspace violation",
    "airspace breach",
    "military attack",
    "armed attack",

    # Hungarian
    "támadás",
    "csapás",
    "rakéta",
    "robbanás",
    "dróntámadás",
    "lelőttek",
    "lelőtt",
    "elfogtak",
    "elfogás",
    "légtérsértés",
    "fegyveres támadás",

    # Romanian
    "atac",
    "lovitură",
    "lovitura",
    "rachetă",
    "racheta",
    "explozie",
    "dronă",
    "drona",
    "dronei",
    "doborât",
    "doborâtă",
    "spațiul aerian",
    "spatiul aerian",

    # Cyber / sabotage
    "cyberattack",
    "cyber attack",
    "ransomware",
    "ddos",
    "data breach",
    "sabotage",
    "kibertámadás",
    "szabotázs",
    "cyberatak",
    "küberrünnak",

    # Critical infrastructure disruption / hazardous event
    "industrial accident",
    "chemical leak",
    "gas leak",
    "pipeline leak",
    "blackout",
    "power outage",
    "grid failure",
    "refinery fire",
    "airport closed",
    "airport closure",
    "port closed",
    "port closure",
    "rail disruption",
    "evacuation",
    "emergency shutdown",

    # Hungarian
    "ipari baleset",
    "vegyi szivárgás",
    "gázszivárgás",
    "vezeték sérül",
    "áramszünet",
    "hálózati hiba",
    "finomítótűz",
    "repülőtér lezár",
    "kikötő lezár",
    "vasúti fennakadás",
    "kiürítés",

    # Generic major fire terms. These require context below.
    "major fire",
    "large fire",
    "tűz",
    "incendiu",
    "požiar",
    "požár",
    "pożar",
    "gaisras",
    "ugunsgrēks",
    "tulekahju",
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
    Strict 8-country CEE location gate.

    GDELT coordinates/geocodes are not accepted as standalone proof.
    The article text must support the monitored country or a monitored city.

    local_events.geojson is already produced by the dedicated strict
    8-country whitelist, so its validated country property is accepted.
    """
    source_file = str(event.get("_source_file") or "").lower()
    props_country = canonical_country(event.get("country"))
    text = f"{event.get('title', '')} {event.get('summary', '')}"

    if source_file == "local_events.geojson":
        return props_country in TARGET_COUNTRIES

    if source_file in ALWAYS_RELEVANT_SOURCE_FILES:
        return True

    city_hits = explicit_target_cities(text)

    if city_hits:
        city_countries = {country for _, country in city_hits}

        if len(city_countries) != 1:
            return False

        detected_country = next(iter(city_countries))

        if props_country and props_country != detected_country:
            return False

        # A concrete CEE city is strong enough even when another country is
        # mentioned as actor/context.
        return True

    country_hits = list(dict.fromkeys(explicit_target_countries(text)))

    if len(country_hits) != 1:
        return False

    detected_country = country_hits[0]

    if props_country and props_country != detected_country:
        return False

    # Country-only evidence is not enough when the article clearly focuses
    # on an external country/region.
    if has_outside_area_focus(text):
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
    Second defensive layer for infrastructure proximity.

    An event is allowed to affect infrastructure proximity only when there
    is actual security / disruption evidence. Ordinary news is rejected even
    if it is geocoded directly onto an infrastructure asset.
    """
    source_file = str(event.get("_source_file") or "").lower()
    category = normalize_text(event.get("category"))
    text = f"{event.get('title', '')} {event.get('summary', '')}"

    # Natural hazards / disaster feeds are incident feeds by definition.
    if source_file in ALWAYS_RELEVANT_SOURCE_FILES:
        return True

    strong_category = category in STRONG_SECURITY_CATEGORIES
    event_signal = contains_any(text, SECURITY_EVENT_TERMS)
    context_signal = contains_any(text, SECURITY_CONTEXT_TERMS)
    negative_signal = contains_any(text, NEGATIVE_CONTEXT_TERMS)

    # Strong explicit incident wording always wins.
    if event_signal:
        # Generic "fire" or similarly broad wording should have context unless
        # the source/category itself already marks a strong security event.
        generic_fire_only = (
            contains_any(
                text,
                [
                    "major fire",
                    "large fire",
                    "tűz",
                    "incendiu",
                    "požiar",
                    "požár",
                    "pożar",
                    "gaisras",
                    "ugunsgrēks",
                    "tulekahju",
                ],
            )
            and not contains_any(
                text,
                [
                    "attack",
                    "strike",
                    "explosion",
                    "blast",
                    "robbanás",
                    "explozie",
                    "sabotage",
                    "szabotázs",
                    "cyberattack",
                    "kibertámadás",
                    "blackout",
                    "áramszünet",
                    "evacuation",
                    "kiürítés",
                    "missile",
                    "rakéta",
                    "drone",
                    "drón",
                    "dronă",
                    "drona",
                    "shahed",
                ],
            )
        )

        if generic_fire_only and not context_signal and not strong_category:
            return False

        return True

    # A strong security category can be sufficient, except when the text is
    # clearly ordinary management/economic/construction reporting.
    if strong_category:
        return not negative_signal

    # Contextual categories need BOTH infrastructure/security context and an
    # incident/disruption signal. Category or proximity alone is not enough.
    if category in CONTEXTUAL_SECURITY_CATEGORIES:
        return context_signal and not negative_signal and event_signal

    return False


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
                "security": "security-relevant incidents only",
                "gdelt_geocode": "not accepted as standalone location evidence",
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

