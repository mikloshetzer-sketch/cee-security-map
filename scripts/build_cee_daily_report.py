import json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
DOCS = ROOT / "docs"

REPORT_DATA_DIR = DATA / "reports"
REPORT_HTML_DIR = DOCS / "reports"

SUMMARY = DATA / "summary.json"
WEEKLY = DATA / "weekly.json"
RISK = DATA / "risk_daily.json"
META = DATA / "meta.json"

LOCAL_EVENTS = DATA / "local_events.geojson"
INFRA_PROX = DATA / "infra_proximity.json"


def ensure_dirs():
    REPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_HTML_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def utc_now():
    return datetime.now(timezone.utc)


def esc(value):
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def risk_class(level):
    level = str(level or "normal").lower()
    if level in {"critical", "tense", "elevated", "normal", "high", "medium", "watch"}:
        return level
    return "normal"


def country_region(country):
    c = str(country or "")
    if c in {"Estonia", "Latvia", "Lithuania"}:
        return "Baltikum"
    if c in {"Poland", "Czech Republic", "Czechia", "Slovakia", "Hungary"}:
        return "V4 térség"
    if c in {"Romania"}:
        return "Fekete-tengeri perem"
    return "Egyéb CEE"


def top_sector_from_matches(matches):
    cats = Counter()
    for m in matches:
        infra = m.get("infrastructure", {})
        cat = infra.get("category") or infra.get("subtype") or "infrastructure"
        cats[str(cat)] += 1
    return cats.most_common(1)[0][0] if cats else "infrastruktúra"


def build_report():
    summary = load_json(SUMMARY, {})
    weekly = load_json(WEEKLY, {})
    risk = load_json(RISK, {})
    meta = load_json(META, {})
    local_events = load_json(LOCAL_EVENTS, {"features": []})
    infra = load_json(INFRA_PROX, {"matches": []})

    countries = risk.get("countries", [])
    countries_sorted = sorted(
        countries,
        key=lambda x: x.get("overall_score", 0),
        reverse=True
    )

    matches = infra.get("matches", [])
    top_prox = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)[:12]
    local = local_events.get("features", [])[:12]

    region_counter = Counter()
    for c in countries_sorted:
        region_counter[country_region(c.get("country"))] += c.get("overall_score", 0)

    return {
        "generated_utc": utc_now().isoformat(),
        "headline": "CEE Infrastructure & Security Daily Brief",
        "region": risk.get("region", {}),
        "top_countries": countries_sorted[:8],
        "summary_bullets": summary.get("bullets", []),
        "weekly_bullets": weekly.get("bullets", []),
        "top_local_events": local,
        "top_infrastructure_matches": top_prox,
        "all_matches": matches,
        "meta": meta,
        "region_counter": dict(region_counter),
        "top_sector": top_sector_from_matches(matches),
    }


def save_report_json(report):
    today = utc_now().strftime("%Y-%m-%d")

    (REPORT_DATA_DIR / "cee_daily_report_latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    (REPORT_DATA_DIR / f"cee_daily_report_{today}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def html_country_cards(countries):
    if not countries:
        return "<p>Nincs elérhető országkockázati adat.</p>"

    out = []

    for c in countries:
        level = risk_class(c.get("overall"))
        drivers = ", ".join(c.get("drivers", [])) or "automatikus OSINT jelzés"
        out.append(f"""
        <div class="country-card {level}">
          <div class="country-name">{esc(c.get("country"))}</div>
          <div class="country-score">{round(c.get("overall_score", 0), 2)}</div>
          <div class="country-level">{esc(c.get("overall"))}</div>
          <div class="country-drivers">{esc(drivers)}</div>
        </div>
        """)

    return "\n".join(out)


def html_local_events(events):
    if not events:
        return "<p>Nincs kiemelt lokális esemény.</p>"

    out = []

    for item in events:
        p = item.get("properties", {})
        out.append(f"""
        <div class="event-card">
          <div class="event-kicker">{esc(p.get("country"))} • {esc(p.get("category"))} • {esc(p.get("severity"))}</div>
          <h3>{esc(p.get("title"))}</h3>
          <p>{esc(p.get("summary"))}</p>
          <a href="{esc(p.get("url") or "#")}" target="_blank" rel="noopener">Forrás megnyitása</a>
        </div>
        """)

    return "\n".join(out)


def html_proximity(matches):
    if not matches:
        return "<p>Nincs infrastruktúra-közeli kiemelt találat.</p>"

    out = []

    for m in matches:
        infra = m.get("infrastructure", {})
        event = m.get("event", {})
        level = risk_class(m.get("level"))

        out.append(f"""
        <div class="infra-card {level}">
          <div class="event-kicker">{esc(infra.get("country"))} • {esc(infra.get("category"))} • {round(m.get("distance_km", 0), 1)} km</div>
          <h3>{esc(infra.get("name"))}</h3>
          <p>{esc(event.get("title"))}</p>
          <div class="level-pill {level}">{esc(m.get("level"))}</div>
          <a href="{esc(event.get("url") or "#")}" target="_blank" rel="noopener">Forrás megnyitása</a>
        </div>
        """)

    return "\n".join(out)


def html_region_dashboard(report):
    countries = report.get("top_countries", [])
    groups = defaultdict(list)

    for c in countries:
        groups[country_region(c.get("country"))].append(c)

    region_names = ["Baltikum", "V4 térség", "Fekete-tengeri perem", "Egyéb CEE"]
    colors = {
        "Baltikum": "blue",
        "V4 térség": "green",
        "Fekete-tengeri perem": "orange",
        "Egyéb CEE": "red",
    }

    out = []

    for name in region_names:
        rows = groups.get(name, [])
        total = round(sum(x.get("overall_score", 0) for x in rows), 2)
        top = rows[0].get("country") if rows else "—"
        color = colors.get(name, "blue")

        out.append(f"""
        <div class="region-card {color}">
          <div class="region-head">{esc(name)}</div>
          <div class="region-body">
            <div class="big-number">{total}</div>
            <div class="small-label">összesített risk score</div>
            <div class="region-line"><b>Fő fókusz:</b> {esc(top)}</div>
            <div class="mini-bar"><span style="width:{min(100, total * 12)}%"></span></div>
          </div>
        </div>
        """)

    return "\n".join(out)


def build_html(report):
    region = report.get("region", {})
    generated = report.get("generated_utc", "")
    today = utc_now().strftime("%Y-%m-%d")

    matches = report.get("all_matches", [])
    critical_count = sum(1 for m in matches if str(m.get("level")).lower() == "critical")
    high_count = sum(1 for m in matches if str(m.get("level")).lower() == "high")
    local_count = len(report.get("top_local_events", []))
    top_country = report.get("top_countries", [{}])[0].get("country", "—")
    top_sector = report.get("top_sector", "infrastruktúra")

    summary_items = "".join(f"<li>{esc(x)}</li>" for x in report.get("summary_bullets", [])[:8])
    weekly_items = "".join(f"<li>{esc(x)}</li>" for x in report.get("weekly_bullets", [])[:5])

    return f"""<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CEE Infrastructure & Security Daily Brief</title>

<style>
:root {{
  --dark:#07111f;
  --dark2:#101b2d;
  --panel:#ffffff;
  --soft:#f3f6fa;
  --text:#172033;
  --muted:#65758b;
  --blue:#2f66e8;
  --green:#18a957;
  --orange:#f97316;
  --red:#dc2626;
  --line:#e4eaf2;
}}

* {{ box-sizing:border-box; }}

body {{
  margin:0;
  background:#dde3ea;
  color:var(--text);
  font-family: Arial, Helvetica, sans-serif;
  line-height:1.55;
}}

.page {{
  max-width:1320px;
  margin:0 auto;
  background:#f8fafc;
  min-height:100vh;
}}

.hero {{
  background:
    radial-gradient(circle at 70% 20%, rgba(79,117,255,0.22), transparent 28%),
    linear-gradient(135deg, #07111f 0%, #132039 100%);
  color:white;
  padding:48px;
  display:flex;
  justify-content:space-between;
  gap:24px;
  align-items:center;
}}

.hero h1 {{
  margin:0;
  font-size:42px;
  line-height:1.05;
  letter-spacing:-0.03em;
}}

.hero p {{
  color:#cbd5e1;
  font-size:18px;
  margin:16px 0 0;
}}

.date-box {{
  border:1px solid rgba(255,255,255,0.18);
  border-radius:18px;
  padding:24px 34px;
  min-width:230px;
  text-align:center;
  background:rgba(255,255,255,0.04);
}}

.date-box .label {{
  color:#94a3b8;
  font-size:13px;
  text-transform:uppercase;
  font-weight:800;
}}

.date-box .date {{
  font-size:28px;
  font-weight:900;
  margin-top:8px;
}}

.content {{
  padding:34px 46px 54px;
}}

.actions {{
  display:flex;
  justify-content:flex-end;
  gap:12px;
  margin-bottom:22px;
}}

.btn {{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding:12px 18px;
  border-radius:12px;
  text-decoration:none;
  font-weight:800;
  background:#111827;
  color:white;
}}

.btn.blue {{
  background:#2563eb;
}}

.kpi-grid {{
  display:grid;
  grid-template-columns:repeat(4, 1fr);
  gap:18px;
  margin-bottom:26px;
}}

.kpi {{
  background:white;
  border:1px solid var(--line);
  border-radius:18px;
  padding:22px;
  box-shadow:0 10px 26px rgba(15,23,42,0.06);
}}

.kpi .label {{
  font-size:13px;
  font-weight:900;
  text-transform:uppercase;
  color:#334155;
}}

.kpi .value {{
  font-size:36px;
  font-weight:900;
  color:#2563eb;
  margin-top:8px;
}}

.kpi .note {{
  color:var(--muted);
  font-size:14px;
}}

.section {{
  background:white;
  border:1px solid var(--line);
  border-radius:20px;
  padding:26px;
  margin-top:26px;
  box-shadow:0 10px 26px rgba(15,23,42,0.05);
}}

.section h2 {{
  margin:0 0 18px;
  font-size:26px;
}}

.summary-box {{
  border-left:6px solid #2563eb;
  background:#f8fbff;
  padding:18px;
  border-radius:14px;
}}

.summary-box li {{
  margin-bottom:10px;
}}

.country-grid,
.region-grid,
.card-grid {{
  display:grid;
  grid-template-columns:repeat(4, 1fr);
  gap:18px;
}}

.country-card,
.region-card,
.event-card,
.infra-card {{
  border-radius:18px;
  overflow:hidden;
  background:white;
  border:1px solid var(--line);
  box-shadow:0 8px 20px rgba(15,23,42,0.05);
}}

.country-card {{
  padding:20px;
  border-top:8px solid #2563eb;
}}

.country-card.normal {{ border-top-color:var(--green); }}
.country-card.elevated {{ border-top-color:#d97706; }}
.country-card.tense {{ border-top-color:var(--orange); }}
.country-card.critical {{ border-top-color:var(--red); }}

.country-name {{
  font-size:18px;
  font-weight:900;
}}

.country-score {{
  font-size:34px;
  font-weight:900;
  color:#2563eb;
  margin-top:8px;
}}

.country-level {{
  display:inline-block;
  margin-top:8px;
  padding:6px 12px;
  border-radius:999px;
  background:#eef2ff;
  color:#1d4ed8;
  font-size:12px;
  font-weight:900;
  text-transform:uppercase;
}}

.country-drivers {{
  color:var(--muted);
  font-size:13px;
  margin-top:12px;
}}

.region-head {{
  color:white;
  padding:20px;
  font-size:22px;
  font-weight:900;
  text-transform:uppercase;
}}

.region-card.blue .region-head {{ background:var(--blue); }}
.region-card.green .region-head {{ background:var(--green); }}
.region-card.orange .region-head {{ background:var(--orange); }}
.region-card.red .region-head {{ background:var(--red); }}

.region-body {{
  padding:20px;
  background:#f8fafc;
}}

.big-number {{
  font-size:38px;
  color:#2563eb;
  font-weight:900;
}}

.small-label,
.region-line {{
  color:#475569;
  font-size:14px;
  margin-top:8px;
}}

.mini-bar {{
  height:9px;
  background:#e2e8f0;
  border-radius:999px;
  overflow:hidden;
  margin-top:18px;
}}

.mini-bar span {{
  display:block;
  height:100%;
  background:#2563eb;
  border-radius:999px;
}}

.event-card,
.infra-card {{
  padding:20px;
}}

.event-kicker {{
  font-size:12px;
  text-transform:uppercase;
  color:#64748b;
  font-weight:900;
  margin-bottom:8px;
}}

.event-card h3,
.infra-card h3 {{
  margin:0 0 10px;
  font-size:18px;
}}

.event-card p,
.infra-card p {{
  color:#475569;
  font-size:14px;
}}

.event-card a,
.infra-card a {{
  color:#2563eb;
  font-weight:800;
  text-decoration:none;
}}

.level-pill {{
  display:inline-block;
  padding:6px 12px;
  border-radius:999px;
  font-size:12px;
  font-weight:900;
  text-transform:uppercase;
  margin:4px 0 12px;
}}

.level-pill.critical {{ background:#fee2e2; color:#991b1b; }}
.level-pill.high {{ background:#ffedd5; color:#9a3412; }}
.level-pill.medium {{ background:#fef3c7; color:#92400e; }}
.level-pill.watch {{ background:#dcfce7; color:#166534; }}

.visual-dashboard {{
  margin-top:34px;
  background:
    radial-gradient(circle at top right, rgba(37,99,235,0.20), transparent 28%),
    linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  color:white;
  border-radius:26px;
  padding:34px;
  box-shadow:0 20px 50px rgba(15,23,42,0.25);
}}

.vd-header {{
  display:flex;
  justify-content:space-between;
  gap:24px;
  align-items:flex-start;
  margin-bottom:26px;
}}

.vd-kicker {{
  font-size:12px;
  font-weight:900;
  letter-spacing:0.12em;
  color:#93c5fd;
}}

.visual-dashboard h2 {{
  margin:6px 0 8px;
  font-size:36px;
  line-height:1.05;
}}

.visual-dashboard p {{
  margin:0;
  color:#cbd5e1;
  max-width:760px;
}}

.vd-date {{
  background:rgba(255,255,255,0.10);
  border:1px solid rgba(255,255,255,0.18);
  padding:14px 18px;
  border-radius:16px;
  font-size:20px;
  font-weight:900;
  white-space:nowrap;
}}

.vd-grid {{
  display:grid;
  grid-template-columns:1.4fr repeat(3, 1fr);
  gap:16px;
}}

.vd-main,
.vd-card,
.vd-wide {{
  background:rgba(255,255,255,0.10);
  border:1px solid rgba(255,255,255,0.16);
  border-radius:18px;
  padding:20px;
}}

.vd-main {{
  grid-row:span 2;
}}

.vd-label {{
  color:#cbd5e1;
  text-transform:uppercase;
  font-size:12px;
  font-weight:900;
  letter-spacing:0.08em;
}}

.vd-big {{
  font-size:52px;
  font-weight:900;
  margin-top:12px;
}}

.vd-sub {{
  color:#cbd5e1;
  font-size:14px;
  margin-top:8px;
}}

.vd-number {{
  font-size:34px;
  font-weight:900;
}}

.vd-text {{
  margin-top:8px;
  color:#dbeafe;
  font-size:14px;
}}

.vd-card.red {{ border-top:5px solid #ef4444; }}
.vd-card.orange {{ border-top:5px solid #f97316; }}
.vd-card.blue {{ border-top:5px solid #3b82f6; }}
.vd-card.green {{ border-top:5px solid #22c55e; }}

.vd-wide {{
  grid-column:span 3;
}}

.vd-sector {{
  font-size:34px;
  font-weight:900;
  margin-top:10px;
}}

.vd-regions {{
  margin-top:18px;
  display:grid;
  grid-template-columns:repeat(4, 1fr);
  gap:16px;
}}

.visual-dashboard .region-card {{
  box-shadow:none;
}}

.vd-footer {{
  margin-top:20px;
  padding-top:16px;
  border-top:1px solid rgba(255,255,255,0.18);
  display:flex;
  justify-content:space-between;
  color:#cbd5e1;
  font-size:13px;
}}

.footer {{
  margin-top:44px;
  background:#101827;
  color:#cbd5e1;
  border-radius:18px;
  padding:22px 28px;
  display:flex;
  justify-content:space-between;
  gap:20px;
}}

.footer b {{
  color:white;
}}

@media(max-width:1000px) {{
  .hero {{
    flex-direction:column;
    align-items:flex-start;
    padding:34px 24px;
  }}

  .content {{
    padding:24px;
  }}

  .kpi-grid,
  .country-grid,
  .region-grid,
  .card-grid,
  .vd-grid,
  .vd-regions {{
    grid-template-columns:1fr;
  }}

  .vd-main,
  .vd-wide {{
    grid-column:auto;
    grid-row:auto;
  }}

  .vd-header,
  .footer,
  .vd-footer {{
    flex-direction:column;
  }}
}}
</style>
</head>

<body>
<div class="page">

  <header class="hero">
    <div>
      <h1>CEE Infrastructure &<br>Security Daily Brief</h1>
      <p>Regionális OSINT helyzetkép kritikus infrastruktúrára, lokális eseményekre és kockázati mintázatokra.</p>
    </div>

    <div class="date-box">
      <div class="label">Dátum</div>
      <div class="date">{esc(today)}</div>
      <div class="label">UTC alapú riport</div>
    </div>
  </header>

  <main class="content">

    <div class="actions">
      <a class="btn blue" href="../index.html">Vissza a dashboardra</a>
      <a class="btn" href="cee-daily-report-latest.html">Legfrissebb riport</a>
    </div>

    <section class="kpi-grid">
      <div class="kpi">
        <div class="label">Régiós állapot</div>
        <div class="value">{esc(str(region.get("overall", "normal")).upper())}</div>
        <div class="note">score: {round(region.get("overall_score", 0), 2)}</div>
      </div>

      <div class="kpi">
        <div class="label">Kritikus infra alert</div>
        <div class="value">{critical_count}</div>
        <div class="note">magas: {high_count}</div>
      </div>

      <div class="kpi">
        <div class="label">Fő kockázati ország</div>
        <div class="value" style="font-size:28px;">{esc(top_country)}</div>
        <div class="note">országkockázati sorrend alapján</div>
      </div>

      <div class="kpi">
        <div class="label">Fő szektor</div>
        <div class="value" style="font-size:28px;">{esc(top_sector)}</div>
        <div class="note">infrastruktúra-közeli találatok alapján</div>
      </div>
    </section>

    <section class="section">
      <h2>Rövid napi értékelés</h2>
      <div class="summary-box">
        <ul>{summary_items or "<li>Nincs elérhető napi kivonat.</li>"}</ul>
      </div>
    </section>

    <section class="section">
      <h2>Fő régiós blokkok</h2>
      <div class="region-grid">
        {html_region_dashboard(report)}
      </div>
    </section>

    <section class="section">
      <h2>Top risk országok</h2>
      <div class="country-grid">
        {html_country_cards(report.get("top_countries", []))}
      </div>
    </section>

    <section class="section">
      <h2>Infrastruktúra-közeli incidensek</h2>
      <div class="card-grid">
        {html_proximity(report.get("top_infrastructure_matches", []))}
      </div>
    </section>

    <section class="section">
      <h2>Helyi forrásokból azonosított események</h2>
      <div class="card-grid">
        {html_local_events(report.get("top_local_events", []))}
      </div>
    </section>

    <section class="section">
      <h2>Heti trendjelzések</h2>
      <div class="summary-box">
        <ul>{weekly_items or "<li>Nincs elérhető heti trendjelzés.</li>"}</ul>
      </div>
    </section>

    <section class="visual-dashboard">
      <div class="vd-header">
        <div>
          <div class="vd-kicker">BLOG VISUAL DASHBOARD</div>
          <h2>CEE Security & Infrastructure Snapshot</h2>
          <p>Napi régiós kockázati kép kritikus infrastruktúra, lokális események és OSINT-jelzések alapján.</p>
        </div>
        <div class="vd-date">{esc(today)}</div>
      </div>

      <div class="vd-grid">
        <div class="vd-main">
          <div class="vd-label">Régiós állapot</div>
          <div class="vd-big">{esc(str(region.get("overall", "normal")).upper())}</div>
          <div class="vd-sub">Regional score: {round(region.get("overall_score", 0), 2)}</div>
        </div>

        <div class="vd-card red">
          <div class="vd-number">{critical_count}</div>
          <div class="vd-text">kritikus infrastruktúra alert</div>
        </div>

        <div class="vd-card orange">
          <div class="vd-number">{high_count}</div>
          <div class="vd-text">magas szintű kapcsolat</div>
        </div>

        <div class="vd-card blue">
          <div class="vd-number">{local_count}</div>
          <div class="vd-text">kiemelt lokális esemény</div>
        </div>

        <div class="vd-card green">
          <div class="vd-number">{esc(top_country)}</div>
          <div class="vd-text">fő kockázati ország</div>
        </div>

        <div class="vd-wide">
          <div class="vd-label">Fő szektor</div>
          <div class="vd-sector">{esc(top_sector)}</div>
          <div class="vd-sub">Infrastruktúra-közeli találatok domináns kategóriája</div>
        </div>
      </div>

      <div class="vd-regions">
        {html_region_dashboard(report)}
      </div>

      <div class="vd-footer">
        <span>Törésvonalak • CEE Security Map</span>
        <span>Automatikus OSINT-alapú helyzetkép</span>
      </div>
    </section>

    <footer class="footer">
      <div>
        <b>Módszertani megjegyzés</b><br>
        Automatikus OSINT-alapú CEE infrastruktúra- és biztonsági jelentés. A források kézi ellenőrzése javasolt.
      </div>
      <div>
        <b>Törésvonalak</b><br>
        CEE Security Map
      </div>
    </footer>

  </main>
</div>
</body>
</html>"""


def save_html(report):
    today = utc_now().strftime("%Y-%m-%d")
    html = build_html(report)

    (REPORT_HTML_DIR / "cee-daily-report-latest.html").write_text(html, encoding="utf-8")
    (REPORT_HTML_DIR / f"cee-daily-report-{today}.html").write_text(html, encoding="utf-8")


def main():
    ensure_dirs()
    report = build_report()
    save_report_json(report)
    save_html(report)
    print("CEE daily report created")


if __name__ == "__main__":
    main()
