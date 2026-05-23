import json
from pathlib import Path
from datetime import datetime, timezone

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


def risk_level(score):
    if score >= 8:
        return "critical"
    if score >= 5:
        return "tense"
    if score >= 2:
        return "elevated"
    return "normal"


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

    top_local = local_events.get("features", [])[:10]

    top_prox = sorted(
        infra.get("matches", []),
        key=lambda x: x.get("score", 0),
        reverse=True
    )[:10]

    generated = utc_now().isoformat()

    report = {
        "generated_utc": generated,
        "headline": "CEE Security Daily Report",
        "region": risk.get("region", {}),
        "top_countries": countries_sorted[:8],
        "summary_bullets": summary.get("bullets", []),
        "weekly_bullets": weekly.get("bullets", []),
        "top_local_events": top_local,
        "top_infrastructure_matches": top_prox,
        "meta": meta
    }

    return report


def save_report_json(report):
    today = utc_now().strftime("%Y-%m-%d")

    latest_path = REPORT_DATA_DIR / "cee_daily_report_latest.json"
    dated_path = REPORT_DATA_DIR / f"cee_daily_report_{today}.json"

    latest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    dated_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def html_country_table(countries):
    rows = []

    for c in countries:
        rows.append(f"""
        <tr>
            <td>{c.get("country", "")}</td>
            <td>{c.get("overall", "")}</td>
            <td>{round(c.get("overall_score", 0), 2)}</td>
            <td>{", ".join(c.get("drivers", []))}</td>
        </tr>
        """)

    return "\n".join(rows)


def html_local_events(events):
    rows = []

    for item in events:
        props = item.get("properties", {})

        rows.append(f"""
        <div class="event-card">
            <div class="event-title">
                {props.get("title", "")}
            </div>

            <div class="event-meta">
                {props.get("country", "")}
                |
                {props.get("category", "")}
                |
                {props.get("severity", "")}
            </div>

            <div class="event-summary">
                {props.get("summary", "")}
            </div>

            <a href="{props.get("url", "#")}" target="_blank">
                Source
            </a>
        </div>
        """)

    return "\n".join(rows)


def html_proximity(matches):
    rows = []

    for m in matches:
        infra = m.get("infrastructure", {})
        event = m.get("event", {})

        rows.append(f"""
        <div class="infra-card">
            <div class="infra-title">
                {infra.get("name", "")}
            </div>

            <div class="infra-meta">
                {infra.get("country", "")}
                |
                {m.get("level", "")}
                |
                {round(m.get("distance_km", 0), 1)} km
            </div>

            <div class="infra-event">
                {event.get("title", "")}
            </div>

            <a href="{event.get("url", "#")}" target="_blank">
                Source
            </a>
        </div>
        """)

    return "\n".join(rows)


def build_html(report):
    region = report.get("region", {})
    countries = report.get("top_countries", [])
    summary_bullets = report.get("summary_bullets", [])
    weekly_bullets = report.get("weekly_bullets", [])

    local_html = html_local_events(report.get("top_local_events", []))
    prox_html = html_proximity(report.get("top_infrastructure_matches", []))

    summary_html = "".join(
        [f"<li>{x}</li>" for x in summary_bullets]
    )

    weekly_html = "".join(
        [f"<li>{x}</li>" for x in weekly_bullets]
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CEE Security Daily Report</title>

<style>

body {{
    margin: 0;
    background: #0f172a;
    color: #e2e8f0;
    font-family: Arial, sans-serif;
}}

.container {{
    max-width: 1400px;
    margin: auto;
    padding: 24px;
}}

h1 {{
    margin-top: 0;
}}

.section {{
    margin-top: 28px;
    background: #111827;
    padding: 18px;
    border-radius: 12px;
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
}}

.event-card,
.infra-card {{
    background: #1e293b;
    padding: 14px;
    border-radius: 10px;
}}

.event-title,
.infra-title {{
    font-weight: bold;
    margin-bottom: 8px;
}}

.event-meta,
.infra-meta {{
    color: #94a3b8;
    margin-bottom: 8px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th, td {{
    border-bottom: 1px solid #334155;
    padding: 10px;
    text-align: left;
}}

a {{
    color: #60a5fa;
}}

.badge {{
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: #1e293b;
}}

</style>
</head>

<body>

<div class="container">

<h1>CEE Security Daily Report</h1>

<div class="badge">
Generated: {report.get("generated_utc", "")}
</div>

<div class="section">
<h2>Regional Risk Snapshot</h2>

<p>
Overall regional status:
<strong>{region.get("overall", "normal")}</strong>
</p>

<p>
Regional score:
<strong>{round(region.get("overall_score", 0), 2)}</strong>
</p>

</div>

<div class="section">
<h2>Top Risk Countries</h2>

<table>
<thead>
<tr>
<th>Country</th>
<th>Status</th>
<th>Score</th>
<th>Drivers</th>
</tr>
</thead>

<tbody>
{html_country_table(countries)}
</tbody>

</table>
</div>

<div class="section">
<h2>Daily Executive Summary</h2>

<ul>
{summary_html}
</ul>

</div>

<div class="section">
<h2>Weekly Trend Signals</h2>

<ul>
{weekly_html}
</ul>

</div>

<div class="section">
<h2>Top Local Infrastructure Events</h2>

<div class="grid">
{local_html}
</div>

</div>

<div class="section">
<h2>Infrastructure Proximity Alerts</h2>

<div class="grid">
{prox_html}
</div>

</div>

</div>

</body>
</html>
"""

    return html


def save_html(report):
    today = utc_now().strftime("%Y-%m-%d")

    html = build_html(report)

    latest = REPORT_HTML_DIR / "cee-daily-report-latest.html"
    dated = REPORT_HTML_DIR / f"cee-daily-report-{today}.html"

    latest.write_text(html, encoding="utf-8")
    dated.write_text(html, encoding="utf-8")


def main():
    ensure_dirs()

    report = build_report()

    save_report_json(report)
    save_html(report)

    print("CEE daily report created")


if __name__ == "__main__":
    main()
