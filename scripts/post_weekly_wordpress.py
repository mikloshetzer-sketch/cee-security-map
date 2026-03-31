import json
import os
from datetime import datetime, timezone

import requests

# ===== CONFIG =====
DEFAULT_WORDPRESS_SITE = "toresvonalak.wordpress.com"
MAP_URL = "https://mikloshetzer-sketch.github.io/cee-security-map/"
WEEKLY_FILE = "data/weekly.json"
META_FILE = "data/meta.json"


# ===== LOAD DATA =====
def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def esc(text) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def top_keys(data, limit=3):
    if not isinstance(data, dict):
        return []
    items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in items[:limit] if k]


def format_list_hu(items):
    items = [str(x) for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} és {items[1]}"
    return f"{', '.join(items[:-1])} és {items[-1]}"


def build_intro(weekly: dict) -> str:
    counts = weekly.get("counts", {})
    category_counts = weekly.get("category_counts", {})
    country_counts = weekly.get("country_counts", {})

    total_events = counts.get("total") or counts.get("events") or counts.get("items")
    top_categories = top_keys(category_counts, 3)
    top_countries = top_keys(country_counts, 3)

    parts = []

    if total_events and top_categories:
        parts.append(
            f"Az elmúlt hétben a közép– és kelet-európai biztonsági környezetben {total_events} releváns jelzés került a heti kivonatba, "
            f"amelyek közül különösen a {format_list_hu(top_categories)} témák rajzolódtak ki hangsúlyosan."
        )
    elif top_categories:
        parts.append(
            f"Az elmúlt hétben a közép– és kelet-európai biztonsági környezetben elsősorban a {format_list_hu(top_categories)} témák határozták meg a regionális képet."
        )
    else:
        parts.append(
            "Az elmúlt hétben a közép– és kelet-európai biztonsági környezetben több, egymással párhuzamosan zajló fejlemény rajzolódott ki."
        )

    if top_countries:
        parts.append(
            f"A rendelkezésre álló jelzések alapján a figyelem leginkább {format_list_hu(top_countries)} irányába összpontosult, "
            f"ami arra utal, hogy ezekben az országokban sűrűsödtek a regionális biztonsági szempontból jelentősebb történések."
        )
    else:
        parts.append(
            "A heti mintázatok alapján nem egyetlen domináns válsághelyzet, hanem több, különböző területeken jelentkező nyomáspont határozta meg a regionális képet."
        )

    return " ".join(parts)


weekly = load_json(WEEKLY_FILE)
meta = load_json(META_FILE)

bullets = weekly.get("bullets", [])
generated = weekly.get("generated_utc") or meta.get("generated_utc") or "-"
date_label = datetime.now(timezone.utc).strftime("%Y.%m.%d")
intro_text = build_intro(weekly)

# ===== POST CONTENT =====
title = f"Közép–Kelet Európa biztonsági helyzet – heti jelentés ({date_label})"

bullets_html = "".join(
    [
        f'<li style="margin:0 0 12px 0;text-align:justify;">{esc(b)}</li>'
        for b in bullets
    ]
)

content = f"""
<div style="background:#4b5563;padding:40px 20px;">
  <div style="max-width:1000px;margin:0 auto;display:flex;flex-direction:column;gap:22px;">

    <!-- HEADER -->
    <div style="
        background:linear-gradient(135deg,#475569,#334155);
        padding:26px 28px;
        border-radius:22px;
        color:#f8fafc;
        box-shadow:0 12px 30px rgba(0,0,0,0.22);
    ">
      <div style="font-size:12px;text-transform:uppercase;letter-spacing:1.4px;color:#cbd5e1;">
        CEE Security Monitor
      </div>
      <div style="font-size:30px;font-weight:700;line-height:1.2;margin-top:8px;">
        {esc(title)}
      </div>
      <div style="margin-top:10px;font-size:15px;line-height:1.6;color:#e2e8f0;">
        Heti automatizált összefoglaló a közép– és kelet-európai biztonsági környezetről.
      </div>
      <div style="margin-top:12px;font-size:14px;line-height:1.6;color:#cbd5e1;">
        <strong>Frissítés:</strong> {esc(generated)}
      </div>
    </div>

    <!-- MÓDSZER -->
    <section style="margin:0;">
      <div style="
          background:#e5e7eb;
          color:#0f172a;
          padding:18px 22px;
          border-radius:16px;
          box-shadow:0 6px 18px rgba(0,0,0,0.16);
          margin:0 0 14px 0;
      ">
        <div style="font-size:22px;font-weight:700;line-height:1.3;">
          Módszer
        </div>
      </div>

      <div style="
          background:rgba(255,255,255,0.08);
          border:1px solid rgba(255,255,255,0.12);
          border-radius:16px;
          padding:22px 24px;
          box-shadow:0 8px 20px rgba(0,0,0,0.12);
      ">
        <p style="margin:0;font-size:15px;line-height:1.8;color:#e2e8f0;text-align:justify;">
          A heti brief nyílt forrású információk strukturált feldolgozásán alapul. Az összesítés célja nem a teljes körű eseménylista megjelenítése, hanem a főbb regionális biztonsági mintázatok, visszatérő témák és geopolitikai jelentőségű trendek kiemelése. Az automatikusan generált összefoglaló ezért tájékozódási és előszűrési célokat szolgál, a források és következtetések kézi ellenőrzése továbbra is indokolt.
        </p>
      </div>
    </section>

    <!-- JELENTÉS -->
    <section style="margin:0;">
      <div style="
          background:#e5e7eb;
          color:#0f172a;
          padding:18px 22px;
          border-radius:16px;
          box-shadow:0 6px 18px rgba(0,0,0,0.16);
          margin:0 0 14px 0;
      ">
        <div style="font-size:22px;font-weight:700;line-height:1.3;">
          Jelentés
        </div>
      </div>

      <div style="
          background:rgba(255,255,255,0.08);
          border:1px solid rgba(255,255,255,0.12);
          border-radius:16px;
          padding:22px 24px;
          box-shadow:0 8px 20px rgba(0,0,0,0.12);
      ">

        <p style="margin:0 0 16px 0;font-size:16px;line-height:1.8;color:#e2e8f0;text-align:justify;">
          {esc(intro_text)}
        </p>

        <ul style="margin:0 0 0 22px;padding:0;color:#f1f5f9;line-height:1.8;font-size:16px;">
          {bullets_html}
        </ul>

      </div>
    </section>

    <!-- PROJEKT -->
    <section style="margin:0;">
      <div style="
          background:#e5e7eb;
          color:#0f172a;
          padding:18px 22px;
          border-radius:16px;
          box-shadow:0 6px 18px rgba(0,0,0,0.16);
          margin:0 0 14px 0;
      ">
        <div style="font-size:22px;font-weight:700;line-height:1.3;">
          Projekt
        </div>
      </div>

      <div style="
          background:rgba(255,255,255,0.08);
          border:1px solid rgba(255,255,255,0.12);
          border-radius:16px;
          padding:22px 24px;
          box-shadow:0 8px 20px rgba(0,0,0,0.12);
      ">
        <p style="margin:0;font-size:15px;line-height:1.8;color:#e2e8f0;">
          <a href="{MAP_URL}" target="_blank" rel="noopener noreferrer" style="color:#93c5fd;font-weight:600;text-decoration:none;">
            Projekt megnyitása
          </a>
        </p>
      </div>
    </section>

  </div>
</div>
"""


def publish() -> None:
    token = (os.getenv("WPCOM_ACCESS_TOKEN") or "").strip()
    if not token:
        raise SystemExit("ERROR: Missing env var WPCOM_ACCESS_TOKEN")

    wordpress_site = (os.getenv("WPCOM_SITE") or DEFAULT_WORDPRESS_SITE).strip()
    if not wordpress_site:
        raise SystemExit("ERROR: Missing site (set WPCOM_SITE or DEFAULT_WORDPRESS_SITE)")

    url = f"https://public-api.wordpress.com/rest/v1.1/sites/{wordpress_site}/posts/new"

    headers = {
        "Authorization": f"Bearer {token}",
    }

    payload = {
        "title": title,
        "content": content,
        "status": "publish",
    }

    r = requests.post(url, headers=headers, data=payload, timeout=30)

    if r.status_code != 200:
        print("WP POST ERROR:", r.status_code, r.text)
        raise SystemExit(1)

    j = r.json()
    print("WordPress post created:", j.get("URL") or j.get("url") or j)


if __name__ == "__main__":
    publish()
