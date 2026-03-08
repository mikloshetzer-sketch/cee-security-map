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


weekly = load_json(WEEKLY_FILE)
meta = load_json(META_FILE)

bullets = weekly.get("bullets", [])
generated = weekly.get("generated_utc") or meta.get("generated_utc") or "-"
date_label = datetime.now(timezone.utc).strftime("%Y.%m.%d")

# ===== POST CONTENT =====
title = f"Közép–Kelet Európa biztonsági helyzet – heti jelentés ({date_label})"

bullets_html = "".join(
    [
        f'<li style="margin:0 0 12px 0;text-align:justify;">{b}</li>'
        for b in bullets
    ]
)

content = f"""
<div style="background:#4b5563;padding:40px 20px;">
  <div style="max-width:1000px;margin:0 auto;display:flex;flex-direction:column;gap:22px;">

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
        {title}
      </div>
      <div style="margin-top:10px;font-size:15px;line-height:1.6;color:#e2e8f0;">
        Heti automatizált összefoglaló a közép– és kelet-európai biztonsági környezetről.
      </div>
    </div>

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
          Heti biztonsági kivonat
        </div>
      </div>

      <div style="padding:2px 8px 0 8px;">
        <p style="margin:0 0 16px 0;font-size:16px;line-height:1.8;color:#f1f5f9;text-align:justify;">
          <strong>Frissítés:</strong> {generated}
        </p>

        <div style="
            background:rgba(255,255,255,0.08);
            border:1px solid rgba(255,255,255,0.12);
            border-radius:16px;
            padding:22px 24px;
            box-shadow:0 8px 20px rgba(0,0,0,0.12);
        ">
          <ul style="margin:0 0 0 22px;padding:0;color:#f1f5f9;line-height:1.8;font-size:16px;">
            {bullets_html}
          </ul>
        </div>
      </div>
    </section>

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
          Interaktív térkép
        </div>
      </div>

      <div style="padding:2px 8px 0 8px;">
        <p style="margin:0 0 16px 0;font-size:16px;line-height:1.8;color:#f1f5f9;text-align:justify;">
          Az alábbi beágyazott térkép vizuálisan támogatja a heti incidensminták és regionális fókuszpontok gyors áttekintését.
        </p>

        <div style="
            background:#f8fafc;
            padding:16px;
            border-radius:18px;
            box-shadow:0 10px 24px rgba(0,0,0,0.18);
            overflow:hidden;
        ">
          <iframe
            src="{MAP_URL}"
            width="100%"
            height="720"
            style="border:0;border-radius:12px;"
            loading="lazy">
          </iframe>
        </div>
      </div>
    </section>

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
          Megjegyzés
        </div>
      </div>

      <div style="padding:2px 8px 0 8px;">
        <p style="margin:0;font-size:14px;line-height:1.8;color:#e2e8f0;text-align:justify;">
          Automatikus OSINT kivonat. A linkelt források és következtetések kézi ellenőrzése minden esetben javasolt.
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
