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


def safe_html_paragraphs(text_or_html: str) -> str:
    if not text_or_html:
        return ""
    return text_or_html.strip()


weekly = load_json(WEEKLY_FILE)
meta = load_json(META_FILE)

generated = weekly.get("generated_utc") or meta.get("generated_utc") or "-"
date_label = datetime.now(timezone.utc).strftime("%Y.%m.%d")

title = f"Közép–Kelet Európa biztonsági helyzet – heti brief ({date_label})"

headline = weekly.get("headline") or "Közép–Kelet Európa heti biztonsági brief"
weekly_assessment_html = safe_html_paragraphs(
    weekly.get("weekly_assessment") or ""
)
methodology_html = safe_html_paragraphs(
    weekly.get("methodology_html") or ""
)

# fallback, ha valamiért még nincs új struktúra
if not weekly_assessment_html:
    bullets = weekly.get("bullets", [])
    bullet_paragraphs = "".join(
        [
            f'<p style="margin:0 0 16px 0;font-size:16px;line-height:1.8;color:#f1f5f9;text-align:justify;">{b}</p>'
            for b in bullets
        ]
    )
    weekly_assessment_html = bullet_paragraphs

if not methodology_html:
    methodology_text = weekly.get("methodology") or (
        "A heti brief nyílt forrású információk strukturált feldolgozásán alapul. "
        "Az automatikus kimenet tájékoztató jellegű, ezért a kiemelt állítások és linkelt források "
        "kézi ellenőrzése minden esetben javasolt."
    )
    methodology_html = (
        f'<p style="margin:0;font-size:15px;line-height:1.8;color:#f1f5f9;text-align:justify;">'
        f"{methodology_text}</p>"
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
        {headline}
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
          Heti jelentés
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
            color:#f1f5f9;
        ">
          {weekly_assessment_html}
          <p style="margin:18px 0 0 0;font-size:15px;line-height:1.8;text-align:justify;">
            <a href="{MAP_URL}" target="_blank" rel="noopener" style="color:#cfe7ff;font-weight:600;text-decoration:underline;">
              Projekt megnyitása
            </a>
          </p>
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
          Módszertan
        </div>
      </div>

      <div style="padding:2px 8px 0 8px;">
        <div style="
            background:rgba(255,255,255,0.08);
            border:1px solid rgba(255,255,255,0.12);
            border-radius:16px;
            padding:22px 24px;
            box-shadow:0 8px 20px rgba(0,0,0,0.12);
            color:#f1f5f9;
        ">
          {methodology_html}
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
