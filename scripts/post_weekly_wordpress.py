import json
import os
import requests
from datetime import datetime, timezone

# ===== CONFIG =====
DEFAULT_WORDPRESS_SITE = "toresvonalak.wordpress.com"  # javasolt ékezet nélkül
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
generated = weekly.get("generated_utc") or meta.get("generated_utc")

date_label = datetime.now(timezone.utc).strftime("%Y.%m.%d")

# ===== POST CONTENT =====
title = f"Közép–Kelet Európa biztonsági helyzet – heti jelentés ({date_label})"
bullets_html = "".join([f"<li>{b}</li>" for b in bullets])

content = f"""
<h2>Heti biztonsági kivonat</h2>

<p><b>Frissítés:</b> {generated}</p>

<ul>
{bullets_html}
</ul>

<h3>Interaktív térkép</h3>

<iframe src="{MAP_URL}" width="100%" height="720" style="border:0;border-radius:12px;"></iframe>

<p style="font-size:12px;opacity:0.7;">
Automatikus OSINT kivonat. A linkelt források kézi ellenőrzése javasolt.
</p>
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
