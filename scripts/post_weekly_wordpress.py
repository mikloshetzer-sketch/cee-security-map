import json
import requests
from datetime import datetime, timezone

# ===== CONFIG =====
WORDPRESS_SITE = "torésvonalak.wordpress.com"   # ide a saját site-od!
ACCESS_TOKEN = None  # GitHub secretből jön
MAP_URL = "https://mikloshetzer-sketch.github.io/cee-security-map/"
WEEKLY_FILE = "data/weekly.json"
META_FILE = "data/meta.json"

# ===== LOAD DATA =====
def load_json(path):
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

# ===== POST =====
def publish():
    global ACCESS_TOKEN

    ACCESS_TOKEN = requests.get(
        "https://api.github.com/repos",
        headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}
    )  # dummy: token check

    token = ACCESS_TOKEN

    url = f"https://public-api.wordpress.com/rest/v1.1/sites/{WORDPRESS_SITE}/posts/new"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "title": title,
        "content": content,
        "status": "publish"
    }

    r = requests.post(url, headers=headers, data=payload)

    if r.status_code != 200:
        print("WP POST ERROR:", r.text)
        raise SystemExit(1)

    print("WordPress post created:", r.json().get("URL"))

if __name__ == "__main__":
    publish()
