#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import datetime, timezone

import requests


WPCOM_TOKEN_URL = "https://public-api.wordpress.com/oauth2/token"
WPCOM_API_BASE = "https://public-api.wordpress.com/rest/v1.1"


def must_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def opt_env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iso_dt(s: str) -> datetime:
    # weekly.json generated_utc: "2026-02-17T08:09:59Z"
    if not s:
        return datetime.now(timezone.utc)
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def get_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    r = requests.post(
        WPCOM_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {data}")
    return token


def wp_post_create(site: str, token: str, title: str, content_html: str, status: str, tags: str, categories: str) -> dict:
    url = f"{WPCOM_API_BASE}/sites/{site}/posts/new"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "title": title,
        "content": content_html,
        "status": status,
    }
    if tags:
        payload["tags"] = tags
    if categories:
        payload["categories"] = categories

    r = requests.post(url, headers=headers, data=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def esc(s: str) -> str:
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def build_post_html(weekly: dict, map_url: str) -> tuple[str, str]:
    # Title
    gen = iso_dt(weekly.get("generated_utc", ""))
    y, w, _ = gen.isocalendar()
    date_hu = gen.astimezone(timezone.utc).strftime("%Y-%m-%d")
    title = f"Közép–Kelet Európa biztonsági monitor – heti kivonat (ISO {y}-W{w:02d} • {date_hu})"

    # Bullets
    bullets = weekly.get("bullets") or []
    bullets_html = ""
    if bullets:
        bullets_html = "<ul>\n" + "\n".join(f"<li>{esc(b)}</li>" for b in bullets) + "\n</ul>\n"
    else:
        bullets_html = "<p><em>Nincs heti kivonat adat.</em></p>"

    # Examples
    examples = weekly.get("examples") or []
    examples_html = ""
    if examples:
        rows = []
        for ex in examples[:8]:
            t = esc(ex.get("time_utc", ""))
            tt = esc(ex.get("title", ""))
            u = ex.get("url", "")
            dom = esc(ex.get("domain", ""))
            if u:
                rows.append(f"<li><b>{t}</b> — <a href=\"{esc(u)}\" target=\"_blank\" rel=\"noopener\">{tt}</a> <span style=\"opacity:.7\">({dom})</span></li>")
            else:
                rows.append(f"<li><b>{t}</b> — {tt} <span style=\"opacity:.7\">({dom})</span></li>")
        examples_html = "<h3>Példák (válogatás)</h3>\n<ul>\n" + "\n".join(rows) + "\n</ul>\n"

    # Map embed + fallback
    # WordPress.com néha szűri az iframe-et. Ezért adunk fallback linket is.
    map_block = textwrap.dedent(f"""
    <h3>Interaktív térkép</h3>
    <p>
      <a href="{esc(map_url)}" target="_blank" rel="noopener"><b>Megnyitás új lapon</b></a>
    </p>
    <div style="position:relative; padding-top:62%; width:100%; max-width:1100px;">
      <iframe
        src="{esc(map_url)}"
        style="position:absolute; inset:0; width:100%; height:100%; border:0; border-radius:12px;"
        loading="lazy"
        referrerpolicy="no-referrer"
        allowfullscreen>
      </iframe>
    </div>
    <p style="font-size:12px; opacity:.8; margin-top:8px;">
      Ha a WordPress.com kiszedi az iframe-et, használd a fenti “Megnyitás új lapon” linket.
    </p>
    """)

    content = textwrap.dedent(f"""
    <p style="font-size:12px; opacity:.8;">
      Automatikusan generált heti kivonat a CEE (8 ország) térképes OSINT-monitorból.
    </p>

    <h3>Heti összefoglaló</h3>
    {bullets_html}

    {examples_html}

    {map_block}
    """)

    return title, content


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weekly_path = os.path.join(repo_root, "data", "weekly.json")

    if not os.path.exists(weekly_path):
        raise RuntimeError(f"weekly.json not found at: {weekly_path}")

    weekly = load_json(weekly_path)

    client_id = must_env("WPCOM_CLIENT_ID")
    client_secret = must_env("WPCOM_CLIENT_SECRET")
    refresh_token = must_env("WPCOM_REFRESH_TOKEN")
    site = must_env("WPCOM_SITE")

    status = opt_env("WPCOM_POST_STATUS", "publish")  # or "draft"
    map_url = opt_env("MAP_URL", "https://mikloshetzer-sketch.github.io/cee-security-map/")

    # opcionális: témák/kategóriák (WP.com oldalon létezzenek, különben létrehozza / vagy ignorálhat)
    tags = opt_env("WPCOM_TAGS", "cee,security,osint,weekly")
    categories = opt_env("WPCOM_CATEGORIES", "Biztonságpolitika")

    token = get_access_token(client_id, client_secret, refresh_token)

    title, html = build_post_html(weekly, map_url)
    created = wp_post_create(site, token, title, html, status, tags, categories)

    post_url = created.get("URL") or created.get("url") or ""
    post_id = created.get("ID") or created.get("id") or ""
    print(f"Created post. ID={post_id} URL={post_url}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        raise
