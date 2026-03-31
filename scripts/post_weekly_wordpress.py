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


def clean_bullet(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lstrip("-• ").split())


# ===== NARRATÍV GENERÁLÁS =====
def build_report(weekly: dict) -> list:

    bullets = [clean_bullet(b) for b in weekly.get("bullets", []) if clean_bullet(b)]

    counts = weekly.get("counts", {})
    category_counts = weekly.get("category_counts", {})
    country_counts = weekly.get("country_counts", {})

    total = counts.get("total") or counts.get("events") or counts.get("items")
    top_categories = top_keys(category_counts, 3)
    top_countries = top_keys(country_counts, 3)

    paragraphs = []

    # 1. bevezető
    if total and top_categories:
        paragraphs.append(
            f"Az elmúlt hétben a közép– és kelet-európai biztonsági környezetben {total} releváns jelzés került az elemzésbe. "
            f"A fejlemények közül különösen a {format_list_hu(top_categories)} témák rajzolódtak ki hangsúlyosan, "
            f"ami arra utal, hogy ezek a területek jelenleg fokozott figyelmet kapnak a regionális diskurzusban."
        )
    else:
        paragraphs.append(
            "Az elmúlt hétben a közép– és kelet-európai biztonsági környezetben több, egymással párhuzamosan zajló fejlemény rajzolódott ki, "
            "amelyek együttesen alakították a térség aktuális biztonsági képét."
        )

    # 2. földrajzi fókusz
    if top_countries:
        paragraphs.append(
            f"A jelzések földrajzi megoszlása alapján a figyelem leginkább {format_list_hu(top_countries)} irányába koncentrálódott. "
            f"Ez arra utal, hogy ezekben az országokban vagy régiókban sűrűsödtek azok az események és folyamatok, "
            f"amelyek rövid távon is hatással lehetnek a biztonsági környezet alakulására."
        )

    # 3. események narratív összefűzése
    if bullets:
        selected = bullets[:6]

        combined = " ".join(selected)

        paragraphs.append(
            f"A heti események részletesebb vizsgálata alapján megállapítható, hogy {combined}. "
            f"Ezek az események nem elszigetelten értelmezhetők, hanem egy tágabb regionális dinamikába illeszkednek."
        )

    # 4. értelmezés
    paragraphs.append(
        "Összességében a vizsgált időszakban nem egyetlen domináns válsághelyzet határozta meg a térséget, "
        "hanem több, egymással összefüggő nyomáspont jelent meg. "
        "A biztonsági környezet továbbra is differenciált képet mutat, ahol a politikai, gazdasági és infrastruktúrális tényezők egyaránt szerepet játszanak."
    )

    # 5. rövid outlook
    paragraphs.append(
        "Rövid távon a jelenlegi trendek fennmaradása valószínűsíthető, különösen azon területeken, ahol a jelzések sűrűsége tartósan magas marad. "
        "Ez indokolja a régió folyamatos monitorozását és az események kontextusba helyezett értelmezését."
    )

    return paragraphs


# ===== LOAD =====
weekly = load_json(WEEKLY_FILE)
meta = load_json(META_FILE)

generated = weekly.get("generated_utc") or meta.get("generated_utc") or "-"
date_label = datetime.now(timezone.utc).strftime("%Y.%m.%d")

report_paragraphs = build_report(weekly)

paragraphs_html = "".join(
    [
        f'<p style="margin:0 0 16px 0;font-size:16px;line-height:1.8;color:#e2e8f0;text-align:justify;">{esc(p)}</p>'
        for p in report_paragraphs
    ]
)

# ===== TITLE =====
title = f"Közép–Kelet Európa biztonsági helyzet – heti jelentés ({date_label})"


# ===== CONTENT =====
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
      <div style="font-size:30px;font-weight:700;">
        {esc(title)}
      </div>
      <div style="margin-top:10px;color:#e2e8f0;">
        Heti automatizált biztonsági elemzés
      </div>
      <div style="margin-top:10px;color:#cbd5e1;">
        <strong>Frissítés:</strong> {esc(generated)}
      </div>
    </div>

    <!-- JELENTÉS -->
    <section>
      <div style="background:#e5e7eb;color:#0f172a;padding:18px;border-radius:16px;">
        <strong>Jelentés</strong>
      </div>

      <div style="background:rgba(255,255,255,0.08);padding:22px;border-radius:16px;">
        {paragraphs_html}

        <p style="margin-top:20px;">
          <a href="{MAP_URL}" target="_blank" style="color:#93c5fd;font-weight:600;">
            Projekt megnyitása
          </a>
        </p>
      </div>
    </section>

    <!-- MÓDSZERTAN -->
    <section>
      <div style="background:#e5e7eb;color:#0f172a;padding:18px;border-radius:16px;">
        <strong>Módszertan</strong>
      </div>

      <div style="background:rgba(255,255,255,0.08);padding:22px;border-radius:16px;">
        <p style="color:#e2e8f0;text-align:justify;">
          A heti jelentés nyílt forrású információk feldolgozásán alapul, és a főbb regionális mintázatok azonosítására fókuszál. 
          A cél nem az események teljes körű felsorolása, hanem azok értelmezése és kontextusba helyezése.
        </p>
      </div>
    </section>

  </div>
</div>
"""


# ===== PUBLISH =====
def publish() -> None:
    token = (os.getenv("WPCOM_ACCESS_TOKEN") or "").strip()
    if not token:
        raise SystemExit("ERROR: Missing env var WPCOM_ACCESS_TOKEN")

    wordpress_site = (os.getenv("WPCOM_SITE") or DEFAULT_WORDPRESS_SITE).strip()

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
