import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from streamlit.delta_generator import DeltaGenerator
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# =========================
# App config
# =========================
st.set_page_config(
    page_title="ASEAN Regulatory Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "ASEAN Regulatory Dashboard: interactive view of regional regulations.",
    },
)

st.markdown(
    """
    <style>
    /* ── Base ── */
    .stApp {
        background-color: #071014;
        color: #f8fafc;
    }

    /* ── Reduce default top padding ── */
    .block-container {
        padding-top: 1.5rem !important;
    }

    /* ── Typography ── */
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4 {
        color: #e2e8f0;
        letter-spacing: 0.01em;
    }

    /* ── Regulation table links ── */
    .regulations-table a {
        color: #7dd3fc;
        text-decoration: none;
        transition: color 0.15s ease;
    }
    .regulations-table a:hover {
        color: #bae6fd;
        text-decoration: underline;
    }

    /* ── Data tables ── */
    .stDataFrame table {
        table-layout: fixed;
        width: 100% !important;
    }
    .stDataFrame th,
    .stDataFrame td {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        text-align: left;
        padding: 6px 10px !important;
    }
    .stDataFrame th {
        font-weight: 600;
        background-color: rgba(255, 255, 255, 0.04);
    }
    .stDataFrame tr:hover td {
        background-color: rgba(255, 255, 255, 0.03);
    }

    /* ── Mobile ── */
    @media (max-width: 768px) {
        .stApp {
            font-size: 14px;
        }
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }

        /* Tighten table cell padding on small screens */
        .stDataFrame th,
        .stDataFrame td {
            padding: 4px 6px !important;
            font-size: 12px !important;
        }

        /* Let the regulations HTML table stack better */
        .regulations-table td {
            display: block;
            width: 100% !important;
            text-align: left !important;
            padding: 4px 0 !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)
st.title("ASEAN Regulatory Dashboard")
st.caption("v3.9 • 2026-06-28")

DATA_DIR = Path("src") / "categories"
ARCHIVE_FILE = Path("src") / "CBregs_audit_log.csv"


def resolve_data_dir(path: Path) -> Path:
    candidates = [
        path,
        Path(__file__).resolve().parent / path,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Data directory not found. Tried: {', '.join(str(p) for p in candidates)}"
    )


def get_csv_cache_key(src_dir: Path) -> tuple:
    csv_files = sorted(src_dir.glob("*.csv"))
    return tuple((f.name, f.stat().st_mtime) for f in csv_files)


# =========================
# Helpers
# =========================
ASEAN_FLAG = {
    "Brunei Darussalam": "🇧🇳",
    "Cambodia": "🇰🇭",
    "Indonesia": "🇮🇩",
    "Lao PDR": "🇱🇦",
    "Laos": "🇱🇦",
    "Malaysia": "🇲🇾",
    "Myanmar": "🇲🇲",
    "Philippines": "🇵🇭",
    "Singapore": "🇸🇬",
    "Thailand": "🇹🇭",
    "Viet Nam": "🇻🇳",
    "Vietnam": "🇻🇳",
    "Timor-Leste": "🇹🇱",
}

COUNTRY_ISO_CODES = {
    "Brunei Darussalam": "BN",
    "Cambodia": "KH",
    "Indonesia": "ID",
    "Lao PDR": "LA",
    "Laos": "LA",
    "Malaysia": "MY",
    "Myanmar": "MM",
    "Philippines": "PH",
    "Singapore": "SG",
    "Thailand": "TH",
    "Viet Nam": "VN",
    "Vietnam": "VN",
    "Timor-Leste": "TL",
}


def country_code_to_emoji(code: str) -> str:
    code = str(code).upper().strip()
    if len(code) != 2 or not code.isalpha():
        return ""
    return "".join(chr(ord(ch) + 0x1F1E6 - ord("A")) for ch in code)


def country_flag(country: str) -> str:
    country = str(country).strip()
    if not country:
        return "🏳️"
    if country in ASEAN_FLAG:
        return ASEAN_FLAG[country]
    iso = COUNTRY_ISO_CODES.get(country)
    if iso:
        return country_code_to_emoji(iso)
    return "🏳️"


META_COL_CANDIDATES = {
    "country": ["Country_std", "country", "Country"],
    "regulator": ["Regulator_std", "regulator", "Regulator"],
    "year": [
        "Year_raw",
        "year",
        "Year",
        "Year approved/implemented",
        "Year Approved/Implemented",
        "Year approved / implemented",
    ],
    "source": ["Source_Original", "Source_URL", "source", "Official Source", "Official source", "Official Source links", "Official source links", "Source", "URL", "Link"],
    "title": [
        "Issuance",
        "Regulation_Title",
        "title",
        "Regulation / Legal Instrument",
        "Regulation / Legal instrument",
        "Primary Legal / Regulatory Framework",
        "Primary Legal/Regulatory Framework",
        "Regulations on fraud risk management",
        "Regulations on consumer protection (payments)",
        "Regulation",
        "Legal Instrument",
    ],
}

_KNOWN_META: set[str] = set(
    META_COL_CANDIDATES["country"]
    + META_COL_CANDIDATES["regulator"]
    + META_COL_CANDIDATES["year"]
    + META_COL_CANDIDATES["source"]
    + META_COL_CANDIDATES["title"]
    + [
        "Category", "Country_std", "Regulator_std", "Year_raw", "Year",
        "Year_sort", "Regulation_Title", "Source_URL", "Entry_ID", "HasData", "ID",
        "Source_Original", "Source_EN", "Amendment_Of",
    ]
)


def get_provision_cols(df: pd.DataFrame, category: str) -> List[str]:
    cat_df = df[df["Category"] == category]
    return [
        c for c in cat_df.columns
        if c not in _KNOWN_META and not cat_df[c].isna().all()
    ]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def extract_year(value) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    # Strip effectivity date qualifiers so "2019 (effective 2022)" → 2019
    s = re.sub(r'\(effectiv\w*[^)]*\)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'effectiv\w*\s+\d[^,;|]*', '', s, flags=re.IGNORECASE)
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return int(m.group(1)) if m else None


def pick_first_existing_col(
    df: pd.DataFrame, candidates: List[str]
) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_title_col(df: pd.DataFrame) -> Optional[str]:
    # 1) Try known title candidates.
    c = pick_first_existing_col(df, META_COL_CANDIDATES["title"])
    if c:
        return c

    # 2) Fallback: pick the first non-meta column whose values look like text.
    known_meta = set(
        META_COL_CANDIDATES["country"]
        + META_COL_CANDIDATES["regulator"]
        + META_COL_CANDIDATES["year"]
        + META_COL_CANDIDATES["source"]
        + ["Regulation ID"]
    )
    for col in df.columns:
        if col in known_meta:
            continue
        if df[col].astype(str).str.len().mean() > 5:
            return col
    return None


def safe_linkify(url) -> str:
    if url is None:
        return ""
    url = str(url).strip()
    if not url or url.lower() in {"nan", "none"}:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
    return url


def html_escape(text: str) -> str:
    text = str(text)
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("\n", " ")
            .replace("\r", " ")
    )


def _render_provision_table(rows: list[dict], columns: list[str]) -> None:
    """Render a resizable provision table via st.components.v1.html()."""
    import html as _hl

    # Build <th> cells — each has a drag handle div
    th_cells = "".join(
        f'<th><span class="col-label">{_hl.escape(str(c))}</span>'
        f'<div class="resizer" title="Drag to resize"></div></th>'
        for c in columns
    )

    # Build <tr> cells
    body_rows = []
    for i, row in enumerate(rows):
        cells = []
        for j, col in enumerate(columns):
            raw = str(row.get(col, ""))
            val = _hl.escape(raw).replace("\r\n", "<br>").replace("\n", "<br>")
            cls = "fc" if j == 0 else "vc"
            cells.append(f'<td class="{cls}">{val}</td>')
        cls = "r0" if i % 2 == 0 else "r1"
        body_rows.append(f'<tr class="{cls}">{"".join(cells)}</tr>')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:"Source Sans Pro",sans-serif;font-size:14px;color:#e2e8f0}}
table{{width:100%;border-collapse:collapse;table-layout:fixed}}
thead tr{{background:#1e293b}}
th{{position:relative;padding:7px 10px 7px 8px;text-align:left;color:#94a3b8;
    font-size:13px;font-weight:600;border-bottom:2px solid #334155;
    white-space:nowrap;overflow:hidden;user-select:none}}
.col-label{{display:block;overflow:hidden;text-overflow:ellipsis;padding-right:6px}}
.resizer{{position:absolute;right:0;top:0;bottom:0;width:5px;cursor:col-resize;
          background:transparent;z-index:1}}
.resizer:hover,.resizer.active{{background:#3b82f6;opacity:.7}}
tr.r0{{background:#0d1526}}
tr.r1{{background:#111e33}}
tr:hover{{background:#1a2d4a}}
td{{padding:7px 10px 7px 8px;vertical-align:top;border-bottom:1px solid #1e293b;
    line-height:1.6;word-break:break-word;overflow-wrap:anywhere}}
td.fc{{color:#7dd3fc;font-size:13px;font-weight:600;word-break:break-word;overflow-wrap:anywhere}}
td.vc{{font-size:13.5px}}
</style>
</head>
<body>
<table id="t">
  <thead><tr>{th_cells}</tr></thead>
  <tbody>{"".join(body_rows)}</tbody>
</table>
<script>
(function(){{
  const tbl = document.getElementById('t');
  const ths = tbl.querySelectorAll('th');

  // Set initial column widths
  const total = tbl.parentElement.offsetWidth || 800;
  const fieldW = Math.min(170, total * 0.18);
  const otherW = (total - fieldW) / Math.max(1, ths.length - 1);
  ths[0].style.width = fieldW + 'px';
  for (let i = 1; i < ths.length; i++) ths[i].style.width = otherW + 'px';

  // Resize logic
  let active = null, startX = 0, startW = 0;
  tbl.querySelectorAll('.resizer').forEach(r => {{
    r.addEventListener('mousedown', e => {{
      active = r;
      startX = e.pageX;
      startW = r.parentElement.offsetWidth;
      r.classList.add('active');
      document.body.style.cursor = 'col-resize';
      e.preventDefault();
    }});
  }});
  document.addEventListener('mousemove', e => {{
    if (!active) return;
    const w = Math.max(60, startW + (e.pageX - startX));
    active.parentElement.style.width = w + 'px';
  }});
  document.addEventListener('mouseup', () => {{
    if (active) {{
      active.classList.remove('active');
      active = null;
      document.body.style.cursor = '';
    }}
  }});

  // Report table height to Streamlit so iframe fits exactly
  let _lastH = 0;
  function sendHeight() {{
    const h = Math.ceil(tbl.getBoundingClientRect().height) + 8;
    if (h > 0 && h !== _lastH) {{ _lastH = h;
      window.parent.postMessage({{isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:h}}, '*');
    }}
  }}
  sendHeight();
  window.addEventListener('load', sendHeight);
  setTimeout(sendHeight, 100);
  setTimeout(sendHeight, 500);
  new ResizeObserver(sendHeight).observe(tbl);
}})();
</script>
</body></html>"""

    # Generous initial estimate; JS grows the iframe to exact height via postMessage.
    # scrolling=True is a fallback: if postMessage undershoots, user can still scroll.
    total_chars = sum(len(str(row.get(col, ""))) for row in rows for col in columns[1:])
    est = max(300, 60 + len(rows) * 60 + (total_chars // 60) * 22)
    components.html(html, height=est, scrolling=True)


def _render_data_table(df: pd.DataFrame, highlighted: set | None = None, row_height: int = 72) -> None:
    """Render a DataFrame as a resizable HTML table; newlines in cells become <br>.
    First column (flag) is centred; second column (country) is bold.
    Rows whose Country value is in `highlighted` are tinted."""
    import html as _hl

    th_cells = "".join(
        f'<th><span class="col-label">{_hl.escape(str(c))}</span>'
        f'<div class="resizer" title="Drag to resize"></div></th>'
        for c in df.columns
    )

    body_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        country_val = str(row.get("Country", "")) if "Country" in df.columns else ""
        is_sel = bool(highlighted and country_val in highlighted)
        cells = []
        for j, col in enumerate(df.columns):
            raw = "" if pd.isna(row[col]) else str(row[col])
            val = _hl.escape(raw).replace("\r\n", "<br>").replace("\n", "<br>")
            if j == 0:
                cells.append(f'<td style="text-align:center;padding:6px 8px;vertical-align:top;">{val}</td>')
            elif j == 1:
                cells.append(f'<td style="font-weight:600;padding:6px 10px;vertical-align:top;white-space:nowrap;">{val}</td>')
            else:
                cells.append(f'<td class="vc">{val}</td>')
        cls = "sel" if is_sel else ("r0" if i % 2 == 0 else "r1")
        body_rows.append(f'<tr class="{cls}">{"".join(cells)}</tr>')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:"Source Sans Pro",sans-serif;font-size:14px;color:#e2e8f0}}
table{{width:100%;border-collapse:collapse;table-layout:fixed}}
thead tr{{background:#1e293b}}
th{{position:relative;padding:7px 10px 7px 8px;text-align:left;color:#94a3b8;
    font-size:13px;font-weight:600;border-bottom:2px solid #334155;
    white-space:nowrap;overflow:hidden;user-select:none}}
.col-label{{display:block;overflow:hidden;text-overflow:ellipsis;padding-right:6px}}
.resizer{{position:absolute;right:0;top:0;bottom:0;width:5px;cursor:col-resize;
          background:transparent;z-index:1}}
.resizer:hover,.resizer.active{{background:#3b82f6;opacity:.7}}
tr.r0{{background:#0d1526}}
tr.r1{{background:#111e33}}
tr.sel{{background:#1a3a5c}}
tr:hover{{background:#1a2d4a}}
tr.sel:hover{{background:#20446b}}
td{{padding:6px 10px;vertical-align:top;border-bottom:1px solid #1e293b;
    line-height:1.6;word-break:break-word;overflow-wrap:anywhere;font-size:13.5px}}
.vc{{font-size:13.5px}}
</style>
</head>
<body>
<table id="t">
  <thead><tr>{th_cells}</tr></thead>
  <tbody>{"".join(body_rows)}</tbody>
</table>
<script>
(function(){{
  const tbl = document.getElementById('t');
  const ths = tbl.querySelectorAll('th');
  const total = tbl.parentElement.offsetWidth || 900;
  const flagW = 48, countryW = 140;
  const rem = total - flagW - countryW;
  const otherW = ths.length > 2 ? rem / (ths.length - 2) : rem;
  if (ths.length > 0) ths[0].style.width = flagW + 'px';
  if (ths.length > 1) ths[1].style.width = countryW + 'px';
  for (let i = 2; i < ths.length; i++) ths[i].style.width = Math.max(80, otherW) + 'px';

  let active = null, startX = 0, startW = 0;
  tbl.querySelectorAll('.resizer').forEach(r => {{
    r.addEventListener('mousedown', e => {{
      active = r; startX = e.pageX; startW = r.parentElement.offsetWidth;
      r.classList.add('active'); document.body.style.cursor = 'col-resize'; e.preventDefault();
    }});
  }});
  document.addEventListener('mousemove', e => {{
    if (!active) return;
    active.parentElement.style.width = Math.max(48, startW + (e.pageX - startX)) + 'px';
  }});
  document.addEventListener('mouseup', () => {{
    if (active) {{ active.classList.remove('active'); active = null; document.body.style.cursor = ''; }}
  }});
  let _lastH = 0;
  function sendH() {{
    const h = Math.ceil(tbl.getBoundingClientRect().height) + 8;
    if (h > 0 && h !== _lastH) {{ _lastH = h;
      window.parent.postMessage({{isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:h}},'*');
    }}
  }}
  sendH();
  window.addEventListener('load', sendH);
  setTimeout(sendH, 150);
  new ResizeObserver(sendH).observe(tbl);
}})();
</script>
</body></html>"""

    est = min(900, max(150, 44 + len(df) * row_height))
    components.html(html, height=est, scrolling=False)


def build_regulations_html_table(df: pd.DataFrame) -> str:
    rows = [
        '<table class="regulations-table" style="width:100%; border-collapse: collapse; margin-bottom:1rem; table-layout: fixed;">',
        "<tbody>",
    ]
    for _, row in df.iterrows():
        title = row.get("Regulation_Title")
        title = html_escape(title) if pd.notna(title) else "—"
        year = row.get("Year")
        year_text = str(int(year)) if pd.notna(year) else "—"
        source_url = safe_linkify(row.get("Source_URL"))
        source_en_url = safe_linkify(row.get("Source_EN")) if pd.notna(row.get("Source_EN")) else ""
        if source_url and source_en_url and source_en_url != source_url:
            source_html = (
                f'<a href="{html_escape(source_url)}" target="_blank" rel="noreferrer">Source</a>'
                f' · <a href="{html_escape(source_en_url)}" target="_blank" rel="noreferrer">EN</a>'
            )
        elif source_url:
            source_html = (
                f'<a href="{html_escape(source_url)}" target="_blank" rel="noreferrer">Source</a>'
            )
        else:
            source_html = "—"
        rows.append(
            "<tr>"
            f"<td style=\"padding:8px 12px 8px 0; vertical-align:top; white-space:normal; word-break:break-word;\">{title}</td>"
            f"<td style=\"padding:8px 12px; vertical-align:top; white-space:nowrap; text-align:right;\">{year_text}</td>"
            f"<td style=\"padding:8px 0 8px 12px; vertical-align:top; white-space:nowrap; text-align:center;\">{source_html}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def get_selected_row_indices(event) -> List[int]:
    """Return all selected row indices from a dataframe selection event."""
    if not event:
        return []
    selection = event.get("selection")
    if not selection:
        return []
    return selection.get("rows") or []


@st.cache_data
def load_cbregs(src_dir: str, cache_key: tuple) -> pd.DataFrame:
    p = Path(src_dir)
    csv_files = sorted(p.glob("*.csv"))
    frames = []

    for csv_file in csv_files:
        sheet = csv_file.stem
        df = pd.read_csv(csv_file, dtype=object)
        df = normalize_columns(df).dropna(how="all").dropna(axis=1, how="all")

        country_col = pick_first_existing_col(df, META_COL_CANDIDATES["country"])
        regulator_col = pick_first_existing_col(df, META_COL_CANDIDATES["regulator"])
        year_col = pick_first_existing_col(df, META_COL_CANDIDATES["year"])
        source_col = pick_first_existing_col(df, META_COL_CANDIDATES["source"])
        title_col = infer_title_col(df)

        # Build standardized view while retaining originals for the country modal
        out = df.copy()
        out["Category"] = sheet

        out["Country_std"] = out[country_col] if country_col else pd.NA
        out["Regulator_std"] = out[regulator_col] if regulator_col else pd.NA
        out["Year_raw"] = out[year_col] if year_col else pd.NA
        out["Year"] = out["Year_raw"].apply(extract_year).astype("Int64")

        if title_col:
            out["Regulation_Title"] = out[title_col].astype(str)
        else:
            out["Regulation_Title"] = pd.NA

        if source_col:
            out["Source_URL"] = out[source_col].astype(str)
        else:
            out["Source_URL"] = pd.NA

        # Source_EN — English translation URL (same as Source_URL if not available)
        if "Source_EN" in df.columns:
            out["Source_EN"] = out["Source_EN"].astype(str)
        else:
            out["Source_EN"] = pd.NA

        # Amendment_Of — reference to original regulation for amendments
        if "Amendment_Of" in df.columns:
            out["Amendment_Of"] = out["Amendment_Of"].astype(str)
        else:
            out["Amendment_Of"] = pd.NA

        frames.append(out)

    all_df = pd.concat(frames, ignore_index=True)

    # Clean
    all_df["Country_std"] = all_df["Country_std"].astype(str).str.strip()
    all_df["Regulator_std"] = all_df["Regulator_std"].astype(str).str.strip()
    all_df["Regulation_Title"] = all_df["Regulation_Title"].astype(str).str.strip()

    # Treat "nan" strings produced by astype(str)
    for c in ["Country_std", "Regulator_std", "Regulation_Title", "Source_URL", "Source_EN", "Amendment_Of"]:
        if c in all_df.columns:
            all_df.loc[all_df[c].str.lower().isin(["nan", "none", ""]), c] = pd.NA

    all_df = ensure_entry_ids(all_df)
    return all_df


def latest_regs_by_country(df: pd.DataFrame, country: str, n: int = 10) -> pd.DataFrame:
    d = df[df["Country_std"] == country].copy()
    d = d.dropna(subset=["Regulation_Title"])
    # Year might be NA; put those last
    d["Year_sort"] = d["Year"].fillna(-1).astype(int)
    d = d.sort_values(["Year_sort", "Regulation_Title"], ascending=[False, True])
    return d.head(n)[["Year", "Regulation_Title", "Category", "Regulator_std", "Source_URL"]]


def build_hover_list(df_country_latest: pd.DataFrame) -> str:
    if df_country_latest.empty:
        return "No regulations found."
    lines = []
    for _, r in df_country_latest.iterrows():
        y = r["Year"]
        y_txt = str(int(y)) if pd.notna(y) else "—"
        title = str(r["Regulation_Title"])
        if len(title) > 60:
            title = title[:57] + "…"
        lines.append(f"{y_txt} — {title}")
    # Plotly hover supports <br>
    return "<br>".join(lines)


def ensure_entry_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Entry_ID" not in df.columns:
        df["Entry_ID"] = pd.NA

    df["Entry_ID"] = df["Entry_ID"].astype(str).replace(["nan", "None", "none"], pd.NA)
    missing = df["Entry_ID"].isna() | (df["Entry_ID"] == "")
    if missing.any():
        df.loc[missing, "Entry_ID"] = [str(uuid.uuid4()) for _ in range(missing.sum())]

    return df


def serialize_value_for_archive(value):
    if pd.isna(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    return str(value)


def serialize_record_for_archive(record: pd.Series) -> dict:
    return {
        k: serialize_value_for_archive(v)
        for k, v in record.items()
        if k is not None
    }


def append_audit_log(
    action: str,
    country: str,
    entry_id: str,
    user: str,
    old_record: Optional[dict],
    new_record: Optional[dict],
) -> None:
    ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "action": action,
        "country": country,
        "entry_id": entry_id,
        "user": user,
        "old_record": (
            json.dumps(old_record, ensure_ascii=False)
            if old_record is not None
            else ""
        ),
        "new_record": (
            json.dumps(new_record, ensure_ascii=False)
            if new_record is not None
            else ""
        ),
    }
    header = not ARCHIVE_FILE.exists()
    pd.DataFrame([row]).to_csv(ARCHIVE_FILE, mode="a", index=False, header=header)


def get_query_params() -> dict[str, str]:
    """Return current URL query parameters as a plain str->str dict.

    Streamlit 1.28+ exposes st.query_params as a Mapping[str, str].
    Older builds had experimental_get_query_params() -> dict[str, list[str]];
    we normalise both shapes to a flat str->str dict.
    """
    # Modern API (Streamlit >= 1.28)
    modern = getattr(st, "query_params", None)
    if modern is not None and not callable(modern):
        # st.query_params is a Mapping-like object, not a function
        return {k: str(v) for k, v in modern.items()}
    # Legacy API
    getter = getattr(st, "experimental_get_query_params", None)
    if callable(getter):
        raw: dict[str, list[str]] = getter()  # type: ignore[assignment]
        return {k: v[0] if v else "" for k, v in raw.items()}
    return {}


def safe_rerun() -> None:
    rerun_func = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun_func):
        rerun_func()


def save_cbregs(df: pd.DataFrame, src_dir: Path) -> None:
    src_dir = Path(src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)
    for category, group in df.groupby("Category", sort=False):
        csv_path = src_dir / f"{category}.csv"
        group.to_csv(csv_path, index=False)


# =========================
# Gap Analysis data layer
# =========================
_GAP_BENCH_FILE   = "gap_benchmarks.csv"
_GAP_MAP_FILE     = "gap_mappings.csv"
_GAP_AUDIT_FILE   = "gap_audit_log.csv"

ASEAN_COUNTRIES_ORDERED = [
    "Brunei Darussalam", "Cambodia", "Indonesia", "Lao PDR",
    "Malaysia", "Myanmar", "Philippines", "Singapore",
    "Thailand", "Timor-Leste", "Viet Nam",
]

GAP_STATUS_OPTIONS = ["Meets", "Partially meets", "Not assessed"]
GAP_STATUS_EMOJI   = {"Meets": "✅", "Partially meets": "⚠️", "Not assessed": "—"}
GAP_STATUS_STYLE   = {
    "Meets":           "background:#14532d;color:#86efac",
    "Partially meets": "background:#78350f;color:#fcd34d",
    "Not assessed":    "background:#0f172a;color:#475569",
}
_GAP_BENCH_COLS = [
    "Benchmark_ID", "Standard", "Category", "Topic", "Provision",
    "Created_By", "Created_At", "Updated_By", "Updated_At",
]
_GAP_MAP_COLS = [
    "Mapping_ID", "Benchmark_ID", "Country", "Status",
    "Entry_IDs", "Notes", "Updated_By", "Updated_At",
]


def get_gap_cache_key(src_dir: Path) -> str:
    b = src_dir / _GAP_BENCH_FILE
    m = src_dir / _GAP_MAP_FILE
    return f"{b.stat().st_mtime if b.exists() else 0}_{m.stat().st_mtime if m.exists() else 0}"


@st.cache_data
def load_gap_benchmarks(src_dir: str, cache_key: str) -> pd.DataFrame:
    path = Path(src_dir) / _GAP_BENCH_FILE
    if not path.exists():
        return pd.DataFrame(columns=_GAP_BENCH_COLS)
    return pd.read_csv(path, dtype=str).fillna("")


@st.cache_data
def load_gap_mappings(src_dir: str, cache_key: str) -> pd.DataFrame:
    path = Path(src_dir) / _GAP_MAP_FILE
    if not path.exists():
        return pd.DataFrame(columns=_GAP_MAP_COLS)
    return pd.read_csv(path, dtype=str).fillna("")


def save_gap_benchmarks(df: pd.DataFrame, src_dir: Path) -> None:
    df.to_csv(src_dir / _GAP_BENCH_FILE, index=False)


def save_gap_mappings(df: pd.DataFrame, src_dir: Path) -> None:
    df.to_csv(src_dir / _GAP_MAP_FILE, index=False)


def append_gap_audit(
    src_dir: Path, action: str, record_type: str,
    record_id: str, before: dict, after: dict, user: str,
) -> None:
    path = src_dir / _GAP_AUDIT_FILE
    row = {
        "Timestamp":   pd.Timestamp.utcnow().isoformat(),
        "User":        user,
        "Action":      action,
        "Record_Type": record_type,
        "Record_ID":   record_id,
        "Before":      json.dumps(before, ensure_ascii=False),
        "After":       json.dumps(after,  ensure_ascii=False),
    }
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _gap_reg_options(df_all: pd.DataFrame, country: str, category: str = "") -> dict[str, str]:
    """Return {Entry_ID: display_label} for a country's regulations, optionally filtered by category."""
    mask = df_all["Country_std"] == country
    if category and category not in ("", "nan", "All"):
        mask &= df_all["Category"] == category
    opts: dict[str, str] = {}
    for _, r in df_all[mask].dropna(subset=["Regulation_Title"]).iterrows():
        eid   = str(r["Entry_ID"])
        title = str(r.get("Regulation_Title", "Untitled"))
        yr    = r.get("Year", "")
        yr_s  = f" ({int(float(yr))})" if pd.notna(yr) and str(yr) not in ("", "nan") else ""
        opts[eid] = f"{title}{yr_s}"
    return opts


def _render_gap_matrix(df_bench: pd.DataFrame, df_map: pd.DataFrame, countries: list[str]) -> None:
    """Render the gap analysis status matrix: benchmarks (rows) × countries (cols)."""
    import html as _hl
    if df_bench.empty:
        st.info("No benchmarks have been defined yet. Admin can add them in the Benchmarks tab.")
        return

    status_lk: dict[tuple[str, str], str] = {}
    if not df_map.empty:
        for _, r in df_map.iterrows():
            status_lk[(str(r["Benchmark_ID"]), str(r["Country"]))] = str(r.get("Status", "Not assessed"))

    short_name = {
        "Brunei Darussalam": "Brunei", "Timor-Leste": "Timor-Leste",
        "Lao PDR": "Lao PDR",
    }

    th = (
        '<th style="min-width:85px;white-space:nowrap">ID</th>'
        '<th style="min-width:130px">Standard</th>'
        '<th style="min-width:180px">Topic / Area</th>'
    )
    for c in countries:
        flag = country_flag(c)
        label = _hl.escape(short_name.get(c, c))
        th += (
            f'<th style="text-align:center;min-width:70px;font-size:11px;line-height:1.4">'
            f'{flag}<br>{label}</th>'
        )

    rows_html = ""
    for i, (_, b) in enumerate(df_bench.iterrows()):
        bid   = str(b.get("Benchmark_ID", ""))
        std   = str(b.get("Standard", ""))
        topic = str(b.get("Topic", ""))
        bg    = "#0d1526" if i % 2 == 0 else "#111e33"
        cells = (
            f'<td style="font-weight:700;color:#7dd3fc;white-space:nowrap;font-size:12px">{_hl.escape(bid)}</td>'
            f'<td style="font-size:11px;color:#94a3b8">{_hl.escape(std)}</td>'
            f'<td style="font-size:13px">{_hl.escape(topic)}</td>'
        )
        for c in countries:
            status = status_lk.get((bid, c), "Not assessed")
            emoji  = GAP_STATUS_EMOJI.get(status, "—")
            style  = GAP_STATUS_STYLE.get(status, GAP_STATUS_STYLE["Not assessed"])
            cells += f'<td style="{style};text-align:center;font-size:16px;padding:5px 3px">{emoji}</td>'
        rows_html += f'<tr style="background:{bg}">{cells}</tr>'

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:"Source Sans Pro",sans-serif;color:#e2e8f0}}
table{{width:100%;border-collapse:collapse}}
thead tr{{background:#1e293b;position:sticky;top:0;z-index:2}}
th{{padding:8px 10px;text-align:left;color:#94a3b8;font-weight:600;border-bottom:2px solid #334155;vertical-align:middle}}
td{{padding:7px 10px;border-bottom:1px solid #1e293b;vertical-align:middle;word-break:break-word}}
tr:hover td{{filter:brightness(1.2)}}
</style></head><body>
<table id="t"><thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody></table>
<script>
(function(){{const tbl=document.getElementById('t');let _lH=0;
function sh(){{const h=Math.ceil(tbl.getBoundingClientRect().height)+8;
if(h>0&&h!==_lH){{_lH=h;window.parent.postMessage({{isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:h}},'*');}}}}
sh();window.addEventListener('load',sh);setTimeout(sh,100);setTimeout(sh,500);
new ResizeObserver(sh).observe(tbl);}})();
</script></body></html>"""

    est = max(300, 60 + len(df_bench) * 52)
    components.html(html, height=est, scrolling=True)


@st.dialog("Edit regulation", width="large")
def edit_regulation_dialog(
    existing_row: pd.Series,
    category_choices: List[str],
    df_all: pd.DataFrame,
    src_dir: Path,
    editor_country: str,
    auth_user: str,
) -> None:
    selected_entry_id = str(existing_row["Entry_ID"])
    save_key = f"edit_saved_{selected_entry_id}"
    confirm_key = f"confirm_delete_{selected_entry_id}"
    st.session_state.setdefault(save_key, False)
    st.session_state.setdefault(confirm_key, False)

    existing_category = (
        existing_row["Category"]
        if pd.notna(existing_row["Category"])
        else category_choices[0]
    )
    edit_provision_cols = get_provision_cols(df_all, existing_category)

    def _s(val) -> str:
        return "" if pd.isna(val) else str(val)

    # Reset field keys when dialog opens for a different entry
    if st.session_state.get("_dialog_entry_id") != selected_entry_id:
        st.session_state["_dialog_entry_id"] = selected_entry_id
        for k in ["_dlg_cat", "_dlg_reg", "_dlg_year", "_dlg_title", "_dlg_url", "_dlg_url_en", "_dlg_amend"]:
            st.session_state.pop(k, None)
        for col in edit_provision_cols:
            st.session_state.pop(f"_dlg_prov_{col}", None)

    # Editable fields (no form — required for real-time change detection)
    edit_category = st.selectbox(
        "Category",
        options=category_choices,
        index=category_choices.index(existing_category) if existing_category in category_choices else 0,
        key="_dlg_cat",
    )
    edit_regulator  = st.text_input("Regulator",           value=_s(existing_row.get("Regulator_std")),    key="_dlg_reg")
    edit_year_raw   = st.text_input("Year",                value=_s(existing_row.get("Year_raw")),         key="_dlg_year")
    edit_title      = st.text_area("Regulation title",     value=_s(existing_row.get("Regulation_Title")), height=120, key="_dlg_title")
    edit_source_url = st.text_input("Source URL",          value=_s(existing_row.get("Source_URL")),       key="_dlg_url")
    edit_source_en  = st.text_input("Source URL (English)", value=_s(existing_row.get("Source_EN")),       key="_dlg_url_en")

    _amend_country = str(existing_row.get("Country_std", ""))
    _amend_cand = df_all[
        (df_all["Country_std"] == _amend_country) &
        (df_all["Category"] == edit_category) &
        df_all["Regulation_Title"].notna() &
        (df_all["Entry_ID"] != selected_entry_id)
    ].sort_values(["Year", "Regulation_Title"], ascending=[False, True])
    _BLANK_OPT = "— (not an amendment)"
    amend_opts = [_BLANK_OPT] + [
        f"{r['Regulation_Title']} ({int(r['Year']) if pd.notna(r['Year']) else '?'})"
        for _, r in _amend_cand.iterrows()
    ]
    _cur_amend = _s(existing_row.get("Amendment_Of"))
    _amend_idx = amend_opts.index(_cur_amend) if _cur_amend in amend_opts else 0
    edit_amendment_of_sel = st.selectbox(
        "Amends (leave blank if original)",
        options=amend_opts,
        index=_amend_idx,
        key="_dlg_amend",
    )
    edit_amendment_of = "" if edit_amendment_of_sel == _BLANK_OPT else edit_amendment_of_sel

    edit_provision_values: dict[str, str] = {}
    if edit_provision_cols:
        st.markdown("**Key provisions**")
        for col in edit_provision_cols:
            edit_provision_values[col] = st.text_area(
                col, value=_s(existing_row.get(col)), height=68, key=f"_dlg_prov_{col}"
            )

    # Detect changes against the original row
    orig = {
        "_dlg_cat":    existing_category,
        "_dlg_reg":    _s(existing_row.get("Regulator_std")),
        "_dlg_year":   _s(existing_row.get("Year_raw")),
        "_dlg_title":  _s(existing_row.get("Regulation_Title")),
        "_dlg_url":    _s(existing_row.get("Source_URL")),
        "_dlg_url_en": _s(existing_row.get("Source_EN")),
        "_dlg_amend":  _cur_amend if _cur_amend in amend_opts else _BLANK_OPT,
        **{f"_dlg_prov_{col}": _s(existing_row.get(col)) for col in edit_provision_cols},
    }
    has_changes = any(st.session_state.get(k, orig[k]) != orig[k] for k in orig)

    if st.button("Save updates", type="primary", disabled=not has_changes, use_container_width=True):
        updated_df = df_all.copy()
        row_index = updated_df.index[updated_df["Entry_ID"] == selected_entry_id][0]
        old_record = serialize_record_for_archive(updated_df.loc[row_index])
        updated_df.at[row_index, "Category"]      = edit_category
        updated_df.at[row_index, "Regulator_std"] = edit_regulator.strip() or pd.NA
        updated_df.at[row_index, "Year_raw"]      = edit_year_raw.strip() or pd.NA
        updated_df.at[row_index, "Year"]          = extract_year(edit_year_raw)
        updated_df.at[row_index, "Regulation_Title"] = edit_title.strip()
        updated_df.at[row_index, "Source_URL"]    = edit_source_url.strip() or pd.NA
        updated_df.at[row_index, "Source_EN"]     = edit_source_en.strip() or pd.NA
        updated_df.at[row_index, "Amendment_Of"]  = edit_amendment_of.strip() or pd.NA
        for col, val in edit_provision_values.items():
            updated_df.at[row_index, col] = val.strip() or pd.NA
        save_cbregs(updated_df, src_dir)
        append_audit_log(
            action="edit",
            country=editor_country,
            entry_id=selected_entry_id,
            user=auth_user,
            old_record=old_record,
            new_record=serialize_record_for_archive(updated_df.loc[row_index]),
        )
        st.session_state[save_key] = True

    if st.session_state[save_key]:
        st.success("Changes saved successfully.")

    if st.button("Close", type="primary", use_container_width=True):
        st.session_state.pop(save_key, None)
        for _cat in df_all["Category"].dropna().unique():
            _cat_key = re.sub(r"[^a-zA-Z0-9]", "_", str(_cat))
            st.session_state.pop(f"editor_table_{_cat_key}", None)
        st.rerun()

    # Marker + scoped CSS: only secondary buttons after this point in the dialog are red
    _arch_marker = f"dlg-arch-{selected_entry_id[:8]}"
    st.markdown(
        f'<div id="{_arch_marker}"></div>'
        f'<style>'
        f'div:has(>div#{_arch_marker}) ~ div [data-testid="baseButton-secondary"]'
        f'{{background-color:#dc2626!important;border-color:#dc2626!important;color:#fff!important}}'
        f'div:has(>div#{_arch_marker}) ~ div [data-testid="baseButton-secondary"]:hover'
        f'{{background-color:#b91c1c!important;border-color:#b91c1c!important}}'
        f'</style>',
        unsafe_allow_html=True,
    )
    st.divider()
    if not st.session_state[confirm_key]:
        if st.button("Archive and delete this record", type="secondary", use_container_width=True):
            st.session_state[confirm_key] = True
            st.rerun()
    else:
        st.warning("Are you sure? This record will be permanently removed from the dataset.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, delete", type="secondary", use_container_width=True):
                updated_df = df_all.copy()
                row_index = updated_df.index[updated_df["Entry_ID"] == selected_entry_id][0]
                old_record = serialize_record_for_archive(updated_df.loc[row_index])
                updated_df = updated_df.drop(index=row_index).reset_index(drop=True)
                save_cbregs(updated_df, src_dir)
                append_audit_log(
                    action="delete",
                    country=editor_country,
                    entry_id=selected_entry_id,
                    user=auth_user,
                    old_record=old_record,
                    new_record=None,
                )
                st.session_state.pop(confirm_key, None)
                st.rerun()
        with col2:
            if st.button("Cancel", type="primary", use_container_width=True):
                st.session_state[confirm_key] = False
                st.rerun()


# =========================
# Load data
# =========================
resolved_data_dir = resolve_data_dir(DATA_DIR)

df_all = load_cbregs(str(resolved_data_dir), get_csv_cache_key(resolved_data_dir))

if df_all.empty:
    st.error("Data loaded but produced no rows.")
    st.stop()

# =========================
# Category filter + download (above tabs)
# =========================
categories = ["All"] + sorted(df_all["Category"].dropna().unique().tolist())

def _on_category_change():
    st.session_state.pop("table_a_country_sel", None)
    st.session_state.pop("table_b_country_sel", None)

_col_pills, _col_dl = st.columns([5, 1], vertical_alignment="bottom")
sel_category = _col_pills.pills(
    "Category",
    options=categories,
    default="All",
    key="category_filter",
    on_change=_on_category_change,
) or "All"

df_f = df_all.copy()
if sel_category != "All":
    df_f = df_f[df_f["Category"] == sel_category]

csv_bytes = df_f.to_csv(index=False).encode("utf-8")
_col_dl.download_button(
    "⬇ Export",
    csv_bytes,
    "asean_regulations_filtered.csv",
    "text/csv",
    use_container_width=True,
)

# =========================
# Country extender (modal)
# =========================
def show_country_modal(country: str, key_suffix: str = ""):
    st.divider()
    st.markdown(f"## {ASEAN_FLAG.get(country, '🏳️')} {country}")

    map_df = pd.DataFrame({"Country": [country], "Selected": [1]})
    country_fig = px.choropleth(
        map_df,
        locations="Country",
        locationmode="country names",
        color="Selected",
        color_continuous_scale=["#facc15", "#facc15"],
        range_color=(0, 1),
    )
    country_fig.update_traces(showscale=False, hovertemplate="<extra></extra>")
    country_fig.update_geos(
        scope="asia",
        projection_type="mercator",
        lonaxis=dict(range=[92, 141]),
        lataxis=dict(range=[-11, 24]),
        visible=True,
        showcoastlines=True,
        coastlinecolor="rgba(255,255,255,0.35)",
        showcountries=True,
        countrycolor="rgba(255,255,255,0.85)",
        showland=True,
        landcolor="rgba(20, 30, 45, 1)",
        showocean=True,
        oceancolor="rgba(10, 16, 26, 1)",
        showlakes=True,
        lakecolor="rgba(10, 16, 26, 1)",
        bgcolor="rgba(0,0,0,0)",
    )
    country_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=220,
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(
        country_fig,
        use_container_width=True,
        key=f"country_modal_chart_{country}_{key_suffix}",
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "responsive": True,
        },
    )

    d = df_f[df_f["Country_std"] == country].copy()
    if d.empty:
        st.info(
            "No regulations found for this country under the current filters."
        )
        return

    # Download button for all country data
    export_cols = [
        c for c in d.columns
        if c not in {"Entry_ID", "Year_raw", "Year_sort", "HasData"}
    ]
    export_df = d[export_cols].rename(columns={
        "Country_std": "Country",
        "Regulator_std": "Regulator",
        "Regulation_Title": "Title",
        "Source_URL": "Source URL",
        "Source_EN": "Source URL (EN)",
        "Amendment_Of": "Amendment Of",
    })
    st.download_button(
        label=f"Download all data for {country}",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{country.lower().replace(' ', '_')}_regulations.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"dl_{country}_{key_suffix}",
    )

    regs = sorted(set(d["Regulator_std"].dropna().tolist()))
    if regs:
        st.markdown("**Regulator:** " + ", ".join(regs))
    else:
        st.markdown("**Regulator:** —")

    st.markdown("### Regulations")
    latest = d.sort_values(["Year", "Regulation_Title"], ascending=[False, True])
    regulation_lines = []
    for _, row in latest.iterrows():
        year = row.get("Year")
        year_text = str(int(year)) if pd.notna(year) else "—"
        title = str(row.get("Regulation_Title", "—"))
        source_link = safe_linkify(row.get("Source_URL"))
        source_en_raw = row.get("Source_EN")
        source_en_link = safe_linkify(source_en_raw) if pd.notna(source_en_raw) else ""
        amendment_of = row.get("Amendment_Of")

        src_parts = []
        if source_link:
            src_parts.append(f"[Source]({source_link})")
        if source_en_link and source_en_link != source_link:
            src_parts.append(f"[EN]({source_en_link})")
        source_text = f" ({' · '.join(src_parts)})" if src_parts else ""

        amend_text = ""
        if pd.notna(amendment_of) and str(amendment_of).strip().lower() not in ("nan", "none", ""):
            amend_text = f" *(amends {amendment_of})*"

        regulation_lines.append(f"- **{year_text}** — {title}{source_text}{amend_text}")
    st.markdown("\n".join(regulation_lines))

    st.markdown("### Key provisions")
    known_meta_cols = set(
        META_COL_CANDIDATES["country"]
        + META_COL_CANDIDATES["regulator"]
        + META_COL_CANDIDATES["year"]
        + META_COL_CANDIDATES["source"]
        + META_COL_CANDIDATES["title"]
        + [
            "Category",
            "Country_std",
            "Regulator_std",
            "Year_raw",
            "Year",
            "Year_sort",
            "Regulation_Title",
            "Source_URL",
            "Source_EN",
            "Amendment_Of",
            "Entry_ID",
        ]
    )

    shown_any_category = False
    for cat in sorted(d["Category"].dropna().unique().tolist()):
        dc = d[d["Category"] == cat].copy()
        detail_cols = [
            c for c in dc.columns
            if c not in known_meta_cols
            and c not in META_COL_CANDIDATES["source"]
        ]
        if not detail_cols:
            continue

        rows = []
        for col in detail_cols:
            values = (
                dc[col]
                .fillna(pd.NA)
                .dropna()
                .astype(str)
                .str.strip()
            )
            unique_values = sorted({
                v for v in values
                if pd.notna(v) and v.lower() not in {"nan", "none", ""}
            })
            if not unique_values:
                continue
            rows.append({"Field": col, "Values": "; ".join(unique_values)})

        if not rows:
            continue

        st.markdown(f"#### {cat}")
        _render_provision_table(rows, ["Field", "Values"])
        shown_any_category = True

    if not shown_any_category:
        st.caption(
            "No key provisions available for this country "
            "under the current filters."
        )

    st.caption(
        "Source links point to the official regulation document. "
        "EN links open the English translation where available."
    )


def show_country_comparison(countries: List[str], key_suffix: str = ""):
    """Side-by-side Key provisions comparison for two or more jurisdictions."""
    st.divider()
    flags_line = "  ·  ".join(
        f"{country_flag(c)} {c}" for c in countries
    )
    st.markdown(f"## Comparing: {flags_line}")

    known_meta_cols = set(
        META_COL_CANDIDATES["country"]
        + META_COL_CANDIDATES["regulator"]
        + META_COL_CANDIDATES["year"]
        + META_COL_CANDIDATES["source"]
        + META_COL_CANDIDATES["title"]
        + [
            "Category",
            "Country_std",
            "Regulator_std",
            "Year_raw",
            "Year",
            "Year_sort",
            "Regulation_Title",
            "Source_URL",
            "Source_EN",
            "Amendment_Of",
            "ID",
            "Entry_ID",
        ]
    )

    # Column header = "flag ISO" e.g. "🇸🇬 SG"
    def col_label(c: str) -> str:
        iso = COUNTRY_ISO_CODES.get(c, c[:2].upper())
        return f"{country_flag(c)} {iso}"

    col_labels = [col_label(c) for c in countries]

    # Gather all categories that appear for any of the selected countries
    all_cats = sorted(
        df_f[df_f["Country_std"].isin(countries)]["Category"].dropna().unique().tolist()
    )

    shown_any_category = False
    for cat in all_cats:
        # Collect detail columns across all countries for this category
        detail_cols_set: list[str] = []
        country_data: dict[str, pd.DataFrame] = {}
        for c in countries:
            dc = df_f[(df_f["Country_std"] == c) & (df_f["Category"] == cat)].copy()
            country_data[c] = dc
            for col in dc.columns:
                if (
                    col not in known_meta_cols
                    and col not in META_COL_CANDIDATES["source"]
                    and col not in detail_cols_set
                ):
                    detail_cols_set.append(col)

        if not detail_cols_set:
            continue

        # Build comparison rows: Field | <flag ISO col per country>
        rows = []
        for col in detail_cols_set:
            row: dict[str, str] = {"Field": col}
            has_any_value = False
            for c, label in zip(countries, col_labels):
                dc = country_data[c]
                if col not in dc.columns:
                    row[label] = "—"
                    continue
                values = (
                    dc[col]
                    .fillna(pd.NA)
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                unique_values = sorted({
                    v for v in values
                    if pd.notna(v) and v.lower() not in {"nan", "none", ""}
                })
                if unique_values:
                    row[label] = "; ".join(unique_values)
                    has_any_value = True
                else:
                    row[label] = "—"
            if has_any_value:
                rows.append(row)

        if not rows:
            continue

        st.markdown(f"#### {cat}")
        _render_provision_table(rows, ["Field"] + col_labels)
        shown_any_category = True

    if not shown_any_category:
        st.caption(
            "No key provisions found for the selected jurisdictions "
            "under the current filters."
        )

    st.caption(
        "Column headers show the flag and ISO code for each jurisdiction."
    )


# =========================
# =========================
# Gap Analysis dialogs
# =========================
@st.dialog("Add benchmark", width="large")
def _dlg_add_benchmark(df_bench: pd.DataFrame, categories: list[str], src_dir: Path, user: str) -> None:
    bid   = st.text_input("Benchmark ID *", placeholder="e.g. AML-01")
    std   = st.text_input("Standard", placeholder="e.g. FATF Recommendations")
    cat   = st.selectbox("Regulatory category", options=[""] + categories)
    topic = st.text_input("Topic / Area")
    prov  = st.text_area("Provision text", height=220)
    if st.button("Add benchmark", type="primary"):
        if not bid.strip():
            st.error("Benchmark ID is required.")
            return
        if bid.strip() in df_bench["Benchmark_ID"].tolist():
            st.error(f"Benchmark ID '{bid.strip()}' already exists.")
            return
        now = pd.Timestamp.utcnow().isoformat()
        new_row = {
            "Benchmark_ID": bid.strip(), "Standard": std.strip(), "Category": cat,
            "Topic": topic.strip(), "Provision": prov.strip(),
            "Created_By": user, "Created_At": now, "Updated_By": user, "Updated_At": now,
        }
        save_gap_benchmarks(pd.concat([df_bench, pd.DataFrame([new_row])], ignore_index=True), src_dir)
        append_gap_audit(src_dir, "add", "benchmark", bid.strip(), {}, new_row, user)
        st.rerun()


@st.dialog("Edit benchmark", width="large")
def _dlg_edit_benchmark(row: pd.Series, df_bench: pd.DataFrame, categories: list[str], src_dir: Path, user: str) -> None:
    bid   = str(row["Benchmark_ID"])
    st.markdown(f"**Benchmark ID:** {bid}")
    std   = st.text_input("Standard",     value=str(row.get("Standard", "")))
    cur_cat = str(row.get("Category", ""))
    cat_opts = [""] + categories
    cat   = st.selectbox("Regulatory category", options=cat_opts,
                         index=cat_opts.index(cur_cat) if cur_cat in cat_opts else 0)
    topic = st.text_input("Topic / Area", value=str(row.get("Topic", "")))
    prov  = st.text_area("Provision text", value=str(row.get("Provision", "")), height=220)
    if st.button("Save", type="primary"):
        before = row.to_dict()
        now    = pd.Timestamp.utcnow().isoformat()
        idx    = df_bench.index[df_bench["Benchmark_ID"] == bid].tolist()
        if idx:
            for col, val in [("Standard", std.strip()), ("Category", cat),
                              ("Topic", topic.strip()), ("Provision", prov.strip()),
                              ("Updated_By", user), ("Updated_At", now)]:
                df_bench.at[idx[0], col] = val
            save_gap_benchmarks(df_bench, src_dir)
            append_gap_audit(src_dir, "edit", "benchmark", bid, before, df_bench.loc[idx[0]].to_dict(), user)
        st.rerun()


# =========================
# Auth resolution (must happen before tabs are created)
# =========================
_qp = get_query_params()
_auth_country = _qp.get("country") or None   # raw value from URL
_auth_user    = _qp.get("user")    or None   # raw value from URL

# Admin: ?country=NA&user=Admin  →  full country dropdown
# Country user: ?country=X&user=Y  →  locked to X
# Unauthenticated: no Editor tab shown
IS_ADMIN        = (_auth_country == "NA" and _auth_user == "Admin")
IS_AUTHENTICATED = bool(_auth_country and _auth_user)

# =========================
# Tabs (Map default)
# =========================
tab_editor: Optional[DeltaGenerator] = None
tab_gap:    Optional[DeltaGenerator] = None
if IS_AUTHENTICATED:
    tab_map, tab_table, tab_gap, tab_editor, tab_guide = st.tabs(
        ["Map", "Table", "Gap Analysis", "Editor", "Guide"]
    )
else:
    tab_map, tab_table, tab_guide = st.tabs(["Map", "Table", "Guide"])

# =========================
# MAP TAB
# =========================
with tab_map:
    # Country counts + hover preview
    by_country = (
        df_f.groupby("Country_std", dropna=True)
        .size()
        .reset_index(name="Regulation_Count")
        .rename(columns={"Country_std": "Country"})
    )

    # Build hover text = latest 10 regs per country
    hover_texts = []
    for c in by_country["Country"].tolist():
        latest10 = latest_regs_by_country(df_f, c, n=10)
        hover_texts.append(build_hover_list(latest10))
    by_country["Latest_10"] = hover_texts

    by_country["HasData"] = 1

    fig = px.choropleth(
        by_country,
        locations="Country",
        locationmode="country names",
        color="HasData",
        color_continuous_scale=["#2563eb", "#2563eb"],
        range_color=(0, 1),
        custom_data=["Regulation_Count", "Latest_10"],
    )

    fig.update_traces(
        hovertemplate="<b>%{location}</b><br><br>%{customdata[1]}<extra></extra>",
        showscale=False,
    )
    
    fig.update_layout(
        title_text="",
        hoverlabel=dict(align="left"),
        dragmode=False,
        hovermode="closest",
        coloraxis_showscale=False,
    )
    
    fig.update_geos(
        scope="asia",
        projection_type="mercator",
        # Lock ASEAN viewport
        lonaxis=dict(range=[92, 141]),
        lataxis=dict(range=[-11, 24]),
        visible=True,
        showcoastlines=True,
        coastlinecolor="rgba(255,255,255,0.35)",
        showcountries=True,
        countrycolor="rgba(255,255,255,0.85)",
        showland=True,
        landcolor="rgba(20, 30, 45, 1)",
        showocean=True,
        oceancolor="rgba(10, 16, 26, 1)",
        showlakes=True,
        lakecolor="rgba(10, 16, 26, 1)",
        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        autosize=True,
        width=None,
        height=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.layout.width = None
    fig.layout.height = None

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="asean_main_map",
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "responsive": True,
        },
    )

    st.caption(
        "Hover a country to preview its 10 most recent regulations "
        "(based on the current filters)."
    )

    # Country selector to open the details panel
    map_country = st.selectbox(
        "Select a country to view details",
        options=["(Select)"] + sorted(c for c in by_country["Country"].tolist() if pd.notna(c) and str(c) != ""),
    )
    if map_country != "(Select)":
        st.session_state["selected_country"] = map_country

    if (
        "selected_country" in st.session_state
        and st.session_state["selected_country"]
    ):
        show_country_modal(st.session_state["selected_country"], key_suffix="map")
        st.session_state["selected_country"] = None


# =========================
# TABLE TAB
# =========================
def _clear_editor_table_keys() -> None:
    """Clear persisted editor row-selection keys so the dialog doesn't reopen."""
    for _k in list(st.session_state.keys()):
        if _k.startswith("editor_table_"):
            del st.session_state[_k]

with tab_table:
    all_sheet_names = sorted(df_all["Category"].dropna().unique().tolist())

    # =========================================================
    # MODE A: Category = All -> summary matrix
    # =========================================================
    if sel_category == "All":
        regs_by_country = (
            df_f.groupby("Country_std")["Regulator_std"]
            .apply(
                lambda x: "\n".join(sorted({str(v) for v in x if pd.notna(v)}))
            )
            .reset_index()
            .rename(columns={"Country_std": "Country", "Regulator_std": "Regulator"})
        )

        counts = (
            df_f.groupby(["Country_std", "Category"])
            .size()
            .reset_index(name="Count")
            .assign(HasReg=True)
            .pivot(index="Country_std", columns="Category", values="HasReg")
            .fillna(False)
            .astype(bool)
            .reset_index()
            .rename(columns={"Country_std": "Country"})
        )

        t = regs_by_country.merge(counts, on="Country", how="outer").fillna({"Regulator": ""})
        for s in all_sheet_names:
            if s not in t.columns:
                t[s] = False

        t["Country"] = t["Country"].fillna("").astype(str)
        t.insert(0, "Flag", t["Country"].map(country_flag))
        t = t[["Flag", "Country"] + all_sheet_names].sort_values("Country")

        CHECK, BLANK = "✓", ""
        for s in all_sheet_names:
            t[s] = t[s].map(lambda x: CHECK if x else BLANK)

        country_options = sorted(c for c in t["Country"].tolist() if c)
        a_sel = st.multiselect(
            "Select countries to compare",
            options=country_options,
            placeholder="Choose one or more countries…",
            key="table_a_country_sel",
            on_change=_clear_editor_table_keys,
        )
        _render_data_table(t, highlighted=set(a_sel), row_height=36)
        if a_sel:
            if len(a_sel) == 1:
                show_country_modal(a_sel[0], key_suffix="table_0")
            else:
                show_country_comparison(a_sel, key_suffix="table_cmp")

    # =========================================================
    # MODE B: Category selected -> country picker then provisions
    # =========================================================
    else:
        d = df_f.copy()

        known_meta_b = set(
            META_COL_CANDIDATES["country"]
            + META_COL_CANDIDATES["regulator"]
            + META_COL_CANDIDATES["year"]
            + META_COL_CANDIDATES["source"]
            + META_COL_CANDIDATES["title"]
            + [
                "Category", "Country_std", "Regulator_std", "Year_raw", "Year",
                "Year_sort", "Regulation_Title", "Source_URL", "Source_EN",
                "Amendment_Of", "Entry_ID", "HasData",
            ]
        )
        detail_cols_b = [c for c in d.columns if c not in known_meta_b]

        country_options = sorted(
            c for c in d["Country_std"].dropna().unique() if str(c).strip()
        )
        b_sel = st.multiselect(
            "Select a country to view",
            options=country_options,
            placeholder="Choose one or more countries…",
            key="table_b_country_sel",
            on_change=_clear_editor_table_keys,
        )

        if not b_sel:
            st.caption("Select a country above to view its key provisions.")
        else:
            # Header line — flags + names
            header = "  ·  ".join(f"{country_flag(c)} {c}" for c in b_sel)
            st.markdown(f"## {header}")

            # Build one row per provision field, one column per country
            col_headers = ["Field"] + [f"{country_flag(c)} {c}" for c in b_sel]
            rows = []
            for col in detail_cols_b:
                if col not in d.columns:
                    continue
                row: dict = {"Field": col}
                has_any = False
                for country in b_sel:
                    dc = d[d["Country_std"] == country]
                    vals = (
                        dc[col].dropna().astype(str).str.strip()
                        .pipe(lambda s: s[~s.str.lower().isin({"nan", "none", ""})])
                        .unique().tolist()
                    )
                    cell = "\n".join(sorted(vals))
                    row[f"{country_flag(country)} {country}"] = cell
                    if cell:
                        has_any = True
                if has_any:
                    rows.append(row)

            if rows:
                _render_provision_table(rows, col_headers)
            else:
                st.caption("No provisions data available for the selected countries and category.")



# =========================
# GAP ANALYSIS TAB  (IS_AUTHENTICATED only)
# =========================
if IS_AUTHENTICATED:
    assert tab_gap is not None
    with tab_gap:
        _gap_cache   = get_gap_cache_key(resolved_data_dir)
        _df_bench    = load_gap_benchmarks(str(resolved_data_dir), _gap_cache)
        _df_maps     = load_gap_mappings(str(resolved_data_dir), _gap_cache)
        _gap_cats    = sorted(df_all["Category"].dropna().unique().tolist())

        # Gap country: admin can select any; country user is locked to their URL param
        _gap_country: Optional[str] = None
        if IS_ADMIN:
            _gap_country = "Admin"  # Admin sees all; per-country selection handled inside Mappings tab
        else:
            _gap_country = _auth_country

        _gsub_overview, _gsub_bench, _gsub_maps = st.tabs(
            ["Overview", "Benchmarks", "Mappings"]
        )

        # ── Overview ──────────────────────────────────────────────
        with _gsub_overview:
            st.caption(
                "✅ Meets · ⚠️ Partially meets · — Not assessed. "
                "Select a benchmark below to see each country's linked regulations."
            )
            _render_gap_matrix(_df_bench, _df_maps, ASEAN_COUNTRIES_ORDERED)

            if not _df_bench.empty:
                st.divider()
                _bench_labels = (
                    _df_bench["Benchmark_ID"] + " — " + _df_bench["Topic"]
                ).tolist()
                _sel_bench_label = st.selectbox(
                    "Select benchmark to view details",
                    options=["(Select)"] + _bench_labels,
                    key="gap_overview_bench_sel",
                )
                if _sel_bench_label != "(Select)":
                    _sel_bid = _sel_bench_label.split(" — ")[0]
                    _brow    = _df_bench[_df_bench["Benchmark_ID"] == _sel_bid].iloc[0]

                    st.markdown(f"### {_sel_bid} — {_brow.get('Topic', '')}")
                    st.caption(f"**Standard:** {_brow.get('Standard', '')}  |  **Category:** {_brow.get('Category', '')}")
                    with st.expander("Full provision text"):
                        st.markdown(str(_brow.get("Provision", "")))

                    # Detail rows: one per country
                    _bench_maps = _df_maps[_df_maps["Benchmark_ID"] == _sel_bid]
                    detail_rows: list[dict] = []
                    for _c in ASEAN_COUNTRIES_ORDERED:
                        _cm = _bench_maps[_bench_maps["Country"] == _c]
                        if _cm.empty:
                            detail_rows.append({
                                "Country": f"{country_flag(_c)} {_c}",
                                "Status":  "—",
                                "Regulations": "",
                                "Notes": "",
                                "Last updated": "",
                            })
                            continue
                        _m = _cm.iloc[0]
                        _status  = str(_m.get("Status", "Not assessed"))
                        _eids    = [e.strip() for e in str(_m.get("Entry_IDs", "")).split(",") if e.strip()]
                        _reg_regs = df_all[df_all["Entry_ID"].isin(_eids)]
                        reg_lines = []
                        for _, _rr in _reg_regs.iterrows():
                            _t   = str(_rr.get("Regulation_Title", "Regulation"))
                            _url = str(_rr.get("Source_URL", ""))
                            _yr  = str(_rr.get("Year", ""))
                            _yr_s = f" ({int(float(_yr))})" if _yr and _yr not in ("nan", "") else ""
                            reg_lines.append(
                                f"[{_t}{_yr_s}]({_url})" if _url and _url != "nan" else f"{_t}{_yr_s}"
                            )
                        _notes   = str(_m.get("Notes", ""))
                        _updated = str(_m.get("Updated_At", ""))[:10]
                        _by      = str(_m.get("Updated_By", ""))
                        detail_rows.append({
                            "Country":      f"{country_flag(_c)} {_c}",
                            "Status":       GAP_STATUS_EMOJI.get(_status, "—") + " " + _status,
                            "Regulations":  "\n".join(reg_lines),
                            "Notes":        _notes if _notes != "nan" else "",
                            "Last updated": f"{_updated} by {_by}" if _updated and _updated != "nan" else "",
                        })
                    _render_provision_table(
                        detail_rows,
                        ["Country", "Status", "Regulations", "Notes", "Last updated"],
                    )

        # ── Benchmarks ────────────────────────────────────────────
        with _gsub_bench:
            if IS_ADMIN:
                _bcol1, _bcol2 = st.columns([4, 1])
                _bcol1.subheader("Benchmark definitions")
                if _bcol2.button("＋ Add benchmark", use_container_width=True, type="primary"):
                    _dlg_add_benchmark(_df_bench, _gap_cats, resolved_data_dir, _auth_user or "Admin")

            if _df_bench.empty:
                st.info("No benchmarks defined yet.")
            else:
                _display_bench = _df_bench[
                    ["Benchmark_ID", "Standard", "Category", "Topic", "Provision",
                     "Updated_By", "Updated_At"]
                ].copy()
                _bench_event = st.dataframe(
                    _display_bench,
                    use_container_width=True,
                    hide_index=True,
                    height=min(500, max(120, 40 + len(_df_bench) * 36)),
                    key="gap_bench_table",
                    on_select="rerun" if IS_ADMIN else "ignore",
                    selection_mode="single-row",
                    column_config={
                        "Benchmark_ID": st.column_config.TextColumn("ID", width="small"),
                        "Standard":     st.column_config.TextColumn("Standard", width="medium"),
                        "Category":     st.column_config.TextColumn("Category", width="small"),
                        "Topic":        st.column_config.TextColumn("Topic / Area", width="large"),
                        "Provision":    st.column_config.TextColumn("Provision", width="large"),
                        "Updated_By":   st.column_config.TextColumn("Updated by", width="small"),
                        "Updated_At":   st.column_config.TextColumn("Updated at", width="small"),
                    },
                )
                if IS_ADMIN:
                    _bsel_idx = (
                        _bench_event.selection.rows
                        if hasattr(_bench_event, "selection") and _bench_event.selection
                        else []
                    )
                    if _bsel_idx:
                        _bsel_row = _df_bench.iloc[_bsel_idx[0]]
                        _bid_sel  = str(_bsel_row["Benchmark_ID"])
                        _bet_col1, _bet_col2 = st.columns(2)
                        if _bet_col1.button("✏️ Edit", key="gap_bench_edit_btn", use_container_width=True):
                            _dlg_edit_benchmark(
                                _bsel_row, _df_bench, _gap_cats, resolved_data_dir, _auth_user or "Admin"
                            )
                        # Delete — only if no mappings reference it
                        _ref_count = len(_df_maps[_df_maps["Benchmark_ID"] == _bid_sel])
                        if _ref_count > 0:
                            _bet_col2.button(
                                f"🗑 Delete (blocked — {_ref_count} mapping(s))",
                                disabled=True,
                                use_container_width=True,
                                help="Remove all country mappings for this benchmark before deleting.",
                            )
                        else:
                            if _bet_col2.button("🗑 Delete", use_container_width=True, type="secondary",
                                                key="gap_bench_del_btn"):
                                _before_del = _bsel_row.to_dict()
                                _updated_bench = _df_bench[_df_bench["Benchmark_ID"] != _bid_sel].reset_index(drop=True)
                                save_gap_benchmarks(_updated_bench, resolved_data_dir)
                                append_gap_audit(
                                    resolved_data_dir, "delete", "benchmark",
                                    _bid_sel, _before_del, {}, _auth_user or "Admin"
                                )
                                st.rerun()

        # ── Mappings ──────────────────────────────────────────────
        with _gsub_maps:
            # Admin picks any country; country user is locked to their own
            if IS_ADMIN:
                _map_country = st.selectbox(
                    "Country", options=ASEAN_COUNTRIES_ORDERED, key="gap_map_country_sel"
                )
            else:
                _map_country = _auth_country or ""
                st.caption(f"Mapping regulations for: {country_flag(_map_country)} **{_map_country}**")

            if _df_bench.empty:
                st.info("No benchmarks available. Ask an admin to add benchmarks first.")
            elif _map_country:
                _mbench_opts = (
                    _df_bench["Benchmark_ID"] + " — " + _df_bench["Topic"]
                ).tolist()
                _sel_mbl = st.selectbox(
                    "Select benchmark to map",
                    options=["(Select)"] + _mbench_opts,
                    key="gap_map_bench_sel",
                )
                if _sel_mbl != "(Select)":
                    _sel_mbid  = _sel_mbl.split(" — ")[0]
                    _mbrow     = _df_bench[_df_bench["Benchmark_ID"] == _sel_mbid].iloc[0]
                    _mcat      = str(_mbrow.get("Category", ""))

                    with st.expander("Benchmark provision", expanded=False):
                        st.markdown(str(_mbrow.get("Provision", "")))

                    # Current mapping
                    _exist_map = _df_maps[
                        (_df_maps["Benchmark_ID"] == _sel_mbid) &
                        (_df_maps["Country"] == _map_country)
                    ]
                    if _exist_map.empty:
                        _cur_status  = "Not assessed"
                        _cur_eids: list[str] = []
                        _cur_notes   = ""
                        _mapping_id  = str(uuid.uuid4())
                        _is_new_map  = True
                    else:
                        _em = _exist_map.iloc[0]
                        _cur_status  = str(_em.get("Status", "Not assessed"))
                        _raw_ids     = str(_em.get("Entry_IDs", ""))
                        _cur_eids    = [e.strip() for e in _raw_ids.split(",") if e.strip()]
                        _cur_notes   = str(_em.get("Notes", ""))
                        _mapping_id  = str(_em.get("Mapping_ID", uuid.uuid4()))
                        _is_new_map  = False
                        st.caption(
                            f"Last updated: {str(_em.get('Updated_At',''))[:10]} "
                            f"by {_em.get('Updated_By','')}"
                        )

                    # Regulation options for this country × category
                    _reg_opts = _gap_reg_options(df_all, _map_country, _mcat)
                    _lbl_to_id = {v: k for k, v in _reg_opts.items()}
                    _default_lbls = [_reg_opts[e] for e in _cur_eids if e in _reg_opts]

                    with st.form(key=f"gap_map_form_{_sel_mbid}_{_map_country}"):
                        _new_status = st.selectbox(
                            "Compliance status",
                            options=GAP_STATUS_OPTIONS,
                            index=GAP_STATUS_OPTIONS.index(_cur_status)
                                  if _cur_status in GAP_STATUS_OPTIONS else 2,
                        )
                        if _reg_opts:
                            _sel_lbls = st.multiselect(
                                "Regulations that address this benchmark",
                                options=list(_reg_opts.values()),
                                default=_default_lbls,
                            )
                        else:
                            st.info(
                                f"No regulations found for {_map_country}"
                                + (f" under category '{_mcat}'" if _mcat else "") + "."
                                + " Add regulations in the Editor tab first."
                            )
                            _sel_lbls = []
                        _new_notes = st.text_area(
                            "Notes (optional)",
                            value=_cur_notes if _cur_notes not in ("", "nan") else "",
                        )
                        _submitted = st.form_submit_button("Save mapping", type="primary")

                    if _submitted:
                        _sel_eids = [_lbl_to_id[l] for l in _sel_lbls if l in _lbl_to_id]
                        _now      = pd.Timestamp.utcnow().isoformat()
                        _new_mrow = {
                            "Mapping_ID":   _mapping_id,
                            "Benchmark_ID": _sel_mbid,
                            "Country":      _map_country,
                            "Status":       _new_status,
                            "Entry_IDs":    ",".join(_sel_eids),
                            "Notes":        _new_notes.strip(),
                            "Updated_By":   _auth_user or "unknown",
                            "Updated_At":   _now,
                        }
                        _before_m = _exist_map.iloc[0].to_dict() if not _is_new_map else {}
                        if _is_new_map:
                            _upd_maps = pd.concat(
                                [_df_maps, pd.DataFrame([_new_mrow])], ignore_index=True
                            )
                        else:
                            _upd_maps = _df_maps.copy()
                            _midx = _upd_maps.index[
                                (_upd_maps["Benchmark_ID"] == _sel_mbid) &
                                (_upd_maps["Country"] == _map_country)
                            ].tolist()
                            for _mi in _midx:
                                for _mk, _mv in _new_mrow.items():
                                    _upd_maps.at[_mi, _mk] = _mv
                        save_gap_mappings(_upd_maps, resolved_data_dir)
                        append_gap_audit(
                            resolved_data_dir,
                            "add" if _is_new_map else "edit",
                            "mapping",
                            f"{_sel_mbid}:{_map_country}",
                            _before_m, _new_mrow,
                            _auth_user or "unknown",
                        )
                        st.success("Mapping saved.")
                        st.rerun()


# =========================
# EDITOR TAB  (only rendered when IS_AUTHENTICATED)
# =========================
if IS_AUTHENTICATED:
    assert tab_editor is not None  # assigned above in the IS_AUTHENTICATED branch
    with tab_editor:
        st.subheader("Country editor & audit log")

        editor_countries = sorted(df_all["Country_std"].dropna().unique().tolist())

        # ── Determine editor_country and display session banner ──────────
        if IS_ADMIN:
            # Admin can switch between any country via a dropdown
            st.success(f"Signed in as **{_auth_user}** · admin access to all countries")
            editor_country: Optional[str] = st.selectbox(
                "Select country to edit",
                options=["(Choose a country)"] + editor_countries,
                index=0,
                key="admin_country_select",
            )
            if editor_country == "(Choose a country)":
                editor_country = None
        else:
            # Country user — locked to their own country
            if _auth_country not in editor_countries:
                st.error(
                    f"Country **{_auth_country}** was not found in the dataset. "
                    "Please contact the administrator."
                )
                st.stop()
            editor_country = _auth_country
            st.success(
                f"Signed in as **{_auth_user}** · editing "
                f"{country_flag(editor_country or '')} **{editor_country}**"
            )

        if not editor_country:
            st.info("Choose a country above to view, edit, or add regulations.")
        else:
            country_rows = df_all[df_all["Country_std"] == editor_country].copy()
            country_rows = country_rows.sort_values(
                ["Year", "Regulation_Title"], ascending=[False, True]
            )

            category_choices = sorted(df_all["Category"].dropna().unique().tolist())

            regulator_name = (
                country_rows["Regulator_std"].dropna().astype(str).str.strip()
                .loc[lambda s: s != ""].iloc[0]
                if not country_rows["Regulator_std"].dropna().empty
                else "current records"
            )
            st.markdown(
                f"#### {country_flag(editor_country or '')} **{editor_country}** — {regulator_name}"
            )

            preview_cols = [
                "Year", "Regulation_Title", "Source_URL",
            ]
            preview_cols = [c for c in preview_cols if c in country_rows.columns]
            if not country_rows.empty:
                st.caption("Click a row to edit or archive that record.")
                selected_entry_id = None
                for cat in sorted(country_rows["Category"].dropna().unique().tolist()):
                    cat_rows = country_rows[country_rows["Category"] == cat].copy()
                    cat_preview = cat_rows[preview_cols].rename(columns={
                        "Regulation_Title": "Title",
                        "Source_URL": "Source URL",
                    }).copy()
                    if "Source URL" in cat_preview.columns:
                        cat_preview["Source URL"] = cat_preview["Source URL"].apply(
                            lambda u: safe_linkify(u) or None
                        )
                    st.markdown(f"**{cat}**")
                    cat_key = re.sub(r"[^a-zA-Z0-9]", "_", cat)
                    cat_event = st.dataframe(
                        cat_preview,
                        use_container_width=True,
                        hide_index=True,
                        height=min(300, max(80, 40 + len(cat_rows) * 35)),
                        key=f"editor_table_{cat_key}",
                        on_select="rerun",
                        selection_mode="single-row",
                        column_config={
                            "Source URL": st.column_config.LinkColumn(
                                "Source URL",
                                display_text="Open",
                            ),
                        },
                    )
                    indices = get_selected_row_indices(cat_event)
                    if indices and selected_entry_id is None:
                        selected_entry_id = cat_rows.iloc[indices[0]]["Entry_ID"]

                if selected_entry_id:
                    existing_row = country_rows[
                        country_rows["Entry_ID"] == selected_entry_id
                    ].iloc[0]
                    edit_regulation_dialog(
                        existing_row=existing_row,
                        category_choices=category_choices,
                        df_all=df_all,
                        src_dir=resolved_data_dir,
                        editor_country=editor_country,
                        auth_user=str(_auth_user),
                    )

            else:
                st.info(
                    "No records exist yet for this country. "
                    "Use the form below to add a new regulation."
                )

            st.markdown("---")
            st.markdown("### Add a new regulation")
            existing_regulators = sorted(
                set(country_rows["Regulator_std"].dropna().astype(str).str.strip().tolist())
            )
            prefilled_regulator = "; ".join(existing_regulators)

            new_category = st.selectbox(
                "Category", options=category_choices, key="add_form_category"
            )
            add_provision_cols = get_provision_cols(df_all, new_category)

            # Amendment dropdown outside the form so it rerenders when category changes
            _add_amend_mask = (df_all["Category"] == new_category) & df_all["Regulation_Title"].notna()
            if editor_country and editor_country != "NA":
                _add_amend_mask &= df_all["Country_std"] == editor_country
            _add_amend_cand = df_all[_add_amend_mask].sort_values(
                ["Year", "Regulation_Title"], ascending=[False, True]
            )
            _ADD_BLANK = "— (not an amendment)"
            add_amend_opts = [_ADD_BLANK] + [
                f"{r['Regulation_Title']} ({int(r['Year']) if pd.notna(r['Year']) else '?'})"
                for _, r in _add_amend_cand.iterrows()
            ]
            new_amendment_of_sel = st.selectbox(
                "Amends (leave blank if original)",
                options=add_amend_opts,
                key="add_form_amend",
            )
            new_amendment_of = "" if new_amendment_of_sel == _ADD_BLANK else new_amendment_of_sel

            with st.form("add_regulation_form"):
                new_regulator = st.text_input(
                    "Regulator", value=prefilled_regulator, disabled=True
                )
                new_year_raw = st.text_input("Year", value="")
                new_title = st.text_area("Regulation title", value="", height=120)
                new_source_url = st.text_input("Source URL", value="")
                new_source_en  = st.text_input("Source URL (English)", value="")
                add_provision_values: dict[str, str] = {}
                if add_provision_cols:
                    st.markdown("**Key provisions**")
                    for col in add_provision_cols:
                        add_provision_values[col] = st.text_area(
                            col, value="", height=68, key=f"add_prov_{col}"
                        )
                add_submitted = st.form_submit_button("Add regulation")

            if add_submitted:
                if not new_title.strip():
                    st.warning("A regulation title is required to add a new record.")
                else:
                    new_row: dict[str, object] = {col: pd.NA for col in df_all.columns}
                    new_row["Entry_ID"] = str(uuid.uuid4())
                    new_row["Category"] = new_category
                    new_row["Country_std"] = editor_country
                    new_row["Regulator_std"] = new_regulator.strip() or pd.NA
                    new_row["Year_raw"] = new_year_raw.strip() or pd.NA
                    new_row["Year"] = extract_year(new_year_raw)
                    new_row["Regulation_Title"] = new_title.strip()
                    new_row["Source_URL"] = new_source_url.strip() or pd.NA
                    new_row["Source_EN"]  = new_source_en.strip() or pd.NA
                    new_row["Amendment_Of"] = new_amendment_of.strip() or pd.NA
                    for col, val in add_provision_values.items():
                        new_row[col] = val.strip() or pd.NA

                    updated_df = pd.concat(
                        [df_all, pd.DataFrame([new_row])], ignore_index=True, sort=False
                    )
                    save_cbregs(updated_df, resolved_data_dir)
                    append_audit_log(
                        action="add",
                        country=editor_country,
                        entry_id=str(new_row["Entry_ID"]),
                        user=str(_auth_user),
                        old_record=None,
                        new_record=serialize_record_for_archive(pd.Series(new_row)),
                    )
                    st.success("New regulation added and archived successfully.")
                    safe_rerun()

            st.markdown("---")
            st.markdown("### Recent audit history")
            if ARCHIVE_FILE.exists():
                audit_df = pd.read_csv(ARCHIVE_FILE)
                audit_df["timestamp"] = pd.to_datetime(
                    audit_df["timestamp"], errors="coerce"
                )
                # Admin sees all; country user sees only their own entries
                if not IS_ADMIN:
                    audit_df = audit_df[audit_df["country"] == editor_country]
                st.dataframe(
                    audit_df.sort_values("timestamp", ascending=False).head(20),
                    use_container_width=True,
                    height=380,
                )
            else:
                st.info(
                    "No audit history exists yet. "
                    "Add or edit a record to create the archive log."
                )


# =========================
# GUIDE TAB
# =========================
with tab_guide:
    st.subheader("How-to Guide")
    st.caption("Expand a section below to learn how to use the dashboard.")

    with st.expander("Viewing & navigating the dashboard", expanded=False):
        st.markdown("""
### Overview
The dashboard is split into three main tabs — **Map**, **Table**, and **Editor** — and a
**sidebar** that controls which data is shown across all views.

---

### Sidebar filters
- **Category** — Select a regulatory category (e.g. *Consumer Protection*, *Fraud Risk*)
  from the dropdown to narrow the entire dashboard to that topic. Choose **All** to see
  every category at once.
- **Select all countries** — Tick this checkbox to include every ASEAN country. Untick it
  to enable the multi-select list below.
- **Country** — Pick one or more countries manually. The map, table, and country detail
  panels all update instantly.
- **Download filtered data** — Exports the currently filtered dataset as a CSV file.

---

### Map tab
1. The choropleth map highlights every country that has at least one regulation in the
   current filter set.
2. **Hover** over any highlighted country to see a tooltip showing the number of
   regulations and a preview of the 10 most recent ones.
3. Use the **"Open a country details popup"** selector below the map to open a full
   country detail panel, which includes:
   - A zoomed country map
   - The responsible regulator(s)
   - A full list of regulations with clickable source links
   - A **Key provisions** table broken down by category

---

### Table tab
**When Category = All:**
- A summary matrix is shown with one row per country.
- Each category column displays a **✓** if that country has at least one regulation in
  that category.
- Click one or more rows to open a full country detail panel for each selected country.

**When a specific Category is selected:**
- The table shows the actual worksheet fields for that category, grouped by country.
- Click one or more rows to open a country detail panel for each selection.
        """)

    with st.expander("Managing records with the Editor", expanded=False):
        st.markdown("""
### Overview
The **Editor** tab lets authorised country representatives add, update, or delete
regulation records. Every change is automatically written to an audit log.

---

### Selecting a country account
- If the application is accessed via a URL that includes `?country=Vietnam&user=Alice`,
  the editor will automatically authenticate to that country and display the user name.
- Otherwise, use the **Country account** dropdown to choose the country you are editing
  on behalf of.

---

### Viewing existing records
Once a country is selected, a table at the top of the tab lists all current regulations
for that country, showing: Entry ID, Category, Regulator, Year, Title, and Source URL.

---

### Adding a new regulation
1. Scroll to **"Add a new country-specific regulation"**.
2. Fill in the form fields:
   - **Category** — The regulatory topic this regulation belongs to.
   - **Regulator** — The government body or authority responsible.
   - **Year** — The year the regulation was approved or implemented.
   - **Regulation title** — The full or short name of the legal instrument.
   - **Source URL** — A direct link to the official source document.
3. Click **Add regulation**. The record is saved to the data file and logged in the
   audit history immediately.

---

### Editing an existing regulation
1. Scroll to **"Edit or archive an existing regulation"**.
2. Use the **"Select a record to edit"** dropdown — entries are listed by their Entry ID
   and a short excerpt of the title.
3. Update any fields in the form that appears, then click **Save updates**.
   The old version of the record is preserved in the audit log.

---

### Archiving (deleting) a regulation
1. Select the record using the dropdown as described above.
2. Click **Archive and delete this record**.
   The record is removed from the live dataset but its full details are stored in the
   audit log so the deletion can always be reviewed.

---

### Audit history
The bottom of the Editor tab displays the **20 most recent audit events**, showing the
timestamp, action type (add / edit / delete), country, user, and the before/after record
data in JSON format.
        """)





