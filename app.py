import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, cast
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# App config
# =========================
st.set_page_config(
    page_title="ASEAN Regulatory Dashboard",
    layout="wide",
    initial_sidebar_state="auto",
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
st.caption("v3 • 2026-06-15")

DATA_FILE = Path("src") / "CBregs.xlsx"
ARCHIVE_FILE = Path("src") / "CBregs_audit_log.csv"


def resolve_data_path(path: Path) -> Path:
    candidates = [
        path,
        Path(__file__).resolve().parent / path,
        Path(__file__).resolve().parent / "src" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Data file not found. Tried: {', '.join(str(p) for p in candidates)}"
    )


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
    "country": ["country", "Country"],
    "regulator": ["regulator", "Regulator"],
    "year": [
        "year",
        "Year",
        "Year approved/implemented",
        "Year Approved/Implemented",
        "Year approved / implemented",
    ],
    "source": ["source", "Official Source", "Official source", "Official Source links", "Official source links", "Source", "URL", "Link"],
    "title": [
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def extract_year(value) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip()
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
        source_html = (
            f'<a href="{html_escape(source_url)}" target="_blank" rel="noreferrer">Source</a>'
            if source_url else "—"
        )
        rows.append(
            "<tr>"
            f"<td style=\"padding:8px 12px 8px 0; vertical-align:top; white-space:normal; word-break:break-word;\">{title}</td>"
            f"<td style=\"padding:8px 12px; vertical-align:top; white-space:nowrap; text-align:right;\">{year_text}</td>"
            f"<td style=\"padding:8px 0 8px 12px; vertical-align:top; white-space:nowrap; text-align:center;\">{source_html}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def get_selected_row_index(event):
    if not event:
        return None
    selection = event.get("selection")
    if not selection:
        return None
    rows = selection.get("rows")
    return rows[0] if rows else None


@st.cache_data
def load_cbregs(file_path: Path, file_mtime: float) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        p = resolve_data_path(p)

    xls = pd.ExcelFile(p, engine="openpyxl")
    frames = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(p, sheet_name=sheet, engine="openpyxl")
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

        frames.append(out)

    all_df = pd.concat(frames, ignore_index=True)

    # Clean
    all_df["Country_std"] = all_df["Country_std"].astype(str).str.strip()
    all_df["Regulator_std"] = all_df["Regulator_std"].astype(str).str.strip()
    all_df["Regulation_Title"] = all_df["Regulation_Title"].astype(str).str.strip()

    # Treat "nan" strings produced by astype(str)
    for c in ["Country_std", "Regulator_std", "Regulation_Title", "Source_URL"]:
        all_df.loc[all_df[c].str.lower().isin(["nan", "none"]), c] = pd.NA

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


def get_query_params() -> dict[str, list[str]]:
    getter = getattr(st, "experimental_get_query_params", None)
    if callable(getter):
        return cast(dict[str, list[str]], getter())
    return {}


def safe_rerun() -> None:
    rerun_func = getattr(st, "experimental_rerun", None)
    if callable(rerun_func):
        rerun_func()


def save_cbregs(df: pd.DataFrame, file_path: Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False, sheet_name="All Regulations")


# =========================
# Load data
# =========================
resolved_data_path = resolve_data_path(DATA_FILE)

df_all = load_cbregs(resolved_data_path, resolved_data_path.stat().st_mtime)

if df_all.empty:
    st.error("Data loaded but produced no rows.")
    st.stop()

# =========================
# Sidebar filters (ORDER: Category -> Country)
# =========================
st.sidebar.header("Filters")
st.sidebar.write("Refine the dashboard by category and country.")

categories = ["All"] + sorted(df_all["Category"].dropna().unique().tolist())
sel_category = st.sidebar.selectbox("Category", options=categories, index=0)

# Apply category filter first so subsequent widgets use the right dataset
df_f = df_all.copy()
if sel_category != "All":
    df_f = df_f[df_f["Category"] == sel_category]

# Country list
countries = sorted(df_f["Country_std"].dropna().unique().tolist())

# ---- init defaults (must happen BEFORE widgets with these keys are created)
st.session_state.setdefault("country_select_all", True)
st.session_state.setdefault("selected_countries", countries.copy())

# Keep selected list valid when category changes
st.session_state["selected_countries"] = [
    c for c in st.session_state["selected_countries"] if c in countries
]
if not st.session_state["selected_countries"]:
    st.session_state["selected_countries"] = countries.copy()


def _toggle_select_all():
    if st.session_state["country_select_all"]:
        st.session_state["selected_countries"] = countries.copy()


def _countries_changed():
    st.session_state["country_select_all"] = (
        set(st.session_state["selected_countries"]) == set(countries)
    )

# ---- widgets
st.sidebar.checkbox(
    "Select all countries",
    key="country_select_all",
    on_change=_toggle_select_all,
)

st.sidebar.multiselect(
    "Country",
    options=countries,
    key="selected_countries",
    on_change=_countries_changed,
)

if countries:
    csv_bytes = df_f.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "Download filtered data",
        csv_bytes,
        "asean_regulations_filtered.csv",
        "text/csv",
        use_container_width=True,
    )
else:
    st.sidebar.info("No countries available for the current filter selection.")

# ---- apply filter
if st.session_state["selected_countries"]:
    df_f = df_f[df_f["Country_std"].isin(st.session_state["selected_countries"])]

st.divider()

# =========================
# Country popup (modal)
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
    country_fig.update_traces(showscale=False)
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
    })
    st.download_button(
        label=f"Download all data for {country}",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{country.lower().replace(' ', '_')}_regulations.csv",
        mime="text/csv",
        use_container_width=True,
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
        source_text = f" ([Source]({source_link}))" if source_link else ""
        regulation_lines.append(f"- **{year_text}** — {title}{source_text}")
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
            "ID",
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
        detail_df = pd.DataFrame(rows)
        detail_height = min(600, max(200, 60 + len(detail_df) * 32))
        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
            height=detail_height,
        )
        shown_any_category = True

    if not shown_any_category:
        st.caption(
            "No key provisions available for this country "
            "under the current filters."
        )

    st.caption(
        "Links shown as 'Source' are taken directly from the "
        "'Official Source' column in CBregs.xlsx."
    )


# =========================
# Tabs (Map default)
# =========================
tab_map, tab_table, tab_editor, tab_guide = st.tabs(["Map", "Table", "Editor", "Guide"])

# =========================
# MAP TAB
# =========================
with tab_map:
    # Country counts + hover preview
    by_country = (
        df_f.groupby("Country_std", dropna=False)
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
        hovertemplate="<b>%{location}</b><br>Regulations: %{customdata[0]}<br><br>%{customdata[1]}<extra></extra>",
        showscale=False,
    )
    
    fig.update_layout(
        title_text="ASEAN regulations by country",
        title_x=0,
        title_xanchor="left",
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
        "Open a country details popup",
        options=["(Select)"] + sorted(by_country["Country"].tolist()),
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
with tab_table:
    all_sheet_names = sorted(df_all["Category"].dropna().unique().tolist())

    # =========================================================
    # MODE A: Category = All -> summary matrix
    # =========================================================
    if sel_category == "All":
        regs_by_country = (
            df_f.groupby("Country_std")["Regulator_std"]
            .apply(
                lambda x: ", ".join(sorted(set(x.dropna().tolist())))
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

        t.insert(0, "Flag", t["Country"].map(lambda x: country_flag(str(x))))
        t = t[["Flag", "Country", "Regulator"] + all_sheet_names].sort_values("Country")

        CHECK, BLANK = "✓", ""
        for s in all_sheet_names:
            t[s] = t[s].map(lambda x: CHECK if x else BLANK)

        st.caption("Select a row to preview and open a country popup.")
        table_height = min(700, max(300, 40 + len(t) * 28))
        event = st.dataframe(
            t,
            use_container_width=True,
            hide_index=True,
            height=table_height,
            on_select="rerun",
            selection_mode="single-row",
        )

        idx = get_selected_row_index(event)
        if idx is not None:
            selected_country = t.iloc[idx]["Country"]
            st.session_state["selected_country"] = selected_country

    # =========================================================
    # MODE B: Category selected -> show actual worksheet columns
    # =========================================================
    else:
        st.caption(f"Showing worksheet fields for: {sel_category}")
        
        d = df_f.copy()
        
        regs_by_country = (
            d.groupby("Country_std")["Regulator_std"]
            .apply(
                lambda x: ", ".join(sorted(set(x.dropna().tolist())))
            )
            .reset_index()
            .rename(columns={"Country_std": "Country", "Regulator_std": "Regulator"})
        )

        exclude_columns = {
            "ID", "title", "year", "Year", "Year_raw", "source",
            "country", "regulator", "Category", "Country_std", "Regulator_std",
            "Source_URL", "Regulation_Title",
        }
        cols_to_concat = [c for c in d.columns if c not in exclude_columns]
        drop_for_grouping = [
            c for c in ["ID", "title", "year", "source"] if c in d.columns
        ]
        provs = (
            d
            .drop(columns=drop_for_grouping)
            .groupby(['Country_std', 'Category'], dropna=False)
            .agg({
                c: lambda x: (
                    pd.NA
                    if x.dropna().empty
                    else " | ".join(x.dropna().astype(str).str.strip().unique())
                )
                for c in cols_to_concat
            })
            .reset_index().rename(columns={"Country_std": "Country"})
        )

        t = regs_by_country.merge(provs, on="Country", how="outer").fillna({"Regulator": ""})

        t.insert(0, "Flag", t["Country"].map(lambda x: country_flag(str(x))))
        extra_cols = [
            col for col in t.columns
            if col not in {"Flag", "Country", "Regulator", "Category", "Regulation_Title"}
        ]
        t = (
            t[["Flag", "Country", "Regulator"] + extra_cols]
            .sort_values("Country")
            .dropna(axis=1, how="all")
        )

        st.caption("Select a row to preview and open a country popup.")
        table_height = min(700, max(300, 40 + len(t) * 28))
        event = st.dataframe(
            t,
            use_container_width=True,
            hide_index=True,
            height=table_height,
            on_select="rerun",
            selection_mode="single-row",
        )

        idx = get_selected_row_index(event)
        if idx is not None:
            selected_country = t.iloc[idx]["Country"]
            st.session_state["selected_country"] = selected_country

    if (
        "selected_country" in st.session_state
        and st.session_state["selected_country"]
    ):
        show_country_modal(st.session_state["selected_country"], key_suffix="table")
        st.session_state["selected_country"] = None


# =========================
# EDITOR TAB
# =========================
with tab_editor:
    st.subheader("Country editor & audit log")

    editor_countries = sorted(df_all["Country_std"].dropna().unique().tolist())
    query_params = get_query_params()
    requested_country = query_params.get("country", [None])[0] if query_params else None
    requested_user = query_params.get("user", ["anonymous"])[0] if query_params else "anonymous"

    if requested_country and requested_country not in editor_countries:
        st.warning(
            "The country from the sign-in context is not available in the "
            "dataset. Please select a valid country below."
        )
        requested_country = None

    if requested_country:
        st.success(f"Signed in as **{requested_user}** for **{requested_country}**")

    country_options = ["(Choose your country)"] if not requested_country else []
    country_choice = st.selectbox(
        "Country account",
        country_options + editor_countries,
        index=0,
        help=(
            "If your website passes a login context, use query parameters like "
            "?country=Vietnam&user=Alice. Otherwise, choose a country to edit "
            "locally."
        ),
    )
    editor_country = requested_country or (country_choice if country_choice != "(Choose your country)" else None)

    if not editor_country:
        st.info("Select a country account to view, edit, archive, or add country-specific regulations.")
    else:
        country_rows = df_all[df_all["Country_std"] == editor_country].copy()
        country_rows = country_rows.sort_values(["Year", "Regulation_Title"], ascending=[False, True])

        st.markdown(f"#### Editing records for {country_flag(editor_country)} **{editor_country}**")

        preview_cols = [
            "Entry_ID",
            "Category",
            "Regulator_std",
            "Year",
            "Regulation_Title",
            "Source_URL",
        ]
        preview_cols = [c for c in preview_cols if c in country_rows.columns]
        if not country_rows.empty:
            st.dataframe(
                country_rows[preview_cols].rename(columns={
                    "Regulator_std": "Regulator",
                    "Regulation_Title": "Title",
                    "Source_URL": "Source URL",
                }),
                use_container_width=True,
                height=min(400, max(220, 60 + len(country_rows) * 30)),
            )
        else:
            st.info("No records exist yet for this country. Use the form below to add a new regulation.")

        st.markdown("---")
        st.markdown("### Add a new country-specific regulation")
        category_choices = sorted(df_all["Category"].dropna().unique().tolist())
        with st.form("add_regulation_form"):
            new_category = st.selectbox(
                "Category",
                options=category_choices,
                index=0,
            )
            new_regulator = st.text_input("Regulator", value="")
            new_year_raw = st.text_input("Year", value="")
            new_title = st.text_area("Regulation title", value="", height=120)
            new_source_url = st.text_input("Source URL", value="")
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

                updated_df = pd.concat([df_all, pd.DataFrame([new_row])], ignore_index=True, sort=False)
                save_cbregs(updated_df, resolved_data_path)
                append_audit_log(
                    action="add",
                    country=editor_country,
                    entry_id=new_row["Entry_ID"],
                    user=requested_user,
                    old_record=None,
                    new_record=serialize_record_for_archive(pd.Series(new_row)),
                )
                st.success("New regulation added and archived successfully.")
                safe_rerun()

        st.markdown("---")
        st.markdown("### Edit or archive an existing regulation")

        entry_options = ["(Select an existing entry)"]
        for _, row in country_rows.iterrows():
            title_excerpt = str(row.get("Regulation_Title", "")).strip()[:80]
            entry_options.append(f"{row['Entry_ID']} | {title_excerpt}")

        selected_entry_label = st.selectbox("Select a record to edit", options=entry_options)
        selected_entry_id = None
        if selected_entry_label and selected_entry_label != "(Select an existing entry)":
            selected_entry_id = selected_entry_label.split(" | ", 1)[0]

        if selected_entry_id:
            existing_row = country_rows[country_rows["Entry_ID"] == selected_entry_id].iloc[0]
            existing_category = existing_row["Category"] if pd.notna(existing_row["Category"]) else category_choices[0]
            with st.form("edit_regulation_form"):
                edit_category = st.selectbox(
                    "Category",
                    options=category_choices,
                    index=category_choices.index(existing_category)
                    if existing_category in category_choices
                    else 0,
                )
                edit_regulator = st.text_input(
                    "Regulator",
                    value=(
                        ""
                        if pd.isna(existing_row.get("Regulator_std"))
                        else str(existing_row.get("Regulator_std"))
                    ),
                )
                edit_year_raw = st.text_input(
                    "Year",
                    value=(
                        ""
                        if pd.isna(existing_row.get("Year_raw"))
                        else str(existing_row.get("Year_raw"))
                    ),
                )
                edit_title = st.text_area(
                    "Regulation title",
                    value="" if pd.isna(existing_row.get("Regulation_Title")) else str(existing_row.get("Regulation_Title")),
                    height=120,
                )
                edit_source_url = st.text_input(
                    "Source URL",
                    value="" if pd.isna(existing_row.get("Source_URL")) else str(existing_row.get("Source_URL")),
                )
                save_submitted = st.form_submit_button("Save updates")

            if save_submitted:
                updated_df = df_all.copy()
                row_index = updated_df.index[updated_df["Entry_ID"] == selected_entry_id][0]
                old_record = serialize_record_for_archive(updated_df.loc[row_index])

                updated_df.at[row_index, "Category"] = edit_category
                updated_df.at[row_index, "Regulator_std"] = edit_regulator.strip() or pd.NA
                updated_df.at[row_index, "Year_raw"] = edit_year_raw.strip() or pd.NA
                updated_df.at[row_index, "Year"] = extract_year(edit_year_raw)
                updated_df.at[row_index, "Regulation_Title"] = edit_title.strip()
                updated_df.at[row_index, "Source_URL"] = edit_source_url.strip() or pd.NA

                save_cbregs(updated_df, resolved_data_path)
                append_audit_log(
                    action="edit",
                    country=editor_country,
                    entry_id=selected_entry_id,
                    user=requested_user,
                    old_record=old_record,
                    new_record=serialize_record_for_archive(updated_df.loc[row_index]),
                )
                st.success("Regulation updated and archived successfully.")
                safe_rerun()

            if st.button("Archive and delete this record"):
                updated_df = df_all.copy()
                row_index = updated_df.index[updated_df["Entry_ID"] == selected_entry_id][0]
                old_record = serialize_record_for_archive(updated_df.loc[row_index])
                updated_df = updated_df.drop(index=row_index).reset_index(drop=True)
                save_cbregs(updated_df, resolved_data_path)
                append_audit_log(
                    action="delete",
                    country=editor_country,
                    entry_id=selected_entry_id,
                    user=requested_user,
                    old_record=old_record,
                    new_record=None,
                )
                st.success("Regulation archived and deleted successfully.")
                safe_rerun()

        st.markdown("---")
        st.markdown("### Recent audit history")
        if ARCHIVE_FILE.exists():
            audit_df = pd.read_csv(ARCHIVE_FILE)
            audit_df["timestamp"] = pd.to_datetime(audit_df["timestamp"], errors="coerce")
            st.dataframe(
                audit_df.sort_values("timestamp", ascending=False).head(20),
                use_container_width=True,
                height=380,
            )
        else:
            st.info("No audit history exists yet. Add or edit a record to create the archive log.")


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
- Click any row to open the full country detail panel below the table.

**When a specific Category is selected:**
- The table shows the actual worksheet fields for that category, grouped by country.
- Click any row to open the country detail panel.
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





