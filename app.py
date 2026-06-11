import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# App config
# =========================
st.set_page_config(page_title="ASEAN Regulatory Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stDataFrame td, .stDataFrame th, .element-container div[data-testid="stMarkdownContainer"] {
        font-family: "Segoe UI Emoji", "Apple Color Emoji", "Segoe UI Symbol", "Noto Color Emoji", sans-serif;
    }
    div[role="dialog"], section[data-testid="stDialog"] {
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        max-width: 100vw !important;
        height: 100vh !important;
        max-height: 100vh !important;
        min-height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
        left: 0 !important;
        top: 0 !important;
        background: rgba(0, 0, 0, 0.72) !important;
        background-color: rgba(0, 0, 0, 0.72) !important;
        box-shadow: none !important;
        overflow: auto !important;
    }
    div[role="dialog"] .element-container, section[data-testid="stDialog"] .element-container {
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 1rem 1rem 1rem !important;
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    div[role="dialog"] .stDataFrame, section[data-testid="stDialog"] .stDataFrame {
        width: auto !important;
        min-width: auto !important;
    }
    div[role="dialog"] table, section[data-testid="stDialog"] table {
        width: auto !important;
        max-width: 100% !important;
        table-layout: auto !important;
    }
    div[role="dialog"] td, div[role="dialog"] th,
    section[data-testid="stDialog"] td, section[data-testid="stDialog"] th {
        white-space: normal !important;
        word-break: break-word !important;
    }
    div[role="dialog"] h1, div[role="dialog"] h2,
    section[data-testid="stDialog"] h1, section[data-testid="stDialog"] h2 {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("ASEAN Regulatory Dashboard")
st.caption("v2 • 2026-01-30")

DATA_FILE = Path("src") / "CBregs.xlsx"


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
    "year": ["year", "Year", "Year approved/implemented", "Year Approved/Implemented", "Year approved / implemented"],
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

def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def infer_title_col(df: pd.DataFrame) -> Optional[str]:
    # 1) Try known title candidates
    c = pick_first_existing_col(df, META_COL_CANDIDATES["title"])
    if c:
        return c

    # 2) Fallback: pick first non-meta-ish column with text values
    known_meta = set(META_COL_CANDIDATES["country"] + META_COL_CANDIDATES["regulator"] +
                     META_COL_CANDIDATES["year"] + META_COL_CANDIDATES["source"] + ["Regulation ID"])
    for col in df.columns:
        if col in known_meta:
            continue
        # Heuristic: choose first column that looks like a name/title field (string-ish)
        if df[col].astype(str).str.len().mean() > 5:
            return col
    return None

def safe_linkify(url) -> str:
    if url is None:
        return ""
    url = str(url).strip()
    if not url or url.lower() in {"nan", "none"}:
        return ""
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
        '<table class="regulations-table" style="width:100%; border-collapse: collapse; margin-bottom:1rem;">',
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


# =========================
# Load data
# =========================
# df_all = load_cbregs(DATA_FILE)
resolved_data_path = resolve_data_path(DATA_FILE)

df_all = load_cbregs(resolved_data_path, resolved_data_path.stat().st_mtime)

if df_all.empty:
    st.error("Data loaded but produced no rows.")
    st.stop()

# =========================
# Sidebar filters (ORDER: Category -> Year -> Country -> Regulator)
# =========================
st.sidebar.header("Filters")

categories = ["All"] + sorted(df_all["Category"].dropna().unique().tolist())
sel_category = st.sidebar.selectbox("Category", options=categories, index=0)

df_f = df_all.copy()
if sel_category != "All":
    df_f = df_f[df_f["Category"] == sel_category]

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

# ---- apply filter
if st.session_state["selected_countries"]:
    df_f = df_f[df_f["Country_std"].isin(st.session_state["selected_countries"])]

st.divider()

# =========================
# Tabs (Map default)
# =========================
tab_map, tab_table = st.tabs(["Map", "Table"])

# =========================
# MAP TAB
# =========================
with tab_map:
    # st.subheader("")

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

    # Neutral shading (no ranking): single constant value
    by_country["_fill"] = 1
    
    fig = px.choropleth(
    by_country,
    locations="Country",
    locationmode="country names",
    color="_fill",
    custom_data=["Latest_10"],     # <-- add this
    )

    fig.update_traces(
    hovertemplate="%{location}<br>%{customdata[0]}<extra></extra>"
    )
    
    # Remove legend/colorbar entirely |dispable box/lasso zoom
    fig.update_layout(
        hoverlabel=dict(align="left"),
        dragmode=False,
        hovermode='closest',
        coloraxis_showscale=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True))
    
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    
    

    fig.update_geos(
        scope="asia",
        projection_type="mercator",
    
        # --- LOCK ASEAN VIEWPORT ---
        lonaxis=dict(range=[92, 141]),
        lataxis=dict(range=[-11, 24]),
    
        # --- Disable interactions ---
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,      # removes zoom / pan / lasso / camera icons
            "scrollZoom": False,         # disables scroll wheel zoom
            "doubleClick": False,        # disables double-click zoom reset
        }
    )


    st.caption("Hover a country to preview its 10 most recent regulations (based on the current filters).")

    # Country selector to open modal (map click is harder without extra packages)
    map_country = st.selectbox("Open a country details popup", options=["(Select)"] + sorted(by_country["Country"].tolist()))
    if map_country != "(Select)":
        st.session_state["selected_country"] = map_country


# =========================
# TABLE TAB
# =========================

with tab_table:
    all_sheet_names = sorted(df_all["Category"].dropna().unique().tolist())

    # =========================================================
    # MODE A: Category = All -> keep your current summary matrix
    # =========================================================
    if sel_category == "All":
        regs_by_country = (
            df_f.groupby("Country_std")["Regulator_std"]
            .apply(lambda x: ", ".join(sorted(set([v for v in x.dropna().tolist()]))))
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
        event = st.dataframe(
            t,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=520,
        )

        idx = get_selected_row_index(event)
        if idx is not None:
            selected_country = t.iloc[idx]["Country"]
            st.session_state["selected_country"] = selected_country

            st.markdown(f"### Preview: {selected_country}")
            latest10 = latest_regs_by_country(df_f, selected_country, n=10)
            if latest10.empty:
                st.info("No regulations found for this country under the current filters.")
            else:
                preview_lines = []
                for _, r in latest10.iterrows():
                    y = r["Year"]
                    y_txt = str(int(y)) if pd.notna(y) else "—"
                    preview_lines.append(f"- **{y_txt}** — {r['Regulation_Title']}")
                st.markdown("\n".join(preview_lines))

    # =========================================================
    # MODE B: Category selected -> show actual worksheet columns
    # =========================================================
    else:
        st.caption(f"Showing worksheet fields for: {sel_category}")
        
        d = df_f.copy()
        
        regs_by_country = (
            d.groupby("Country_std")["Regulator_std"]
            .apply(lambda x: ", ".join(sorted(set([v for v in x.dropna().tolist()]))))
            .reset_index()
            .rename(columns={"Country_std": "Country", "Regulator_std": "Regulator"})
        )

        cols_to_concat = d.drop(columns=['ID', 'title', 'year', 'Year', 'Year_raw', 'source', 
                                         'country', 'regulator','Category','Country_std', 'Regulator_std', 
                                         'Source_URL']).columns
        provs = (
            d
            .drop(columns=['ID', 'title', 'year', 'source'])
            .groupby(['Country_std','Category'], dropna=False)
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
        t = (t[["Flag", "Country", "Regulator"] 
             + [col for col in t.columns if col not in ["Flag", "Country", "Regulator", "Category", "Regulation_Title"]]]
             .sort_values("Country").dropna(axis=1, how="all"))

        st.caption("Select a row to preview and open a country popup.")
        event = st.dataframe(
            t,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=520,
        )

        idx = get_selected_row_index(event)
        if idx is not None:
            selected_country = t.iloc[idx]["Country"]
            st.session_state["selected_country"] = selected_country

            st.markdown(f"### Preview: {selected_country}")
            latest10 = latest_regs_by_country(df_f, selected_country, n=10)
            if latest10.empty:
                st.info("No regulations found for this country under the current filters.")
            else:
                preview_lines = []
                for _, r in latest10.iterrows():
                    y = r["Year"]
                    y_txt = str(int(y)) if pd.notna(y) else "—"
                    preview_lines.append(f"- **{y_txt}** — {r['Regulation_Title']}")
                st.markdown("\n".join(preview_lines))



# =========================
# Country popup (modal)
# =========================
@st.dialog("Country regulations")
def country_dialog(country: str):
    st.markdown(f"## {ASEAN_FLAG.get(country, '🏳️')} {country}")

    d = df_f[df_f["Country_std"] == country].copy()
    if d.empty:
        st.info("No regulations found for this country under the current filters.")
        return

    regs = sorted(set(x for x in d["Regulator_std"].dropna().tolist()))
    st.markdown("**Regulator:** " + (", ".join(regs) if regs else "—"))

    st.markdown("## Regulations")
    regs_table_html = build_regulations_html_table(d.sort_values(["Year", "Regulation_Title"], ascending=[False, True]))
    st.markdown(regs_table_html, unsafe_allow_html=True)

    st.markdown("## Key Provisions")

    known_meta_cols = set(
        META_COL_CANDIDATES["country"] +
        META_COL_CANDIDATES["regulator"] +
        META_COL_CANDIDATES["year"] +
        META_COL_CANDIDATES["source"] +
        META_COL_CANDIDATES["title"] +
        ["Category", "Country_std", "Regulator_std", "Year_raw", "Year", "Year_sort", "Regulation_Title", "Source_URL", "ID"]
    )

    shown_any_category = False
    for cat in sorted(d["Category"].dropna().unique().tolist()):
        dc = d[d["Category"] == cat].copy()

        detail_cols = [c for c in dc.columns if c not in known_meta_cols and c not in META_COL_CANDIDATES["source"]]
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
            unique_values = sorted({v for v in values if pd.notna(v) and v.lower() not in {"nan", "none", ""}})
            if not unique_values:
                continue
            rows.append({"Field": col, "Values": "; ".join(unique_values)})

        if not rows:
            continue

        st.markdown(f"### {cat}")
        detail_df = pd.DataFrame(rows)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        shown_any_category = True

    if not shown_any_category:
        st.caption("No key provisions available for this country under the current filters.")

    st.caption("Links shown as 'Source' are taken directly from the 'Official Source' column in CBregs.xlsx.")


# Fire dialog if a country is chosen
if "selected_country" in st.session_state and st.session_state["selected_country"]:
    country_dialog(st.session_state["selected_country"])
    # optional: clear after showing (keeps UX from reopening on every rerun)
    st.session_state["selected_country"] = None
