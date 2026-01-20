import re
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Payments Consumer Protection Dashboard", layout="wide")
st.title("Payments Consumer Protection Dashboard")

# =========================
# Column setup (based on your sheet)
# =========================
META_COLS = [
    "Country",
    "Regulator",
    "Regulations on consumer protection (payments)",
    "Year approved/implemented",
    "Official source links",
]

TOPIC_COLS = [
    "Fair treatment & market conduct",
    "Safeguarding of Funds",
    "Transparency & disclosure",
    "Product design, suitability & distribution",
    "Pricing & fees",
    "Consumer data protection & privacy",
    "Consumer redress & dispute resolution",
    "Complaints Handling",
    "Consumer education & awareness",
    "Vulnerable & special consumer groups",
    "Reporting requirements",
    "Sanctions",
    "Other protections",
]

ALL_EXPECTED = META_COLS[:-1] + TOPIC_COLS + [META_COLS[-1]]  # include Official source links at end


# =========================
# Helpers
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Trim whitespace in headers and collapse multiple spaces
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df

def coerce_year(series: pd.Series) -> pd.Series:
    """
    Attempts to extract a 4-digit year from whatever is in the Year column.
    Keeps NaN if none found.
    """
    def extract_year(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        m = re.search(r"(19\d{2}|20\d{2})", s)
        return int(m.group(1)) if m else pd.NA
    return series.apply(extract_year).astype("Int64")

def cell_is_covered(x, treat_partial_as_covered=True) -> bool:
    """
    Coverage heuristic:
    - Non-empty text is covered
    - Treat common negatives as NOT covered
    - Optionally treat 'Partial' as covered
    """
    if pd.isna(x):
        return False
    s = str(x).strip()
    if s == "":
        return False

    s_low = s.lower()

    # Common "not covered" tokens (extend as needed)
    negatives = {"no", "none", "n/a", "na", "not applicable", "not covered", "nil", "-"}
    if s_low in negatives:
        return False

    # If it’s explicitly partial
    if "partial" in s_low:
        return treat_partial_as_covered

    # If it’s explicitly yes/covered
    if s_low in {"yes", "y", "covered", "in place", "implemented"}:
        return True

    # Otherwise: any substantive text counts as covered
    return True

def build_coverage_matrix(df: pd.DataFrame, treat_partial_as_covered=True) -> pd.DataFrame:
    cov = pd.DataFrame(index=df.index)
    for c in TOPIC_COLS:
        if c in df.columns:
            cov[c] = df[c].apply(lambda x: cell_is_covered(x, treat_partial_as_covered))
        else:
            cov[c] = False
    return cov


# =========================
# Load data (file upload to avoid hardcoding paths)
# =========================
uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if not uploaded:
    st.info("Upload your Excel file to begin.")
    st.stop()

@st.cache_data
def load_first_sheet(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=0, engine="openpyxl")
    df = normalize_columns(df)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

df_raw = load_first_sheet(uploaded)

# Basic validation
missing = [c for c in ALL_EXPECTED if c not in df_raw.columns]
if missing:
    st.warning(
        "Some expected columns were not found. The app will still run, but those fields will be empty/ignored.\n\n"
        + "Missing:\n- " + "\n- ".join(missing)
    )

df = df_raw.copy()

# Coerce Year
if "Year approved/implemented" in df.columns:
    df["Year (parsed)"] = coerce_year(df["Year approved/implemented"])
else:
    df["Year (parsed)"] = pd.Series([pd.NA] * len(df), dtype="Int64")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Filters")

treat_partial_as_covered = st.sidebar.toggle("Count 'Partial' as covered", value=True)

countries = sorted([c for c in df.get("Country", pd.Series(dtype=str)).dropna().unique()])
regulators = sorted([r for r in df.get("Regulator", pd.Series(dtype=str)).dropna().unique()])

sel_countries = st.sidebar.multiselect("Country", options=countries, default=countries)
sel_regulators = st.sidebar.multiselect("Regulator", options=regulators, default=regulators)

# Year range
years = df["Year (parsed)"].dropna().astype(int)
if len(years) > 0:
    y_min, y_max = int(years.min()), int(years.max())
    year_range = st.sidebar.slider("Year (parsed) range", min_value=y_min, max_value=y_max, value=(y_min, y_max))
else:
    year_range = None
    st.sidebar.caption("No parseable years found (looking for 4-digit years).")

# Topic filter (show only rows that cover selected topics)
topic_filter = st.sidebar.multiselect("Must cover topics (optional)", options=TOPIC_COLS)

# Free text search (regulation title/notes)
search_text = st.sidebar.text_input("Search text (regulations / notes)", value="").strip().lower()

# Apply filters
f = df.copy()

if "Country" in f.columns and sel_countries:
    f = f[f["Country"].isin(sel_countries)]

if "Regulator" in f.columns and sel_regulators:
    f = f[f["Regulator"].isin(sel_regulators)]

if year_range and "Year (parsed)" in f.columns:
    f = f[f["Year (parsed)"].notna()]
    f = f[(f["Year (parsed)"] >= year_range[0]) & (f["Year (parsed)"] <= year_range[1])]

if search_text and "Regulations on consumer protection (payments)" in f.columns:
    f = f[f["Regulations on consumer protection (payments)"].astype(str).str.lower().str.contains(search_text, na=False)]

# Coverage matrix + topic filtering
cov = build_coverage_matrix(f, treat_partial_as_covered=treat_partial_as_covered)
if topic_filter:
    mask = cov[topic_filter].all(axis=1)
    f = f[mask]
    cov = cov.loc[f.index]

# Derived metrics
if len(f) > 0:
    f = f.copy()
    f["Topics covered (count)"] = cov.sum(axis=1).astype(int)
    f["Coverage (%)"] = (f["Topics covered (count)"] / len(TOPIC_COLS) * 100).round(1)
else:
    st.warning("No rows match your filters.")
    st.stop()

# =========================
# KPI row
# =========================
k1, k2, k3, k4 = st.columns(4)

k1.metric("Rows (jurisdictions / entries)", f"{len(f):,}")

if "Country" in f.columns:
    k2.metric("Countries", f"{f['Country'].nunique():,}")
else:
    k2.metric("Countries", "—")

if "Regulator" in f.columns:
    k3.metric("Regulators", f"{f['Regulator'].nunique():,}")
else:
    k3.metric("Regulators", "—")

k4.metric("Avg coverage", f"{f['Coverage (%)'].mean():.1f}%")

st.divider()

# =========================
# Charts
# =========================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Coverage by topic (share of rows covered)")
    topic_rates = (cov.mean(axis=0) * 100).round(1).sort_values(ascending=False).reset_index()
    topic_rates.columns = ["Topic", "Covered %"]
    fig_topics = px.bar(topic_rates, x="Covered %", y="Topic", orientation="h")
    st.plotly_chart(fig_topics, use_container_width=True)

with right:
    st.subheader("Coverage distribution")
    fig_hist = px.histogram(f, x="Coverage (%)", nbins=12)
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# =========================
# Country / Regulator leaderboards (optional)
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("Average coverage by country")
    if "Country" in f.columns:
        by_country = (
            f.groupby("Country", dropna=False)["Coverage (%)"]
            .mean()
            .sort_values(ascending=False)
            .round(1)
            .reset_index()
        )
        fig_country = px.bar(by_country.head(30), x="Coverage (%)", y="Country", orientation="h")
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.caption("Column 'Country' not found.")

with c2:
    st.subheader("Average coverage by regulator")
    if "Regulator" in f.columns:
        by_reg = (
            f.groupby("Regulator", dropna=False)["Coverage (%)"]
            .mean()
            .sort_values(ascending=False)
            .round(1)
            .reset_index()
        )
        fig_reg = px.bar(by_reg.head(30), x="Coverage (%)", y="Regulator", orientation="h")
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.caption("Column 'Regulator' not found.")

st.divider()

# =========================
# Table view
# =========================
st.subheader("Filtered data (with coverage metrics)")

# Show a compact table: meta + metrics + source links
show_cols = []
for c in ["Country", "Regulator", "Regulations on consumer protection (payments)", "Year approved/implemented", "Year (parsed)"]:
    if c in f.columns:
        show_cols.append(c)

show_cols += ["Topics covered (count)", "Coverage (%)"]

if "Official source links" in f.columns:
    show_cols.append("Official source links")

st.dataframe(
    f[show_cols].sort_values(["Coverage (%)", "Topics covered (count)"], ascending=False),
    use_container_width=True,
    height=420,
)

with st.expander("Show topic coverage matrix (True/False)"):
    st.dataframe(cov.astype(bool), use_container_width=True, height=420)
