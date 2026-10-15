"""Smart Air Quality & Environmental Health Monitor.

Streamlit dashboard for the GTU "Python for Data Science" project:
Pandas data cleaning, NumPy/SciPy descriptive statistics, Matplotlib/Seaborn
visualisation, and a LangChain + Groq AI layer (health advisory + data chat).

Run with:  streamlit run app.py
"""

import ast
import contextlib
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Smart Air Quality Monitor",
    page_icon="🌫️",
    layout="wide",
)

# Used when the live model list can't be fetched from Groq; preferred order otherwise.
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
# Groq models that are not general chat models (speech, safety, agentic systems).
NON_CHAT_MARKERS = ("whisper", "tts", "guard", "orpheus", "playai", "distil", "compound")

# Known column-name variants across popular Kaggle Indian AQI datasets
# (e.g. "city_day.csv" from "Air Quality Data in India (2015-2020)" and
# "data.csv" from "India Air Quality Data").
COLUMN_ALIASES = {
    "City": ["city", "location", "station", "stationname", "station_name"],
    "State": ["state"],
    "Date": ["date", "datetime", "sampling_date", "timestamp", "day"],
    "AQI": ["aqi", "aqi_value", "overall_aqi"],
    "PM2.5": ["pm2.5", "pm2_5", "pm25", "pm 2.5"],
    "PM10": ["pm10", "rspm"],
    "NO2": ["no2"],
    "SO2": ["so2"],
    "CO": ["co"],
    "O3": ["o3", "ozone"],
    "NH3": ["nh3"],
}

POLLUTANT_COLS = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column variants to a standard set of names."""
    lookup = {c.strip().lower(): c for c in df.columns}
    rename = {}
    for standard, aliases in COLUMN_ALIASES.items():
        if standard in df.columns:
            continue
        for alias in aliases:
            if alias in lookup and lookup[alias] not in rename:
                rename[lookup[alias]] = standard
                break
    return df.rename(columns=rename)


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_and_clean(file_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    """Load the CSV, standardise columns and handle missing sensor readings."""
    from io import BytesIO

    raw = pd.read_csv(BytesIO(file_bytes), encoding_errors="replace", low_memory=False)
    df = standardise_columns(raw)

    report = {"rows_raw": len(df), "missing_before": int(df.isna().sum().sum())}

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
        df = df.dropna(subset=["Date"])

    pollutants = [c for c in POLLUTANT_COLS if c in df.columns]
    for col in pollutants:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where every pollutant reading is missing (dead sensor).
    if pollutants:
        df = df.dropna(subset=pollutants, how="all")

    # Fill remaining gaps with the median of the same city (robust to outliers),
    # then with the overall column median.
    for col in pollutants:
        if "City" in df.columns:
            df[col] = df[col].fillna(df.groupby("City")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates().reset_index(drop=True)

    report.update(
        rows_clean=len(df),
        missing_after=int(df[pollutants].isna().sum().sum()) if pollutants else 0,
        pollutants=pollutants,
    )
    return df, report


def make_sample_data() -> bytes:
    """Generate a synthetic dataset in the Kaggle city_day.csv format."""
    rng = np.random.default_rng(42)
    cities = {
        "Delhi": ("Delhi", 210), "Lucknow": ("Uttar Pradesh", 190),
        "Patna": ("Bihar", 180), "Ahmedabad": ("Gujarat", 150),
        "Kolkata": ("West Bengal", 130), "Mumbai": ("Maharashtra", 100),
        "Hyderabad": ("Telangana", 95), "Chennai": ("Tamil Nadu", 80),
        "Bengaluru": ("Karnataka", 75), "Shillong": ("Meghalaya", 40),
    }
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    season = 1 + 0.45 * np.cos(2 * np.pi * (dates.dayofyear - 15) / 365)  # winter peak
    frames = []
    for city, (state, base) in cities.items():
        pm25 = np.clip(base * 0.55 * season * rng.lognormal(0, 0.25, len(dates)), 5, None)
        pm10 = pm25 * rng.uniform(1.4, 2.0, len(dates))
        aqi = np.clip(base * season * rng.lognormal(0, 0.2, len(dates)), 15, 500)
        frame = pd.DataFrame({
            "City": city, "State": state, "Date": dates.strftime("%Y-%m-%d"),
            "PM2.5": pm25.round(2), "PM10": pm10.round(2),
            "NO2": (pm25 * rng.uniform(0.2, 0.4, len(dates))).round(2),
            "SO2": rng.uniform(5, 25, len(dates)).round(2),
            "CO": rng.uniform(0.3, 2.5, len(dates)).round(2),
            "O3": rng.uniform(15, 60, len(dates)).round(2),
            "AQI": aqi.round(0),
        })
        # Simulate ~5% missing sensor readings.
        for col in ["PM2.5", "PM10", "NO2", "AQI"]:
            frame.loc[rng.random(len(frame)) < 0.05, col] = np.nan
        frames.append(frame)
    return pd.concat(frames).to_csv(index=False).encode()


def aqi_category(aqi: float) -> str:
    """Indian National AQI category (CPCB)."""
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def compute_statistics(df: pd.DataFrame, pollutants: list[str]) -> dict:
    """Measures of central tendency and dispersion for each pollutant."""
    rows = []
    for col in pollutants:
        values = df[col].dropna().to_numpy()
        if values.size == 0:
            continue
        mode_res = stats.mode(np.round(values), keepdims=False)
        rows.append({
            "Pollutant": col,
            "Mean": np.mean(values),
            "Median": np.median(values),
            "Mode": float(mode_res.mode),
            "Std Dev": np.std(values, ddof=1),
            "Variance": np.var(values, ddof=1),
            "Min": np.min(values),
            "Max": np.max(values),
            "Range": np.ptp(values),
            "IQR": stats.iqr(values),
            "Skewness": stats.skew(values),
            "Kurtosis": stats.kurtosis(values),
        })
    table = pd.DataFrame(rows).set_index("Pollutant") if rows else pd.DataFrame()

    summary = {"table": table}
    target = "AQI" if "AQI" in pollutants else (pollutants[0] if pollutants else None)
    summary["target"] = target
    if target:
        summary["mean"] = df[target].mean()
        summary["max"] = df[target].max()
        summary["std"] = df[target].std()
        if "City" in df.columns:
            by_city = df.groupby("City")[target].mean().sort_values(ascending=False)
            summary["by_city"] = by_city
            summary["worst_city"] = by_city.index[0]
            summary["worst_city_value"] = by_city.iloc[0]
            summary["best_city"] = by_city.index[-1]
            summary["best_city_value"] = by_city.iloc[-1]

    # Primary pollutant = particulate/gas with the highest mean relative to its
    # Indian 24-hour standard (NAAQS).
    standards = {"PM2.5": 60, "PM10": 100, "NO2": 80, "SO2": 80, "O3": 100, "CO": 2, "NH3": 400}
    ratios = {p: df[p].mean() / s for p, s in standards.items() if p in pollutants}
    summary["primary_pollutant"] = max(ratios, key=ratios.get) if ratios else "Unknown"
    summary["exceedance"] = ratios
    return summary


def get_llm(api_key: str, model: str, temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(api_key=api_key, model=model, temperature=temperature)


@st.cache_data(ttl=600, show_spinner=False)
def list_groq_models(api_key: str) -> list[str]:
    """Chat models currently available on the user's Groq account."""
    try:
        from groq import Groq

        models = Groq(api_key=api_key).models.list().data
        ids = sorted(
            m.id for m in models
            if getattr(m, "active", True) and not any(k in m.id.lower() for k in NON_CHAT_MARKERS)
        )
        preferred = [m for m in FALLBACK_MODELS if m in ids]
        return preferred + [m for m in ids if m not in preferred] or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


# --------------------------------------------------------------------------- #
# Data chat agent (LangChain tool calling + local Python execution)
# --------------------------------------------------------------------------- #
def python(code: str) -> str:
    """Run Python code on the pandas DataFrame `df` (pandas is `pd`, numpy is `np`).
    Use print() to show results; the value of the last line is also returned."""


def run_python(code: str, namespace: dict) -> str:
    """Execute model-written code against the dataframe and capture its output."""
    code = code.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
    buffer = io.StringIO()
    try:
        tree = ast.parse(code)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(buffer):
            exec(compile(tree, "<agent>", "exec"), namespace)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<agent>", "eval"), namespace)
                if value is not None:
                    print(value)
    except Exception as exc:
        buffer.write(f"{type(exc).__name__}: {exc}")
    output = buffer.getvalue().strip() or "(no output - use print() to show results)"
    return output[:4000]


DATA_AGENT_PROMPT = """You are an air quality data analyst. A pandas DataFrame named `df` holds \
Indian air quality sensor readings ({rows} rows).

Columns and dtypes:
{dtypes}

First rows:
{head}

Rules:
- ALWAYS answer by calling the `python` tool with code that computes the result from `df`; never guess.
- `pd` and `np` are already imported. `Date` (if present) is already datetime.
- Keep the code short and print() the results.
- After you get the tool output, reply with a clear final answer in plain English including the numbers."""

# Groq errors raised when a model emits a malformed or unexpected tool call.
TOOL_CALL_ERRORS = ("tool_use_failed", "Failed to parse tool call", "called a tool",
                    "not in request.tools", "tool call validation failed")


def ask_data_agent(question: str, data: pd.DataFrame, history: list[dict],
                   api_key: str, model: str, max_steps: int = 8) -> str:
    """Answer a question about `data` with a tool-calling loop on a Groq model.

    Replaces langchain_experimental's pandas agent, whose prompts made Groq models
    fail with "Failed to parse tool call arguments as JSON" or
    "Tool choice is none, but model called a tool".
    """
    llm = get_llm(api_key, model, temperature=0).bind_tools([python])
    retry_llm = get_llm(api_key, model, temperature=0.6).bind_tools([python])
    namespace = {"df": data.copy(), "pd": pd, "np": np}

    messages = [SystemMessage(DATA_AGENT_PROMPT.format(
        rows=len(data),
        dtypes="\n".join(f"- {c}: {t}" for c, t in data.dtypes.astype(str).items()),
        head=data.head(3).to_string(),
    ))]
    for msg in history[-6:]:
        messages.append(HumanMessage(msg["content"]) if msg["role"] == "user" else AIMessage(msg["content"]))
    messages.append(HumanMessage(question))

    retries = 0
    for _ in range(max_steps):
        try:
            response = (retry_llm if retries else llm).invoke(messages)
        except Exception as exc:
            if any(marker in str(exc) for marker in TOOL_CALL_ERRORS) and retries < 3:
                retries += 1
                messages.append(HumanMessage(
                    'Your last tool call was malformed. Call the `python` tool again with valid '
                    'JSON arguments like {"code": "print(df.shape)"} and short, simple code.'
                ))
                continue
            raise
        messages.append(response)

        if not response.tool_calls:
            if response.text.strip():
                return response.text
            messages.append(HumanMessage("Please give your final answer."))
            continue

        for call in response.tool_calls:
            args = call.get("args") or {}
            code = args.get("code") or next((v for v in args.values() if isinstance(v, str)), "")
            messages.append(ToolMessage(run_python(code, namespace), tool_call_id=call["id"]))

    return "⚠️ I couldn't reach a final answer for that question. Try rephrasing it more specifically."


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input(label="Groq API Key", type="password",
                            help="Get a free key at https://console.groq.com/keys")
    available_models = list_groq_models(api_key) if api_key else FALLBACK_MODELS
    model_choice = st.selectbox("Groq Model", available_models + ["Custom..."],
                                help="Loaded live from your Groq account once a key is entered")
    if model_choice == "Custom...":
        model_name = st.text_input("Custom model ID", value=FALLBACK_MODELS[0])
    else:
        model_name = model_choice

    st.divider()
    uploaded = st.file_uploader("Upload Air Quality CSV", type=["csv"])
    use_sample = st.checkbox("Use built-in sample data", value=uploaded is None,
                             help="Synthetic data in the Kaggle city_day.csv format")
    st.caption("Recommended dataset: Kaggle *Air Quality Data in India (2015–2020)* — `city_day.csv`")

# Session state for chat history (Tab 4)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "advisory" not in st.session_state:
    st.session_state.advisory = None

st.title("🌫️ Smart Air Quality & Environmental Health Monitor")
st.caption("Python for Data Science · Pandas · NumPy · SciPy · Matplotlib · Seaborn · LangChain · Groq")

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    source_name = uploaded.name
elif use_sample:
    file_bytes = make_sample_data()
    source_name = "sample_city_day.csv"
else:
    st.info("👈 Upload an air quality CSV (or tick *Use built-in sample data*) to begin.")
    st.stop()

try:
    df_clean, clean_report = load_and_clean(file_bytes)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

pollutants = clean_report["pollutants"]
if not pollutants:
    st.error(
        "No pollutant columns found. Expected at least one of: "
        + ", ".join(POLLUTANT_COLS)
    )
    st.stop()

# Optional filters (city & date)
with st.sidebar:
    st.divider()
    st.subheader("🔎 Filters")
    df = df_clean
    if "City" in df.columns:
        all_cities = sorted(df["City"].dropna().astype(str).unique())
        selected_cities = st.multiselect("Cities", all_cities, placeholder="All cities")
        if selected_cities:
            df = df[df["City"].astype(str).isin(selected_cities)]
    if "Date" in df.columns and not df.empty:
        dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
        date_range = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            df = df[(df["Date"] >= start) & (df["Date"] <= end)]

if df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

summary = compute_statistics(df, pollutants)
target = summary["target"]

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & Statistics",
    "📈 Pollution Visualization",
    "📝 AI Health Advisory",
    "💬 Chat with Sensor Data",
])

# --------------------------------------------------------------------------- #
# Tab 1: Data & Statistics
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader(f"Dataset: `{source_name}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Average {target}", f"{summary['mean']:.1f}",
              help=f"Category: {aqi_category(summary['mean'])}" if target == "AQI" else None)
    c2.metric(f"Highest {target}", f"{summary['max']:.1f}")
    c3.metric(f"Std Dev of {target}", f"{summary['std']:.1f}")
    if "PM2.5" in df.columns:
        c4.metric("Highest PM2.5 recorded", f"{df['PM2.5'].max():.1f} µg/m³")
    else:
        c4.metric("Records", f"{len(df):,}")

    if "worst_city" in summary:
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Most polluted city", summary["worst_city"], f"{summary['worst_city_value']:.1f} avg {target}",
                  delta_color="inverse")
        c6.metric("Cleanest city", summary["best_city"], f"{summary['best_city_value']:.1f} avg {target}",
                  delta_color="off")
        c7.metric("Primary pollutant", summary["primary_pollutant"])
        c8.metric("Records (filtered)", f"{len(df):,}")

    with st.expander("🧹 Data cleaning report", expanded=False):
        st.write(f"- Rows in raw file: **{clean_report['rows_raw']:,}**")
        st.write(f"- Missing values before cleaning: **{clean_report['missing_before']:,}**")
        st.write("- Rows with **all** pollutant readings missing were dropped; remaining gaps were "
                 "filled with the **city-wise median** (falling back to the overall median).")
        st.write(f"- Rows after cleaning: **{clean_report['rows_clean']:,}**")
        st.write(f"- Missing pollutant values after cleaning: **{clean_report['missing_after']:,}**")
        st.write(f"- Pollutant columns detected: `{', '.join(pollutants)}`")

    st.markdown("#### Descriptive statistics (central tendency & dispersion)")
    st.dataframe(summary["table"].style.format("{:.2f}"), width="stretch")

    st.markdown("#### Cleaned data")
    st.dataframe(df, width="stretch", height=350)
    st.download_button("⬇️ Download cleaned CSV", df.to_csv(index=False).encode(),
                       "cleaned_air_quality.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Tab 2: Visualization
# --------------------------------------------------------------------------- #
with tab2:
    sns.set_theme(style="whitegrid")
    metric = st.selectbox("Pollutant to visualise", pollutants, index=pollutants.index(target))

    if "City" in df.columns:
        top_n = st.slider("Number of cities", 3, 20, 5)
        st.markdown(f"#### Top {top_n} most polluted cities by mean {metric}")
        city_means = df.groupby("City")[metric].mean().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(x=city_means.values, y=city_means.index.astype(str), hue=city_means.index.astype(str),
                    palette="Reds_r", legend=False, ax=ax)
        ax.set_xlabel(f"Mean {metric}")
        ax.set_ylabel("City")
        for i, v in enumerate(city_means.values):
            ax.text(v, i, f" {v:.1f}", va="center")
        st.pyplot(fig)
        plt.close(fig)

    if "Date" in df.columns:
        st.markdown(f"#### {metric} trend over time")
        freq = st.radio("Resample by", ["Daily", "Weekly", "Monthly"], index=2, horizontal=True)
        rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if "City" in df.columns:
            top_cities = df.groupby("City")[metric].mean().nlargest(5).index
            trend = (df[df["City"].isin(top_cities)]
                     .set_index("Date").groupby("City")[metric].resample(rule).mean().reset_index())
            sns.lineplot(data=trend, x="Date", y=metric, hue="City", ax=ax)
            ax.legend(title="City (top 5)", bbox_to_anchor=(1.01, 1), loc="upper left")
        else:
            trend = df.set_index("Date")[metric].resample(rule).mean().reset_index()
            sns.lineplot(data=trend, x="Date", y=metric, ax=ax)
        if metric == "AQI":
            for level, label in [(100, "Satisfactory"), (200, "Moderate"), (300, "Poor")]:
                ax.axhline(level, ls="--", lw=0.8, color="grey")
                ax.text(ax.get_xlim()[0], level, f" {label} limit", va="bottom", fontsize=8, color="grey")
        ax.set_ylabel(metric)
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"#### Distribution of {metric}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[metric], kde=True, color="tab:orange", ax=ax)
        ax.axvline(df[metric].mean(), color="red", ls="--", label="Mean")
        ax.axvline(df[metric].median(), color="blue", ls=":", label="Median")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        if len(pollutants) > 1:
            st.markdown("#### Pollutant correlation heatmap")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[pollutants].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

# --------------------------------------------------------------------------- #
# Tab 3: AI Environmental Advisory
# --------------------------------------------------------------------------- #
ADVISORY_PROMPT = PromptTemplate.from_template(
    """You are a senior Government Environmental Scientist at the State Pollution Control Board.
Write an official PUBLIC HEALTH ADVISORY on air quality, based STRICTLY on the statistics below.
Do not invent any numbers that are not given.

DATA SUMMARY
- Dataset period: {period}
- Locations covered: {locations}
- Mean {target}: {mean:.1f} (Indian AQI category: {category})
- Maximum {target} recorded: {max:.1f}
- Standard deviation of {target}: {std:.1f}
- Most polluted location: {worst_city} (mean {target} {worst_value})
- Cleanest location: {best_city} (mean {target} {best_value})
- Primary pollutant (highest ratio to Indian NAAQS 24-hr limit): {primary_pollutant}
- Pollutant mean / NAAQS limit ratios: {exceedance}

FORMAT
1. Title and date line ("Issued on the basis of monitored data")
2. Overall air quality status (2-3 sentences)
3. Key findings (bullet points with the numbers above)
4. Health impacts for the general public and sensitive groups (children, elderly, people with asthma/heart disease)
5. Recommended precautions for citizens
6. Recommended actions for local authorities
7. Closing statement

Use a formal, clear, government tone. Use Markdown headings."""
)

with tab3:
    st.subheader("📝 Official Environmental Health Advisory")
    st.write("The selected Groq model writes a formal public-health advisory using **only** the "
             "statistics computed from the current (filtered) dataset.")

    if not api_key:
        st.warning("Enter your Groq API key in the sidebar to generate the advisory.")
    elif st.button("Generate Official Health Advisory", type="primary"):
        period = (f"{df['Date'].min():%d %b %Y} to {df['Date'].max():%d %b %Y}"
                  if "Date" in df.columns else "Not specified")
        locations = df["City"].nunique() if "City" in df.columns else "Not specified"
        prompt_vars = dict(
            period=period,
            locations=locations,
            target=target,
            mean=summary["mean"],
            max=summary["max"],
            std=summary["std"],
            category=aqi_category(summary["mean"]) if target == "AQI" else "N/A",
            worst_city=summary.get("worst_city", "N/A"),
            worst_value=f"{summary['worst_city_value']:.1f}" if "worst_city" in summary else "N/A",
            best_city=summary.get("best_city", "N/A"),
            best_value=f"{summary['best_city_value']:.1f}" if "best_city" in summary else "N/A",
            primary_pollutant=summary["primary_pollutant"],
            exceedance=", ".join(f"{k}: {v:.2f}x" for k, v in summary["exceedance"].items()) or "N/A",
        )
        try:
            chain = ADVISORY_PROMPT | get_llm(api_key, model_name, temperature=0.3)
            with st.spinner(f"Generating advisory with `{model_name}`..."):
                st.session_state.advisory = chain.invoke(prompt_vars).content
        except Exception as exc:
            st.error(f"Failed to generate advisory: {exc}")

    if st.session_state.advisory:
        st.divider()
        st.markdown(st.session_state.advisory)
        st.download_button("⬇️ Download advisory (.md)", st.session_state.advisory,
                           "health_advisory.md", "text/markdown")

# --------------------------------------------------------------------------- #
# Tab 4: Chat with Sensor Data
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("💬 Chat with Sensor Data")
    st.caption('Try: *"Which city has the lowest average PM10?"* or *"How many days had an AQI over 300?"*')

    if st.button("Clear chat"):
        st.session_state.messages = []

    chat_box = st.container(height=480)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the air quality data...", disabled=not api_key)
    if not api_key:
        st.warning("Enter your Groq API key in the sidebar to chat with the data.")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Analysing the data..."):
                    try:
                        answer = ask_data_agent(question, df, st.session_state.messages[:-1],
                                                api_key, model_name)
                    except Exception as exc:
                        answer = f"⚠️ Sorry, I couldn't answer that query. Error: `{exc}`"
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
