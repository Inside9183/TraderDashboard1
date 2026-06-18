import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import perf_counter

import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


st.set_page_config(
    page_title="Hermes AI | Macro Deployment Gate",
    page_icon="L1",
    layout="wide",
)


FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
VIX_DATASET_URL = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"

WEIGHTS = {
    "VIX Level": 0.25,
    "VIX Term Structure": 0.20,
    "Market Breadth": 0.20,
    "Credit Spreads": 0.15,
    "Fear Sentiment": 0.10,
    "Factor Crowding": 0.10,
}

LARGE_CAP_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "BRK-B",
    "JPM", "LLY", "V", "MA", "NFLX", "XOM", "COST", "WMT", "PG", "JNJ",
    "HD", "ABBV", "BAC", "KO", "ORCL", "CRM", "MRK", "CVX", "AMD", "PEP",
    "ADBE", "TMO", "LIN", "MCD", "CSCO", "ABT", "ACN", "DHR", "WFC", "QCOM",
    "TXN", "AMGN", "PM", "IBM", "GE", "CAT", "ISRG", "INTU", "VZ", "NOW",
    "NEE", "DIS", "PFE", "GS", "RTX", "SPGI", "UBER", "LOW", "UNP", "BKNG",
    "T", "PGR", "SYK", "HON", "TJX", "BLK", "ETN", "LMT", "ELV", "SCHW",
    "VRTX", "C", "MDT", "CB", "ADP", "PANW", "BSX", "DE", "ADI", "MMC",
    "REGN", "PLD", "AMAT", "FI", "KLAC", "MU", "LRCX", "GILD", "SO", "MO",
    "ICE", "DUK", "SHW", "ZTS", "WM", "MCO", "EQIX", "APO", "CME", "PH",
    "NKE", "CDNS", "SNPS", "CL", "TT", "UPS", "MAR", "CMG", "AON", "PYPL",
    "HCA", "ORLY", "MS", "USB", "PNC", "FDX", "EOG", "SLB", "COF", "AIG",
]


@dataclass(frozen=True)
class Signal:
    name: str
    value: str
    score: float | None
    weight: float
    date: str
    source: str
    status: str


@dataclass(frozen=True)
class Diagnostic:
    source: str
    status: str
    rows: int
    latest_date: str
    latency_ms: int
    message: str


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, float(value)))


def linear_score(value, best, worst, higher_is_better=False):
    if best == worst:
        return 50.0

    if higher_is_better:
        score = (value - worst) / (best - worst) * 100
    else:
        score = (worst - value) / (worst - best) * 100

    return clamp(score)


def rolling_z_score(series, lookback=252):
    clean = pd.Series(series).dropna().tail(lookback)
    if len(clean) < 30:
        return None

    std = clean.std()
    if std == 0 or math.isnan(std):
        return None

    return float((clean.iloc[-1] - clean.mean()) / std)


def classify(score):
    if score is None:
        return "Unavailable", "--", "One or more required live signals did not load."
    if score >= 70:
        return "Full Deploy", "100%", "Full deployment permitted."
    if score >= 40:
        return "Reduced", "60%", "Reduce sizing and raise the bar for new positions."
    return "Defensive", "25%", "No new longs. Scanner disabled."


def get_fred_api_key():
    for key in ["FRED_API_KEY", "fred_api_key", "FRED_KEY", "fred_key"]:
        try:
            value = st.secrets.get(key)
            if value:
                return str(value).strip(), f"st.secrets['{key}']"
        except Exception:
            pass

    for key in ["FRED_API_KEY", "fred_api_key", "FRED_KEY", "fred_key"]:
        value = os.environ.get(key)
        if value:
            return value.strip(), f"os.environ['{key}']"

    return "", "not found"


def unavailable_signal(name, source, message):
    return Signal(
        name=name,
        value="--",
        score=None,
        weight=WEIGHTS[name],
        date="--",
        source=source,
        status=message,
    )


def observations_to_frame(observations, value_name):
    rows = []
    for item in observations:
        raw_value = item.get("value")
        if raw_value in (None, "."):
            continue
        try:
            rows.append({"date": pd.to_datetime(item["date"]), value_name: float(raw_value)})
        except ValueError:
            continue

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=900)
def fetch_vix_dataset():
    start = perf_counter()
    response = requests.get(VIX_DATASET_URL, timeout=20)
    response.raise_for_status()

    from io import StringIO

    frame = pd.read_csv(StringIO(response.text))
    frame.columns = [column.lower() for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").dropna(subset=["close"])
    latest_date = frame["date"].iloc[-1].strftime("%Y-%m-%d")

    diagnostic = Diagnostic(
        source="datasets/finance-vix",
        status="OK",
        rows=len(frame),
        latest_date=latest_date,
        latency_ms=int((perf_counter() - start) * 1000),
        message="VIX daily close loaded from GitHub raw CSV.",
    )
    return frame, diagnostic


@st.cache_data(ttl=900)
def fetch_fred_series(series_id, api_key, years=3):
    start = perf_counter()
    start_date = (datetime.now(timezone.utc) - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "observation_start": start_date,
    }

    response = requests.get(FRED_BASE_URL, params=params, timeout=20)
    response.raise_for_status()
    observations = response.json().get("observations", [])
    frame = observations_to_frame(observations, series_id)

    if frame.empty:
        raise RuntimeError(f"No numeric observations returned for {series_id}.")

    latest_date = frame["date"].iloc[-1].strftime("%Y-%m-%d")
    diagnostic = Diagnostic(
        source=f"FRED {series_id}",
        status="OK",
        rows=len(frame),
        latest_date=latest_date,
        latency_ms=int((perf_counter() - start) * 1000),
        message="FRED API observations loaded.",
    )
    return frame, diagnostic


def extract_close_prices(raw):
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"]
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        close = raw[["Close"]]

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.loc[:, ~close.columns.duplicated()]
    close = close.sort_index()
    return close


@st.cache_data(ttl=21600)
def fetch_large_cap_prices():
    if yf is None:
        raise RuntimeError("yfinance is not installed.")

    start = perf_counter()
    raw = yf.download(
        LARGE_CAP_UNIVERSE,
        period="18mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    close = extract_close_prices(raw)
    close = close.dropna(axis=1, thresh=220)

    if close.shape[1] < 100:
        raise RuntimeError(f"Only {close.shape[1]} large-cap symbols returned enough history.")

    latest_date = close.index[-1].strftime("%Y-%m-%d")
    diagnostic = Diagnostic(
        source="Yahoo Finance via yfinance",
        status="OK",
        rows=close.shape[0],
        latest_date=latest_date,
        latency_ms=int((perf_counter() - start) * 1000),
        message=f"{close.shape[1]} liquid large-cap symbols loaded.",
    )
    return close, diagnostic


def build_vix_signals(vix_frame):
    latest = float(vix_frame["close"].iloc[-1])
    latest_date = vix_frame["date"].iloc[-1].strftime("%Y-%m-%d")
    trailing_year = vix_frame.tail(252)
    percentile = float((trailing_year["close"] <= latest).mean() * 100)
    vix_score = clamp(100 - percentile)

    if latest < 15:
        vix_score = clamp(vix_score + 5)
    elif latest > 30:
        vix_score = clamp(vix_score - 10)

    vix_20_days_ago = float(vix_frame["close"].iloc[-20])
    vix_roc = ((latest / vix_20_days_ago) - 1) * 100
    fear_score = linear_score(vix_roc, best=-30, worst=50)

    return (
        Signal(
            name="VIX Level",
            value=f"{latest:.2f}",
            score=round(vix_score, 1),
            weight=WEIGHTS["VIX Level"],
            date=latest_date,
            source="datasets/finance-vix",
            status="Live daily close",
        ),
        Signal(
            name="Fear Sentiment",
            value=f"{vix_roc:.1f}%",
            score=round(fear_score, 1),
            weight=WEIGHTS["Fear Sentiment"],
            date=latest_date,
            source="20-day VIX ROC",
            status="Live daily proxy",
        ),
        latest,
    )


def build_term_structure_signal(vix_value, vxv_frame):
    vix3m = float(vxv_frame["VXVCLS"].iloc[-1])
    ratio = vix_value / vix3m
    score = linear_score(ratio, best=0.85, worst=1.15)

    return Signal(
        name="VIX Term Structure",
        value=f"{ratio:.3f}",
        score=round(score, 1),
        weight=WEIGHTS["VIX Term Structure"],
        date=vxv_frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        source="FRED VXVCLS",
        status=f"VIX3M {vix3m:.2f}",
    )


def build_credit_signal(credit_frame):
    credit_z = rolling_z_score(credit_frame["BAMLH0A0HYM2"], lookback=252)
    if credit_z is None:
        raise RuntimeError("Not enough credit spread history for z-score.")

    score = linear_score(credit_z, best=-2, worst=2)
    return Signal(
        name="Credit Spreads",
        value=f"{credit_z:.2f} z",
        score=round(score, 1),
        weight=WEIGHTS["Credit Spreads"],
        date=credit_frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        source="FRED BAMLH0A0HYM2",
        status="1-year z-score",
    )


def build_breadth_and_crowding_signals(close):
    sma_200 = close.rolling(200).mean()
    latest_close = close.iloc[-1]
    latest_sma = sma_200.iloc[-1]
    valid = latest_close.notna() & latest_sma.notna()

    if valid.sum() < 100:
        raise RuntimeError("Not enough symbols for breadth calculation.")

    breadth = float((latest_close[valid] > latest_sma[valid]).mean() * 100)
    breadth_score = linear_score(breadth, best=80, worst=30, higher_is_better=True)

    returns = close.pct_change(fill_method=None).dropna(how="all")
    momentum = (close.iloc[-1] / close.iloc[-61] - 1).dropna().sort_values(ascending=False)

    if len(momentum) < 100:
        raise RuntimeError("Not enough symbols for factor crowding baskets.")

    long_names = momentum.head(50).index.tolist()
    short_names = momentum.tail(50).index.tolist()
    long_basket = returns[long_names].mean(axis=1)
    short_basket = returns[short_names].mean(axis=1)
    rolling_corr = long_basket.rolling(60).corr(short_basket).dropna()

    if rolling_corr.empty:
        raise RuntimeError("Could not calculate factor crowding correlation.")

    crowding_corr = float(rolling_corr.iloc[-1])
    crowding_score = linear_score(crowding_corr, best=0.3, worst=-0.8, higher_is_better=True)
    latest_date = close.index[-1].strftime("%Y-%m-%d")

    return (
        Signal(
            name="Market Breadth",
            value=f"{breadth:.1f}%",
            score=round(breadth_score, 1),
            weight=WEIGHTS["Market Breadth"],
            date=latest_date,
            source="Large-cap universe via yfinance",
            status=f"{int(valid.sum())} symbols",
        ),
        Signal(
            name="Factor Crowding",
            value=f"{crowding_corr:.3f}",
            score=round(crowding_score, 1),
            weight=WEIGHTS["Factor Crowding"],
            date=latest_date,
            source="Top/bottom 50 momentum baskets",
            status="60-day rolling correlation",
        ),
        long_names[:10],
        short_names[:10],
    )


@st.cache_data(ttl=900)
def get_market_data():
    fred_api_key, fred_key_location = get_fred_api_key()
    diagnostics = []
    warnings = []
    signals_by_name = {}
    crowding_longs = []
    crowding_shorts = []
    vix_value = None

    try:
        vix_frame, diagnostic = fetch_vix_dataset()
        diagnostics.append(diagnostic)
        vix_signal, fear_signal, vix_value = build_vix_signals(vix_frame)
        signals_by_name[vix_signal.name] = vix_signal
        signals_by_name[fear_signal.name] = fear_signal
    except Exception as error:
        warnings.append(f"VIX source failed: {error}")
        signals_by_name["VIX Level"] = unavailable_signal("VIX Level", "datasets/finance-vix", "Unavailable")
        signals_by_name["Fear Sentiment"] = unavailable_signal("Fear Sentiment", "20-day VIX ROC", "Unavailable")
        diagnostics.append(Diagnostic("datasets/finance-vix", "ERROR", 0, "--", 0, str(error)))

    if fred_api_key:
        try:
            vxv_frame, diagnostic = fetch_fred_series("VXVCLS", fred_api_key, years=3)
            diagnostics.append(diagnostic)
            if vix_value is None:
                raise RuntimeError("VIX is unavailable, so term structure cannot be calculated.")
            signals_by_name["VIX Term Structure"] = build_term_structure_signal(vix_value, vxv_frame)
        except Exception as error:
            warnings.append(f"VIX term structure failed: {error}")
            signals_by_name["VIX Term Structure"] = unavailable_signal("VIX Term Structure", "FRED VXVCLS", "Unavailable")
            diagnostics.append(Diagnostic("FRED VXVCLS", "ERROR", 0, "--", 0, str(error)))

        try:
            credit_frame, diagnostic = fetch_fred_series("BAMLH0A0HYM2", fred_api_key, years=3)
            diagnostics.append(diagnostic)
            signals_by_name["Credit Spreads"] = build_credit_signal(credit_frame)
        except Exception as error:
            warnings.append(f"Credit spread failed: {error}")
            signals_by_name["Credit Spreads"] = unavailable_signal("Credit Spreads", "FRED BAMLH0A0HYM2", "Unavailable")
            diagnostics.append(Diagnostic("FRED BAMLH0A0HYM2", "ERROR", 0, "--", 0, str(error)))
    else:
        warnings.append("FRED_API_KEY was not detected in Streamlit Secrets.")
        signals_by_name["VIX Term Structure"] = unavailable_signal("VIX Term Structure", "FRED VXVCLS", "FRED key missing")
        signals_by_name["Credit Spreads"] = unavailable_signal("Credit Spreads", "FRED BAMLH0A0HYM2", "FRED key missing")

    try:
        close, diagnostic = fetch_large_cap_prices()
        diagnostics.append(diagnostic)
        breadth_signal, crowding_signal, crowding_longs, crowding_shorts = build_breadth_and_crowding_signals(close)
        signals_by_name[breadth_signal.name] = breadth_signal
        signals_by_name[crowding_signal.name] = crowding_signal
    except Exception as error:
        warnings.append(f"Breadth/crowding failed: {error}")
        signals_by_name["Market Breadth"] = unavailable_signal("Market Breadth", "Large-cap universe via yfinance", "Unavailable")
        signals_by_name["Factor Crowding"] = unavailable_signal("Factor Crowding", "Top/bottom momentum baskets", "Unavailable")
        diagnostics.append(Diagnostic("Yahoo Finance via yfinance", "ERROR", 0, "--", 0, str(error)))

    ordered_signals = [signals_by_name[name] for name in WEIGHTS]
    if all(signal.score is not None for signal in ordered_signals):
        composite = sum(signal.score * signal.weight for signal in ordered_signals)
    else:
        composite = None

    zone, allocation, guidance = classify(composite)

    return {
        "signals": ordered_signals,
        "composite": composite,
        "zone": zone,
        "allocation": allocation,
        "guidance": guidance,
        "diagnostics": diagnostics,
        "warnings": warnings,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "fred_key_detected": bool(fred_api_key),
        "fred_key_location": fred_key_location,
        "crowding_longs": crowding_longs,
        "crowding_shorts": crowding_shorts,
    }


def score_color(score):
    if score is None:
        return "#8d96aa"
    if score >= 70:
        return "#30d983"
    if score >= 40:
        return "#f2c94c"
    return "#ff5364"


def format_score(score):
    if score is None:
        return "--"
    return f"{score:.1f}"


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: #080b12;
            color: #f7f9fc;
        }
        div[data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
        }
        .top-rule {
            height: 4px;
            background: #7d6bff;
            margin: -1.2rem -3rem 1.4rem;
        }
        .title-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #1b2130;
            padding-bottom: 20px;
            margin-bottom: 22px;
            gap: 20px;
        }
        .title-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .badge {
            background: #23c8d2;
            color: #06242a;
            font-weight: 900;
            border-radius: 6px;
            padding: 13px 14px;
        }
        .title {
            font-size: 46px;
            line-height: 1;
            font-weight: 900;
            letter-spacing: 0;
        }
        .subtitle {
            color: #9aa8d6;
            margin-top: 8px;
            font-weight: 800;
        }
        .timestamp {
            color: #9aa8d6;
            text-align: right;
            font-size: 13px;
            line-height: 1.45;
        }
        .metric-box {
            background: #111725;
            border: 1px solid #202b40;
            border-radius: 8px;
            padding: 22px;
            min-height: 136px;
        }
        .metric-label {
            color: #9aa8d6;
            font-size: 13px;
            font-weight: 900;
            text-transform: uppercase;
        }
        .metric-value {
            color: #f7f9fc;
            font-size: 38px;
            font-weight: 900;
            margin-top: 12px;
        }
        .metric-help {
            color: #c5cee2;
            margin-top: 8px;
        }
        .signal-card {
            background: #111725;
            border: 1px solid #202b40;
            border-radius: 8px;
            padding: 18px;
            min-height: 146px;
        }
        .signal-name {
            color: #f7f9fc;
            font-weight: 900;
            font-size: 16px;
        }
        .signal-score {
            font-size: 31px;
            font-weight: 900;
            margin-top: 10px;
        }
        .signal-meta {
            color: #9aa8d6;
            font-size: 13px;
            margin-top: 7px;
        }
        .source-box {
            background: #0d121d;
            border: 1px solid #202b40;
            border-radius: 8px;
            color: #b9c2d6;
            padding: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, help_text, color="#f7f9fc"):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(signal):
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-name">{signal.name}</div>
            <div class="signal-score" style="color:{score_color(signal.score)};">{format_score(signal.score)}</div>
            <div class="signal-meta">Value: {signal.value}</div>
            <div class="signal-meta">Weight: {signal.weight:.0%} · Date: {signal.date}</div>
            <div class="signal-meta">{signal.status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_styles()

    data = get_market_data()

    st.markdown(
        f"""
        <div class="top-rule"></div>
        <div class="title-row">
            <div class="title-left">
                <div class="badge">L1</div>
                <div>
                    <div class="title">MACRO DEPLOYMENT GATE</div>
                    <div class="subtitle">source-backed signals · weighted composite · deployment zone</div>
                </div>
            </div>
            <div class="timestamp">
                Last refresh<br>{data["updated_at"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Refresh data", use_container_width=False):
        st.cache_data.clear()
        st.rerun()

    score = data["composite"]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Composite Score", format_score(score), "weighted 0-100", score_color(score))
    with m2:
        metric_card("Deployment Zone", data["zone"], data["guidance"])
    with m3:
        metric_card("Capital Allocation", data["allocation"], "gate output")
    with m4:
        status = "Connected" if data["fred_key_detected"] else "Missing"
        metric_card("FRED API", status, data["fred_key_location"])

    st.write("")
    card_columns = st.columns(3)
    for index, signal in enumerate(data["signals"]):
        with card_columns[index % 3]:
            render_signal_card(signal)

    st.write("")
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.subheader("Signal Table")
        signal_frame = pd.DataFrame(
            [
                {
                    "Signal": signal.name,
                    "Value": signal.value,
                    "Score": signal.score,
                    "Weight": signal.weight,
                    "Date": signal.date,
                    "Source": signal.source,
                    "Status": signal.status,
                }
                for signal in data["signals"]
            ]
        )
        st.dataframe(signal_frame, use_container_width=True, hide_index=True)

        st.subheader("Source Diagnostics")
        diagnostics_frame = pd.DataFrame(
            [
                {
                    "Source": item.source,
                    "Status": item.status,
                    "Rows": item.rows,
                    "Latest Date": item.latest_date,
                    "Latency ms": item.latency_ms,
                    "Message": item.message,
                }
                for item in data["diagnostics"]
            ]
        )
        st.dataframe(diagnostics_frame, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Sources")
        st.markdown(
            """
            <div class="source-box">
                <b>VIX:</b> datasets/finance-vix GitHub CSV<br>
                <b>VIX3M:</b> FRED VXVCLS<br>
                <b>Credit:</b> FRED BAMLH0A0HYM2<br>
                <b>Breadth:</b> liquid large-cap universe via yfinance<br>
                <b>Fear:</b> 20-day VIX rate of change<br>
                <b>Crowding:</b> top/bottom 50 momentum baskets
            </div>
            """,
            unsafe_allow_html=True,
        )

        if data["crowding_longs"] or data["crowding_shorts"]:
            with st.expander("Factor Baskets"):
                st.write("Top momentum:", ", ".join(data["crowding_longs"]))
                st.write("Bottom momentum:", ", ".join(data["crowding_shorts"]))

        if data["warnings"]:
            with st.expander("Source Warnings", expanded=True):
                for warning in data["warnings"]:
                    st.warning(warning)


if __name__ == "__main__":
    main()