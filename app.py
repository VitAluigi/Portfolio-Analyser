# Portfolio Analysis App
# pip install streamlit yfinance pandas numpy plotly openpyxl
# streamlit run app.py

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #0E1117; }
    .metric-card { background: #161B22; border: 1px solid #21262D; border-radius: 12px; padding: 16px 20px; text-align: center; }
    .metric-label { font-size: 11px; color: #8B949E; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
    .metric-value { font-size: 22px; font-weight: 700; font-family: 'DM Mono', monospace; }
    .metric-sub { font-size: 11px; color: #8B949E; margin-top: 4px; }
    .positive { color: #3FB950; } .negative { color: #F85149; } .neutral { color: #58A6FF; }
    [data-testid="stSidebar"] { background: #0D1117; border-right: 1px solid #21262D; }
    .stTabs [data-baseweb="tab-list"] { background: #161B22; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8B949E; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: #21262D; color: #E6EDF3; border-radius: 6px; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; color: #E6EDF3; }
    .bm-card { background: #161B22; border: 2px solid #21262D; border-radius: 14px; padding: 14px 18px; cursor: pointer; text-align: center; transition: all 0.2s; }
    .bm-card:hover { border-color: #3FB950; }
    .bm-card-selected { border-color: #3FB950 !important; background: #1a2e1a !important; }
    .bm-flag { font-size: 24px; }
    .bm-name { font-size: 12px; font-weight: 700; color: #E6EDF3; margin-top: 4px; }
    .bm-ticker { font-size: 10px; color: #8B949E; }
    .rf-badge { background: #1a2e1a; border: 1px solid #3FB950; border-radius: 8px; padding: 6px 14px; display: inline-block; font-family: 'DM Mono', monospace; font-size: 13px; color: #3FB950; }
    .stButton>button { background: linear-gradient(135deg, #238636, #2EA043); color: white; border: none; border-radius: 8px; font-weight: 700; font-family: 'Syne', sans-serif; }
</style>
""", unsafe_allow_html=True)

# Constants
BENCHMARKS = {
    "S&P 500": {"ticker": "^GSPC", "rf_ticker": "^TNX", "rf_name": "T-Note 10y", "rf_fallback": 4.20},
    "NASDAQ 100": {"ticker": "^NDX", "rf_ticker": "^TNX", "rf_name": "T-Note 10y", "rf_fallback": 4.20},
    "FTSE MIB": {"ticker": "FTSEMIB.MI","rf_ticker": "ITALY10YR=X", "rf_name": "BTP 10y", "rf_fallback": 3.70},
    "DAX": {"ticker": "^GDAXI", "rf_ticker": "^DE10YB=RR", "rf_name": "Bund 10y", "rf_fallback": 2.50},
    "Euro Stoxx 50":{"ticker": "^STOXX50E", "rf_ticker": "^DE10YB=RR", "rf_name": "Bund 10y", "rf_fallback": 2.50},
    "FTSE 100": {"ticker": "^FTSE", "rf_ticker": "^GB10YB=RR", "rf_name": "Gilt 10y", "rf_fallback": 4.30},
    "Nikkei 225": {"ticker": "^N225", "rf_ticker": "^JP10YB=RR", "rf_name": "JGB 10y", "rf_fallback": 1.50},
    "CAC 40": {"ticker": "^FCHI", "rf_ticker": "^FR10YB=RR", "rf_name": "OAT 10y", "rf_fallback": 3.20},
    "Russell 2000": {"ticker": "^RUT", "rf_ticker": "^TNX", "rf_name": "T-Note 10y", "rf_fallback": 4.20},
    "MSCI World": {"ticker": "URTH", "rf_ticker": "^TNX", "rf_name": "T-Note 10y", "rf_fallback": 4.20},
}

C = {
    "portfolio": "#3FB950", "benchmark": "#58A6FF", "danger": "#F85149",
    "warning": "#D29922", "neutral": "#8B949E", "surface": "#161B22",
}
PIE_COLORS = ["#58A6FF","#3FB950","#BC8CFF","#D29922","#8B949E","#39D353","#7EE787","#FFA657","#FF7B72"]
VOL_WINDOW = 30

# Risk-free fetcher
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_rf_rate(rf_ticker, fallback):
    """Fetch current 10y government bond yield from Yahoo Finance."""
    try:
        data = yf.download(rf_ticker, period="5d", auto_adjust=True, progress=False)
        if data.empty:
            return fallback
        close = data["Close"].squeeze()
        val = float(close.dropna().iloc[-1])
        # ^TNX and similar return yield in % already (e.g. 4.25)
        return round(val, 2)
    except Exception:
        return fallback

@st.cache_data(show_spinner=False)
def load_benchmark(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df["Close"].squeeze().rename("benchmark")

# Helpers
def plotly_layout(height=500, title=None):
    layout = dict(
        height=height,
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
        font=dict(family="Syne, sans-serif", size=11, color="#E6EDF3"),
        margin=dict(t=50 if title else 30, b=40, l=50, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    orientation="h", y=1.02, x=0.5, xanchor="center"),
        hovermode="x unified",
        xaxis=dict(gridcolor="#21262D", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#21262D", showgrid=True, zeroline=False),
    )
    if title:
        layout["title"] = dict(text=f"<b>{title}</b>", font=dict(size=14), x=0.02)
    return layout

def color_val(v, is_pct=True):
    fmt = f"{v:+.1%}" if is_pct else f"{v:+.2f}"
    return fmt, "positive" if v > 0 else ("negative" if v < 0 else "neutral")

def metric_html(label, ptf_fmt, ptf_cls, bm_fmt=None, bm_cls=None):
    bm_part = f'<div class="metric-sub"><span class="{bm_cls}">{bm_fmt}</span> benchmark</div>' if bm_fmt else ""
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {ptf_cls}">{ptf_fmt}</div>{bm_part}</div>'

# Data pipeline
def compute_features(df, bm_series, vol_window=VOL_WINDOW):
    port = df["Total"]
    combined = pd.concat([bm_series, port], axis=1).dropna()
    combined.columns = ["benchmark", "portfolio"]
    combined["bm_ret"] = combined["benchmark"].pct_change()
    combined["port_ret"] = combined["portfolio"].pct_change()
    combined = combined.dropna()
    combined["bm_cum"] = (1 + combined["bm_ret"]).cumprod() - 1
    combined["port_cum"] = (1 + combined["port_ret"]).cumprod() - 1
    combined["bm_vol"] = combined["bm_ret"].rolling(vol_window).std() * np.sqrt(252)
    combined["port_vol"] = combined["port_ret"].rolling(vol_window).std() * np.sqrt(252)
    combined["bm_dd"] = combined["benchmark"] / combined["benchmark"].cummax() - 1
    combined["port_dd"] = combined["portfolio"] / combined["portfolio"].cummax() - 1
    etf_cols = [c for c in df.columns if c != "Total"]
    for col in etf_cols:
        combined[col] = df[col].reindex(combined.index).ffill()
    return combined, etf_cols

def compute_metrics(db, rf):
    port = db["port_ret"].dropna()
    bm = db["bm_ret"].dropna()
    def stats(r):
        mu = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sh = (mu - rf) / vol if vol > 0 else 0
        down = r[r < 0]
        so = (mu - rf) / (down.std() * np.sqrt(252)) if len(down) > 1 else 0
        cum = (1 + r).cumprod()
        mdd = float((cum / cum.cummax()).min() - 1)
        ca = mu / abs(mdd) if mdd < 0 else 0
        cagr = (1 + (1+r).prod()-1) ** (365/max(len(r),1)) - 1
        return dict(mu=mu, vol=vol, sharpe=sh, sortino=so, mdd=mdd,
                    calmar=ca, cagr=cagr, cum=(1+r).prod()-1,
                    skew=r.skew(), kurt=r.kurtosis())
    aligned = pd.concat([port, bm], axis=1).dropna()
    cov = aligned.cov()
    beta = cov.iloc[0,1] / cov.iloc[1,1] if cov.iloc[1,1] > 0 else 1
    alpha = (port.mean() - beta * bm.mean()) * 252
    return dict(port=stats(port), bm=stats(bm), beta=beta, alpha=alpha)

def split_periods(db):
    periods = {}
    for yr in sorted(db.index.year.unique()):
        sub = db[db.index.year == yr].copy()
        if len(sub) > 5:
            sub["port_cum"] = (1 + sub["port_ret"]).cumprod() - 1
            sub["bm_cum"] = (1 + sub["bm_ret"]).cumprod() - 1
            periods[str(yr)] = sub
    full = db.copy()
    full["port_cum"] = (1 + full["port_ret"]).cumprod() - 1
    full["bm_cum"] = (1 + full["bm_ret"]).cumprod() - 1
    periods["Total"] = full
    return periods

# Tab 1: Performance
def tab_performance(db, m, bm_name, rf):
    # Date range filter
    min_date = db.index.min().date()
    max_date = db.index.max().date()
    c_from, c_to, _ = st.columns([1.5, 1.5, 7])
    d_from = c_from.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="perf_from")
    d_to = c_to.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="perf_to")
    if d_from >= d_to:
        st.warning("Start date must be before end date.")
        return

    # Filter and recompute from the selected start date
    db = db[(db.index.date >= d_from) & (db.index.date <= d_to)].copy()
    if len(db) < 2:
        st.warning("Period too short.")
        return
    db["port_cum"] = (1 + db["port_ret"]).cumprod() - 1
    db["bm_cum"] = (1 + db["bm_ret"]).cumprod() - 1
    db["port_dd"] = db["portfolio"] / db["portfolio"].cummax() - 1
    db["bm_dd"] = db["benchmark"] / db["benchmark"].cummax() - 1

    port_r = db["port_ret"].dropna()
    bm_r = db["bm_ret"].dropna()
    beta = db[["port_ret","bm_ret"]].dropna().cov().iloc[0,1] / bm_r.var() if bm_r.var() > 0 else 1
    m = {
        "port": {
            "cum": float((1 + port_r).prod() - 1),
            "cagr": float((1 + (1+port_r).prod()-1) ** (365/max(len(db),1)) - 1),
            "sharpe": float((port_r.mean()*252 - rf) / (port_r.std()*np.sqrt(252))) if port_r.std()>0 else 0,
            "mdd": float((db["portfolio"]/db["portfolio"].cummax()).min()-1),
        },
        "bm": {
            "cum": float((1 + bm_r).prod() - 1),
            "cagr": float((1 + (1+bm_r).prod()-1) ** (365/max(len(db),1)) - 1),
            "sharpe": float((bm_r.mean()*252 - rf) / (bm_r.std()*np.sqrt(252))) if bm_r.std()>0 else 0,
            "mdd": float((db["benchmark"]/db["benchmark"].cummax()).min()-1),
        },
        "beta": beta,
        "alpha": float((port_r.mean() - beta * bm_r.mean()) * 252),
    }
    st.caption(f"Period: {d_from.strftime('%d %b %Y')} to {d_to.strftime('%d %b %Y')} — {len(db)} trading days")

    cols = st.columns(5)
    kpis = [
        ("Cumulative Return", m["port"]["cum"], m["bm"]["cum"], True),
        ("CAGR", m["port"]["cagr"], m["bm"]["cagr"], True),
        ("Sharpe Ratio", m["port"]["sharpe"], m["bm"]["sharpe"], False),
        ("Max Drawdown", m["port"]["mdd"], m["bm"]["mdd"], True),
        ("Beta / Alpha", m["beta"], m["alpha"], False),
    ]
    for col, (label, pv, bv, is_pct) in zip(cols, kpis):
        pf, pc = color_val(pv, is_pct)
        bf, bc = color_val(bv, is_pct)
        if label == "Beta / Alpha":
            pf, pc = f"{pv:.2f} / {bv:+.2%}", "neutral"
            bf, bc = None, None
        col.markdown(metric_html(label, pf, pc, bf, bc), unsafe_allow_html=True)

    st.markdown("---")
    dates = db.index
    port_arr = db["port_cum"].values * 100
    bm_arr = db["bm_cum"].values * 100

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dates, y=bm_arr,   line=dict(color=C["benchmark"], width=2, dash="dash"), name=bm_name))
    fig1.add_trace(go.Scatter(x=dates, y=port_arr, line=dict(color=C["portfolio"], width=2), name="Portfolio"))
    fig1.add_trace(go.Scatter(
        x=list(dates)+list(dates[::-1]),
        y=list(np.where(port_arr>=bm_arr,port_arr,bm_arr))+list(np.where(port_arr>=bm_arr,bm_arr,port_arr))[::-1],
        fill="toself", fillcolor="rgba(63,185,80,0.12)", line=dict(width=0),
        showlegend=False, hovertemplate="<extra></extra>"))
    fig1.add_trace(go.Scatter(
        x=list(dates)+list(dates[::-1]),
        y=list(np.where(port_arr<bm_arr,port_arr,bm_arr))+list(np.where(port_arr<bm_arr,bm_arr,port_arr))[::-1],
        fill="toself", fillcolor="rgba(248,81,73,0.10)", line=dict(width=0),
        showlegend=False, hovertemplate="<extra></extra>"))
    fig1.add_hline(y=0, line=dict(color=C["neutral"], width=0.8, dash="dot"))
    fig1.update_layout(**plotly_layout(380, "Cumulative Return (%)"))
    fig1.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates, y=db["bm_dd"]*100, fill="tozeroy", fillcolor="rgba(88,166,255,0.2)", line=dict(color=C["benchmark"], width=1.5), name=bm_name))
    fig2.add_trace(go.Scatter(x=dates, y=db["port_dd"]*100, fill="tozeroy", fillcolor="rgba(63,185,80,0.3)", line=dict(color=C["portfolio"], width=2), name="Portfolio"))
    fig2.update_layout(**plotly_layout(300, "Drawdown (%)"))
    fig2.update_yaxes(ticksuffix="%")
    c1.plotly_chart(fig2, use_container_width=True)

    ret_vals = db["port_ret"] * 100
    fig3 = go.Figure(go.Bar(x=dates, y=ret_vals,
        marker_color=[C["portfolio"] if v >= 0 else C["danger"] for v in ret_vals],
        marker_line_width=0, showlegend=False))
    fig3.update_layout(**plotly_layout(300, "Daily Returns (%)"))
    fig3.update_yaxes(ticksuffix="%")
    c2.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=dates, y=db["port_vol"]*100, line=dict(color=C["portfolio"], width=2), name="Portfolio"))
    fig4.add_trace(go.Scatter(x=dates, y=db["bm_vol"]*100, line=dict(color=C["benchmark"], width=2, dash="dash"), name=bm_name))
    fig4.update_layout(**plotly_layout(280, f"Rolling Volatility {VOL_WINDOW}d (%)"))
    fig4.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig4, use_container_width=True)

# Tab 2: Composizione
def tab_composizione(db, etf_cols, m):
    if not etf_cols:
        st.info("No asset columns found.")
        return

    last = db[etf_cols].iloc[-1]
    c1, c2 = st.columns(2)

    fig_pie = go.Figure(go.Pie(
        labels=etf_cols, values=last.values,
        marker=dict(colors=PIE_COLORS[:len(etf_cols)], line=dict(color="#0E1117", width=2)),
        textinfo="label+percent", hole=0.4,
        hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>"))
    fig_pie.update_layout(height=380, paper_bgcolor=C["surface"],
        font=dict(family="Syne, sans-serif", size=10, color="#E6EDF3"),
        margin=dict(t=40, b=20, l=20, r=20),
        title=dict(text="<b>Current Composition</b>", font=dict(size=14), x=0.02),
        showlegend=False)
    c1.plotly_chart(fig_pie, use_container_width=True)

    etf_perf = {col: db[col].dropna().iloc[-1]/db[col].dropna().iloc[0]-1
                for col in etf_cols if len(db[col].dropna()) > 1}
    etf_sorted = sorted(etf_perf.items(), key=lambda x: x[1])
    vals = [v*100 for _,v in etf_sorted]
    fig_etf = go.Figure(go.Bar(
        x=vals, y=[k for k,_ in etf_sorted], orientation="h",
        marker_color=[C["portfolio"] if v>=0 else C["danger"] for v in vals],
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        textfont=dict(size=9, color="#E6EDF3")))
    fig_etf.add_vline(x=0, line=dict(color=C["neutral"], width=0.8))
    fig_etf.update_layout(**plotly_layout(380, "Performance per Asset (%)"))
    fig_etf.update_xaxes(ticksuffix="%")
    c2.plotly_chart(fig_etf, use_container_width=True)

    monthly = db["port_ret"].resample("ME").apply(lambda x: (1+x).prod()-1)
    fig_m = go.Figure(go.Bar(
        x=[d.strftime("%b-%y") for d in monthly.index], y=monthly*100,
        marker_color=[C["portfolio"] if v>=0 else C["danger"] for v in monthly],
        text=[f"{v*100:.1f}%" for v in monthly], textposition="outside",
        textfont=dict(size=8, color=C["neutral"])))
    fig_m.add_hline(y=0, line=dict(color=C["neutral"], width=0.8))
    fig_m.update_layout(**plotly_layout(300, "Monthly Returns (%)"))
    fig_m.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_m, use_container_width=True)

    c3, c4 = st.columns(2)
    port_r = db["port_ret"].dropna()*100
    bm_r = db["bm_ret"].dropna()*100
    bins = np.linspace(min(port_r.min(),bm_r.min()), max(port_r.max(),bm_r.max()), 50)
    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(x=port_r, xbins=dict(start=bins[0],end=bins[-1],size=bins[1]-bins[0]),
        marker_color=C["portfolio"], opacity=0.7, name=f"Portfolio σ={port_r.std():.2f}%"))
    fig_d.add_trace(go.Histogram(x=bm_r, xbins=dict(start=bins[0],end=bins[-1],size=bins[1]-bins[0]),
        marker_color=C["benchmark"], opacity=0.5, name=f"Benchmark σ={bm_r.std():.2f}%"))
    fig_d.add_vline(x=0, line=dict(color=C["neutral"], dash="dot"))
    fig_d.update_layout(**plotly_layout(300, f"Distribuzione — Kurt:{m['port']['kurt']:.1f} Skew:{m['port']['skew']:.2f}"), barmode="overlay")
    c3.plotly_chart(fig_d, use_container_width=True)

    c4.markdown("**Risk Metrics**")
    c4.dataframe(pd.DataFrame({
        "Metrica":    ["Sharpe","Sortino","Calmar","Max DD","Vol ann.","CAGR","Beta","Alpha ann."],
        "Portfolio":[f"{m['port']['sharpe']:.2f}", f"{m['port']['sortino']:.2f}",
                       f"{m['port']['calmar']:.2f}", f"{m['port']['mdd']:.1%}",
                       f"{m['port']['vol']:.1%}",    f"{m['port']['cagr']:.1%}",
                       f"{m['beta']:.2f}",           f"{m['alpha']:+.1%}"],
    }), hide_index=True, use_container_width=True)

# Tab 3: Volatilità
def tab_volatilita(db, bm_name):
    c1, c2 = st.columns(2)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=db.index, y=db["port_vol"]*100, line=dict(color=C["portfolio"], width=2), name="Portfolio"))
    fig1.add_trace(go.Scatter(x=db.index, y=db["bm_vol"]*100, line=dict(color=C["benchmark"], width=2, dash="dash"), name=bm_name))
    fig1.update_layout(**plotly_layout(320, f"Rolling Volatility {VOL_WINDOW}d (%)"))
    fig1.update_yaxes(ticksuffix="%")
    c1.plotly_chart(fig1, use_container_width=True)

    vol_ratio = (db["port_vol"]/db["bm_vol"]).dropna()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=vol_ratio.index, y=vol_ratio, fill="tozeroy",
        fillcolor="rgba(210,153,34,0.15)", line=dict(color=C["warning"], width=2), name="Ratio vol"))
    fig2.add_hline(y=1, line=dict(color=C["neutral"], dash="dash"), annotation_text="Parity")
    fig2.add_hline(y=vol_ratio.mean(), line=dict(color=C["portfolio"], dash="dot"),
        annotation_text=f"Mean {vol_ratio.mean():.2f}")
    fig2.update_layout(**plotly_layout(320, "Portfolio Vol / Benchmark Vol"))
    c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    monthly_vol = db["port_ret"].resample("ME").std() * np.sqrt(252) * 100
    fig3 = go.Figure(go.Bar(
        x=[d.strftime("%b-%y") for d in monthly_vol.index], y=monthly_vol,
        marker_color=[C["danger"] if v>15 else C["warning"] if v>8 else C["portfolio"] for v in monthly_vol],
        showlegend=False))
    fig3.add_hline(y=15, line=dict(color=C["danger"], dash="dash"), annotation_text="15%")
    fig3.add_hline(y=8,  line=dict(color=C["warning"], dash="dash"), annotation_text="8%")
    fig3.update_layout(**plotly_layout(300, "Annualised Monthly Vol (%)"))
    fig3.update_yaxes(ticksuffix="%")
    c3.plotly_chart(fig3, use_container_width=True)

    roll_corr = db["port_ret"].rolling(60).corr(db["bm_ret"]).dropna()
    fig4 = go.Figure(go.Scatter(x=roll_corr.index, y=roll_corr, fill="tozeroy",
        fillcolor="rgba(88,166,255,0.15)", line=dict(color=C["benchmark"], width=2), name="60d Correlation"))
    fig4.add_hline(y=roll_corr.mean(), line=dict(color=C["neutral"], dash="dash"),
        annotation_text=f"Mean {roll_corr.mean():.2f}")
    fig4.update_yaxes(range=[-1, 1])
    fig4.update_layout(**plotly_layout(300, f"Rolling Correlation 60d with {bm_name}"))
    c4.plotly_chart(fig4, use_container_width=True)

# Tab 4: Periodi
def tab_periodi(db, bm_name, rf):
    periods = split_periods(db)
    period_names = [k for k in periods if k != "Total"] + ["Total"]

    cols = st.columns(len(period_names))
    for col, pname in zip(cols, period_names):
        sub = periods[pname]
        port_r = sub["port_ret"].dropna()
        bm_r = sub["bm_ret"].dropna()
        cum_p = (1+port_r).prod()-1
        sh_p = (port_r.mean()*252-rf)/(port_r.std()*np.sqrt(252)) if port_r.std()>0 else 0
        alpha = (port_r.mean()-bm_r.mean())*252
        pf, pc = color_val(cum_p)
        af, ac = color_val(alpha)
        col.markdown(f'''<div class="metric-card">
            <div class="metric-label">{pname}</div>
            <div class="metric-value {pc}">{pf}</div>
            <div class="metric-sub">Sharpe: {sh_p:.2f}</div>
            <div class="metric-sub"><span class="{ac}">Alpha: {af}</span></div>
        </div>''', unsafe_allow_html=True)

    st.markdown("---")
    n = len(period_names)
    fig = make_subplots(rows=1, cols=n, subplot_titles=period_names, shared_yaxes=False)
    pc_list = [C["benchmark"], C["portfolio"], C["warning"], C["neutral"]]
    for idx, pname in enumerate(period_names):
        sub = periods[pname]
        fig.add_trace(go.Scatter(x=sub.index, y=sub["bm_cum"]*100,
            line=dict(color=C["benchmark"], width=1.5, dash="dash"),
            name=bm_name if idx==0 else None, showlegend=(idx==0)), row=1, col=idx+1)
        fig.add_trace(go.Scatter(x=sub.index, y=sub["port_cum"]*100,
            line=dict(color=pc_list[idx%len(pc_list)], width=2),
            name="Portfolio" if idx==0 else None, showlegend=(idx==0)), row=1, col=idx+1)
    fig.update_layout(height=350, paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
        font=dict(family="Syne, sans-serif", size=10, color="#E6EDF3"),
        margin=dict(t=40,b=40,l=50,r=20), hovermode="x unified")
    fig.update_yaxes(ticksuffix="%", gridcolor="#21262D")
    fig.update_xaxes(gridcolor="#21262D")
    st.plotly_chart(fig, use_container_width=True)

    monthly = db["port_ret"].resample("ME").apply(lambda x: (1+x).prod()-1)*100
    df_m = monthly.to_frame("ret")
    df_m["year"] = df_m.index.year
    df_m["month"] = df_m.index.month
    years = sorted(df_m["year"].unique())
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    z_vals, t_vals = [], []
    for yr in years:
        rz, rt = [], []
        for mo in range(1,13):
            sel = df_m[(df_m["year"]==yr)&(df_m["month"]==mo)]
            if len(sel):
                v = sel["ret"].iloc[0]; rz.append(v); rt.append(f"{v:+.1f}%")
            else:
                rz.append(None); rt.append("")
        z_vals.append(rz); t_vals.append(rt)
    flat = [v for row in z_vals for v in row if v is not None]
    vmax = max(abs(max(flat)), abs(min(flat)), 3) if flat else 3
    fig_heat = go.Figure(go.Heatmap(
        z=z_vals, x=month_labels, y=[str(y) for y in years],
        colorscale=[[0,C["danger"]],[0.5,"#161B22"],[1,C["portfolio"]]],
        zmin=-vmax, zmax=vmax, text=t_vals, texttemplate="%{text}",
        textfont=dict(size=10), colorbar=dict(ticksuffix="%"), hoverongaps=False))
    fig_heat.update_layout(height=max(200,80*len(years)), paper_bgcolor=C["surface"],
        font=dict(family="Syne, sans-serif", size=11, color="#E6EDF3"),
        margin=dict(t=40,b=40,l=60,r=20),
        title=dict(text="<b>Monthly Returns Heatmap (%)</b>", font=dict(size=13), x=0.02))
    st.plotly_chart(fig_heat, use_container_width=True)

# Tab 5: Monte Carlo
def tab_montecarlo(db, etf_cols, rf):
    if len(etf_cols) < 2:
        st.info("At least 2 assets required for Monte Carlo optimisation.")
        return

    n_sims = st.slider("Number of simulations", 1000, 20000, 5000, 1000)

    st.markdown("**Per-asset weight constraints**")
    st.caption("Set the minimum and maximum weight for each asset. Current weights are shown for reference.")

    # Compute current weights
    prices_ref = db[etf_cols].replace(0, np.nan).ffill().bfill()
    last_ref = prices_ref.iloc[-1]
    w_ref = (last_ref / last_ref.sum() * 100).round(1)

    bounds = {}
    n_per_row = 3
    asset_chunks = [etf_cols[i:i+n_per_row] for i in range(0, len(etf_cols), n_per_row)]
    for chunk in asset_chunks:
        ui_cols = st.columns(n_per_row)
        for ui_col, asset in zip(ui_cols, chunk):
            curr_w = int(w_ref[asset])
            with ui_col:
                st.markdown(f"**{asset}**")
                st.caption(f"Current: {curr_w}%")
                mn = st.number_input("Min %", 0, 95, 0, 5, key=f"mn_{asset}")
                mx = st.number_input("Max %", mn, 100, min(max(curr_w + 10, 20), 100), 5, key=f"mx_{asset}")
                bounds[asset] = (mn / 100, mx / 100)

    sum_mins = sum(v[0] for v in bounds.values())
    sum_maxs = sum(v[1] for v in bounds.values())
    if sum_mins > 1.0:
        st.error(f"Sum of minimums = {sum_mins*100:.0f}% > 100%. Reduce the minimums.")
        return
    if sum_maxs < 1.0:
        st.error(f"Sum of maximums = {sum_maxs*100:.0f}% < 100%. Increase the maximums.")
        return

    if not st.button("Run simulation"):
        return

    with st.spinner("Running simulation..."):
        prices = db[etf_cols].replace(0, np.nan).ffill().bfill()
        returns = prices.pct_change().replace([np.inf,-np.inf], np.nan).dropna()
        ret_mat = returns.values
        N = ret_mat.shape[1]
        last = prices.iloc[-1]
        w_curr = (last/last.sum()).values

        min_w = np.array([bounds[a][0] for a in etf_cols])
        max_w = np.array([bounds[a][1] for a in etf_cols])

        def dirichlet_capped(n):
            for _ in range(500):
                g = -np.log(np.maximum(np.random.random(n), 1e-15))
                w = g / g.sum()
                # Scale into [min_w, max_w] range
                w = min_w + w * (max_w - min_w)
                w = np.clip(w, min_w, max_w)
                w = w / w.sum()
                if np.all(w >= min_w - 1e-6) and np.all(w <= max_w + 1e-6):
                    return w
            return np.clip(w, min_w, max_w) / np.clip(w, min_w, max_w).sum()

        def ptf_stats(w):
            daily = ret_mat @ w
            ret = float(daily.mean()*252)
            vol = float(daily.std()*np.sqrt(252))
            cum = np.cumprod(1+daily)
            mdd = float(np.min(cum/np.maximum.accumulate(cum))-1)
            ca = ret/abs(mdd) if mdd < -1e-6 else 99
            return ret, vol, float((ret-rf)/vol if vol>0 else -99), ca

        results = []
        best_sh = {"sharpe": -99}
        best_ca = {"calmar": -99}

        for seed in [42, 99]:
            np.random.seed(seed)
            for _ in range(n_sims):
                w = dirichlet_capped(N)
                ret, vol, sh, ca = ptf_stats(w)
                entry = {"ret":ret,"vol":vol,"sharpe":sh,"calmar":ca,"w":w}
                results.append(entry)
                if seed==42 and sh>best_sh["sharpe"]: best_sh=entry
                if seed==99 and ca>best_ca["calmar"]: best_ca=entry

        cr, cv, csh, cca = ptf_stats(w_curr)
        curr = {"ret":cr,"vol":cv,"sharpe":csh,"calmar":cca,"w":w_curr}

        sh_vals = [r["sharpe"] for r in results]
        sh_min, sh_max = min(sh_vals), max(sh_vals)
        norm = [(s-sh_min)/(sh_max-sh_min+1e-9) for s in sh_vals]
        step = max(1, len(results)//3000)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[r["vol"]*100 for r in results[::step]],
            y=[r["ret"]*100 for r in results[::step]],
            mode="markers",
            marker=dict(size=3, color=[norm[i] for i in range(0,len(results),step)],
                colorscale=[[0,"#1C2B3A"],[0.5,"#1F6FEB"],[1,"#3FB950"]],
                showscale=True, colorbar=dict(title="Sharpe",len=0.5,tickformat=".2f")),
            name="Simulated portfolios",
            hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"))

        for entry, label, color, symbol in [
            (curr, "Current",    "#8B949E", "circle"),
            (best_sh, "Max Sharpe", "#3FB950", "star"),
            (best_ca, "Max Calmar", "#F8B73E", "diamond"),
        ]:
            fig.add_trace(go.Scatter(
                x=[entry["vol"]*100], y=[entry["ret"]*100],
                mode="markers+text",
                marker=dict(size=15, color=color, symbol=symbol, line=dict(width=2,color="white")),
                text=[label], textposition="top right", textfont=dict(size=10,color=color),
                name=f"{label} (Sh:{entry['sharpe']:.2f} Ca:{entry['calmar']:.2f})",
                hovertemplate=(f"{label}<br>Ret: {entry['ret']*100:.2f}%<br>"
                               f"Vol: {entry['vol']*100:.2f}%<br>"
                               f"Sharpe: {entry['sharpe']:.3f}<br>"
                               f"Calmar: {entry['calmar']:.3f}<extra></extra>")))

        fig.update_layout(height=520, paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
            font=dict(family="Syne, sans-serif", size=10, color="#E6EDF3"),
            margin=dict(t=50,b=60,l=60,r=20),
            title=dict(text=f"<b>Efficient Frontier — {len(results):,} simulations</b>", font=dict(size=14), x=0.02),
            xaxis=dict(title="Volatility (%)", ticksuffix="%", gridcolor="#21262D"),
            yaxis=dict(title="Return (%)", ticksuffix="%", gridcolor="#21262D"),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Optimal allocation**")
        rows = []
        for i, col in enumerate(etf_cols):
            rows.append({"Asset":col, "Current":f"{w_curr[i]*100:.1f}%",
                         "Max Sharpe":f"{best_sh['w'][i]*100:.1f}%",
                         "Max Calmar":f"{best_ca['w'][i]*100:.1f}%"})
        rows.append({"Asset":"—","Current":"—","Max Sharpe":"—","Max Calmar":"—"})
        for label, key, is_pct in [
            ("Ann. Return","ret",True),("Ann. Volatility","vol",True),
            ("Sharpe Ratio","sharpe",False),("Calmar Ratio","calmar",False)]:
            rows.append({"Asset":label,
                "Current":   f"{curr[key]*100:.2f}%"    if is_pct else f"{curr[key]:.3f}",
                "Max Sharpe":f"{best_sh[key]*100:.2f}%" if is_pct else f"{best_sh[key]:.3f}",
                "Max Calmar":f"{best_ca[key]*100:.2f}%" if is_pct else f"{best_ca[key]:.3f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# MAIN
def main():
    pwd = st.text_input("Enter access password", type="password")
    if pwd != st.secrets["password"]:
        st.info("Enter the password to access the app. Request access via LinkedIn.")
        st.stop()
    with st.sidebar:
        st.markdown("# Portfolio Analyzer")
        st.markdown("---")
        uploaded = st.file_uploader("Upload your file", type=["xlsx","xls","csv"],
            help="Excel or CSV with dates and asset values")

        if uploaded is None:
            st.markdown("""
**How to use:**
1. Upload an Excel or CSV file
2. Map the columns
3. Choose a benchmark

**Expected format:**
- One date column
- One or more columns with asset values
            """)
            st.stop()

        try:
            preview = pd.read_csv(uploaded, nrows=3) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded, nrows=3)
            uploaded.seek(0)
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

        all_cols = preview.columns.tolist()
        st.markdown("**Map columns**")
        date_col   = st.selectbox("Date column", all_cols, index=0)
        value_cols = st.multiselect("Asset columns",
            [c for c in all_cols if c != date_col],
            default=[c for c in all_cols if c != date_col])
        if not value_cols:
            st.warning("Select at least one asset column."); st.stop()

        sheet_name = None
        if uploaded.name.endswith((".xlsx",".xls")):
            try:
                xf = pd.ExcelFile(uploaded)
                sheet_name = st.selectbox("Excel sheet", xf.sheet_names) if len(xf.sheet_names)>1 else xf.sheet_names[0]
                uploaded.seek(0)
            except: pass

    # Selezione benchmark e RF nella pagina principale
    st.markdown("## Choose Benchmark")
    st.markdown("Seleziona l'indice di riferimento per il tuo portafoglio. Il tasso risk-free viene scaricato automaticamente.")

    bm_keys = list(BENCHMARKS.keys())
    n_cols = 5
    rows_bm = [bm_keys[i:i+n_cols] for i in range(0, len(bm_keys), n_cols)]

    if "selected_bm" not in st.session_state:
        st.session_state.selected_bm = "S&P 500"

    for row in rows_bm:
        cols_bm = st.columns(len(row))
        for col_ui, bm_key in zip(cols_bm, row):
            bm = BENCHMARKS[bm_key]
            is_sel = st.session_state.selected_bm == bm_key
            border = "#3FB950" if is_sel else "#21262D"
            bg = "#1a2e1a" if is_sel else "#161B22"
            if col_ui.button(bm_key, key=f"bm_{bm_key}",
                             use_container_width=True):
                st.session_state.selected_bm = bm_key
                st.rerun()
            # Visual feedback via markdown
            col_ui.markdown(
                f'<div style="background:{bg};border:2px solid {border};border-radius:10px;'
                f'padding:6px;text-align:center;margin-top:-8px;font-size:10px;color:#8B949E;">'
                f'{BENCHMARKS[bm_key]["rf_name"]}</div>',
                unsafe_allow_html=True)

    bm_name = st.session_state.selected_bm
    bm_info = BENCHMARKS[bm_name]
    bm_ticker = bm_info["ticker"]

    # Fetch RF
    with st.spinner(f"Loading {bm_info['rf_name']}..."):
        rf_pct = fetch_rf_rate(bm_info["rf_ticker"], bm_info["rf_fallback"])

    c_rf1, c_rf2, _ = st.columns([1.5, 1.5, 7])
    c_rf1.markdown(
        f'<div class="rf-badge">{bm_info["rf_name"]}<br>'
        f'<span style="font-size:20px;font-weight:700">{rf_pct:.2f}%</span></div>',
        unsafe_allow_html=True)
    rf_override = c_rf2.number_input("Override RF (%)", 0.0, 15.0,
        float(rf_pct), 0.1, key="rf_override",
        help="Manually override if you want a different value")
    rf = rf_override / 100

    st.markdown("---")

    # Load data
    with st.spinner("Loading data..."):
        try:
            raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded, sheet_name=sheet_name or 0)
            raw[date_col] = pd.to_datetime(raw[date_col])
            raw = raw.set_index(date_col).sort_index()
            for c in value_cols:
                raw[c] = pd.to_numeric(raw[c], errors="coerce").replace(0,np.nan).ffill().bfill()
            raw["Total"] = raw[value_cols].sum(axis=1)
            start = raw.index.min().strftime("%Y-%m-%d")
            end = raw.index.max().strftime("%Y-%m-%d")
            bm_raw = load_benchmark(bm_ticker, start, end)
            db, etf_cols = compute_features(raw, bm_raw)
            m = compute_metrics(db, rf)
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

    # Header
    st.markdown(f"""
    <h1 style='margin-bottom:4px'>Portfolio Analysis</h1>
    <p style='color:#8B949E;margin-top:0'>
        {db.index[0].strftime('%d %b %Y')}: {db.index[-1].strftime('%d %b %Y')} &nbsp;·&nbsp;
        {len(db)} trading days &nbsp;·&nbsp;
        Benchmark: <strong style='color:#58A6FF'>{bm_name}</strong> &nbsp;·&nbsp;
        RF: <strong style='color:#3FB950'>{rf:.2%}</strong> ({bm_info["rf_name"]})
    </p>""", unsafe_allow_html=True)

    # Tabs
    t1, t2, t3, t4, t5 = st.tabs(["Performance", "Composition", "Volatility", "Periods", "Optimisation"])
    with t1: tab_performance(db, m, bm_name, rf)
    with t2: tab_composizione(db, etf_cols, m)
    with t3: tab_volatilita(db, bm_name)
    with t4: tab_periodi(db, bm_name, rf)
    with t5: tab_montecarlo(db, etf_cols, rf)

if __name__ == "__main__":
    main()
