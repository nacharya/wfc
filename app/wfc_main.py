import os
import sys
import json
import base64
import datetime as dtt
from os.path import expanduser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import logzero
from logzero import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wfc_config import wfc_config
from wfc_ds import wfc_dgather, wfc_dpredict, wfc_visual, StockNow, StockDataGather
from wfc_news import wfc_feed

CONFIG_PATH = expanduser("~/.wfc")
IMAGES_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

# ----------------------------------------------------------------- page setup

st.set_page_config(
    page_title="WFC — Wealth Forecast",
    page_icon=os.path.join(IMAGES_DIR, 'favicon.ico'),
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


@st.cache_resource
def _init_logging():
    logzero.logfile("wfc.log", maxBytes=1048576, backupCount=3)

_init_logging()


# ------------------------------------------------------------ helpers

@st.cache_data(show_spinner=False)
def _logo_b64():
    path = os.path.join(IMAGES_DIR, 'main.png')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return ""


def load_config():
    if not os.path.isfile(CONFIG_PATH):
        return None
    with open(CONFIG_PATH) as f:
        return wfc_config(f.read())


def save_config(cfg):
    cfg.save(CONFIG_PATH)


def _available_tickers(cfg):
    data_dir = cfg.data_location()
    out = []
    if os.path.isdir(data_dir):
        for d in sorted(os.listdir(data_dir)):
            if os.path.exists(os.path.join(data_dir, d, f"{d}-stock-data.csv")):
                out.append(d)
    return out


# ---------------------------------------------------------------- titlebar

def _render_titlebar():
    """Inject a fixed-position titlebar and push page content below it."""
    logo_b64  = _logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" '
        'style="height:38px;width:auto;display:block">'
        if logo_b64 else "📈"
    )

    st.markdown(f"""
    <style>
    /* ── remove Streamlit chrome ───────────────────────────────── */
    #MainMenu {{ visibility: hidden; }}
    footer    {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ display: none !important; }}

    /* ── push all page content below the fixed titlebar ─────────── */
    .main .block-container {{
        padding-top:   62px  !important;
        padding-left:  1rem  !important;
        padding-right: 1rem  !important;
        max-width:     100%  !important;
    }}

    /* ── fixed titlebar ─────────────────────────────────────────── */
    #wfc-titlebar {{
        position:      fixed;
        top:           0;
        left:          0;
        right:         0;
        height:        54px;
        z-index:       999;
        display:       flex;
        align-items:   center;
        gap:           12px;
        padding:       0 1.5rem;
        background:    linear-gradient(135deg, #0d2137 0%, #1a3558 100%);
        border-bottom: 2px solid #2d6a9f;
        box-shadow:    0 2px 8px rgba(0,0,0,0.4);
    }}
    #wfc-titlebar .wfc-title {{
        color:       white;
        font-size:   1.4em;
        font-weight: 700;
        flex:        1;
        letter-spacing: 0.01em;
    }}
    #wfc-titlebar .wfc-sub {{
        color:     rgba(255,255,255,0.5);
        font-size: 0.82em;
    }}

    /* ── tab bar styling ─────────────────────────────────────────── */
    div[data-testid="stTabs"] {{
        margin-top: 0;
    }}
    div[data-testid="stTabs"] > div:first-child {{
        gap: 4px;
    }}
    div[data-baseweb="tab"] {{
        padding:       10px 20px !important;
        font-size:     0.95em   !important;
        font-weight:   500      !important;
        border-radius: 6px 6px 0 0 !important;
    }}
    </style>

    <div id="wfc-titlebar">
        {logo_html}
        <span class="wfc-title">Wealth Forecast</span>
        <span class="wfc-sub">Portfolio Analysis &amp; Prediction</span>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------- setup page

def show_setup():
    st.title("WFC — First Run Setup")
    st.warning(f"No config file found at `{CONFIG_PATH}`.")
    st.info("Create a minimal config to get started, then add your portfolios and tickers.")
    if st.button("Create default config", type="primary"):
        sample = {
            "basename":      "user",
            "data_location": "data/",
            "pfolios": [{"NAME": "MY_PORTFOLIO", "TICKERS": [], "RELATED": [], "EXCHG": "NSDQ"}],
            "purchased": [],
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(sample, f, indent=4)
        st.success(f"Created {CONFIG_PATH} — reload the page.")
        st.rerun()


# ---------------------------------------------------------------- Portfolio

def show_portfolio(cfg):
    # ── Portfolios & Tickers ─────────────────────────────────────────────
    st.markdown("**Portfolios & Tickers**")
    for pf_name, pf in cfg.pf_dict.items():
        tickers = pf.tickers()
        ticker_str = "  ·  " + ", ".join(tickers) if tickers else "  ·  _no tickers_"
        with st.expander(f"📁 {pf_name}  ({pf.exchg}){ticker_str}", expanded=False):
            ca, cb, cc, cd, ce, cf = st.columns([3, 1, 3, 1, 2, 1])
            new_tk = ca.text_input("Add ticker", key=f"add_tk_{pf_name}",
                                   placeholder="Ticker symbol",
                                   label_visibility="collapsed").upper().strip()
            cb.markdown("<br>", unsafe_allow_html=True)
            if cb.button("Add", key=f"add_btn_{pf_name}") and new_tk:
                cfg.ticker_add(pf_name, new_tk)
                save_config(cfg)
                st.rerun()

            del_tk = cc.selectbox("Remove ticker", [""] + tickers, key=f"del_sel_{pf_name}",
                                  label_visibility="collapsed")
            cd.markdown("<br>", unsafe_allow_html=True)
            if cd.button("Remove", key=f"del_btn_{pf_name}") and del_tk:
                cfg.ticker_del(pf_name, del_tk)
                save_config(cfg)
                st.rerun()

            cf.markdown("<br>", unsafe_allow_html=True)
            if cf.button("🗑️", key=f"del_pf_{pf_name}",
                         help=f"Delete portfolio {pf_name}"):
                cfg.portfolio_del(pf_name)
                save_config(cfg)
                st.rerun()

    # Add new portfolio
    with st.expander("➕  Add portfolio", expanded=False):
        pa, pb, pc, pd_ = st.columns([3, 2, 1, 4])
        new_pf   = pa.text_input("Name", key="new_pf_name",
                                 placeholder="Portfolio name",
                                 label_visibility="collapsed").upper().strip()
        new_exch = pb.text_input("Exchange", key="new_pf_exchg",
                                 placeholder="Exchange (e.g. NSDQ)",
                                 label_visibility="collapsed").upper().strip() or "NSDQ"
        pc.markdown("<br>", unsafe_allow_html=True)
        if pc.button("Add", key="add_pf_btn") and new_pf:
            if cfg.portfolio_add(new_pf, new_exch):
                save_config(cfg)
                st.rerun()
            else:
                pd_.warning(f"'{new_pf}' already exists.")

    st.divider()

    # ── Portfolio selector for owned / add-remove sections ───────────────
    pf_names = list(cfg.pf_dict.keys())
    sel_pf   = st.selectbox("Portfolio", pf_names, key="pf_owned_sel",
                             label_visibility="collapsed") if pf_names else None

    if sel_pf:
        pf_tickers = cfg.pf(sel_pf).tickers()

        # ── Owned Positions ──────────────────────────────────────────────
        st.markdown("**Owned Positions**")
        pf_owned = [o for o in cfg.purchased if o['ticker'] in pf_tickers]
        if pf_owned:
            for owned in pf_owned:
                tkr     = owned['ticker']
                current = StockNow(tkr).get('Close')
                label   = f"📈 {tkr}" + (f"  —  ${current:.2f}" if current else "")
                with st.expander(label, expanded=False):
                    total_cost, total_shares = 0.0, 0
                    for purchase in owned['ticker_purchase']:
                        for date, details in purchase.items():
                            count, price  = details[0], details[1]
                            cost          = count * price
                            total_cost   += cost
                            total_shares += count
                            st.write(f"  {date}: {count} shares @ ${price:.2f}  (cost ${cost:,.0f})")
                    if current and total_shares > 0:
                        mkt  = current * total_shares
                        gain = mkt - total_cost
                        st.metric("Market Value", f"${mkt:,.0f}",
                                  f"{gain:+,.0f} ({gain/total_cost*100:+.1f}%)")
        else:
            st.info(f"No owned positions in {sel_pf}.")

        st.divider()

        # ── Add / Remove Positions ───────────────────────────────────────
        st.markdown("**Add / Remove Positions**")
        col_add, col_del = st.columns(2)

        with col_add:
            st.caption("Add a new position")
            with st.form("add_owned_form"):
                tk_options = [""] + pf_tickers
                tk = st.selectbox("Ticker", tk_options, key="add_pos_ticker") if pf_tickers else st.text_input("Ticker", key="add_pos_ticker_txt")
                if isinstance(tk, str):
                    tk = tk.upper().strip()
                date  = st.date_input("Purchase date", value=dtt.date.today())
                count = st.number_input("Shares", min_value=1, value=100, step=1)
                price = st.number_input("Price paid per share ($)", min_value=0.01,
                                        value=100.0, step=0.01)
                if st.form_submit_button("Add Position") and tk:
                    cfg.owned_add(tk, str(date), str(count), str(price))
                    save_config(cfg)
                    st.success(f"Added {count} × {tk} @ ${price:.2f}")
                    st.rerun()

        with col_del:
            st.caption("Remove an existing position")
            owned_tickers = [o['ticker'] for o in pf_owned]
            if owned_tickers:
                with st.form("del_owned_form"):
                    del_ticker = st.selectbox("Ticker", owned_tickers, key="del_pos_ticker")
                    dates = []
                    for x in cfg.purchased:
                        if x['ticker'] == del_ticker:
                            for p in x['ticker_purchase']:
                                dates.extend(p.keys())
                    del_date = st.selectbox("Purchase date", dates, key="del_pos_date") if dates else None
                    if st.form_submit_button("Remove") and del_date:
                        cfg.owned_del(del_ticker, del_date)
                        save_config(cfg)
                        st.success(f"Removed {del_ticker} position from {del_date}")
                        st.rerun()
            else:
                st.info(f"No positions to remove in {sel_pf}.")


# ---------------------------------------------------------------- Gather Data

def show_gather(cfg):
    st.header("Gather Stock Data")
    col_form, col_data = st.columns([1, 2])

    with col_form:
        pf_names = list(cfg.pf_dict.keys())
        pf_sel   = st.selectbox("Portfolio", pf_names)

        default_end   = dtt.date.today()
        default_start = default_end - dtt.timedelta(days=365 * 2)
        start = st.date_input("Start date", value=default_start)
        end   = st.date_input("End date",   value=default_end)
        force = st.checkbox("Force re-download", value=False)

        if st.button("Gather Data", type="primary"):
            tickers = cfg.pf(pf_sel).tickers()
            if not tickers:
                st.error("No tickers in this portfolio. Add some first.")
            else:
                progress = st.progress(0)
                status   = st.empty()
                ok, fail = [], []
                for i, tkr in enumerate(tickers):
                    status.text(f"Downloading {tkr} …")
                    sdg = StockDataGather(tkr, str(start), str(end), cfg.data_location())
                    (ok if sdg.gather(force=force) else fail).append(tkr)
                    progress.progress((i + 1) / len(tickers))
                status.empty()
                if ok:
                    st.success(f"Downloaded: {', '.join(ok)}")
                if fail:
                    st.warning(f"Failed: {', '.join(fail)}")

    with col_data:
        st.subheader("Available Data")
        rows     = []
        data_dir = cfg.data_location()
        if os.path.isdir(data_dir):
            for d in sorted(os.listdir(data_dir)):
                csv = os.path.join(data_dir, d, f"{d}-stock-data.csv")
                if os.path.exists(csv):
                    try:
                        df = pd.read_csv(csv, usecols=['Date', 'Close'], parse_dates=['Date'])
                        rows.append({
                            'Ticker':       d,
                            'From':         str(df['Date'].min().date()),
                            'To':           str(df['Date'].max().date()),
                            'Trading days': len(df),
                            'Last Close':   f"${df['Close'].iloc[-1]:.2f}",
                        })
                    except Exception:
                        pass
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.info("No data gathered yet.")


# ---------------------------------------------------------------- Predict

def show_predict(cfg):
    st.header("Price Prediction")
    available = _available_tickers(cfg)

    col_form, col_result = st.columns([1, 2])
    with col_form:
        ticker = (st.selectbox("Ticker", available, key="predict_ticker")
                  if available else
                  st.text_input("Ticker (gather data first)", key="predict_ticker_txt").upper().strip())

        mode = st.radio("Future target", ["Specific date", "Days from today"])
        if mode == "Specific date":
            future_str = str(st.date_input(
                "Future date", value=dtt.date.today() + dtt.timedelta(days=14)))
        else:
            days       = st.slider("Days from today", 1, 180, 14)
            future_str = str(dtt.date.today() + dtt.timedelta(days=days))
            st.caption(f"Predicting for: **{future_str}**")

        predict_btn = st.button("Run Prediction", type="primary")

    with col_result:
        if predict_btn and ticker:
            with st.spinner(f"Running {ticker} predictions …"):
                try:
                    results = wfc_dpredict(cfg, ticker, future_str).predict()
                    if not results:
                        st.error("Prediction failed — ensure data has been gathered.")
                        return

                    current = StockNow(ticker).get('Close')
                    if current:
                        st.metric("Current Price", f"${current:.2f}")

                    rows = []
                    for algo in ["linear", "gbr", "rfr"]:
                        if algo not in results:
                            continue
                        r   = results[algo]
                        row = {'Algorithm': r['label'],
                               f'Predicted ({future_str})': f"${r['price']:.2f}"}
                        if current:
                            row['Change'] = f"{(r['price'] - current) / current * 100:+.1f}%"
                        row['R²']   = f"{r['r2']:.3f}"
                        row['RMSE'] = f"${r['rmse']:.2f}"
                        row['MAE']  = f"${r['mae']:.2f}"
                        rows.append(row)

                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

                    if 'ensemble' in results:
                        ens   = results['ensemble']['price']
                        delta = f"{(ens - current) / current * 100:+.1f}%" if current else None
                        st.metric(f"Ensemble — {future_str}", f"${ens:.2f}", delta)

                except FileNotFoundError:
                    st.error(f"No data for **{ticker}**. Go to 'Gather Data' first.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception(e)


# ---------------------------------------------------------------- News

def _sentiment_badge(compound):
    if compound >= 0.05:
        bg, label = "#27ae60", "POSITIVE"
    elif compound <= -0.05:
        bg, label = "#e74c3c", "NEGATIVE"
    else:
        bg, label = "#7f8c8d", "NEUTRAL"
    return (f'<span style="background:{bg};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.75em;font-weight:bold">{label}</span>')


def _render_article(a):
    badge      = _sentiment_badge(a['compound'])
    title_link = (f'<a href="{a["url"]}" target="_blank">{a["title"]}</a>'
                  if a['url'] else a['title'])
    meta       = (f"<small>{a['publisher']}"
                  + (f" · {a['published']}" if a['published'] else "")
                  + "</small>")
    st.markdown(f"{badge} &nbsp; {title_link}<br>{meta}", unsafe_allow_html=True)
    if a['summary'] and a['summary'] != a['title']:
        st.caption(a['summary'])
    st.markdown(
        f"<small>compound: <b>{a['compound']:+.3f}</b> &nbsp;|&nbsp; "
        f"+{a['pos']:.2f} / -{a['neg']:.2f} / ~{a['neu']:.2f}</small>",
        unsafe_allow_html=True,
    )
    st.divider()


def show_news(cfg):
    st.header("News & Sentiment")
    owned_tickers = [x['ticker'] for x in cfg.purchased]

    col_form, col_news = st.columns([1, 3])
    with col_form:
        choices  = ["All owned"] + owned_tickers + ["Custom ticker…"]
        selected = st.selectbox("Ticker", choices, key="news_ticker_sel")
        if selected == "Custom ticker…":
            custom           = st.text_input("Enter ticker").upper().strip()
            tickers_to_fetch = [custom] if custom else []
        elif selected == "All owned":
            tickers_to_fetch = owned_tickers
        else:
            tickers_to_fetch = [selected]

        sort_by   = st.radio("Sort articles by", ["Date (newest first)", "Compound score"])
        max_show  = st.slider("Max articles to show", 5, 50, 20)
        fetch_btn = st.button("Fetch News", type="primary")

    with col_news:
        if fetch_btn and tickers_to_fetch:
            for tkr in tickers_to_fetch:
                with st.spinner(f"Fetching news for {tkr} …"):
                    feed = wfc_feed(tkr)
                    feed.get_news()

                st.subheader(f"{tkr} — {len(feed.articles)} articles")
                if not feed.articles:
                    st.warning("No news articles found.")
                    continue

                pos, neg, neu = feed.get_pos(), feed.get_neg(), feed.get_neu()
                overall       = feed.overall_sentiment()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Overall Sentiment", overall.upper())
                c2.metric("Positive score",    f"{pos:.2f}")
                c3.metric("Negative score",    f"{neg:.2f}")
                c4.metric("Neutral score",     f"{neu:.2f}")

                fig, ax = plt.subplots(figsize=(5, 2))
                ax.bar(['Positive', 'Negative', 'Neutral'], [pos, neg, neu],
                       color=['#27ae60', '#e74c3c', '#7f8c8d'])
                ax.set_title(f"{tkr} Sentiment Distribution")
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                articles_sorted = (
                    sorted(feed.articles, key=lambda a: a['compound'], reverse=True)
                    if sort_by == "Compound score" else feed.articles
                )

                tab_all, tab_pos, tab_neg = st.tabs(
                    ["All Articles", "Top Positive", "Top Negative"])
                with tab_all:
                    for a in articles_sorted[:max_show]:
                        _render_article(a)
                with tab_pos:
                    for a in feed.get_headlines('positive', max_show):
                        _render_article(a)
                with tab_neg:
                    for a in feed.get_headlines('negative', max_show):
                        _render_article(a)

                st.divider()


# ---------------------------------------------------------------- Charts

def show_charts(cfg):
    st.header("Stock Charts")
    available = _available_tickers(cfg)

    if not available:
        st.warning("No data available. Use 'Gather Data' first.")
        return

    col_ctrl, col_chart = st.columns([1, 4])
    with col_ctrl:
        ticker     = st.selectbox("Ticker", available, key="charts_ticker")
        chart_type = st.radio("Chart type",
                               ["Closing Price", "Volume", "Daily % Change"])
        show_btn   = st.button("Show Chart", type="primary")

    with col_chart:
        if show_btn:
            cmd_map = {"Closing Price": "closing",
                       "Volume": "volume",
                       "Daily % Change": "daily"}
            fig = wfc_visual(cfg, ticker).plot(cmd_map[chart_type])
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.error("Could not generate chart.")


# ---------------------------------------------------------------- main

def main():
    _render_titlebar()

    cfg = load_config()
    if cfg is None:
        show_setup()
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio",
        "📥 Gather Data",
        "🔮 Predict",
        "📰 News & Sentiment",
        "📈 Charts",
    ])
    with tab1: show_portfolio(cfg)
    with tab2: show_gather(cfg)
    with tab3: show_predict(cfg)
    with tab4: show_news(cfg)
    with tab5: show_charts(cfg)


main()
