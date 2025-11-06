# app.py
"""
Indian Stock Analyzer - NSE preferred, Finnhub fallback, Yahoo final fallback.
Optional OpenAI enrichment for parsing/sanity-checking raw responses.

Install these (requirements.txt):
streamlit
pandas
yfinance
python-dotenv
requests
beautifulsoup4
newsapi-python==0.2.7
lxml
openai
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import requests
import re
import json

# Optional: OpenAI for parsing/enrichment
try:
    import openai
except Exception:
    openai = None

# NewsAPI
try:
    from newsapi import NewsApiClient
except Exception:
    NewsApiClient = None

# Load env if running locally (not used on Streamlit Cloud when using st.secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Indian Stock Analyzer (NSE->Finnhub->Yahoo)", page_icon="📈", layout="wide")

# ------------------------ Utils ------------------------
def get_secret(key):
    """Read secret from Streamlit secrets or environment (TOML vs .env compatibility)"""
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

def to_number(x):
    """Robust conversion of heterogeneous inputs to float (handles commas, parentheses, currency tokens)."""
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s in {"-", "—", "None", "nan", "NaN"}:
            return 0.0
        # remove common tokens
        for token in ["₹", "Rs.", "Rs", "rs", "Cr", "L", "lakhs", "crore", "INR", ","]:
            s = s.replace(token, "")
        negative = False
        if s.startswith("(") and s.endswith(")"):
            negative = True
            s = s[1:-1]
        s = re.sub(r"[^\d\.\-eE]", "", s)
        if s in ("", "-", "."):
            return 0.0
        val = float(s)
        return -val if negative else val
    except Exception:
        return 0.0

# ------------------------ NSE helper ------------------------
def fetch_nse_session():
    """Session with headers and cookie handshake for NSE endpoints (best-effort)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com"
    }
    s = requests.Session()
    s.headers.update(headers)
    try:
        s.get("https://www.nseindia.com", timeout=5)
    except Exception:
        pass
    return s

@st.cache_data(ttl=900)
def get_financial_data_nse(symbol):
    """
    Try NSE JSON endpoint for 'results' which contains quarterly results.
    Returns dict with keys: income_statement (DataFrame metrics x quarters), balance_sheet (None), cashflow (None), last_updated, source
    """
    stock = symbol.replace(".NS", "").replace(".BO", "")
    session = fetch_nse_session()
    try:
        url = f"https://www.nseindia.com/api/results-equity?symbol={stock}"
        r = session.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        qdata = j.get("data", []) or []
        if not qdata:
            return None

        rows = []
        for i, q in enumerate(qdata[:8]):
            label = q.get("quarter") or q.get("period") or q.get("fiscalPeriod") or q.get("resultDate") or f"Q{i+1}"
            total_income = to_number(q.get("totalIncome") or q.get("totalRevenue") or q.get("revenue") or q.get("Total Income"))
            gross_profit = to_number(q.get("grossProfit") or q.get("grossProfitLoss") or q.get("Gross Profit"))
            operating_income = to_number(q.get("operatingProfit") or q.get("operatingProfitLoss") or q.get("Operating Income"))
            ebitda = to_number(q.get("ebitda") or q.get("EBITDA"))
            net_profit = to_number(q.get("netProfit") or q.get("profitAfterTax") or q.get("Net Income"))
            rows.append({
                "Quarter": label,
                "Total Revenue": total_income,
                "Gross Profit": gross_profit,
                "Operating Income": operating_income,
                "EBITDA": ebitda,
                "Net Income": net_profit
            })

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("Quarter").T  # metrics x quarters
        return {"income_statement": df, "balance_sheet": None, "cashflow": None,
                "last_updated": str(df.columns[0]) if df.shape[1] > 0 else None, "source": "NSE"}
    except Exception:
        return None

# ------------------------ Finnhub integration ------------------------
@st.cache_data(ttl=1800)
def fetch_financials_finnhub(symbol):
    """
    Use Finnhub's 'financials-reported' endpoint as a robust fallback.
    Returns dict similar to NSE function or (None) on failure.
    """
    key = get_secret("FINNHUB_API_KEY")
    if not key:
        return None
    base = "https://finnhub.io/api/v1"
    # Accept either SYMBOL or NSE:SYMBOL forms; for Indian tickers try without suffix
    short = symbol.replace(".NS", "").replace(".BO", "")
    url = f"{base}/stock/financials-reported?symbol={short}&token={key}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        jd = r.json()
        rows = []
        data_list = jd.get("data") or jd.get("result") or []
        for entry in data_list[:8]:
            period = entry.get("reportDate") or entry.get("period") or entry.get("fiscalDate") or entry.get("filingDate") or entry.get("date")
            report = entry.get("report", entry) if isinstance(entry, dict) else entry
            total_revenue = to_number(report.get("totalRevenue") or report.get("revenue") or report.get("Total Revenue") or report.get("total_revenue"))
            ebitda = to_number(report.get("ebitda") or report.get("EBITDA"))
            net_income = to_number(report.get("netIncome") or report.get("net_profit") or report.get("profitAfterTax"))
            rows.append({"Quarter": period or "NA", "Total Revenue": total_revenue, "EBITDA": ebitda, "Net Income": net_income})
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("Quarter").T
        return {"income_statement": df, "balance_sheet": None, "cashflow": None,
                "last_updated": str(df.columns[0]) if df.shape[1] > 0 else None, "source": "Finnhub"}
    except Exception:
        return None

# ------------------------ Optional OpenAI parsing/enrichment ------------------------
def openai_summarize_financials(raw_obj):
    """
    Optional: call OpenAI to normalize messy JSON into an array of quarter objects.
    Returns parsed python object or None. This is only enrichment/sanity-check — not a data source.
    """
    key = get_secret("OPENAI_API_KEY")
    if not key or openai is None:
        return None
    try:
        openai.api_key = key
        prompt = f"""
You are a precise JSON extractor. Given this raw JSON: {json.dumps(raw_obj)[:5000]}
Return only valid JSON array of objects. Each object should contain these keys:
'quarter' (label or date string), 'total_revenue' (number or null), 'ebitda' (number or null), 'net_income' (number or null).
Sanitize thousands separators and parentheses, e.g. "(1,234)" -> -1234.
Return only JSON.
"""
        # Use ChatCompletion (Chat API) with a deterministic temperature
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # use the model you have access to; adjust if necessary
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.0
        )
        out = resp["choices"][0]["message"]["content"].strip()
        # Try extract JSON substring
        m = re.search(r"(\[.*\])", out, re.S)
        json_text = m.group(1) if m else out
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        return None

# ------------------------ Yahoo fallback ------------------------
@st.cache_data(ttl=1800)
def get_financial_data_yf_fallback(symbol):
    """Use yfinance as final fallback. Returns same shaped dict as other fetchers."""
    try:
        ticker = yf.Ticker(symbol)
        # yfinance attributes vary; try several
        quarterly_income = getattr(ticker, "quarterly_income_stmt", None) or getattr(ticker, "quarterly_financials", None)
        quarterly_balance = getattr(ticker, "quarterly_balance_sheet", None) or getattr(ticker, "quarterly_balance", None)
        quarterly_cashflow = getattr(ticker, "quarterly_cashflow", None)
        income_t = quarterly_income.T if (quarterly_income is not None and not getattr(quarterly_income, "empty", True)) else None
        balance_t = quarterly_balance.T if (quarterly_balance is not None and not getattr(quarterly_balance, "empty", True)) else None
        cash_t = quarterly_cashflow.T if (quarterly_cashflow is not None and not getattr(quarterly_cashflow, "empty", True)) else None
        data_date = None
        try:
            if income_t is not None:
                data_date = income_t.columns[0].strftime("%Y-%m-%d") if hasattr(income_t.columns[0], "strftime") else str(income_t.columns[0])
        except Exception:
            data_date = None
        return {"income_statement": income_t, "balance_sheet": balance_t, "cashflow": cash_t, "last_updated": data_date, "source": "Yahoo"}, None
    except Exception as e:
        return None, str(e)

# ------------------------ Unified fetcher: NSE -> Finnhub -> Yahoo ------------------------
@st.cache_data(ttl=1800)
def get_financials_unified(symbol):
    """
    Try NSE -> Finnhub -> Yahoo. Return (data_dict, error_or_None)
    """
    # 1) NSE for .NS/.BO
    try:
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            n = get_financial_data_nse(symbol)
            if n:
                return n, None
    except Exception:
        pass

    # 2) Finnhub if key present
    try:
        f = fetch_financials_finnhub(symbol)
        if f:
            return f, None
    except Exception:
        pass

    # 3) Yahoo fallback
    try:
        yf_res, err = get_financial_data_yf_fallback(symbol)
        if yf_res:
            return yf_res, None
        return None, err or "No financials found"
    except Exception as e:
        return None, str(e)

# ------------------------ Metrics calculation (defensive) ------------------------
def calculate_metrics_with_warning(financials):
    metrics = {}
    warnings = []
    inc = financials.get("income_statement") if financials else None

    def get_latest_prior(df):
        if df is None:
            return None, None
        # if columns represent quarters (common in our shape)
        try:
            if df.shape[1] >= 1:
                latest = df.iloc[:, 0]
                prior = df.iloc[:, 1] if df.shape[1] > 1 else None
                return latest, prior
        except Exception:
            pass
        try:
            # rows could be quarters
            if df.shape[0] >= 1:
                latest = df.iloc[-1]
                prior = df.iloc[-2] if df.shape[0] > 1 else None
                return latest, prior
        except Exception:
            pass
        return None, None

    latest, prior = get_latest_prior(inc)
    if latest is not None:
        metrics["Quarter Date"] = financials.get("last_updated", "Unknown")
        def safe_val(series, name_candidates):
            if series is None:
                return 0.0
            for n in name_candidates:
                if n in series.index:
                    return to_number(series.get(n))
            try:
                return to_number(series.iloc[0])
            except Exception:
                return 0.0

        revenue_latest = safe_val(latest, ["Total Revenue", "totalIncome", "Revenue", "TotalRevenue", "total_revenue"])
        metrics["Revenue (Latest)"] = revenue_latest

        if prior is not None:
            revenue_prev = safe_val(prior, ["Total Revenue", "totalIncome", "Revenue"])
            metrics["Revenue (YoY)"] = revenue_prev
            if revenue_prev and abs(revenue_prev) > 1e-9:
                metrics["Revenue Growth %"] = ((revenue_latest - revenue_prev) / abs(revenue_prev)) * 100
            else:
                metrics["Revenue Growth %"] = None
        else:
            metrics["Revenue (YoY)"] = 0
            metrics["Revenue Growth %"] = None

        gp = safe_val(latest, ["Gross Profit", "grossProfit"])
        metrics["Gross Profit"] = gp
        metrics["Gross Margin %"] = (gp / revenue_latest * 100) if revenue_latest and abs(revenue_latest) > 1e-9 else None

        op = safe_val(latest, ["Operating Income", "operatingProfit"])
        metrics["Operating Income"] = op
        metrics["Operating Margin %"] = (op / revenue_latest * 100) if revenue_latest and abs(revenue_latest) > 1e-9 else None

        ebitda = safe_val(latest, ["EBITDA", "ebitda"])
        metrics["EBITDA"] = ebitda
        metrics["EBITDA Margin %"] = (ebitda / revenue_latest * 100) if revenue_latest and abs(revenue_latest) > 1e-9 else None

        net = safe_val(latest, ["Net Income", "netProfit", "net_income"])
        metrics["Net Income"] = net
        metrics["Net Margin %"] = (net / revenue_latest * 100) if revenue_latest and abs(revenue_latest) > 1e-9 else None

        if prior is not None:
            net_prev = safe_val(prior, ["Net Income", "netProfit", "net_income"])
            metrics["Net Income (YoY)"] = net_prev
            metrics["Net Income Growth %"] = ((net - net_prev) / abs(net_prev) * 100) if net_prev and abs(net_prev) > 1e-9 else None

        # freshness check
        lu = financials.get("last_updated")
        if lu:
            try:
                if isinstance(lu, str) and len(lu) >= 8 and lu[0].isdigit():
                    parsed = datetime.strptime(lu[:10], "%Y-%m-%d")
                    months_old = (datetime.now() - parsed).days / 30
                    if months_old > 6:
                        warnings.append(f"⚠️ Data is {int(months_old)} months old (source: {financials.get('source')}).")
                else:
                    warnings.append("⚠️ Financial date label available but not standard. Verify freshness.")
            except Exception:
                warnings.append("⚠️ Could not verify data freshness.")
    else:
        warnings.append("⚠️ No income statement data available from preferred sources (NSE / Finnhub / Yahoo).")

    # Balance and cashflow best-effort (kept simple)
    if financials and financials.get("balance_sheet") is not None:
        try:
            bal = financials.get("balance_sheet")
            latest_bal = bal.iloc[-1]
            if "Total Assets" in bal.columns:
                metrics["Total Assets"] = to_number(latest_bal.get("Total Assets", 0))
            if "Stockholders Equity" in bal.columns:
                metrics["Shareholders Equity"] = to_number(latest_bal.get("Stockholders Equity", 0))
        except Exception:
            pass

    if financials and financials.get("cashflow") is not None:
        try:
            cf = financials.get("cashflow")
            latest_cf = cf.iloc[-1]
            if "Operating Cash Flow" in cf.columns:
                metrics["Operating Cash Flow"] = to_number(latest_cf.get("Operating Cash Flow", 0))
        except Exception:
            pass

    return metrics, warnings

# ------------------------ News & company info helpers ------------------------
@st.cache_data(ttl=900)
def get_company_info(symbol):
    try:
        t = yf.Ticker(symbol)
        return getattr(t, "info", {}) or {}
    except Exception:
        return {}

@st.cache_data(ttl=900)
def get_comprehensive_news(symbol, company_name):
    all_news = []
    # Yahoo news
    try:
        t = yf.Ticker(symbol)
        yf_news = getattr(t, "news", None)
        if yf_news:
            for item in yf_news:
                all_news.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', 'Yahoo Finance'),
                    'link': item.get('link', ''),
                    'date': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M') if item.get('providerPublishTime') else 'Unknown',
                    'timestamp': item.get('providerPublishTime', 0),
                    'description': item.get('summary', '') or ''
                })
    except Exception:
        pass

    # NewsAPI
    try:
        key = get_secret("NEWSAPI_KEY")
        if key and NewsApiClient is not None:
            napi = NewsApiClient(api_key=key)
            q = company_name if company_name else symbol.replace('.NS','').replace('.BO','')
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            articles = napi.get_everything(q=q, from_param=from_date, language='en', sort_by='publishedAt', page_size=50)
            if articles and articles.get('articles'):
                for art in articles['articles']:
                    pub = art.get('publishedAt')
                    ts = 0
                    if pub:
                        try:
                            ts = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ").timestamp()
                        except:
                            ts = 0
                    all_news.append({
                        'title': art.get('title',''),
                        'publisher': art.get('source',{}).get('name',''),
                        'link': art.get('url',''),
                        'date': pub or 'Unknown',
                        'timestamp': ts,
                        'description': art.get('description','') or ''
                    })
    except Exception:
        pass

    all_news.sort(key=lambda x: x.get('timestamp',0), reverse=True)
    # dedupe
    unique, seen = [], set()
    for n in all_news:
        k = (n.get('title') or '')[:80].lower()
        if k not in seen:
            unique.append(n)
            seen.add(k)
    return unique[:30]

# ------------------------ Formatting helpers ------------------------
def format_number(num):
    if pd.isna(num) or num is None:
        return "N/A"
    try:
        num = float(num)
    except Exception:
        return "N/A"
    if abs(num) >= 1e7:
        return f"₹{num/1e7:.2f} Cr"
    elif abs(num) >= 1e5:
        return f"₹{num/1e5:.2f} L"
    else:
        return f"₹{num:,.0f}"

def analyze_news_sentiment_advanced(news_items):
    if not news_items:
        return {"positive": [], "negative": [], "neutral": [], "earnings": []}
    positive_keywords = ['growth','profit','surge','beat','outperform','upgrade','contract','buyback','dividend']
    negative_keywords = ['loss','decline','miss','downgrade','lawsuit','delay','recall','layoff','investigation']
    earnings_keywords = ['earnings','quarterly','q1','q2','q3','q4','results','earnings call','guidance']
    categorized = {"positive": [], "negative": [], "neutral": [], "earnings": []}
    for item in news_items:
        text = (item.get('title','') + ' ' + item.get('description','')).lower()
        is_earn = any(k in text for k in earnings_keywords)
        pos = sum(1 for k in positive_keywords if k in text)
        neg = sum(1 for k in negative_keywords if k in text)
        obj = {'title': item.get('title',''), 'publisher': item.get('publisher',''), 'link': item.get('link',''), 'date': item.get('date',''), 'description': item.get('description','')}
        if is_earn:
            obj['sentiment'] = 'positive' if pos>neg else 'negative' if neg>pos else 'neutral'
            categorized['earnings'].append(obj)
        elif pos>neg:
            categorized['positive'].append(obj)
        elif neg>pos:
            categorized['negative'].append(obj)
        else:
            categorized['neutral'].append(obj)
    return categorized

# ------------------------ Streamlit UI ------------------------
st.title("📊 Indian Stock Financial Analyzer (NSE → Finnhub → Yahoo)")

st.sidebar.warning("""
⚠️ Data accuracy: For Indian tickers Yahoo can be stale. 
Preferred flow: NSE JSON (free) -> Finnhub (recommended) -> Yahoo fallback.
Add FINNHUB_API_KEY and OPENAI_API_KEY to Streamlit Secrets for best results.
""")

col1, col2 = st.columns([2,1])
with col1:
    symbol_input = st.text_input("Enter Stock Symbol", value="SUNPHARMA.NS", help="Use .NS for NSE / .BO for BSE")
with col2:
    st.markdown("**Verify Data At:**")
    st.markdown("[BSE India](https://www.bseindia.com) | [NSE India](https://www.nseindia.com)")

if st.button("🔍 Analyze Stock"):
    if not symbol_input:
        st.warning("Please enter a ticker symbol (e.g. SUNPHARMA.NS)")
    else:
        symbol = symbol_input.upper().strip()
        with st.spinner("Fetching company info and financials..."):
            company_info = get_company_info(symbol)
            company_name = company_info.get('longName') or symbol.replace('.NS','').replace('.BO','')
            financials, err = get_financials_unified(symbol)
            news = get_comprehensive_news(symbol, company_name)

        if err:
            st.error(f"Error fetching financials: {err}")

        st.header(f"🏢 {company_name}")
        if financials and financials.get('last_updated'):
            st.info(f"📅 Financial data last updated: {financials.get('last_updated')} (source: {financials.get('source')})")
        else:
            st.info("📅 Financial data last updated: None (no reliable quarterly data found)")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Sector", company_info.get('sector', 'N/A'))
        with c2:
            st.metric("Industry", company_info.get('industry', 'N/A'))
        with c3:
            st.metric("Employees", f"{company_info.get('fullTimeEmployees', 'N/A'):,}" if company_info.get('fullTimeEmployees') else 'N/A')
        with c4:
            st.metric("Market Cap", format_number(company_info.get('marketCap', 0)))

        metrics, warnings = calculate_metrics_with_warning(financials or {})
        if warnings:
            for w in warnings:
                st.warning(w)
            st.info("Tip: open 'View Raw Financial Statements' to inspect the source table.")

        if metrics:
            st.markdown("---")
            st.header("📈 Financial Metrics")
            if metrics.get("Quarter Date"):
                st.caption(f"Data as of: {metrics.get('Quarter Date')}")
            # Revenue / Profit
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                if 'Revenue (Latest)' in metrics:
                    rg = metrics.get('Revenue Growth %')
                    growth = f"+{rg:.1f}%" if rg is not None and rg>0 else f"{rg:.1f}%" if rg is not None else "-"
                    st.metric("Revenue (Latest Qtr)", format_number(metrics['Revenue (Latest)']), growth)
            with rc2:
                if 'Net Income' in metrics:
                    ng = metrics.get('Net Income Growth %')
                    growth = f"+{ng:.1f}%" if ng is not None and ng>0 else f"{ng:.1f}%" if ng is not None else "-"
                    st.metric("Net Income", format_number(metrics['Net Income']), growth)
            with rc3:
                if 'EBITDA' in metrics:
                    st.metric("EBITDA", format_number(metrics['EBITDA']))
            with rc4:
                if 'Operating Income' in metrics:
                    st.metric("Operating Income", format_number(metrics['Operating Income']))

            # Margins
            st.subheader("📊 Margin Analysis")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                gm = metrics.get('Gross Margin %')
                st.metric("Gross Margin", f"{gm:.2f}%" if gm is not None else "-")
            with m2:
                om = metrics.get('Operating Margin %')
                st.metric("Operating Margin", f"{om:.2f}%" if om is not None else "-")
            with m3:
                em = metrics.get('EBITDA Margin %')
                st.metric("EBITDA Margin", f"{em:.2f}%" if em is not None else "-")
            with m4:
                nm = metrics.get('Net Margin %')
                st.metric("Net Margin", f"{nm:.2f}%" if nm is not None else "-")

            # Balance & Cash
            st.subheader("🏦 Balance Sheet & Cash Flow (best-effort)")
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if 'Total Assets' in metrics:
                    st.metric("Total Assets", format_number(metrics['Total Assets']))
            with b2:
                if 'Shareholders Equity' in metrics:
                    st.metric("Shareholders Equity", format_number(metrics['Shareholders Equity']))
            with b3:
                if 'Current Ratio' in metrics:
                    st.metric("Current Ratio", f"{metrics['Current Ratio']:.2f}")
            with b4:
                if 'Debt to Equity' in metrics:
                    st.metric("Debt to Equity", f"{metrics['Debt to Equity']:.2f}")

        # News section
        st.markdown("---")
        st.header("📰 Recent News & Announcements")
        if news:
            categorized = analyze_news_sentiment_advanced(news)
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("📢 Earnings News", len(categorized['earnings']))
            n2.metric("🟢 Positive News", len(categorized['positive']))
            n3.metric("🔴 Negative News", len(categorized['negative']))
            n4.metric("⚪ Neutral News", len(categorized['neutral']))

            if categorized['earnings']:
                with st.expander("📢 Earnings Calls & Results - MOST RECENT", expanded=True):
                    for item in categorized['earnings']:
                        sentiment_icon = "🟢" if item.get('sentiment') == 'positive' else "🔴" if item.get('sentiment') == 'negative' else "⚪"
                        st.markdown(f"{sentiment_icon} **[{item['title']}]({item['link']})**")
                        st.caption(f"📅 {item['date']} | 📰 {item['publisher']}")
                        if item.get('description'):
                            st.markdown(f"*{item['description']}*")
                        st.markdown("---")
        else:
            st.info("No recent news found.")

        # Raw tables for debugging / inspection
        with st.expander("📋 View Raw Financial Statements (NSE → Finnhub → Yahoo)", expanded=True):
            t1, t2, t3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            with t1:
                if financials and financials.get('income_statement') is not None:
                    st.dataframe(financials['income_statement'], use_container_width=True)
                else:
                    st.write("No income statement available.")
            with t2:
                if financials and financials.get('balance_sheet') is not None:
                    st.dataframe(financials['balance_sheet'], use_container_width=True)
                else:
                    st.write("No balance sheet available.")
            with t3:
                if financials and financials.get('cashflow') is not None:
                    st.dataframe(financials['cashflow'], use_container_width=True)
                else:
                    st.write("No cashflow available.")

st.markdown("---")
st.markdown("<small>Not financial advice. For critical numbers always verify with BSE/NSE or company filings.</small>", unsafe_allow_html=True)
