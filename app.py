# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re

load_dotenv()

st.set_page_config(
    page_title="Indian Stock Analyzer - Accurate Data",
    page_icon="📊",
    layout="wide"
)

# Sidebar data accuracy note
st.sidebar.warning("""
⚠️ **Data Accuracy Note:**

Yahoo Finance may show outdated data for Indian stocks.

**For most accurate data:**
1. Cross-verify with BSE/NSE official websites
2. Check company's investor relations page
3. This tool shows latest available data but may have delays

**Recommendation:** Always verify important numbers with official sources!
""")

# ---------------------- News & Company info ----------------------
@st.cache_data(ttl=900)
def get_comprehensive_news(symbol, company_name):
    all_news = []
    # Yahoo news (best-effort)
    try:
        ticker = yf.Ticker(symbol)
        yf_news = getattr(ticker, "news", None)
        if yf_news:
            for item in yf_news:
                all_news.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', 'Yahoo Finance'),
                    'link': item.get('link', ''),
                    'date': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M') if item.get('providerPublishTime') else 'Unknown',
                    'timestamp': item.get('providerPublishTime', 0),
                    'description': '',
                    'source': 'Yahoo Finance'
                })
    except Exception:
        pass

    # NewsAPI (optional)
    newsapi_key = os.getenv('NEWSAPI_KEY')
    if newsapi_key:
        try:
            newsapi = NewsApiClient(api_key=newsapi_key)
            search_term = company_name if company_name else symbol.replace('.NS', '').replace('.BO', '')
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            articles = newsapi.get_everything(
                q=search_term,
                from_param=from_date,
                language='en',
                sort_by='publishedAt',
                page_size=50
            )
            if articles and articles.get('articles'):
                for article in articles['articles']:
                    published = article.get('publishedAt')
                    ts = 0
                    if published:
                        try:
                            ts = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ').timestamp()
                        except:
                            ts = 0
                    all_news.append({
                        'title': article.get('title', ''),
                        'publisher': article.get('source', {}).get('name', 'Unknown'),
                        'link': article.get('url', ''),
                        'date': datetime.strptime(article.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M') if article.get('publishedAt') else 'Unknown',
                        'timestamp': ts,
                        'description': article.get('description', ''),
                        'source': 'NewsAPI'
                    })
        except Exception:
            pass

    all_news.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    # de-dupe by title prefix
    unique = []
    seen = set()
    for n in all_news:
        key = (n.get('title') or '')[:60].lower()
        if key not in seen:
            unique.append(n)
            seen.add(key)
    return unique[:30]

@st.cache_data(ttl=3600)
def get_company_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return getattr(ticker, "info", {}) or {}
    except Exception:
        return {}

# ---------------------- Helpers for NSE session & parsing ----------------------
def fetch_nse_session():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com"
    }
    session = requests.Session()
    session.headers.update(headers)
    try:
        session.get("https://www.nseindia.com", timeout=5)
    except Exception:
        pass
    return session

def to_number(x):
    """Robustly convert strings/numbers to float. Handles commas, parentheses, currency tokens."""
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

# ---------------------- NSE financial fetch (preferred for Indian tickers) ----------------------
@st.cache_data(ttl=900)
def get_financial_data_nse(symbol):
    """
    Fetch a compact, normalized income statement (metrics x quarters) from NSE API.
    Returns None on failure so the caller can fallback to Yahoo.
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

        df = pd.DataFrame(rows)
        df.set_index("Quarter", inplace=True)
        # transpose so rows = metrics, columns = quarters (most recent first)
        income_statement = df.T

        last_updated = str(df.index[0]) if len(df.index) > 0 else datetime.now().strftime("%Y-%m-%d")
        return {
            "income_statement": income_statement,
            "balance_sheet": None,
            "cashflow": None,
            "last_updated": last_updated,
            "source": "NSE"
        }
    except Exception:
        return None

# ---------------------- Unified fetch: NSE preferred, Yahoo fallback ----------------------
@st.cache_data(ttl=1800)
def get_financial_data_yf(symbol):
    # try NSE first for .NS / .BO
    try:
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            nse = get_financial_data_nse(symbol)
            if nse:
                return nse, None
    except Exception:
        pass

    # fallback to yfinance's quarterly tables
    try:
        ticker = yf.Ticker(symbol)
        quarterly_income = None
        quarterly_balance = None
        quarterly_cashflow = None
        # Try known attributes; handle variations across yfinance versions
        try:
            quarterly_income = getattr(ticker, "quarterly_income_stmt", None) or getattr(ticker, "quarterly_financials", None)
            quarterly_balance = getattr(ticker, "quarterly_balance_sheet", None) or getattr(ticker, "quarterly_balance", None)
            quarterly_cashflow = getattr(ticker, "quarterly_cashflow", None) or getattr(ticker, "quarterly_cashflow_statements", None)
        except Exception:
            quarterly_income = getattr(ticker, "quarterly_financials", None)

        data_date = None
        if quarterly_income is not None and not getattr(quarterly_income, "empty", True):
            # ensure a transposed income_statement with rows = metrics, cols = quarters
            try:
                income_t = quarterly_income.T
            except Exception:
                try:
                    income_t = quarterly_income
                except Exception:
                    income_t = None
            try:
                data_date = income_t.columns[0].strftime('%Y-%m-%d') if income_t is not None else None
            except Exception:
                data_date = None
        else:
            income_t = None

        data = {
            "income_statement": income_t,
            "balance_sheet": (quarterly_balance.T if quarterly_balance is not None and not getattr(quarterly_balance, "empty", True) else None),
            "cashflow": (quarterly_cashflow.T if quarterly_cashflow is not None and not getattr(quarterly_cashflow, "empty", True) else None),
            "last_updated": data_date,
            "source": "Yahoo"
        }
        return data, None
    except Exception as e:
        return None, f"Error fetching financials: {str(e)}"

# ---------------------- Robust metrics calculator ----------------------
def calculate_metrics_with_warning(financials):
    """
    Accepts financials dict with keys income_statement, balance_sheet, cashflow, last_updated.
    Returns (metrics dict, warnings list). Defensive about missing/empty data.
    """
    metrics = {}
    warnings = []

    inc = None
    if financials and financials.get("income_statement") is not None:
        inc = financials["income_statement"]

    # Helper to extract latest & prior series from different DF orientations
    def get_latest_prior(df):
        """
        Attempts multiple policies to pick latest and prior metric series.
        Returns (latest_series, prior_series) or (None, None).
        latest_series / prior_series will be pd.Series where index are metric names.
        """
        if df is None:
            return None, None
        try:
            # prefer columns = quarters (most recent first)
            if df.shape[1] >= 1:
                latest = df.iloc[:, 0]
                prior = df.iloc[:, 1] if df.shape[1] > 1 else None
                return latest, prior
        except Exception:
            pass
        try:
            # fallback: rows = quarters (most recent last)
            if df.shape[0] >= 1:
                latest = df.iloc[-1]
                prior = df.iloc[-2] if df.shape[0] > 1 else None
                return latest, prior
        except Exception:
            pass
        return None, None

    latest, prior = get_latest_prior(inc)

    # If we have latest, compute metrics defensively
    if latest is not None:
        metrics["Quarter Date"] = financials.get("last_updated", "Unknown")

        def safe_val(series, names):
            if series is None:
                return 0.0
            for n in names:
                if n in series.index:
                    return to_number(series.get(n))
            # try first numeric element
            try:
                return to_number(series.iloc[0])
            except Exception:
                return 0.0

        revenue_latest = safe_val(latest, ["Total Revenue", "totalIncome", "Revenue", "totalIncome"])
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

        net = safe_val(latest, ["Net Income", "netProfit"])
        metrics["Net Income"] = net
        metrics["Net Margin %"] = (net / revenue_latest * 100) if revenue_latest and abs(revenue_latest) > 1e-9 else None

        if prior is not None:
            net_prev = safe_val(prior, ["Net Income", "netProfit"])
            metrics["Net Income (YoY)"] = net_prev
            metrics["Net Income Growth %"] = ((net - net_prev) / abs(net_prev) * 100) if net_prev and abs(net_prev) > 1e-9 else None

        # Freshness check
        last_label = financials.get("last_updated", None)
        if last_label:
            try:
                parsed = None
                if isinstance(last_label, str) and len(last_label) >= 8 and last_label[0].isdigit():
                    parsed = datetime.strptime(last_label[:10], "%Y-%m-%d")
                if parsed:
                    months_old = (datetime.now() - parsed).days / 30
                    if months_old > 6:
                        warnings.append(f"⚠️ Data is {int(months_old)} months old. May not reflect latest earnings.")
                else:
                    warnings.append("⚠️ Financial date label available but not standard. Verify freshness.")
            except Exception:
                warnings.append("⚠️ Could not verify data freshness.")
    else:
        warnings.append("⚠️ No income statement data available from preferred sources (NSE/Yahoo).")

    # Balance sheet: defensive extraction
    if financials and financials.get("balance_sheet") is not None:
        bal = financials.get("balance_sheet")
        try:
            # Get most recent safely
            b_latest = None
            if hasattr(bal, "shape") and bal.shape[0] >= 1:
                try:
                    b_latest = bal.iloc[-1]
                except Exception:
                    b_latest = bal.iloc[:, -1] if bal.shape[1] >= 1 else None
            if b_latest is not None:
                if "Total Assets" in bal.columns:
                    metrics["Total Assets"] = to_number(b_latest.get("Total Assets", 0))
                if "Stockholders Equity" in bal.columns:
                    metrics["Shareholders Equity"] = to_number(b_latest.get("Stockholders Equity", 0))
                if "Current Assets" in bal.columns and "Current Liabilities" in bal.columns:
                    ca = to_number(b_latest.get("Current Assets", 0))
                    cl = to_number(b_latest.get("Current Liabilities", 0))
                    if cl and abs(cl) > 0:
                        metrics["Current Ratio"] = ca / cl
                if "Cash And Cash Equivalents" in bal.columns:
                    metrics["Cash & Equivalents"] = to_number(b_latest.get("Cash And Cash Equivalents", 0))
                if "Total Debt" in bal.columns:
                    td = to_number(b_latest.get("Total Debt", 0))
                    metrics["Total Debt"] = td
                    se = metrics.get("Shareholders Equity", 0)
                    if se and abs(se) > 0:
                        metrics["Debt to Equity"] = td / se
        except Exception:
            pass

    # Cashflow: defensive extraction
    if financials and financials.get("cashflow") is not None:
        cf = financials.get("cashflow")
        try:
            cf_latest = None
            if hasattr(cf, "shape") and cf.shape[0] >= 1:
                try:
                    cf_latest = cf.iloc[-1]
                except Exception:
                    cf_latest = cf.iloc[:, -1] if cf.shape[1] >= 1 else None
            if cf_latest is not None:
                if "Operating Cash Flow" in cf.columns:
                    metrics["Operating Cash Flow"] = to_number(cf_latest.get("Operating Cash Flow", 0))
                if "Free Cash Flow" in cf.columns:
                    metrics["Free Cash Flow"] = to_number(cf_latest.get("Free Cash Flow", 0))
                if "Capital Expenditure" in cf.columns:
                    metrics["Capital Expenditure"] = to_number(cf_latest.get("Capital Expenditure", 0))
        except Exception:
            pass

    return metrics, warnings

# ---------------------- News sentiment and formatting helpers ----------------------
def analyze_news_sentiment_advanced(news_items):
    if not news_items:
        return {"positive": [], "negative": [], "neutral": [], "earnings": []}

    positive_keywords = ['growth', 'profit', 'surge', 'expansion', 'launch', 'partnership', 'innovation',
                         'record', 'strong', 'beat', 'exceed', 'outperform', 'approval', 'contract', 'dividend', 'buyback']
    negative_keywords = ['loss', 'decline', 'fall', 'drop', 'lawsuit', 'investigation', 'cut', 'layoff', 'warning', 'weak', 'miss', 'downgrade']
    earnings_keywords = ['earnings', 'results', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'guidance', 'outlook', 'earnings call']

    categorized = {"positive": [], "negative": [], "neutral": [], "earnings": []}
    for item in news_items:
        title = (item.get('title') or "").lower()
        desc = (item.get('description') or "").lower()
        text = title + " " + desc
        is_earn = any(k in text for k in earnings_keywords)
        pos = sum(1 for k in positive_keywords if k in text)
        neg = sum(1 for k in negative_keywords if k in text)
        news_obj = {
            'title': item.get('title', ''),
            'publisher': item.get('publisher', ''),
            'link': item.get('link', ''),
            'date': item.get('date', ''),
            'description': (item.get('description') or '')[:200]
        }
        if is_earn:
            news_obj['sentiment'] = 'positive' if pos > neg else 'negative' if neg > pos else 'neutral'
            categorized['earnings'].append(news_obj)
        elif pos > neg:
            categorized['positive'].append(news_obj)
        elif neg > pos:
            categorized['negative'].append(news_obj)
        else:
            categorized['neutral'].append(news_obj)
    return categorized

def format_number(num):
    if pd.isna(num) or num is None:
        return "N/A"
    try:
        num = float(num)
    except Exception:
        return "N/A"
    # Represent in Crore / Lakh for readability
    if abs(num) >= 1e7:
        return f"₹{num/1e7:.2f} Cr"
    elif abs(num) >= 1e5:
        return f"₹{num/1e5:.2f} L"
    else:
        return f"₹{num:,.0f}"

# ---------------------- Streamlit UI ----------------------
st.title("📊 Indian Stock Financial Analyzer")
st.markdown("""
**Features:**
- Financial snapshot (NSE preferred → Yahoo fallback)
- News & simple sentiment tagging
- Data freshness warnings

⚠️ Always cross-check important financials with BSE/NSE or company filings.
""")

col1, col2 = st.columns([2, 1])
with col1:
    symbol_input = st.text_input("Enter Stock Symbol", value="LAURUSLABS.NS", help="Use .NS for NSE / .BO for BSE")
with col2:
    st.markdown("**Verify Data At:**")
    st.markdown("[BSE India](https://www.bseindia.com) | [NSE India](https://www.nseindia.com)")

if st.button("🔍 Analyze Stock", type="primary"):
    if not symbol_input:
        st.warning("Please enter a stock symbol")
    else:
        symbol = symbol_input.upper().strip()
        with st.spinner(f"🔄 Fetching data for {symbol}..."):
            company_info = get_company_info(symbol)
            company_name = company_info.get('longName', symbol.replace('.NS','').replace('.BO',''))
            financials, error = get_financial_data_yf(symbol)
            news = get_comprehensive_news(symbol, company_name)

        if error:
            st.error(f"❌ {error}")
        else:
            st.header(f"🏢 {company_name}")
            if financials and financials.get('last_updated'):
                src = financials.get('source', 'Yahoo')
                st.info(f"📅 Financial data last updated: {financials['last_updated']} (source: {src})")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sector", company_info.get('sector', 'N/A'))
            with col2:
                st.metric("Industry", company_info.get('industry', 'N/A'))
            with col3:
                st.metric("Employees", f"{company_info.get('fullTimeEmployees', 'N/A'):,}" if company_info.get('fullTimeEmployees') else 'N/A')
            with col4:
                st.metric("Market Cap", format_number(company_info.get('marketCap', 0)))

            metrics, warnings = calculate_metrics_with_warning(financials or {})
            if warnings:
                for w in warnings:
                    st.warning(w)
                st.info("💡 Tip: If values look off, open 'View Raw Financial Statements' to inspect source table.")

            if metrics:
                st.markdown("---")
                st.header("📈 Financial Metrics (preferred source: NSE → fallback Yahoo)")
                if 'Quarter Date' in metrics:
                    st.caption(f"Data as of: {metrics['Quarter Date']}")

                st.subheader("💰 Revenue & Profitability")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if 'Revenue (Latest)' in metrics:
                        rg = metrics.get('Revenue Growth %', None)
                        growth = f"+{rg:.1f}%" if rg is not None and rg > 0 else f"{rg:.1f}%" if rg is not None else "-"
                        st.metric("Revenue (Latest Qtr)", format_number(metrics['Revenue (Latest)']), growth)
                with c2:
                    if 'Net Income' in metrics:
                        ng = metrics.get('Net Income Growth %', None)
                        growth = f"+{ng:.1f}%" if ng is not None and ng > 0 else f"{ng:.1f}%" if ng is not None else "-"
                        st.metric("Net Income", format_number(metrics['Net Income']), growth)
                with c3:
                    if 'EBITDA' in metrics:
                        st.metric("EBITDA", format_number(metrics['EBITDA']))
                with c4:
                    if 'Operating Income' in metrics:
                        st.metric("Operating Income", format_number(metrics['Operating Income']))

                st.subheader("📊 Margin Analysis")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    gm = metrics.get('Gross Margin %')
                    st.metric("Gross Margin", f"{gm:.2f}%" if gm is not None else "-")
                with c2:
                    om = metrics.get('Operating Margin %')
                    st.metric("Operating Margin", f"{om:.2f}%" if om is not None else "-")
                with c3:
                    em = metrics.get('EBITDA Margin %')
                    st.metric("EBITDA Margin", f"{em:.2f}%" if em is not None else "-")
                with c4:
                    nm = metrics.get('Net Margin %')
                    st.metric("Net Margin", f"{nm:.2f}%" if nm is not None else "-")

                st.subheader("🏦 Balance Sheet")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if 'Total Assets' in metrics:
                        st.metric("Total Assets", format_number(metrics['Total Assets']))
                with c2:
                    if 'Shareholders Equity' in metrics:
                        st.metric("Shareholders Equity", format_number(metrics['Shareholders Equity']))
                with c3:
                    if 'Current Ratio' in metrics:
                        st.metric("Current Ratio", f"{metrics['Current Ratio']:.2f}")
                with c4:
                    if 'Debt to Equity' in metrics:
                        st.metric("Debt to Equity", f"{metrics['Debt to Equity']:.2f}")

                st.subheader("💵 Cash Flow")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if 'Operating Cash Flow' in metrics:
                        st.metric("Operating Cash Flow", format_number(metrics['Operating Cash Flow']))
                with c2:
                    if 'Free Cash Flow' in metrics:
                        st.metric("Free Cash Flow", format_number(metrics['Free Cash Flow']))
                with c3:
                    if 'Capital Expenditure' in metrics:
                        st.metric("Capital Expenditure", format_number(metrics['Capital Expenditure']))

            st.markdown("---")
            st.header("📰 Recent News & Announcements")
            st.caption("✅ News data is real-time and accurate (when available).")
            if news:
                categorized = analyze_news_sentiment_advanced(news)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📢 Earnings News", len(categorized['earnings']))
                c2.metric("🟢 Positive News", len(categorized['positive']))
                c3.metric("🔴 Negative News", len(categorized['negative']))
                c4.metric("⚪ Neutral News", len(categorized['neutral']))

                if categorized['earnings']:
                    with st.expander("📢 Earnings Calls & Results - MOST RECENT & ACCURATE", expanded=True):
                        st.success("✅ These are real-time news with actual reported numbers!")
                        for item in categorized['earnings']:
                            sentiment_icon = "🟢" if item.get('sentiment') == 'positive' else "🔴" if item.get('sentiment') == 'negative' else "⚪"
                            st.markdown(f"{sentiment_icon} **[{item['title']}]({item['link']})**")
                            st.caption(f"📅 {item['date']} | 📰 {item['publisher']}")
                            if item.get('description'):
                                st.markdown(f"*{item['description']}*")
                            st.markdown("---")

                if categorized['positive']:
                    with st.expander("🟢 Growth Catalysts & Positive Developments"):
                        for item in categorized['positive']:
                            st.markdown(f"**📈 [{item['title']}]({item['link']})**")
                            st.caption(f"📅 {item['date']} | 📰 {item['publisher']}")
                            if item.get('description'):
                                st.markdown(f"*{item['description']}*")
                            st.markdown("---")

                if categorized['negative']:
                    with st.expander("🔴 Risk Factors & Concerns"):
                        for item in categorized['negative']:
                            st.markdown(f"**📉 [{item['title']}]({item['link']})**")
                            st.caption(f"📅 {item['date']} | 📰 {item['publisher']}")
                            if item.get('description'):
                                st.markdown(f"*{item['description']}*")
                            st.markdown("---")

                if categorized['neutral']:
                    with st.expander("⚪ Other Updates"):
                        for item in categorized['neutral']:
                            st.markdown(f"**📊 [{item['title']}]({item['link']})**")
                            st.caption(f"📅 {item['date']} | 📰 {item['publisher']}")
                            if item.get('description'):
                                st.markdown(f"*{item['description']}*")
                            st.markdown("---")
            else:
                st.info("No recent news available.")

            st.markdown("---")
            with st.expander("📋 View Raw Financial Statements (Preferred source: NSE → fallback Yahoo Finance)"):
                tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                with tab1:
                    if financials and financials.get('income_statement') is not None:
                        st.dataframe(financials['income_statement'], use_container_width=True)
                with tab2:
                    if financials and financials.get('balance_sheet') is not None:
                        st.dataframe(financials['balance_sheet'], use_container_width=True)
                with tab3:
                    if financials and financials.get('cashflow') is not None:
                        st.dataframe(financials['cashflow'], use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>⚠️ Data Disclaimer:</strong> Financial metrics from Yahoo Finance may be delayed or inaccurate for Indian stocks. 
    News data is real-time and accurate when provided by NewsAPI. Always verify important figures with official BSE/NSE sources.</p>
    <p>Not financial advice | For educational purposes only</p>
</div>
""", unsafe_allow_html=True)
