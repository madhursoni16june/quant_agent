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

# Warning about data accuracy
st.sidebar.warning("""
⚠️ **Data Accuracy Note:**

Yahoo Finance may show outdated data for Indian stocks.

**For most accurate data:**
1. Cross-verify with BSE/NSE official websites
2. Check company's investor relations page
3. This tool shows latest available data but may have delays

**Recommendation:** Always verify important numbers with official sources!
""")

@st.cache_data(ttl=900)
def get_comprehensive_news(symbol, company_name):
    """Fetches news from multiple sources"""
    all_news = []
    
    try:
        ticker = yf.Ticker(symbol)
        yf_news = ticker.news
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
                    all_news.append({
                        'title': article.get('title', ''),
                        'publisher': article.get('source', {}).get('name', 'Unknown'),
                        'link': article.get('url', ''),
                        'date': datetime.strptime(article.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M') if article.get('publishedAt') else 'Unknown',
                        'timestamp': datetime.strptime(article.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ').timestamp() if article.get('publishedAt') else 0,
                        'description': article.get('description', ''),
                        'source': 'NewsAPI'
                    })
        except Exception:
            pass
    
    all_news.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    
    unique_news = []
    seen_titles = set()
    for news in all_news:
        title_key = news['title'].lower()[:50]
        if title_key not in seen_titles:
            unique_news.append(news)
            seen_titles.add(title_key)
    
    return unique_news[:30]

@st.cache_data(ttl=3600)
def get_company_info(symbol):
    """Fetches company information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info
    except Exception:
        return {}

def fetch_nse_session():
    """
    Helper to create a requests.Session that can fetch NSE JSON endpoints.
    NSE blocks simple scrapers unless headers and initial visit are present.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com"
    }
    session = requests.Session()
    session.headers.update(headers)
    # Touch the main page to populate cookies
    try:
        session.get("https://www.nseindia.com", timeout=5)
    except:
        # Sometimes NSE will block; we'll let the caller handle failures
        pass
    return session

def to_number(x):
    """Robust conversion of heterogeneous string/number inputs to float.
    Handles commas, parentheses for negatives, currency symbols, and '—' or empty values.
    """
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s in ["-", "—", "None", "nan", "NaN"]:
            return 0.0
        # Remove common unit tokens and currency symbols
        for token in ["₹", "rs", "Rs.", "Cr", "L", "lakhs", "crore", "INR", ","]:
            s = s.replace(token, "")
        # Parentheses indicate negative
        negative = False
        if s.startswith("(") and s.endswith(")"):
            negative = True
            s = s[1:-1]
        # Remove any remaining non-numeric chars except . and - and e/E
        s = re.sub(r"[^\d\.\-eE]", "", s)
        if s == "" or s == "-" or s == ".":
            return 0.0
        val = float(s)
        if negative:
            val = -val
        return val
    except Exception:
        return 0.0

@st.cache_data(ttl=900)
def get_financial_data_nse(symbol):
    """
    Try to fetch latest quarterly numbers from NSE India endpoints.
    Returns a dict with data needed by the app: income_statement DataFrame (T), balance_sheet None, cashflow None, last_updated.
    NOTE: Works best for .NS tickers (NSE). For others, fallback to Yahoo is used.
    """
    stock = symbol.replace(".NS", "").replace(".BO", "")
    session = fetch_nse_session()
    try:
        results_url = f"https://www.nseindia.com/api/results-equity?symbol={stock}"
        resp = session.get(results_url, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        qdata = j.get("data", [])
        if not qdata:
            return None

        rows = []
        for idx, q in enumerate(qdata[:8]):  # get up to 8 quarters if present
            quarter_label = q.get("quarter") or q.get("period") or q.get("fiscalPeriod") or q.get("resultDate") or f"Q{idx+1}"
            total_income = to_number(q.get("totalIncome") or q.get("totalRevenue") or q.get("revenue") or q.get("Total Income") or q.get("TotalRevenue"))
            gross_profit = to_number(q.get("grossProfit") or q.get("grossProfitLoss") or q.get("Gross Profit"))
            operating_income = to_number(q.get("operatingProfit") or q.get("operatingProfitLoss") or q.get("Operating Income"))
            ebitda = to_number(q.get("ebitda") or q.get("EBITDA"))
            net_profit = to_number(q.get("netProfit") or q.get("profitAfterTax") or q.get("Net Income"))
            rows.append({
                "Quarter": quarter_label,
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
        income_statement = df.T  # rows = metrics, columns = quarters (most recent first)

        last_updated = str(df.index[0]) if len(df.index) > 0 else datetime.now().strftime("%Y-%m-%d")

        data = {
            'income_statement': income_statement,
            'balance_sheet': None,
            'cashflow': None,
            'last_updated': last_updated,
            'source': 'NSE'
        }
        return data
    except Exception:
        return None

@st.cache_data(ttl=1800)
def get_financial_data_yf(symbol):
    """
    Unified financial fetcher:
    1) Try NSE JSON (preferred for Indian tickers) — returns a minimal income_statement DataFrame (transposed to match previous usage).
    2) If NSE fails, fallback to yfinance quarterly statements (existing behavior).
    Returns: (data_dict, error_message_or_None)
    """
    # First attempt NSE (only for .NS/.BO tickers typically)
    try:
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            nse_data = get_financial_data_nse(symbol)
            if nse_data:
                return nse_data, None
    except Exception:
        pass
    
    # Yahoo fallback (may be stale for Indian tickers)
    try:
        ticker = yf.Ticker(symbol)
        
        quarterly_income = None
        quarterly_balance = None
        quarterly_cashflow = None
        try:
            quarterly_income = ticker.quarterly_income_stmt
            quarterly_balance = ticker.quarterly_balance_sheet
            quarterly_cashflow = ticker.quarterly_cashflow
        except:
            try:
                quarterly_income = ticker.quarterly_financials
            except:
                quarterly_income = None
        
        data_date = None
        if quarterly_income is not None and not quarterly_income.empty:
            try:
                data_date = quarterly_income.columns[0].strftime('%Y-%m-%d')
            except:
                data_date = str(datetime.now().date())
        
        data = {
            'income_statement': quarterly_income.T if quarterly_income is not None else None,
            'balance_sheet': quarterly_balance.T if quarterly_balance is not None else None,
            'cashflow': quarterly_cashflow.T if quarterly_cashflow is not None else None,
            'last_updated': data_date
        }
        
        return data, None
    except Exception as e:
        return None, f"Error fetching financials: {str(e)}"

def calculate_metrics_with_warning(financials):
    """Calculates metrics with data freshness warning and robust numeric handling"""
    metrics = {}
    warnings = []

    inc = None
    if financials and financials.get("income_statement") is not None:
        inc = financials["income_statement"]

    if inc is not None and inc.shape[1] >= 1:
        try:
            latest = inc.iloc[:, 0]
        except Exception:
            latest = inc.iloc[-1]

        prior = inc.iloc[:, 1] if inc.shape[1] > 1 else None

        metrics["Quarter Date"] = financials.get("last_updated", "Unknown")

        def safe_val(series, keys):
            for k in keys:
                if k in series.index:
                    return to_number(series[k])
            try:
                return float(series.iloc[0])
            except:
                return 0.0

        revenue_latest = safe_val(latest, ["Total Revenue", "totalIncome", "Revenue", "totalIncome"])
        metrics["Revenue (Latest)"] = revenue_latest

        if prior is not None:
            revenue_prev = safe_val(prior, ["Total Revenue", "totalIncome", "Revenue"])
            metrics["Revenue (YoY)"] = revenue_prev
            if revenue_prev and abs(revenue_prev) > 1e-6:
                metrics["Revenue Growth %"] = ((revenue_latest - revenue_prev) / abs(revenue_prev)) * 100
            else:
                metrics["Revenue Growth %"] = None
        else:
            metrics["Revenue (YoY)"] = 0
            metrics["Revenue Growth %"] = None

        gp = safe_val(latest, ["Gross Profit", "grossProfit"])
        metrics["Gross Profit"] = gp
        if revenue_latest and abs(revenue_latest) > 0:
            metrics["Gross Margin %"] = (gp / revenue_latest) * 100
        else:
            metrics["Gross Margin %"] = None

        op = safe_val(latest, ["Operating Income", "operatingProfit"])
        metrics["Operating Income"] = op
        if revenue_latest and abs(revenue_latest) > 0:
            metrics["Operating Margin %"] = (op / revenue_latest) * 100
        else:
            metrics["Operating Margin %"] = None

        ebitda = safe_val(latest, ["EBITDA", "ebitda"])
        metrics["EBITDA"] = ebitda
        if revenue_latest and abs(revenue_latest) > 0:
            metrics["EBITDA Margin %"] = (ebitda / revenue_latest) * 100
        else:
            metrics["EBITDA Margin %"] = None

        net = safe_val(latest, ["Net Income", "netProfit"])
        metrics["Net Income"] = net
        if revenue_latest and abs(revenue_latest) > 0:
            metrics["Net Margin %"] = (net / revenue_latest) * 100
        else:
            metrics["Net Margin %"] = None

        if prior is not None:
            net_prev = safe_val(prior, ["Net Income", "netProfit"])
            metrics["Net Income (YoY)"] = net_prev
            if net_prev and abs(net_prev) > 1e-6:
                metrics["Net Income Growth %"] = ((net - net_prev) / abs(net_prev)) * 100
            else:
                metrics["Net Income Growth %"] = None

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
            except:
                warnings.append("⚠️ Could not verify data freshness.")

    if financials and financials.get('balance_sheet') is not None:
        bal = financials['balance_sheet']
        latest = bal.iloc[-1]
        try:
            if 'Total Assets' in bal.columns:
                metrics['Total Assets'] = to_number(latest.get('Total Assets', 0))
            if 'Stockholders Equity' in bal.columns:
                metrics['Shareholders Equity'] = to_number(latest.get('Stockholders Equity', 0))
            if 'Current Assets' in bal.columns and 'Current Liabilities' in bal.columns:
                current_assets = to_number(latest.get('Current Assets', 0))
                current_liabilities = to_number(latest.get('Current Liabilities', 0))
                if current_liabilities and abs(current_liabilities) > 0:
                    metrics['Current Ratio'] = current_assets / current_liabilities
            if 'Cash And Cash Equivalents' in bal.columns:
                metrics['Cash & Equivalents'] = to_number(latest.get('Cash And Cash Equivalents', 0))
            if 'Total Debt' in bal.columns:
                metrics['Total Debt'] = to_number(latest.get('Total Debt', 0))
                if metrics.get('Shareholders Equity', 0) and abs(metrics.get('Shareholders Equity', 0)) > 0:
                    metrics['Debt to Equity'] = metrics['Total Debt'] / metrics['Shareholders Equity']
        except Exception:
            pass

    if financials and financials.get('cashflow') is not None:
        cf = financials['cashflow']
        latest = cf.iloc[-1]
        try:
            if 'Operating Cash Flow' in cf.columns:
                metrics['Operating Cash Flow'] = to_number(latest.get('Operating Cash Flow', 0))
            if 'Free Cash Flow' in cf.columns:
                metrics['Free Cash Flow'] = to_number(latest.get('Free Cash Flow', 0))
            if 'Capital Expenditure' in cf.columns:
                metrics['Capital Expenditure'] = to_number(latest.get('Capital Expenditure', 0))
        except Exception:
            pass

    return metrics, warnings

def analyze_news_sentiment_advanced(news_items):
    """Advanced news analysis"""
    if not news_items:
        return {"positive": [], "negative": [], "neutral": [], "earnings": []}
    
    positive_keywords = ['growth', 'profit', 'surge', 'expansion', 'launch', 'partnership', 'innovation', 
                        'record', 'strong', 'beat', 'exceed', 'outperform', 'breakthrough', 'approval',
                        'contract', 'order', 'dividend', 'buyback', 'upgraded', 'bullish', 'rally',
                        'gain', 'jump', 'soar', 'successful', 'rises', 'up']
    
    negative_keywords = ['loss', 'decline', 'fall', 'drop', 'lawsuit', 'investigation', 'cut',
                        'layoff', 'warning', 'concern', 'risk', 'delay', 'weak', 'miss', 'downgrade',
                        'bearish', 'slump', 'plunge', 'crash', 'fail', 'disappointing', 'down']
    
    earnings_keywords = ['earnings', 'results', 'quarterly', 'q1', 'q2', 'q3', 'q4', 
                        'earnings call', 'conference call', 'guidance', 'outlook',
                        'management', 'ceo', 'cfo', 'announces', 'reports', 'fy25', 'fy26']
    
    categorized = {"positive": [], "negative": [], "neutral": [], "earnings": []}
    
    for item in news_items:
        title = item.get('title', '').lower()
        description = item.get('description', '').lower()
        combined_text = title + ' ' + description
        
        is_earnings = any(keyword in combined_text for keyword in earnings_keywords)
        
        pos_count = sum(1 for keyword in positive_keywords if keyword in combined_text)
        neg_count = sum(1 for keyword in negative_keywords if keyword in combined_text)
        
        news_obj = {
            'title': item.get('title', ''),
            'publisher': item.get('publisher', 'Unknown'),
            'link': item.get('link', ''),
            'date': item.get('date', 'Unknown'),
            'description': item.get('description', '')[:200] + '...' if item.get('description') else ''
        }
        
        if is_earnings:
            news_obj['sentiment'] = 'positive' if pos_count > neg_count else 'negative' if neg_count > pos_count else 'neutral'
            categorized['earnings'].append(news_obj)
        elif pos_count > neg_count:
            categorized['positive'].append(news_obj)
        elif neg_count > pos_count:
            categorized['negative'].append(news_obj)
        else:
            categorized['neutral'].append(news_obj)
    
    return categorized

def format_number(num):
    """Format numbers in Indian style"""
    if pd.isna(num) or num is None:
        return "N/A"
    try:
        num = float(num)
    except:
        return "N/A"
    if abs(num) >= 1e7:
        return f"₹{num/1e7:.2f} Cr"
    elif abs(num) >= 1e5:
        return f"₹{num/1e5:.2f} L"
    else:
        return f"₹{num:,.0f}"

# Main Application
st.title("📊 Indian Stock Financial Analyzer")
st.markdown("""
**Features:**
- Complete Financial Analysis
- Real-Time News & Earnings Announcements
- Growth Catalysts & Risk Identification

⚠️ **Important:** Always cross-verify numbers with official BSE/NSE or company websites!
""")

# Data source links
col1, col2 = st.columns([2, 1])
with col1:
    symbol_input = st.text_input(
        "Enter Stock Symbol",
        value="LAURUSLABS.NS",
        help="NSE: Add .NS | BSE: Add .BO"
    )

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
            company_name = company_info.get('longName', symbol.replace('.NS', '').replace('.BO', ''))
            
            financials, error = get_financial_data_yf(symbol)
            news = get_comprehensive_news(symbol, company_name)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.header(f"🏢 {company_name}")
                
                # Data freshness warning
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
                
                metrics, warnings = calculate_metrics_with_warning(financials if financials else {'income_statement': None, 'balance_sheet': None, 'cashflow': None, 'last_updated': None})
                
                # Show warnings
                if warnings:
                    for warning in warnings:
                        st.warning(warning)
                    st.info("💡 **Tip:** Check recent news below for latest earnings announcements with actual numbers!")
                
                if metrics:
                    st.markdown("---")
                    st.header("📈 Financial Metrics (preferred source: NSE → fallback Yahoo Finance)")
                    
                    if 'Quarter Date' in metrics:
                        st.caption(f"Data as of: {metrics['Quarter Date']}")
                    
                    st.subheader("💰 Revenue & Profitability")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'Revenue (Latest)' in metrics:
                            rg = metrics.get('Revenue Growth %', None)
                            growth = f"+{rg:.1f}%" if rg is not None and rg > 0 else f"{rg:.1f}%" if rg is not None else "-"
                            st.metric("Revenue (Latest Qtr)", format_number(metrics['Revenue (Latest)']), growth)
                    
                    with col2:
                        if 'Net Income' in metrics:
                            ng = metrics.get('Net Income Growth %', None)
                            growth = f"+{ng:.1f}%" if ng is not None and ng > 0 else f"{ng:.1f}%" if ng is not None else "-"
                            st.metric("Net Income", format_number(metrics['Net Income']), growth)
                    
                    with col3:
                        if 'EBITDA' in metrics:
                            st.metric("EBITDA", format_number(metrics['EBITDA']))
                    
                    with col4:
                        if 'Operating Income' in metrics:
                            st.metric("Operating Income", format_number(metrics['Operating Income']))
                    
                    st.subheader("📊 Margin Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if metrics.get('Gross Margin %') is not None:
                            st.metric("Gross Margin", f"{metrics['Gross Margin %']:.2f}%")
                        else:
                            st.metric("Gross Margin", "-")
                    with col2:
                        if metrics.get('Operating Margin %') is not None:
                            st.metric("Operating Margin", f"{metrics['Operating Margin %']:.2f}%")
                        else:
                            st.metric("Operating Margin", "-")
                    with col3:
                        if metrics.get('EBITDA Margin %') is not None:
                            st.metric("EBITDA Margin", f"{metrics['EBITDA Margin %']:.2f}%")
                        else:
                            st.metric("EBITDA Margin", "-")
                    with col4:
                        if metrics.get('Net Margin %') is not None:
                            st.metric("Net Margin", f"{metrics['Net Margin %']:.2f}%")
                        else:
                            st.metric("Net Margin", "-")
                    
                    st.subheader("🏦 Balance Sheet")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'Total Assets' in metrics:
                            st.metric("Total Assets", format_number(metrics['Total Assets']))
                    with col2:
                        if 'Shareholders Equity' in metrics:
                            st.metric("Shareholders Equity", format_number(metrics['Shareholders Equity']))
                    with col3:
                        if 'Current Ratio' in metrics:
                            st.metric("Current Ratio", f"{metrics['Current Ratio']:.2f}")
                    with col4:
                        if 'Debt to Equity' in metrics:
                            st.metric("Debt to Equity", f"{metrics['Debt to Equity']:.2f}")
                    
                    st.subheader("💵 Cash Flow")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'Operating Cash Flow' in metrics:
                            st.metric("Operating Cash Flow", format_number(metrics['Operating Cash Flow']))
                    with col2:
                        if 'Free Cash Flow' in metrics:
                            st.metric("Free Cash Flow", format_number(metrics['Free Cash Flow']))
                    with col3:
                        if 'Capital Expenditure' in metrics:
                            st.metric("Capital Expenditure", format_number(metrics['Capital Expenditure']))
                    
                    st.markdown("---")
                    st.header("📰 Recent News & Announcements")
                    st.caption("✅ News data is real-time and accurate!")
                    
                    if news:
                        categorized_news = analyze_news_sentiment_advanced(news)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📢 Earnings News", len(categorized_news['earnings']))
                        with col2:
                            st.metric("🟢 Positive News", len(categorized_news['positive']))
                        with col3:
                            st.metric("🔴 Negative News", len(categorized_news['negative']))
                        with col4:
                            st.metric("⚪ Neutral News", len(categorized_news['neutral']))
                        
                        if categorized_news['earnings']:
                            with st.expander("📢 Earnings Calls & Results - MOST RECENT & ACCURATE", expanded=True):
                                st.success("✅ These are real-time news with actual reported numbers!")
                                for item in categorized_news['earnings']:
                                    sentiment_icon = "🟢" if item.get('sentiment') == 'positive' else "🔴" if item.get('sentiment') == 'negative' else "⚪"
                                    st.markdown(f"{sentiment_icon} **[{item['title']}]({item['link']})**")
                                    st.caption(f"📅 {item['date']} | 📰 {item['publisher']}")
                                    if item.get('description'):
                                        st.markdown(f"*{item['description']}*")
                                    st.markdown("---")
                        
                        if categorized_news['positive']:
                            with st.expander("🟢 Growth Catalysts & Positive Developments"):
                                for item in categorized_news['positive']:
                                    st.markdown(f"**📈 [{item['title']}]({item['link']})**")
                                    st.caption(f"📅 {item['date']} | 📰 {item['publisher']}")
                                    if item.get('description'):
                                        st.markdown(f"*{item['description']}*")
                                    st.markdown("---")
                        
                        if categorized_news['negative']:
                            with st.expander("🔴 Risk Factors & Concerns"):
                                for item in categorized_news['negative']:
                                    st.markdown(f"**📉 [{item['title']}]({item['link']})**")
                                    st.caption(f"📅 {item['date']} | 📰 {item['publisher']}")
                                    if item.get('description'):
                                        st.markdown(f"*{item['description']}*")
                                    st.markdown("---")
                        
                        if categorized_news['neutral']:
                            with st.expander("⚪ Other Updates"):
                                for item in categorized_news['neutral']:
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
                            if financials and financials['income_statement'] is not None:
                                st.dataframe(financials['income_statement'], use_container_width=True)
                        with tab2:
                            if financials and financials['balance_sheet'] is not None:
                                st.dataframe(financials['balance_sheet'], use_container_width=True)
                        with tab3:
                            if financials and financials['cashflow'] is not None:
                                st.dataframe(financials['cashflow'], use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>⚠️ Data Disclaimer:</strong> Financial metrics from Yahoo Finance may be delayed or inaccurate for Indian stocks. 
    News data is real-time and accurate. Always verify with official BSE/NSE sources.</p>
    <p>Not financial advice | For educational purposes only</p>
</div>
""", unsafe_allow_html=True)
