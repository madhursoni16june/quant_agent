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
    except:
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
        except Exception as e:
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
    except:
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
        # Primary result endpoint (quarterly results)
        # This endpoint can change — if it fails we return None and let fallback take over.
        results_url = f"https://www.nseindia.com/api/results-equity?symbol={stock}"
        resp = session.get(results_url, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        
        # j.get('data') usually contains list of result rows (latest first)
        qdata = j.get("data", [])
        if not qdata:
            return None  # let fallback handle
        
        # Build a minimal income_statement DataFrame from the first two quarters (if available)
        rows = []
        for idx, q in enumerate(qdata[:4]):  # get up to 4 quarters if present
            # NSE keys can vary depending on listing; attempt common ones
            quarter_label = q.get("quarter") or q.get("period") or q.get("fiscalPeriod") or f"Q{idx}"
            # Common numeric keys observed on NSE: totalIncome, netProfit, basicEPS, totalExpenditure etc.
            total_income = q.get("totalIncome") or q.get("totalRevenue") or q.get("revenue") or 0
            gross_profit = q.get("grossProfit") or None
            operating_income = q.get("operatingProfit") or q.get("operatingProfitLoss") or None
            ebitda = q.get("ebitda") or None
            net_profit = q.get("netProfit") or q.get("profitAfterTax") or 0
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
        # Use Quarter as index and transpose to match your previous structure (columns become fields)
        df.set_index("Quarter", inplace=True)
        income_statement = df.T  # matches earlier pattern: DataFrame where rows are metrics, cols are quarters; you'll be transposing where needed
        
        # last_updated - use the latest quarter label or today's date as fallback
        last_updated = df.index[0] if len(df.index) > 0 else datetime.now().strftime("%Y-%m-%d")
        
        data = {
            'income_statement': income_statement,  # already transposed to keep compatibility
            'balance_sheet': None,
            'cashflow': None,
            'last_updated': last_updated
        }
        return data
    except Exception as e:
        # If NSE fetch errors, return None to allow fallback to yahoo
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
        # ignore and try yahoo fallback
        pass
    
    # Yahoo fallback (may be stale for Indian tickers)
    try:
        ticker = yf.Ticker(symbol)
        
        # Attempt to fetch the quarterly DataFrames as before
        quarterly_income = None
        quarterly_balance = None
        quarterly_cashflow = None
        try:
            quarterly_income = ticker.quarterly_income_stmt
            quarterly_balance = ticker.quarterly_balance_sheet
            quarterly_cashflow = ticker.quarterly_cashflow
        except:
            # Some yfinance installs expose differently; try alternative properties
            try:
                quarterly_income = ticker.quarterly_financials
            except:
                quarterly_income = None
        
        # Get the last update timestamp if available
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
    """Calculates metrics with data freshness warning"""
    metrics = {}
    warnings = []
    
    if financials['income_statement'] is not None and len(financials['income_statement']) >= 1:
        inc = financials['income_statement']
        # the income_statement on NSE path is transposed differently: we ensured it's inc where rows are metrics and columns are quarters
        # To be safe, try to get the latest column using iloc[:, 0] if columns represent quarters
        try:
            # If original yahoo path, inc rows are quarter rows, hence we used T earlier; handle common shapes.
            if inc.shape[0] >= 1 and inc.shape[1] >= 1:
                # prefer selecting the most recent quarter as the first column if transposed that way
                latest = inc.iloc[:, 0] if inc.shape[1] > 1 else inc.iloc[:, 0]
                # try to pick a prior quarter for yoy (if available)
                yoy = inc.iloc[:, 1] if inc.shape[1] > 1 else inc.iloc[:, 0]
            else:
                latest = inc.iloc[-1]
                yoy = inc.iloc[-5] if len(inc) >= 5 else inc.iloc[0]
        except Exception:
            # fallback to earlier behavior expecting transposed DataFrame (index = quarters)
            try:
                latest = inc.iloc[-1]
                yoy = inc.iloc[-5] if len(inc) >= 5 else inc.iloc[0]
            except:
                latest = None
                yoy = None
        
        # Get a quarter date/value for freshness - prefer the 'last_updated' key from the financials container
        quarter_date = financials.get('last_updated', None)
        if quarter_date:
            try:
                # attempt to parse if in YYYY-MM-DD else keep original label
                quarter_datetime = None
                if isinstance(quarter_date, str) and len(quarter_date) >= 8:
                    try:
                        quarter_datetime = datetime.strptime(quarter_date[:10], '%Y-%m-%d')
                    except:
                        quarter_datetime = None
                if quarter_datetime:
                    months_old = (datetime.now() - quarter_datetime).days / 30
                    if months_old > 6:
                        warnings.append(f"⚠️ Data is {int(months_old)} months old. May not reflect latest earnings.")
                else:
                    # if quarter_date is not a parsable date, show a generic freshness message
                    warnings.append("⚠️ Financial date label available but not in YYYY-MM-DD format; verify freshness.")
            except:
                warnings.append("⚠️ Could not verify data freshness.")
        else:
            warnings.append("⚠️ Could not determine last updated date for financials.")
        
        # map different column names from NSE to the names this analyzer expects
        # Several possible column names might be present; check for common ones
        def safe_get(series, keys):
            for k in keys:
                if k in series.index:
                    return series.get(k)
            # try numeric-like fallback
            try:
                return series.iloc[0]
            except:
                return 0
        
        if latest is not None:
            metrics['Quarter Date'] = str(quarter_date)[:10] if quarter_date else 'Unknown'
            
            # Revenue
            revenue_val = safe_get(latest, ['Total Revenue', 'totalIncome', 'Total Income', 'Revenue', 'revenue'])
            metrics['Revenue (Latest)'] = revenue_val or 0
            # YoY / prior quarter revenue
            if yoy is not None:
                prev_revenue_val = safe_get(yoy, ['Total Revenue', 'totalIncome', 'Total Income', 'Revenue', 'revenue'])
                metrics['Revenue (YoY)'] = prev_revenue_val or 0
                if metrics['Revenue (YoY)'] != 0:
                    try:
                        metrics['Revenue Growth %'] = ((metrics['Revenue (Latest)'] - metrics['Revenue (YoY)']) / abs(metrics['Revenue (YoY)'])) * 100
                    except:
                        metrics['Revenue Growth %'] = 0
            
            # Gross Profit
            gp_val = safe_get(latest, ['Gross Profit', 'grossProfit', 'GrossProfit'])
            metrics['Gross Profit'] = gp_val or 0
            if metrics.get('Revenue (Latest)', 0) != 0:
                try:
                    metrics['Gross Margin %'] = (metrics['Gross Profit'] / metrics['Revenue (Latest)']) * 100
                except:
                    metrics['Gross Margin %'] = 0
            
            # Operating Income
            op_val = safe_get(latest, ['Operating Income', 'operatingProfit', 'OperatingProfit', 'Operating Income'])
            metrics['Operating Income'] = op_val or 0
            if metrics.get('Revenue (Latest)', 0) != 0:
                try:
                    metrics['Operating Margin %'] = (metrics['Operating Income'] / metrics['Revenue (Latest)']) * 100
                except:
                    metrics['Operating Margin %'] = 0
            
            # EBITDA
            ebitda_val = safe_get(latest, ['EBITDA', 'ebitda'])
            metrics['EBITDA'] = ebitda_val or 0
            if metrics.get('Revenue (Latest)', 0) != 0:
                try:
                    metrics['EBITDA Margin %'] = (metrics['EBITDA'] / metrics['Revenue (Latest)']) * 100
                except:
                    metrics['EBITDA Margin %'] = 0
            
            # Net Income
            net_val = safe_get(latest, ['Net Income', 'netProfit', 'NetIncome', 'Net Income'])
            metrics['Net Income'] = net_val or 0
            if metrics.get('Revenue (Latest)', 0) != 0:
                try:
                    metrics['Net Margin %'] = (metrics['Net Income'] / metrics['Revenue (Latest)']) * 100
                except:
                    metrics['Net Margin %'] = 0
            
            # Net Income YoY and growth if available
            if yoy is not None:
                prev_net = safe_get(yoy, ['Net Income', 'netProfit', 'NetIncome'])
                metrics['Net Income (YoY)'] = prev_net or 0
                if metrics['Net Income (YoY)'] != 0:
                    try:
                        metrics['Net Income Growth %'] = ((metrics['Net Income'] - metrics['Net Income (YoY)']) / abs(metrics['Net Income (YoY)'])) * 100
                    except:
                        metrics['Net Income Growth %'] = 0
    
    # Balance sheet (best-effort from yahoo fallback; NSE returns not provided here)
    if financials.get('balance_sheet') is not None and len(financials['balance_sheet']) >= 1:
        bal = financials['balance_sheet']
        latest = bal.iloc[-1]
        
        if 'Total Assets' in bal.columns:
            metrics['Total Assets'] = latest.get('Total Assets', 0)
        if 'Stockholders Equity' in bal.columns:
            metrics['Shareholders Equity'] = latest.get('Stockholders Equity', 0)
        
        if 'Current Assets' in bal.columns and 'Current Liabilities' in bal.columns:
            current_assets = latest.get('Current Assets', 0)
            current_liabilities = latest.get('Current Liabilities', 0)
            if current_liabilities != 0:
                metrics['Current Ratio'] = current_assets / current_liabilities
        
        if 'Cash And Cash Equivalents' in bal.columns:
            metrics['Cash & Equivalents'] = latest.get('Cash And Cash Equivalents', 0)
        
        if 'Total Debt' in bal.columns:
            metrics['Total Debt'] = latest.get('Total Debt', 0)
            if metrics.get('Shareholders Equity', 0) != 0:
                metrics['Debt to Equity'] = metrics['Total Debt'] / metrics['Shareholders Equity']
    
    # Cashflow (best-effort)
    if financials.get('cashflow') is not None and len(financials['cashflow']) >= 1:
        cf = financials['cashflow']
        latest = cf.iloc[-1]
        
        if 'Operating Cash Flow' in cf.columns:
            metrics['Operating Cash Flow'] = latest.get('Operating Cash Flow', 0)
        if 'Free Cash Flow' in cf.columns:
            metrics['Free Cash Flow'] = latest.get('Free Cash Flow', 0)
        if 'Capital Expenditure' in cf.columns:
            metrics['Capital Expenditure'] = latest.get('Capital Expenditure', 0)
    
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
                    st.info(f"📅 Financial data last updated: {financials['last_updated']} (preferred source: NSE if available, else Yahoo Finance)")
                
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
                            growth = f"+{metrics.get('Revenue Growth %', 0):.1f}%" if metrics.get('Revenue Growth %', 0) > 0 else f"{metrics.get('Revenue Growth %', 0):.1f}%"
                            st.metric("Revenue (Latest Qtr)", format_number(metrics['Revenue (Latest)']), growth)
                    
                    with col2:
                        if 'Net Income' in metrics:
                            growth = f"+{metrics.get('Net Income Growth %', 0):.1f}%" if metrics.get('Net Income Growth %', 0) > 0 else f"{metrics.get('Net Income Growth %', 0):.1f}%"
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
                        if 'Gross Margin %' in metrics:
                            st.metric("Gross Margin", f"{metrics['Gross Margin %']:.2f}%")
                    with col2:
                        if 'Operating Margin %' in metrics:
                            st.metric("Operating Margin", f"{metrics['Operating Margin %']:.2f}%")
                    with col3:
                        if 'EBITDA Margin %' in metrics:
                            st.metric("EBITDA Margin", f"{metrics['EBITDA Margin %']:.2f}%")
                    with col4:
                        if 'Net Margin %' in metrics:
                            st.metric("Net Margin", f"{metrics['Net Margin %']:.2f}%")
                    
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
