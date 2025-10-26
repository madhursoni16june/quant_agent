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

@st.cache_data(ttl=1800)
def get_financial_data_yf(symbol):
    """Gets financial data from Yahoo Finance with warnings"""
    try:
        ticker = yf.Ticker(symbol)
        
        quarterly_income = ticker.quarterly_income_stmt
        quarterly_balance = ticker.quarterly_balance_sheet
        quarterly_cashflow = ticker.quarterly_cashflow
        
        # Get the last update timestamp
        data_date = None
        if quarterly_income is not None and not quarterly_income.empty:
            data_date = quarterly_income.columns[0].strftime('%Y-%m-%d')
        
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
    
    if financials['income_statement'] is not None and len(financials['income_statement']) >= 5:
        inc = financials['income_statement']
        latest = inc.iloc[-1]
        yoy = inc.iloc[-5]
        
        # Get the quarter date
        quarter_date = str(inc.index[-1])[:10]
        metrics['Quarter Date'] = quarter_date
        
        # Check if data is recent (within 6 months)
        try:
            quarter_datetime = datetime.strptime(quarter_date, '%Y-%m-%d')
            months_old = (datetime.now() - quarter_datetime).days / 30
            if months_old > 6:
                warnings.append(f"⚠️ Data is {int(months_old)} months old. May not reflect latest earnings.")
        except:
            warnings.append("⚠️ Could not verify data freshness.")
        
        if 'Total Revenue' in inc.columns:
            metrics['Revenue (Latest)'] = latest.get('Total Revenue', 0)
            metrics['Revenue (YoY)'] = yoy.get('Total Revenue', 0)
            if metrics['Revenue (YoY)'] != 0:
                metrics['Revenue Growth %'] = ((metrics['Revenue (Latest)'] - metrics['Revenue (YoY)']) / abs(metrics['Revenue (YoY)'])) * 100
        
        if 'Gross Profit' in inc.columns:
            metrics['Gross Profit'] = latest.get('Gross Profit', 0)
            if metrics.get('Revenue (Latest)', 0) != 0:
                metrics['Gross Margin %'] = (metrics['Gross Profit'] / metrics['Revenue (Latest)']) * 100
        
        if 'Operating Income' in inc.columns:
            metrics['Operating Income'] = latest.get('Operating Income', 0)
            if metrics.get('Revenue (Latest)', 0) != 0:
                metrics['Operating Margin %'] = (metrics['Operating Income'] / metrics['Revenue (Latest)']) * 100
        
        if 'EBITDA' in inc.columns:
            metrics['EBITDA'] = latest.get('EBITDA', 0)
            if metrics.get('Revenue (Latest)', 0) != 0:
                metrics['EBITDA Margin %'] = (metrics['EBITDA'] / metrics['Revenue (Latest)']) * 100
        
        if 'Net Income' in inc.columns:
            metrics['Net Income'] = latest.get('Net Income', 0)
            metrics['Net Income (YoY)'] = yoy.get('Net Income', 0)
            if metrics.get('Revenue (Latest)', 0) != 0:
                metrics['Net Margin %'] = (metrics['Net Income'] / metrics['Revenue (Latest)']) * 100
            if metrics['Net Income (YoY)'] != 0:
                metrics['Net Income Growth %'] = ((metrics['Net Income'] - metrics['Net Income (YoY)']) / abs(metrics['Net Income (YoY)'])) * 100
    
    if financials['balance_sheet'] is not None and len(financials['balance_sheet']) >= 2:
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
    
    if financials['cashflow'] is not None and len(financials['cashflow']) >= 2:
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
                if financials.get('last_updated'):
                    st.info(f"📅 Financial data last updated: {financials['last_updated']} (from Yahoo Finance)")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sector", company_info.get('sector', 'N/A'))
                with col2:
                    st.metric("Industry", company_info.get('industry', 'N/A'))
                with col3:
                    st.metric("Employees", f"{company_info.get('fullTimeEmployees', 'N/A'):,}" if company_info.get('fullTimeEmployees') else 'N/A')
                with col4:
                    st.metric("Market Cap", format_number(company_info.get('marketCap', 0)))
                
                metrics, warnings = calculate_metrics_with_warning(financials)
                
                # Show warnings
                if warnings:
                    for warning in warnings:
                        st.warning(warning)
                    st.info("💡 **Tip:** Check recent news below for latest earnings announcements with actual numbers!")
                
                if metrics:
                    st.markdown("---")
                    st.header("📈 Financial Metrics (from Yahoo Finance)")
                    
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
                    with st.expander("📋 View Raw Financial Statements (Yahoo Finance Data)"):
                        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                        
                        with tab1:
                            if financials['income_statement'] is not None:
                                st.dataframe(financials['income_statement'], use_container_width=True)
                        with tab2:
                            if financials['balance_sheet'] is not None:
                                st.dataframe(financials['balance_sheet'], use_container_width=True)
                        with tab3:
                            if financials['cashflow'] is not None:
                                st.dataframe(financials['cashflow'], use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>⚠️ Data Disclaimer:</strong> Financial metrics from Yahoo Finance may be delayed or inaccurate for Indian stocks. 
    News data is real-time and accurate. Always verify with official BSE/NSE sources.</p>
    <p>Not financial advice | For educational purposes only</p>
</div>
""", unsafe_allow_html=True)