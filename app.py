# app.py - Enhanced Indian Stock Analyzer with Perplexity API
"""
Comprehensive Indian Stock Analyzer using Perplexity API for accurate financial data.
Includes 5-year historical data, PEAD scoring, and extensive financial metrics.

Install requirements:
streamlit
pandas
python-dotenv
requests
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import requests
import json
import re

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Advanced Stock Analyzer with PEAD", page_icon="📊", layout="wide")

# ------------------------ Utils ------------------------
def get_secret(key):
    """Read secret from Streamlit secrets or environment"""
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

def format_number(num):
    """Format numbers in Indian currency format"""
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

# ------------------------ Perplexity API Integration ------------------------
@st.cache_data(ttl=3600)
def query_perplexity(prompt, model="sonar-pro"):
    """
    Query Perplexity API with a financial analysis prompt.
    Returns the API response content.
    """
    api_key = get_secret("PERPLEXITY_API_KEY")
    if not api_key:
        return None, "Perplexity API key not found. Add PERPLEXITY_API_KEY to secrets."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a financial analyst expert specializing in Indian stock markets. Provide accurate, structured financial data with sources. Always include numbers in standardized format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])
        return content, citations
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def get_comprehensive_financials(symbol, company_name):
    """
    Fetch comprehensive 5-year financial data using Perplexity API.
    Returns structured data for all key metrics.
    """
    stock_name = symbol.replace(".NS", "").replace(".BO", "")
    
    prompt = f"""
For {company_name} (NSE: {stock_name}), provide the following financial data for the last 20 quarters (5 years):

**QUARTERLY DATA (Last 20 Quarters):**
1. Quarter name/date
2. Total Revenue/Sales
3. YoY Revenue Growth %
4. QoQ Revenue Growth %
5. Gross Profit
6. Gross Margin %
7. EBITDA
8. EBITDA Margin %
9. YoY EBITDA Growth %
10. QoQ EBITDA Growth %
11. Operating Profit (EBIT)
12. Operating Margin %
13. Net Profit (PAT)
14. Net Margin %
15. YoY Net Profit Growth %
16. QoQ Net Profit Growth %
17. EPS (Earnings Per Share)

**KEY RATIOS (Latest Quarter):**
1. Current Ratio
2. Quick Ratio
3. Debt-to-Equity Ratio
4. Interest Coverage Ratio
5. Return on Equity (ROE) %
6. Return on Assets (ROA) %
7. Return on Capital Employed (ROCE) %
8. Asset Turnover Ratio
9. Inventory Turnover Ratio
10. Receivables Turnover Days
11. Payables Turnover Days
12. Cash Conversion Cycle (Days)

**PROFITABILITY METRICS (Latest Quarter):**
1. Operating Leverage
2. Financial Leverage
3. Total Leverage
4. Break-even Point (if available)
5. Contribution Margin %

**CASH FLOW METRICS (Last 5 Years Annual):**
1. Operating Cash Flow
2. Investing Cash Flow
3. Financing Cash Flow
4. Free Cash Flow
5. Cash Flow from Operations to Net Profit Ratio

**VALUATION METRICS (Current):**
1. Market Cap
2. P/E Ratio
3. P/B Ratio
4. EV/EBITDA
5. Price-to-Sales Ratio
6. Dividend Yield %
7. Payout Ratio %

**RECENT RESULTS:**
- Last 4 quarterly results announcement dates
- Links to results/investor presentations

Format the response as structured JSON or clear tables with actual numbers. Include data sources.
"""
    
    content, citations = query_perplexity(prompt)
    return content, citations

@st.cache_data(ttl=3600)
def get_company_overview(symbol, company_name):
    """Get company overview and business description"""
    stock_name = symbol.replace(".NS", "").replace(".BO", "")
    
    prompt = f"""
For {company_name} (NSE: {stock_name}), provide:

1. **Business Overview:** Brief description of what the company does (2-3 sentences)
2. **Sector:** Primary sector
3. **Industry:** Specific industry
4. **Market Cap:** Current market capitalization
5. **Key Products/Services:** Top 3-5 products or services
6. **Geographic Presence:** Main markets (India/International breakdown)
7. **Number of Employees:** Total employee count
8. **Competitors:** Top 3-5 main competitors in India
9. **Recent Major Developments:** Any significant news in last 3 months (acquisitions, expansions, partnerships)

Keep it concise and factual with sources.
"""
    
    content, citations = query_perplexity(prompt)
    return content, citations

@st.cache_data(ttl=3600)
def calculate_pead_score(symbol, company_name):
    """
    Calculate Post-Earnings Announcement Drift (PEAD) Score.
    This analyzes earnings surprises and subsequent stock performance.
    """
    stock_name = symbol.replace(".NS", "").replace(".BO", "")
    
    prompt = f"""
For {company_name} (NSE: {stock_name}), analyze the Post-Earnings Announcement Drift (PEAD):

1. **Last 8 Quarterly Earnings:**
   - Quarter date
   - Actual EPS vs Analyst Consensus EPS
   - Earnings Surprise % (Beat/Miss)
   - Stock price change 1 day after announcement
   - Stock price change 5 days after announcement
   - Stock price change 30 days after announcement

2. **PEAD Pattern Analysis:**
   - Does the stock show consistent drift in the direction of earnings surprise?
   - Average drift magnitude for positive surprises
   - Average drift magnitude for negative surprises
   - PEAD Score (0-100, where 100 = strongest positive drift pattern)

3. **Market Reaction Quality:**
   - Does market under-react or over-react initially?
   - Time taken for full price adjustment (days)
   - Volatility during announcement periods

4. **Latest Quarter Assessment:**
   - Most recent earnings surprise
   - Current drift status
   - Expected continuation probability

Provide numerical PEAD score and interpretation.
"""
    
    content, citations = query_perplexity(prompt)
    return content, citations

# ------------------------ Streamlit UI ------------------------
st.title("📊 Advanced Indian Stock Analyzer with PEAD Scoring")
st.markdown("### *Powered by Perplexity AI for Accurate Financial Intelligence*")

st.sidebar.info("""
🚀 **Features:**
- 5-year quarterly financial history
- 30+ key financial metrics & ratios
- PEAD (Post-Earnings Announcement Drift) scoring
- QoQ and YoY growth comparisons
- Cash flow analysis
- Valuation metrics
- Recent results with links

⚙️ **Setup:**
Add `PERPLEXITY_API_KEY` to Streamlit Secrets
""")

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    symbol_input = st.text_input("Enter Stock Symbol", value="SUNPHARMA.NS", 
                                 help="Use .NS for NSE or .BO for BSE (e.g., RELIANCE.NS, TCS.BO)")
with col2:
    st.markdown("**Verify Data At:**")
    st.markdown("[BSE India](https://www.bseindia.com) | [NSE India](https://www.nseindia.com)")

if st.button("🔍 Analyze Stock", type="primary"):
    if not symbol_input:
        st.warning("⚠️ Please enter a stock symbol (e.g., SUNPHARMA.NS)")
    else:
        symbol = symbol_input.upper().strip()
        stock_name = symbol.replace(".NS", "").replace(".BO", "")
        
        # Check API key
        if not get_secret("PERPLEXITY_API_KEY"):
            st.error("❌ Perplexity API key not configured. Please add PERPLEXITY_API_KEY to your Streamlit secrets.")
            st.stop()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Company Overview
        status_text.text("📋 Fetching company overview...")
        progress_bar.progress(10)
        company_overview, overview_citations = get_company_overview(symbol, stock_name)
        
        # Step 2: Comprehensive Financials
        status_text.text("💰 Fetching 5-year financial data...")
        progress_bar.progress(40)
        financials, financial_citations = get_comprehensive_financials(symbol, stock_name)
        
        # Step 3: PEAD Score
        status_text.text("📈 Calculating PEAD score...")
        progress_bar.progress(70)
        pead_analysis, pead_citations = calculate_pead_score(symbol, stock_name)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Clear progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # ======================== DISPLAY RESULTS ========================
        
        # Company Header
        st.header(f"🏢 {stock_name}")
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
        
        # Company Overview Section
        st.markdown("---")
        st.subheader("📊 Company Overview")
        if company_overview:
            st.markdown(company_overview)
            if overview_citations:
                with st.expander("📚 Sources for Company Overview"):
                    for i, cite in enumerate(overview_citations, 1):
                        st.markdown(f"{i}. {cite}")
        else:
            st.warning("Unable to fetch company overview")
        
        # Financial Metrics Section
        st.markdown("---")
        st.subheader("💰 Comprehensive Financial Analysis (5-Year Data)")
        if financials:
            st.markdown(financials)
            if financial_citations:
                with st.expander("📚 Sources for Financial Data"):
                    for i, cite in enumerate(financial_citations, 1):
                        st.markdown(f"{i}. {cite}")
        else:
            st.warning("Unable to fetch financial data")
        
        # PEAD Score Section
        st.markdown("---")
        st.subheader("🎯 PEAD (Post-Earnings Announcement Drift) Analysis")
        st.info("""
        **What is PEAD?** PEAD measures how stock prices continue to drift in the direction of an earnings surprise 
        after the initial announcement. A high PEAD score indicates predictable post-earnings momentum, which can be 
        valuable for trading strategies around earnings dates.
        """)
        
        if pead_analysis:
            st.markdown(pead_analysis)
            if pead_citations:
                with st.expander("📚 Sources for PEAD Analysis"):
                    for i, cite in enumerate(pead_citations, 1):
                        st.markdown(f"{i}. {cite}")
        else:
            st.warning("Unable to calculate PEAD score")
        
        # Key Insights Summary
        st.markdown("---")
        st.subheader("💡 AI-Generated Key Insights")
        
        insight_prompt = f"""
Based on the financial analysis of {stock_name}, provide:

1. **Overall Financial Health Score (0-100):** With brief justification
2. **Top 3 Strengths:** What's the company doing well?
3. **Top 3 Concerns:** What are the red flags or areas of concern?
4. **Growth Trajectory:** Is the company in growth, stable, or declining phase?
5. **Competitive Position:** How does it compare to industry peers?
6. **Investment Recommendation:** Based purely on fundamentals (Buy/Hold/Sell with reasoning)

Keep it concise, actionable, and data-driven.
"""
        
        with st.spinner("🤖 Generating AI insights..."):
            insights, insight_citations = query_perplexity(insight_prompt)
            if insights:
                st.markdown(insights)
                if insight_citations:
                    with st.expander("📚 Sources for Insights"):
                        for i, cite in enumerate(insight_citations, 1):
                            st.markdown(f"{i}. {cite}")
        
        # Download Report Option
        st.markdown("---")
        st.subheader("📥 Export Analysis")
        
        report_text = f"""
COMPREHENSIVE STOCK ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
Stock: {stock_name}
Symbol: {symbol}

{'='*80}
COMPANY OVERVIEW
{'='*80}
{company_overview or 'N/A'}

{'='*80}
FINANCIAL ANALYSIS (5-YEAR DATA)
{'='*80}
{financials or 'N/A'}

{'='*80}
PEAD ANALYSIS
{'='*80}
{pead_analysis or 'N/A'}

{'='*80}
KEY INSIGHTS
{'='*80}
{insights or 'N/A'}

{'='*80}
DISCLAIMER
{'='*80}
This report is for informational purposes only and does not constitute financial advice.
Always verify data from official sources before making investment decisions.
"""
        
        st.download_button(
            label="📄 Download Full Report (TXT)",
            data=report_text,
            file_name=f"{stock_name}_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool provides financial analysis for informational purposes only. 
    Not financial advice. Always verify critical data from official sources (BSE/NSE/Company filings) before making investment decisions.</p>
    <p>Powered by Perplexity AI | Data sources cited in expandable sections</p>
</div>
""", unsafe_allow_html=True)
