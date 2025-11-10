# app.py - Clean Table-Based Stock Analyzer with Perplexity API
"""
Indian Stock Analyzer with structured table display and PEAD scoring.
Focused on earnings/results news only.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import requests
import json
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

st.set_page_config(page_title="Stock Analyzer - PEAD Score", page_icon="📊", layout="wide")

# ------------------------ Utils ------------------------
def get_secret(key):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

# ------------------------ Perplexity API ------------------------
@st.cache_data(ttl=3600)
def query_perplexity(prompt, model="sonar-pro"):
    api_key = get_secret("PERPLEXITY_API_KEY")
    if not api_key:
        return None, "API key not found"
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a financial data analyst. Provide structured, numerical data in JSON format. Be precise and concise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 3000
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
def get_financial_metrics_table(symbol, company_name):
    """Get structured financial metrics for last 8 quarters"""
    stock_name = symbol.replace(".NS", "").replace(".BO", "")
    
    prompt = f"""
For {company_name} (NSE: {stock_name}), provide ONLY the following data in JSON format for the last 8 quarters:

{{
  "company": "{company_name}",
  "quarters": [
    {{
      "quarter": "Q2 FY25",
      "result_date": "2024-10-15",
      "forward_pe": 25.6,
      "sales_yoy": 12.5,
      "sales_qoq": 3.2,
      "np_yoy": 15.8,
      "np_qoq": 4.1,
      "ebitda_yoy": 14.2,
      "ebitda_qoq": 3.8,
      "cfo": 1250.5,
      "revenue": 5000,
      "net_profit": 800,
      "ebitda": 1200,
      "eps": 12.5,
      "roe": 18.5,
      "roce": 22.3,
      "debt_to_equity": 0.35,
      "current_ratio": 1.8
    }},
    ... (7 more quarters)
  ]
}}

Provide actual numbers only. All growth rates in percentage. All amounts in Crores INR.
"""
    
    content, citations = query_perplexity(prompt)
    return content, citations

@st.cache_data(ttl=3600)
def get_pead_score(symbol, company_name):
    """Calculate PEAD score"""
    stock_name = symbol.replace(".NS", "").replace(".BO", "")
    
    prompt = f"""
For {company_name} (NSE: {stock_name}), calculate PEAD score (0-100):

Analyze last 8 quarterly results:
1. Earnings surprise % (actual EPS vs consensus)
2. Stock price movement: Day 1, Week 1, Month 1 post-announcement
3. Pattern consistency
4. Calculate PEAD Score (0-100)

Return JSON:
{{
  "pead_score": 76.5,
  "interpretation": "High positive drift",
  "last_8_quarters": [
    {{
      "quarter": "Q2 FY25",
      "surprise_pct": 5.2,
      "drift_1d": 2.1,
      "drift_7d": 4.5,
      "drift_30d": 8.3
    }},
    ...
  ]
}}
"""
    
    content, citations = query_perplexity(prompt)
    return content, citations

@st.cache_data(ttl=3600)
def get_earnings_news_only(symbol, company_name):
    """Get ONLY earnings/results related news"""
    stock_name = symbol.replace(".NS", "").replace(".BO", "")
    
    prompt = f"""
For {company_name} (NSE: {stock_name}), provide ONLY earnings-related news from last 6 months:

Include ONLY:
- Quarterly results announcements
- Earnings call schedules/transcripts
- Financial results press releases
- Profit/loss announcements

EXCLUDE:
- General company news
- Product launches
- Appointments
- Acquisitions

Return JSON:
{{
  "earnings_news": [
    {{
      "date": "2024-10-15",
      "title": "Q2 FY25 Results - Net Profit up 15%",
      "link": "url",
      "source": "NSE/BSE/Company Website"
    }},
    ...
  ]
}}

Maximum 10 most recent items.
"""
    
    content, citations = query_perplexity(prompt)
    return content, citations

# ------------------------ Data Parsing ------------------------
def parse_json_from_text(text):
    """Extract JSON from Perplexity response"""
    try:
        # Try direct parse
        return json.loads(text)
    except:
        # Extract JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

def create_metrics_dataframe(financial_data):
    """Convert financial data to pandas DataFrame"""
    data = parse_json_from_text(financial_data)
    if not data or "quarters" not in data:
        return None
    
    quarters = data["quarters"]
    df = pd.DataFrame(quarters)
    
    # Reorder columns for display
    column_order = [
        "quarter", "result_date", "forward_pe", 
        "sales_yoy", "sales_qoq", "np_yoy", "np_qoq",
        "ebitda_yoy", "ebitda_qoq", "cfo",
        "revenue", "net_profit", "ebitda", "eps",
        "roe", "roce", "debt_to_equity", "current_ratio"
    ]
    
    # Keep only available columns
    available_cols = [col for col in column_order if col in df.columns]
    df = df[available_cols]
    
    return df

# ------------------------ UI ------------------------
st.title("📊 Stock Financial Analyzer - PEAD Score")
st.caption("Clean, structured financial metrics with earnings-focused news")

# Input
col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.text_input("Stock Symbol", value="SUNPHARMA.NS", placeholder="e.g., RELIANCE.NS, TCS.NS")
with col2:
    st.markdown("### ")
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze_btn:
    if not symbol:
        st.error("Please enter a stock symbol")
        st.stop()
    
    if not get_secret("PERPLEXITY_API_KEY"):
        st.error("⚠️ Perplexity API key not configured. Add PERPLEXITY_API_KEY to secrets.")
        st.stop()
    
    symbol = symbol.upper().strip()
    stock_name = symbol.replace(".NS", "").replace(".BO", "")
    
    # Progress
    with st.spinner("🔄 Fetching financial data..."):
        financial_data, fin_citations = get_financial_metrics_table(symbol, stock_name)
        pead_data, pead_citations = get_pead_score(symbol, stock_name)
        news_data, news_citations = get_earnings_news_only(symbol, stock_name)
    
    # Display company header
    st.markdown("---")
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    with col_h1:
        st.markdown(f"## 🏢 {stock_name}")
    with col_h2:
        # Parse PEAD score
        pead_json = parse_json_from_text(pead_data) if pead_data else None
        if pead_json and "pead_score" in pead_json:
            score = pead_json["pead_score"]
            st.metric("PEAD Score", f"{score}/100", 
                     delta="High" if score > 70 else "Medium" if score > 40 else "Low")
        else:
            st.metric("PEAD Score", "N/A")
    with col_h3:
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Financial Metrics Table
    st.markdown("---")
    st.subheader("📈 Financial Metrics (Last 8 Quarters)")
    
    if financial_data:
        df = create_metrics_dataframe(financial_data)
        
        if df is not None and not df.empty:
            # Format percentage and currency columns
            display_df = df.copy()
            
            # Format columns manually
            pct_cols = ["sales_yoy", "sales_qoq", "np_yoy", "np_qoq", "ebitda_yoy", "ebitda_qoq", "roe", "roce"]
            for col in pct_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
            
            curr_cols = ["cfo", "revenue", "net_profit", "ebitda"]
            for col in curr_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"₹{x:.0f} Cr" if pd.notna(x) else "N/A")
            
            if "eps" in display_df.columns:
                display_df["eps"] = display_df["eps"].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "N/A")
            
            if "forward_pe" in display_df.columns:
                display_df["forward_pe"] = display_df["forward_pe"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
            
            if "debt_to_equity" in display_df.columns:
                display_df["debt_to_equity"] = display_df["debt_to_equity"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            if "current_ratio" in display_df.columns:
                display_df["current_ratio"] = display_df["current_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            # Display with clean formatting
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Key metrics summary
            st.markdown("### 📊 Key Highlights (Latest Quarter)")
            latest = df.iloc[0]
            
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Revenue YoY", f"{latest.get('sales_yoy', 0):.1f}%")
            m2.metric("NP YoY", f"{latest.get('np_yoy', 0):.1f}%")
            m3.metric("EBITDA YoY", f"{latest.get('ebitda_yoy', 0):.1f}%")
            m4.metric("ROE", f"{latest.get('roe', 0):.1f}%")
            m5.metric("ROCE", f"{latest.get('roce', 0):.1f}%")
            m6.metric("D/E Ratio", f"{latest.get('debt_to_equity', 0):.2f}")
            
            # Data sources
            if fin_citations:
                with st.expander("📚 Data Sources"):
                    for i, cite in enumerate(fin_citations[:5], 1):
                        st.caption(f"{i}. {cite}")
        else:
            st.warning("⚠️ Unable to parse financial data into table format")
            st.text(financial_data[:500] + "..." if financial_data else "No data")
    else:
        st.error("❌ Failed to fetch financial data")
    
    # PEAD Analysis
    st.markdown("---")
    st.subheader("🎯 PEAD Analysis")
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        if pead_data:
            pead_json = parse_json_from_text(pead_data)
            if pead_json:
                score = pead_json.get("pead_score", "N/A")
                interpretation = pead_json.get("interpretation", "No interpretation")
                
                st.metric("PEAD Score", f"{score}/100" if isinstance(score, (int, float)) else score)
                st.info(f"**Pattern:** {interpretation}")
            else:
                st.text(pead_data[:300] if pead_data else "No data")
    
    with col_p2:
        if pead_data:
            pead_json = parse_json_from_text(pead_data)
            if pead_json and "last_8_quarters" in pead_json:
                pead_df = pd.DataFrame(pead_json["last_8_quarters"])
                
                # Format columns manually
                for col in ["surprise_pct", "drift_1d", "drift_7d", "drift_30d"]:
                    if col in pead_df.columns:
                        pead_df[col] = pead_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                
                st.dataframe(pead_df, use_container_width=True)
    
    # Earnings News ONLY
    st.markdown("---")
    st.subheader("📰 Earnings & Results News Only")
    st.caption("Showing only quarterly results, earnings calls, and profit announcements")
    
    if news_data:
        news_json = parse_json_from_text(news_data)
        if news_json and "earnings_news" in news_json:
            news_items = news_json["earnings_news"]
            
            if news_items:
                for item in news_items[:10]:
                    with st.container():
                        col_n1, col_n2 = st.columns([4, 1])
                        with col_n1:
                            title = item.get("title", "No title")
                            link = item.get("link", "#")
                            st.markdown(f"**[{title}]({link})**")
                            st.caption(f"📅 {item.get('date', 'N/A')} | 📰 {item.get('source', 'N/A')}")
                        with col_n2:
                            st.caption(item.get("date", ""))
                        st.markdown("---")
            else:
                st.info("No recent earnings news found")
        else:
            st.warning("⚠️ Unable to parse news data")
            st.text(news_data[:300] if news_data else "No data")
    else:
        st.error("❌ Failed to fetch earnings news")
    
    # Download option
    st.markdown("---")
    if financial_data:
        st.download_button(
            label="📥 Download Report (JSON)",
            data=financial_data,
            file_name=f"{stock_name}_analysis_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ Not financial advice. Verify data from official sources (NSE/BSE).</p>
    <p>Powered by Perplexity AI</p>
</div>
""", unsafe_allow_html=True)
