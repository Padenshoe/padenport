import streamlit as st
import requests
import re
import time
from datetime import datetime, timedelta
from openai import OpenAI, RateLimitError
import yfinance as yf

# === CONFIG ===
st.set_page_config(page_title="PadenPort", layout="wide")
st.title("📊 PadenPort - Stock News Sentiment Dashboard")

# MODE: 'live' or 'mock'
MODE = st.secrets.get("MODE", "live")

# === API SETUP ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

# === FUNCTIONS ===
def fetch_news(ticker):
    """Return up to 3 recent articles (within the last week) for the ticker."""
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={ticker}&from={from_date}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    )
    res = requests.get(url)
    return res.json().get("articles", [])[:3]

def fetch_stock_info(ticker):
    """Fetch current stock price and a few key metrics using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("regularMarketPrice")
        pe = info.get("trailingPE")
        market_cap = info.get("marketCap")
        high_52 = info.get("fiftyTwoWeekHigh")
        low_52 = info.get("fiftyTwoWeekLow")
        hist = stock.history(period="1y")["Close"]
        return price, pe, market_cap, high_52, low_52, hist
    except Exception:
        return None, None, None, None, None, None

def analyze_article(ticker, article):
    if MODE == "mock":
        return f"🧪 Mock summary for {ticker}.", "neutral"

    content = article.get("content") or article.get("description") or "No content available."
    prompt = f"""
    Summarize the following news about {ticker} and determine if it's good, bad, or neutral for the stock price.

    Title: {article['title']}
    Content: {content}
    """

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            break
        except RateLimitError:
            st.warning("⚠️ Rate limit hit. Waiting 20 seconds before retrying...")
            time.sleep(20)
    else:
        return "⚠️ Failed after 3 retries due to rate limits.", "neutral"

    summary = response.choices[0].message.content
    sentiment = "neutral"
    if re.search(r'\b(good|positive)\b', summary, re.I):
        sentiment = "good"
    elif re.search(r'\b(bad|negative)\b', summary, re.I):
        sentiment = "bad"
    return summary, sentiment

# === UI ===
tickers_input = st.text_input("🖊 Enter ticker symbols (comma-separated)", "")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
tickers = tickers[:2]  # Limit to 2 tickers to stay under 3 RPM

if tickers:
    for ticker in tickers:
        st.subheader(f"📈 {ticker}")
        price, pe, market_cap, high_52, low_52, hist = fetch_stock_info(ticker)
        if price is not None:
            st.write(f"**Price:** ${price} | **PE:** {pe} | **Market Cap:** {market_cap}")
            if high_52 and low_52:
                st.write(f"**52w High:** ${high_52} | **52w Low:** ${low_52}")
            if hist is not None and not hist.empty:
                st.line_chart(hist)
        else:
            st.write("Stock data unavailable.")

        articles = fetch_news(ticker)
        if not articles:
            st.warning("No recent news found.")
            continue
        cols = st.columns(3)
        for col, article in zip(cols, articles):
            with col:
                summary, sentiment = analyze_article(ticker, article)
                if article.get("urlToImage"):
                    st.image(article["urlToImage"], use_column_width=True)
                st.markdown(f"**[{article['title']}]({article['url']})**")
                published = article.get("publishedAt", "")[:10]
                source = article.get("source", {}).get("name", "")
                st.caption(f"{source}, {published}")
                st.write(summary)
                st.write(f"*Sentiment: `{sentiment.upper()}`*")
        st.divider()
