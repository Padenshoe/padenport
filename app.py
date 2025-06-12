import streamlit as st
import requests
import re
import time
from openai import OpenAI, RateLimitError

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
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    return res.json().get("articles", [])[:3]

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
tickers_input = st.text_input("🖊 Enter ticker symbols (comma-separated)", "AAPL, TSLA, NVDA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
tickers = tickers[:2]  # Limit to 2 tickers to stay under 3 RPM

if tickers:
    for ticker in tickers:
        st.subheader(f"📈 {ticker}")
        articles = fetch_news(ticker)
        if not articles:
            st.warning("No recent news found.")
            continue
        for article in articles:
            summary, sentiment = analyze_article(ticker, article)
            st.markdown(
                f"**{article['title']}**  \n"
                f"{summary}  \n"
                f"*Sentiment: `{sentiment.upper()}`*  \n"
                f"[Read more]({article['url']})"
            )
            st.divider()
