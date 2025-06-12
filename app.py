import streamlit as st
import openai
import requests
import re

# Set up page
st.set_page_config(page_title="PadenPort", layout="wide")
st.title("📊 PadenPort - Stock News Sentiment Dashboard")

# Set API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

# --- Functions ---

def fetch_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    return res.json().get("articles", [])[:3]

def analyze_article(ticker, article):
    content = article.get("content") or article.get("description") or "No content available."
    prompt = f"""
    Summarize the following news about {ticker} and determine if it's good, bad, or neutral for the stock price.

    Title: {article['title']}
    Content: {content}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    summary = response.choices[0].message.content
    sentiment = "neutral"
    if re.search(r'\b(good|positive)\b', summary, re.I):
        sentiment = "good"
    elif re.search(r'\b(bad|negative)\b', summary, re.I):
        sentiment = "bad"
    return summary, sentiment

# --- UI ---

tickers_input = st.text_input("🖊 Enter ticker symbols (comma-separated)", "AAPL, TSLA, NVDA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

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
