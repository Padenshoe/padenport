import streamlit as st
import requests
import re
import time
from datetime import datetime, timedelta
from openai import OpenAI, RateLimitError
import yfinance as yf
import pytesseract
from PIL import Image
import io

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

def parse_positions(text):
    """Parse lines like 'AAPL 10' and return dict of ticker -> shares."""
    positions = {}
    for line in text.splitlines():
        match = re.search(r"([A-Za-z]+)\s+(\d+(?:\.\d+)?)", line)
        if match:
            ticker, shares = match.groups()
            positions[ticker.upper()] = positions.get(ticker.upper(), 0) + float(shares)
    return positions

# === UI ===
# --- Sidebar positions input ---
with st.sidebar:
    st.header("📊 My Positions")

    positions = st.session_state.setdefault("positions", {})

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Ticker", key="ticker_entry")
    with col2:
        st.number_input("Shares", min_value=0.0, step=1.0, key="shares_entry")

    def add_position():
        ticker = st.session_state.get("ticker_entry", "").strip()
        shares = st.session_state.get("shares_entry", 0.0)
        if ticker:
            positions[ticker.upper()] = positions.get(ticker.upper(), 0) + shares
        st.session_state["ticker_entry"] = ""
        st.session_state["shares_entry"] = 0.0

    st.button("Add", on_click=add_position)

    uploaded_image = st.file_uploader("Or upload screenshot", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        try:
            text_from_image = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
            st.text_area("OCR Result", text_from_image, height=100, key="ocr")
            ocr_positions = parse_positions(text_from_image)
            for t, s in ocr_positions.items():
                positions[t] = positions.get(t, 0) + s
        except Exception as e:
            st.error(f"OCR failed: {e}")

    total = 0.0
    remove_keys = []
    for t, shares in positions.items():
        price, *_ = fetch_stock_info(t)
        if price:
            value = price * shares
            total += value
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{t}: {shares} × ${price:.2f} = ${value:,.2f}")
            with col2:
                if st.button("Remove", key=f"remove_{t}"):
                    remove_keys.append(t)
    for rk in remove_keys:
        positions.pop(rk, None)

    if positions:
        st.write(f"**Total Value: ${total:,.2f}**")

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
                    st.image(article["urlToImage"], use_container_width=True)
                st.markdown(f"**[{article['title']}]({article['url']})**")
                published = article.get("publishedAt", "")[:10]
                source = article.get("source", {}).get("name", "")
                st.caption(f"{source}, {published}")
                st.write(summary)
                st.write(f"*Sentiment: `{sentiment.upper()}`*")
        st.divider()