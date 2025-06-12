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
import numpy as np
import pandas as pd

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
    query = f"\"{ticker}\" stock"
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={query}&from={from_date}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
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

def parse_positions(text: str) -> dict:
    """Return a mapping of ticker -> shares from raw OCR text.

    The OCR output can vary in column order depending on the source. This parser
    scans each line individually and attempts to pick the numeric token that
    most likely represents the share quantity. If both price and share values
    are present, the first integer-like token after the ticker is used. This
    avoids accidentally grabbing the price when it appears before the share
    count.
    """

    ignore = {
        "shares",
        "share",
        "price",
        "total",
        "cost",
        "qty",
        "quantity",
        "value",
        "s",
        "o",
    }

    share_tokens = {"share", "shares", "shr", "shrs", "sh", "shs"}

    positions: dict[str, float] = {}

    for line in text.splitlines():
        tokens = re.findall(r"[A-Za-z]+|[\d,.]+", line)
        ticker_idx = None
        ticker = None

        for idx, tok in enumerate(tokens):
            if tok.isalpha() and 1 <= len(tok) <= 5 and tok.lower() not in ignore:
                ticker_idx = idx
                ticker = tok.upper()
                break

        if ticker_idx is not None:
            share_token = None

            for idx in range(ticker_idx + 1, len(tokens)):
                tok_lower = tokens[idx].lower()
                if tok_lower in share_tokens:
                    if idx > ticker_idx + 1 and re.fullmatch(r"[\d,.]+", tokens[idx - 1]):
                        share_token = tokens[idx - 1]
                        break
                elif re.fullmatch(r"[\d,.]+", tokens[idx]) and idx + 1 < len(tokens) and tokens[idx + 1].lower() in share_tokens:
                    share_token = tokens[idx]
                    break
                if tokens[idx].lower() in {"share", "shares"}:
                    if idx > ticker_idx + 1 and re.fullmatch(r"[\d,.]+", tokens[idx - 1]):
                        share_token = tokens[idx - 1]
                        break

            if share_token is None:
                numeric_tokens = [t for t in tokens[ticker_idx + 1 :] if re.fullmatch(r"[\d,.]+", t)]
                if numeric_tokens:
                    share_token = next((n for n in numeric_tokens if "." not in n), numeric_tokens[0])

            if share_token is not None:
                try:
                    shares = float(share_token.replace(",", ""))
                    positions[ticker] = positions.get(ticker, 0.0) + shares
                except ValueError:
                    pass

    return positions

def extract_text_from_image(image_bytes):
    """Return OCR text using pytesseract or easyocr as fallback."""
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
    except pytesseract.pytesseract.TesseractNotFoundError:
        try:
            import easyocr
        except ModuleNotFoundError:
            raise RuntimeError(
                "EasyOCR fallback unavailable. Install the 'easyocr' package."
            )
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            img = np.array(Image.open(io.BytesIO(image_bytes)))
            return "\n".join(reader.readtext(img, detail=0, paragraph=True))
        except Exception as e:
            raise RuntimeError(f"EasyOCR failed: {e}")

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
            text_from_image = extract_text_from_image(image_bytes)
            with st.expander("OCR Result", expanded=False):
                st.text_area("Raw text", text_from_image, height=100, key="ocr")

            ocr_positions = parse_positions(text_from_image)
            if ocr_positions:
                summary_rows = []
                for t, s in ocr_positions.items():
                    positions[t] = positions.get(t, 0) + s
                    price, *_ = fetch_stock_info(t)
                    total_val = price * s if price else 0.0
                    summary_rows.append(
                        {"Ticker": t, "Shares": s, "Total Price": f"${total_val:,.2f}"}
                    )

                st.markdown("#### Parsed Summary")
                st.table(pd.DataFrame(summary_rows))
            else:
                st.info("No positions found in screenshot.")
        except Exception as e:
            st.error(f"OCR failed: {e}")

    total = 0.0

    if positions:
        st.markdown("### Portfolio Summary")
        header = st.columns([2, 2, 2, 2])
        header[0].markdown("**Ticker**")
        header[1].markdown("**Shares**")
        header[2].markdown("**Price**")
        header[3].markdown("**Total**")

        for t, shares in list(positions.items()):
            price, *_ = fetch_stock_info(t)
            value = price * shares if price else 0.0
            total += value
            row = st.columns([2, 2, 2, 2])
            row[0].write(t)
            row[1].write(shares)
            row[2].write(f"${price:,.2f}" if price else "N/A")
            row[3].write(f"${value:,.2f}")

        st.write(f"**Total Portfolio: ${total:,.2f}**")
        if st.button("Import Portfolio", key="import_portfolio"):
            st.session_state["tickers_input"] = ", ".join(positions.keys())
    else:
        st.info("No positions added.")


tickers_input = st.text_input("🖊 Enter ticker symbols (comma-separated)", key="tickers_input")
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
