import streamlit as st
import openai
import requests
from PIL import Image
import pytesseract
import re

st.set_page_config(page_title="PadenPort", layout="wide")
st.title("📊 PadenPort - Stock News Sentiment Dashboard")

openai.api_key = st.secrets["OPENAI_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

def extract_tickers_from_image(image):
    text = pytesseract.image_to_string(image)
    tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
    return sorted(set(tickers))

def fetch_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    return res.json().get("articles", [])[:3]

def analyze_article(ticker, article):
    prompt = f"Summarize the following news about {ticker} and determine if it's good, bad, or neutral for the stock price.\n\nTitle: {article['title']}\nContent: {article['content'] or article['description']}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        temperature=0.3
    )
    summary = response["choices"][0]["message"]["content"]
    sentiment = "neutral"
    if re.search(r'\b(good|positive)\b', summary, re.I):
        sentiment = "good"
    elif re.search(r'\b(bad|negative)\b', summary, re.I):
        sentiment = "bad"
    return summary, sentiment

# UI for upload
uploaded_file = st.file_uploader("📷 Upload a screenshot of your portfolio", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Portfolio Screenshot", use_column_width=True)
    tickers = extract_tickers_from_image(image)
    st.success(f"✅ Detected Tickers: {', '.join(tickers)}")

    for ticker in tickers:
        st.subheader(f"📈 {ticker}")
        articles = fetch_news(ticker)
        for article in articles:
            summary, sentiment = analyze_article(ticker, article)
            st.markdown(
    f"**{article['title']}**  \n"
    f"{summary}  \n"
    f"*Sentiment: `{sentiment.upper()}`*  \n"
    f"[Read more]({article['url']})"
)
{summary}  
*Sentiment: `{sentiment.upper()}`*  
[Read more]({article['url']})")
            st.divider()
