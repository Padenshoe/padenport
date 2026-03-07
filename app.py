import difflib
from typing import List, Dict

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Intuit Support Copilot", layout="wide")
st.title("🎧 Intuit Support Copilot")
st.caption(
    "Record customer questions, find your best prepared response, and summarize clearly in real time."
)


def default_response_library() -> List[Dict[str, str]]:
    return [
        {
            "trigger": "How do I reset my Intuit password?",
            "response": "You can reset your password from the Intuit account recovery page. I'll send the secure reset link, then you can verify by email or phone and create a new password.",
            "summary": "Share reset link, verify identity, and create a new password.",
        },
        {
            "trigger": "Why can’t I connect my bank account?",
            "response": "Bank connection issues are usually caused by temporary bank outages, expired login credentials, or security prompts from your bank. Let's refresh the connection and re-enter credentials.",
            "summary": "Check outage, refresh connection, re-enter bank credentials.",
        },
        {
            "trigger": "How do I update billing information?",
            "response": "Go to Account & Settings, then Billing & Subscription. From there you can update your card and billing details securely.",
            "summary": "Navigate to Billing & Subscription and update payment details.",
        },
    ]


def ensure_state() -> None:
    st.session_state.setdefault("library", default_response_library())
    st.session_state.setdefault("transcript", "")


def best_match(question: str, library: List[Dict[str, str]]) -> Dict[str, str] | None:
    if not question.strip() or not library:
        return None

    best_item = None
    best_score = 0.0
    for item in library:
        score = difflib.SequenceMatcher(
            None, question.lower().strip(), item["trigger"].lower().strip()
        ).ratio()
        if score > best_score:
            best_score = score
            best_item = item

    if best_score < 0.35:
        return None
    return {**best_item, "score": f"{best_score:.0%}"}


def get_openai_client() -> OpenAI | None:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def transcribe_audio(audio_bytes: bytes) -> str:
    client = get_openai_client()
    if client is None:
        return ""

    # OpenAI expects a file-like object with a name attribute.
    import io

    buffer = io.BytesIO(audio_bytes)
    buffer.name = "customer_question.wav"

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=buffer,
    )
    return transcript.text


def ai_rephrase_for_customer(question: str, response: str) -> str:
    client = get_openai_client()
    if client is None:
        return ""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Intuit support assistant. Rewrite support notes into a "
                    "clear, empathetic, concise customer-facing explanation in 3-5 bullet points."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Customer question: {question}\n\n"
                    f"Agent prepared response: {response}\n\n"
                    "Return only the final bullet list."
                ),
            },
        ],
    )
    return completion.choices[0].message.content.strip()


ensure_state()

with st.sidebar:
    st.header("📚 Response Library")
    st.write("Build your reusable answers for common Intuit customer questions.")

    with st.expander("Add new response", expanded=False):
        trigger = st.text_input("Customer question pattern")
        response = st.text_area("Your prepared response", height=120)
        summary = st.text_input("Quick summary")

        if st.button("Save response"):
            if trigger and response:
                st.session_state.library.append(
                    {
                        "trigger": trigger.strip(),
                        "response": response.strip(),
                        "summary": summary.strip() or "No quick summary provided.",
                    }
                )
                st.success("Saved to your response library.")
            else:
                st.warning("Please add both a question pattern and response.")

    if st.session_state.library:
        st.markdown("#### Current library")
        for idx, item in enumerate(st.session_state.library, start=1):
            st.markdown(f"**{idx}.** {item['trigger']}")

st.subheader("1) Capture customer question")
recorded_audio = st.audio_input("Record customer question")
manual_question = st.text_area(
    "Or type/paste customer question",
    value=st.session_state.transcript,
    height=120,
)

if recorded_audio is not None:
    st.audio(recorded_audio)
    if st.button("Transcribe recorded audio"):
        with st.spinner("Transcribing audio..."):
            transcript_text = transcribe_audio(recorded_audio.read())

        if transcript_text:
            st.session_state.transcript = transcript_text
            st.success("Audio transcribed successfully.")
            st.rerun()
        else:
            st.warning(
                "No transcript generated. Add OPENAI_API_KEY in Streamlit secrets or type manually."
            )

question_text = manual_question.strip() or st.session_state.transcript.strip()

st.subheader("2) Suggested support response")
match = best_match(question_text, st.session_state.library)
if question_text:
    st.markdown("**Detected customer question:**")
    st.info(question_text)

if match:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Best match confidence:** {match['score']}")
        st.text_area("Prepared response", value=match["response"], height=180)
    with col2:
        st.text_area("Quick summary", value=match["summary"], height=180)

    if st.button("Generate customer-friendly summary"):
        with st.spinner("Generating summary..."):
            rewritten = ai_rephrase_for_customer(question_text, match["response"])
        if rewritten:
            st.markdown("### 3) Customer-friendly summary")
            st.write(rewritten)
        else:
            st.warning(
                "Unable to generate AI summary. Add OPENAI_API_KEY in Streamlit secrets."
            )
elif question_text:
    st.warning(
        "No strong library match found. Add this scenario to the response library in the sidebar."
    )
else:
    st.info("Record audio or type the customer question to get an instant suggested response.")
