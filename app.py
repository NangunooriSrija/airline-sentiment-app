import re
import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Airline Tweet Sentiment", page_icon="✈️")


# ---------- Setup (cached so it runs once) ----------
@st.cache_resource
def load_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    model = joblib.load("sentiment_model.pkl")
    le = joblib.load("label_encoder.pkl")
    stop_words = set(stopwords.words("english")) - {"no", "nor", "not"}
    return model, le, stop_words, WordNetLemmatizer()


model, le, stop_words, lemmatizer = load_resources()


# ---------- Same cleaning function as in training ----------
def clean_text(t):
    t = t.lower()
    t = t.replace("&amp;", " ").replace("’", "'")
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    words = [lemmatizer.lemmatize(w) for w in t.split()
             if w not in stop_words and len(w) > 1]
    return " ".join(words)


EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}

# ---------- UI ----------
st.title("✈️ US Airline Tweet Sentiment")
st.write("Type a tweet about an airline and the model will predict whether it is "
         "**negative**, **neutral** or **positive**.")

examples = {
    "(write my own)": "",
    "Positive example": "Thanks @delta for the smooth flight and amazing crew!",
    "Negative example": "@united my flight was delayed 5 hours and nobody helped us",
    "Neutral example": "@southwestair can you tell me the baggage allowance for tomorrow?",
}
choice = st.selectbox("Try an example", list(examples.keys()))
tweet = st.text_area("Tweet text", value=examples[choice], height=120)

if st.button("Predict sentiment"):
    cleaned = clean_text(tweet)
    if not cleaned:
        st.warning("Please enter a tweet with some words in it.")
    else:
        pred = model.predict([cleaned])[0]
        label = le.inverse_transform([pred])[0]
        st.subheader(f"{EMOJI[label]} {label.capitalize()}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned])[0]
            prob_df = pd.DataFrame({"Sentiment": le.classes_, "Confidence": probs})
            st.metric("Confidence", f"{probs.max():.0%}")
            st.bar_chart(prob_df.set_index("Sentiment"))

        with st.expander("See the cleaned text the model used"):
            st.code(cleaned)
