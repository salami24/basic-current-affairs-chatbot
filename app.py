import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    # Example dataset: replace with your CSV path
    data = pd.read_csv('C:\\Users\\user\\Documents\\UN_Countries_QA_History.csv')

    return pd.DataFrame(data)

df = load_data()

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Precompute embeddings for dataset
corpus_embeddings = model.encode(df["Question"], convert_to_tensor=True)


# ==============================
# Streamlit UI
# ==============================
st.title("🧠 Current Affairs Chatbot")
st.write("Ask me any question from united nation countries!")

user_input = st.text_input("Your question:")

if user_input:
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)
    best_idx = scores.argmax().item()
    confidence = scores[0][best_idx].item()

    st.markdown(f"**Answer:** {df['Answer'].iloc[best_idx]}")
    st.caption(f"Confidence: {confidence:.2f}")
