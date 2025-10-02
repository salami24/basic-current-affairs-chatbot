import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model and dataset
with open("currentaffairs_chatbot_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chatbot_dataset.pkl", "rb") as f:
    df = pickle.load(f)

# Rebuild embeddings
X_vectors = vectorizer.transform(df["Question"])

# Chatbot function
def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X_vectors)
    idx = similarities.argmax()
    return df.iloc[idx]["Answer"]

# Streamlit UI
st.set_page_config(page_title="🌍 UN Countries Chatbot", page_icon="🌍")

st.title("🌍 UN Countries Chatbot")
st.write("Ask me about any UN member state and I’ll tell you the basic info.")

# User input box
user_input = st.text_input("You:", "")

if user_input:
    response = chatbot_response(user_input)
    st.success(f"🤖 {response}")
