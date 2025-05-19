import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Load tokenizer and label encoder
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

MAX_LEN = 100

# Dictionary to map model names to file paths
model_options = {
    "Default Model": "news_category_model.h5",
    "RNN Model": "rnn_model.keras",
    "LSTM Model": "lstm_model.keras"
}

# Streamlit UI
st.title("🧠 News Category Classifier")
st.subheader("Choose a model and enter a news headline to predict its category")

# Dropdown to select model
selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]

# Lazy-load the selected model (only load when needed)
@st.cache_resource
def load_selected_model(path):
    return load_model(path)

model = load_selected_model(selected_model_path)

# Display available categories
categories = label_encoder.classes_
st.write("### Available Categories:")
st.write(", ".join(map(str, categories)))

# Input
user_input = st.text_input("Headline:")

# Prediction
def predict_category(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded)
    category_index = np.argmax(pred, axis=1)[0]
    return label_encoder.inverse_transform([category_index])[0]

if user_input:
    predicted_category = predict_category(user_input)
    st.success(f"**Predicted Category ({selected_model_name}):** {predicted_category}")
