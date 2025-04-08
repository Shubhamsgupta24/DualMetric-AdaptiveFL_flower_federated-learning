import streamlit as st
import tensorflow as tf
import os
import json
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import tokenizer_from_json #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore

# Constants
NUM_CLIENTS = 6
MODELS_DIR = "models"

# Function to preprocess user input (remove punctuation)
def preprocess_text(text):
    text = text.lower().strip()  # Convert to lowercase and strip spaces
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

# Function to load model, tokenizer, and label encoder
@st.cache_resource
def load_client_resources(client_id):
    client_path = os.path.join(MODELS_DIR, f"Client{client_id}")

    # Load model
    model = tf.keras.models.load_model(os.path.join(client_path, "model.keras"), compile=False)
    
    # Load tokenizer
    with open(os.path.join(client_path, "tokenizer.json"), "r") as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)
    
    # Load label encoder
    with open(os.path.join(client_path, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    
    return model, tokenizer, label_encoder

# Load all client models (cached for performance)
@st.cache_resource
def load_all_clients():
    clients = {}
    for i in range(NUM_CLIENTS):
        clients[i] = load_client_resources(i)
    return clients

clients = load_all_clients()

# Streamlit UI
st.title("Federated Learning Client Inference")

# Input box with key to ensure state tracking
user_input = st.text_input("Enter text for classification:", key="user_input")

# Button to classify
if st.button("Classify"):
    if user_input.strip():  # Ensure input is not empty
        cleaned_input = preprocess_text(user_input)  # Clean input text
        results = {}

        try:
            for client_id, (model, tokenizer, label_encoder) in clients.items():
                # Tokenize input
                sequences = tokenizer.texts_to_sequences([cleaned_input])
                if not sequences or sequences == [[]]:  # Handle empty tokenized sequences
                    st.warning(f"Client {client_id}: Unable to process input (empty sequence). Try a different input.")
                    continue

                maxlen = model.input_shape[1]  # Dynamically fetch maxlen
                padded_sequences = pad_sequences(sequences, maxlen=maxlen)  # Match training maxlen

                # Make prediction
                prediction = model.predict(padded_sequences)[0]

                # Decode labels
                if isinstance(label_encoder, dict):  # Ensure compatibility with JSON-loaded encoders
                    label_classes = [label_encoder[str(i)] for i in range(len(prediction))]
                else:  # Standard sklearn label encoder
                    label_classes = label_encoder.classes_

                # Store result
                results[f"Client {client_id}"] = dict(zip(label_classes, prediction))

            # Display results
            for client_id, probabilities in results.items():
                st.subheader(client_id)  # Properly formatted subheader
                
                # Convert results to DataFrame
                df = pd.DataFrame(list(probabilities.items()), columns=["Category", "Probability"])
                df = df.sort_values(by="Probability", ascending=False)  # Sort by probability
                
                # Plot bar chart
                fig, ax = plt.subplots(figsize=(6, 10))
                ax.barh(df["Category"], df["Probability"], color="skyblue")
                ax.set_xlabel("Probability")
                ax.set_ylabel("Category")
                ax.set_title(client_id)
                ax.invert_yaxis()  # Highest probability at the top
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter text before clicking Classify.")
