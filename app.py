import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load the data
@st.cache_data
def load_data():
    with open('bns.csv', 'r') as file:
        data = file.read().split('\n')
    return pd.DataFrame(data, columns=['Details'])

# Load model
@st.cache_data #(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Perform semantic search
def semantic_search(query, data, model, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    data_embeddings = model.encode(data['Details'].tolist(), convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, data_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    return top_results

# Title
st.title("Bharathiya Nyaya Sanhitha Search")

# Load data and model
data = load_data()
model = load_model()

# User input
query = st.text_input("Enter your search query:", "")

# Search and display results
if query:
    top_k = st.slider("Number of results to display:", 1, 10, 5)
    results = semantic_search(query, data, model, top_k=top_k)
    for score, idx in zip(results[0], results[1]):
        st.write(f"**Score:** {score.item():.4f}")
        st.write(f"**Detail:** {data.iloc[idx.item()]['Details']}")
        st.write("---")

