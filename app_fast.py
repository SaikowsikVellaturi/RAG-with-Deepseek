import streamlit as st
import requests
import numpy as np
import faiss
from transformers import pipeline
import pdfplumber  # Faster PDF text extraction
from sentence_transformers import SentenceTransformer  # Faster embeddings

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "API_KEY"  # Replace with your actual API key
DIMENSION = 384  # Smaller embedding dimension for faster processing

index = faiss.IndexFlatL2(DIMENSION)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Faster and smaller model

@st.cache_data
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    return text

@st.cache_data
def generate_embeddings(text):
    return embedding_model.encode(text)

def query_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
    return response.json()

def retrieve_similar_documents(query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return indices

def rag_pipeline(query, context):
    query_embedding = generate_embeddings([query])
    
    relevant_indices = retrieve_similar_documents(query_embedding)
    
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = query_deepseek(prompt)
    
    return response['choices'][0]['message']['content'], relevant_indices

def main():
    st.title("RAG Application with DeepSeek API")
    st.write("Upload your PDF and ask questions based on it.")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
       
        embeddings = generate_embeddings(text)
        index.add(np.array([embeddings]))  # Add to FAISS index

        query = st.text_input("Enter your question based on the document:", "")

        if st.button("Get Answer"):
            with st.spinner("Processing..."):
                answer, relevant_indices = rag_pipeline(query, text)
                st.subheader("Answer:")
                st.write(answer)

if __name__ == "__main__":
    main()