import streamlit as st
import requests
import numpy as np
import faiss
from transformers import pipeline
import PyPDF2  # For extracting text from PDF resumes

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "API_KEY"  # Replace with your actual API key
DIMENSION = 768  # Dimension of embeddings (e.g., BERT embeddings)

# Initialize FAISS index
index = faiss.IndexFlatL2(DIMENSION)

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Initialize DeepSeek API client
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

# Retrieve similar documents using FAISS
def retrieve_similar_documents(query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return indices

# RAG Pipeline
def rag_pipeline(query, context):
    # Step 1: Retrieve relevant documents (replace with actual query embedding)
    query_embedding = np.random.random((1, DIMENSION)).astype('float32')  # Dummy embedding
    relevant_indices = retrieve_similar_documents(query_embedding)
    
    # Step 2: Generate a response using DeepSeek API
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = query_deepseek(prompt)
    
    return response['choices'][0]['message']['content'], relevant_indices

# Streamlit App
def main():
    st.title("Resume-Based RAG Application with DeepSeek API")
    st.write("Upload your resume (PDF) and ask questions based on it.")

    # File upload
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        resume_text = extract_text_from_pdf(uploaded_file)
        # st.subheader("Extracted Resume Text:")
        # st.write(resume_text[:1000] + "...")  # Display first 1000 characters for preview

        # # User input
        query = st.text_input("Enter your question based on the resume:", "What is my work experience?")

        if st.button("Get Answer"):
            with st.spinner("Processing..."):
                # Run RAG pipeline using the resume text as context
                answer, relevant_indices = rag_pipeline(query, resume_text)
                
                # Display results
                st.subheader("Answer:")
                st.write(answer)
                


if __name__ == "__main__":
    main()