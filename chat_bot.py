import streamlit as st
import requests
import pdfplumber  

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "API_KEY" 
RESUME_PATH="SaikowsikVellaturi__DataScience.pdf"

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    return text

def query_deepseek(prompt, context):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant that answers questions based on the following resume: {context}"},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
    return response.json()

def main():
    st.title("Resume Chatbot with DeepSeek API")
    st.write("Chat with the bot to ask questions about my resume.")

    try:
        resume_text = extract_text_from_pdf(RESUME_PATH)
        st.success("Resume loaded successfully!")
    except FileNotFoundError:
        st.error(f"Resume file not found at path: {RESUME_PATH}")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("Chat")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):  # User message on the right
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):  # Assistant message on the left
                st.write(message["content"])

    user_input = st.chat_input("Ask a question about my resume:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            response = query_deepseek(user_input, resume_text)
            bot_response = response['choices'][0]['message']['content']

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

        st.rerun()

if __name__ == "__main__":
    main()