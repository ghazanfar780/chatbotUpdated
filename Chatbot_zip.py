import streamlit as st
import os
import base64
import zipfile
import io
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import fitz  # PyMuPDF
import pandas as pd

# API key is hardcoded
api_key = "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"

# Function to load text from various file types
def load_text(file_stream, file_name):
    if file_name.endswith('.pdf'):
        return load_pdf(file_stream)
    elif file_name.endswith('.csv'):
        return load_csv(file_stream)
    return []

# Function to load PDF and extract text
def load_pdf(file_stream):
    file_bytes = io.BytesIO(file_stream.read())
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            texts.append(page.get_text("text"))
        return texts
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return []

# Function to load CSV and convert to text
def load_csv(file_stream):
    try:
        df = pd.read_csv(file_stream)
        return [df.to_string()]
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return []

# Function to extract and process files from ZIP
def process_zip(uploaded_file):
    docs = []
    with zipfile.ZipFile(uploaded_file) as z:
        for file_name in z.namelist():
            with z.open(file_name) as file_stream:
                texts = load_text(file_stream, file_name)
                docs.extend([Document(text) for text in texts])
    return docs

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

# CSS styles for the chat UI
css = '''
<style>
body {
    font-family: 'Arial', sans-serif;
    background-color: #f5f5f5;
}
.chat-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin-bottom: 10px;
}
.chat-message {
    padding: 1rem; 
    border-radius: 10px; 
    margin-bottom: 1rem; 
    display: flex;
}
.chat-message.user {
    background-color: #2b313e;
    color: #fff;
    align-self: flex-end;
    border-bottom-right-radius: 0;
}
.chat-message.bot {
    background-color: #475063;
    color: #fff;
    align-self: flex-start;
    border-bottom-left-radius: 0;
}
.chat-message .avatar {
  width: 15%;
  margin-right: 10px;
}
.chat-message .avatar img {
  max-width: 50px;
  max-height: 50px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 85%;
  padding: 0 1rem;
}
button {
    background-color: #0084ff;
    border: none;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 5px;
}
button:hover {
    background-color: #005f99;
}
</style>
'''

# HTML templates for chat UI
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/jpeg;base64,{{BOT_IMG}}" />
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/jpeg;base64,{{USER_IMG}}" />
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# Function to display chat message
def display_message(message, is_user=False):
    with open("user.jpg", "rb") as img_file:
        user_image = base64.b64encode(img_file.read()).decode("utf-8")
    with open("bot.jpg", "rb") as img_file:
        bot_image = base64.b64encode(img_file.read()).decode("utf-8")

    if is_user:
        html_content = user_template.replace("{{USER_IMG}}", user_image).replace("{{MSG}}", message)
    else:
        html_content = bot_template.replace("{{BOT_IMG}}", bot_image).replace("{{MSG}}", message)

    st.markdown(html_content, unsafe_allow_html=True)

# Admin section to upload files
def admin_section():
    st.sidebar.title("Admin Area")
    admin_password = st.sidebar.text_input("Enter Admin Password", type="password")
    
    if admin_password == "admin_pass":  # Replace with your own secure password
        uploaded_zip = st.sidebar.file_uploader("Upload a ZIP file", type=["zip"])
        process_button = st.sidebar.button("Process Files")
        
        if process_button and uploaded_zip:
            with st.spinner("Loading and processing ZIP file..."):
                docs = process_zip(uploaded_zip)
                if docs:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(docs)
                    model_name = "all-MiniLM-L6-v2"
                    embeddings = HuggingFaceEmbeddings(model_name=model_name)
                    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                    retriever = vectorstore.as_retriever()
                    st.session_state.retriever = retriever
                    st.success("Files uploaded and processed successfully!")
    else:
        st.sidebar.error("Incorrect password")

# User section to ask questions
def user_section():
    st.title("Search Documents")
    if 'retriever' in st.session_state:
        system_prompt = (
            "You are an assistant for question-answering tasks based on the provided/uploaded documents. "
            "When a query is received, search the content of the documents to find a relevant answer. "
            "If the query is not addressed, provide a response based on your knowledge."
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )

        llm = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

        if st.session_state['generated']:
            for ai_msg, user_msg in zip(st.session_state['generated'], st.session_state['past']):
                if user_msg:
                    display_message(user_msg, is_user=True)
                if ai_msg:
                    display_message(ai_msg)

        user_input = st.text_input("Ask a question about the content:", key="user_input")
        if st.button("Ask") and user_input:
            with st.spinner("Getting the answer..."):
                results = rag_chain.invoke({"input": user_input})
                answer = results['answer']
                st.session_state.past.append(user_input)
                st.session_state.generated.append(answer)
                display_message(user_input, is_user=True)
                display_message(answer)

# Main function to run the Streamlit app
def main():
    st.markdown(css, unsafe_allow_html=True)
    
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []
    
    st.sidebar.title("User Mode")
    mode = st.sidebar.radio("Select Mode", ["Admin", "User"])

    if mode == "Admin":
        admin_section()
    else:
        user_section()

if __name__ == "__main__":
    main()
