import streamlit as st
import os
from datetime import datetime
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from src.prompt import prompt_template
from dotenv import load_dotenv
load_dotenv()

# Page config
st.set_page_config(
    page_title="🩺 Medical Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        padding: 0px;
    }

    .container-box {
        max-width: 900px;
        margin: auto;
        padding: 20px;
    }

    .title-container {
        background-color: #4e73df;
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    .chat-container {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        max-height: 400px;
        overflow-y: auto;
    }

    .message-box {
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        max-width: 90%;
        word-wrap: break-word;
    }

    .user-msg {
        background-color: #4e73df;
        color: white;
        margin-left: auto;
        text-align: right;
    }

    .bot-msg {
        background-color: #f1f3f9;
        color: #333;
        margin-right: auto;
        text-align: left;
        border-left: 5px solid #4e73df;
    }

    .input-container {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 15px;
    }

    .stTextInput>div>div>input {
        padding: 15px;
        font-size: 16px;
        border-radius: 25px;
        border: 2px solid #4e73df;
    }

    .stButton>button {
        background-color: #4e73df;
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #3a57b5;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .footer {
        text-align: center;
        margin-top: 20px;
        color: #6c757d;
        font-size: 14px;
    }

    .timestamp {
        font-size: 12px;
        opacity: 0.6;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# Centered container box
st.markdown('<div class="container-box">', unsafe_allow_html=True)

# Header
st.markdown('<div class="title-container"><h1>🩺 ApneDoctors Chatbot</h1><p>Ask me any medical questions you have</p></div>', unsafe_allow_html=True)

# Load model and vector store
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = FAISS.load_local("faiss_index2/", embeddings, allow_dangerous_deserialization=True)
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}
llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Chat history init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Display
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg_type, text, timestamp in st.session_state.chat_history:
        if msg_type == "user":
            st.markdown(f'<div class="message-box user-msg"><strong>You:</strong> {text}<div class="timestamp">{timestamp}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message-box bot-msg"><strong>Medical Assistant:</strong> {text}<div class="timestamp">{timestamp}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align: center; color: #6c757d; margin-top: 25vh;">Ask me anything about medical topics!</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-container">', unsafe_allow_html=True)
input_col, button_col = st.columns([5, 1])
with input_col:
    user_input = st.text_input("", placeholder="Type your medical query here...", key="input")
with button_col:
    send_button = st.button("Send")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Powered by LangChain + Groq | © 2025 Medical Assistant</div>', unsafe_allow_html=True)

# Handle user input
if (send_button or user_input and user_input != st.session_state.get("last_input", "")) and user_input:
    st.session_state["last_input"] = user_input
    current_time = datetime.now().strftime("%I:%M %p")
    st.session_state.chat_history.append(("user", user_input, current_time))

    with st.spinner("Searching medical knowledge..."):
        response = qa({"query": user_input})

    st.session_state.chat_history.append(("bot", response["result"], current_time))
    st.rerun()

# Close main container
st.markdown('</div>', unsafe_allow_html=True)
