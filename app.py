
import streamlit as st
import os
from datetime import datetime
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from src.prompt import prompt_template
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="🩺 Medical Chatbot", layout="centered")
st.title("🩺 Medical Assistant Chatbot")
st.markdown("<style> .stTextInput>div>div>input { padding: 12px; font-size: 18px; } .message-box { border-radius: 12px; padding: 10px 16px; margin: 8px 0; } .user-msg { background-color: #DCF8C6; text-align: right; } .bot-msg { background-color: #F1F0F0; text-align: left; } </style>", unsafe_allow_html=True)

# Load model & vector store
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = FAISS.load_local("faiss_index/", embeddings, allow_dangerous_deserialization=True)

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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    msg_type, text, timestamp = msg
    if msg_type == "user":
        st.markdown(f"<div class='message-box user-msg'><strong>You:</strong> {text}<br><small>{timestamp}</small></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='message-box bot-msg'><strong>Bot:</strong> {text}<br><small>{timestamp}</small></div>", unsafe_allow_html=True)

# Chat input
user_input = st.text_input("Type your medical query here:", key="input")

if st.button("Send") and user_input:
    current_time = datetime.now().strftime("%I:%M %p")
    st.session_state.chat_history.append(("user", user_input, current_time))
    with st.spinner("Bot is thinking..."):
        response = qa({"query": user_input})
    st.session_state.chat_history.append(("bot", response["result"], current_time))
    st.experimental_rerun()
