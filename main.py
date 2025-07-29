import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FAQ_URL = "https://ti.ua/ua/faq/"
STORES_URL = "https://ti.ua/ua/nashi-magazini/"

def fetch_page_text(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def prepare_corpus():
    faq_text = fetch_page_text(FAQ_URL)
    stores_text = fetch_page_text(STORES_URL)
    return [
        {"source": "FAQ", "text": faq_text},
        {"source": "Stores", "text": stores_text},
    ]


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = []
    metadatas = []
    for doc in docs:
        for chunk in splitter.split_text(doc["text"]):
            texts.append(chunk)
            metadatas.append({"source": doc["source"]})
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore

#LangChain QA
def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa


def process_date_query(query):
    today = datetime.now().strftime("%d.%m.%Y")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d.%m.%Y")
    query = query.replace("сьогодні", today).replace("завтра", tomorrow)
    return query

#Streamlit UI
st.set_page_config(page_title="Чат підтримки TI.UA", page_icon="💬")
st.title("💬 Чат підтримки TI.UA")

if "vectorstore" not in st.session_state:
    with st.spinner("Завантаження даних..."):
        docs = prepare_corpus()
        st.session_state.vectorstore = build_vectorstore(docs)
        st.session_state.qa = get_qa_chain(st.session_state.vectorstore)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ваше питання:", key="user_input")

if st.button("Відправити") and user_input:
    query = process_date_query(user_input)
    qa = st.session_state.qa
    result = qa({"query": query})
    answer = result["result"].strip()
    if len(answer) < 10 or "не знаю" in answer.lower() or "не можу" in answer.lower():
        answer = "Питаю менеджера"
    st.session_state.history.append((user_input, answer))


for q, a in st.session_state.history[::-1]:
    st.markdown(f"**Ви:** {q}")
    st.markdown(f"**Бот:** {a}")
