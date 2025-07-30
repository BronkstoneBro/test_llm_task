from scraper import fetch_page_text
from config import FAQ_URL, STORES_URL, OPENAI_API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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