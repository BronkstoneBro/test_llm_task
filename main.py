import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

FAQ_URL = "https://ti.ua/ua/faq/"
STORES_URL = "https://ti.ua/ua/nashi-magazini/"

def setup_chrome_driver():
    """Setup Chrome driver with options to bypass Cloudflare"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def extract_faq_content(soup):
    """Extract FAQ content from the specific accordion structure"""
    faq_text = []
    

    accordion_sections = soup.find_all("section", class_="section accordeon-section")
    
    for section in accordion_sections:

        accordion_items = section.find_all("div", class_="accordeon_item")
        
        for item in accordion_items:

            question_elem = item.find("div", class_="accordion__button")
            if question_elem:
                question = question_elem.get_text(strip=True)
                

                answer_elem = item.find("span", class_="text-content faq-accordeon")
                if answer_elem:
                    answer = answer_elem.get_text(strip=True)
                    

                    faq_item = f"Питання: {question}\nВідповідь: {answer}\n"
                    faq_text.append(faq_item)
    
    return "\n".join(faq_text)

def fetch_page_text(url):
    """Fetch page content using Selenium to handle dynamic content"""
    driver = None
    try:
        driver = setup_chrome_driver()
        driver.get(url)

        time.sleep(3)
        

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        
        if url == FAQ_URL:
            return extract_faq_content(soup)
        else:
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)
            
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return ""
    finally:
        if driver:
            driver.quit()

def prepare_corpus():
    """Prepare corpus from FAQ and stores pages"""
    faq_text = fetch_page_text(FAQ_URL)
    stores_text = fetch_page_text(STORES_URL)
    return [
        {"source": "FAQ", "text": faq_text},
        {"source": "Stores", "text": stores_text},
    ]

def build_vectorstore(docs):
    """Build vector store from documents"""
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

def get_qa_chain(vectorstore):
    """Create QA chain with specific prompt"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Відповідай українською мовою ТІЛЬКИ на задане питання, використовуючи надану інформацію з FAQ TI.UA.\n"
            "Якщо в контексті є точна відповідь, скопіюй її повністю, не змінюй і не скорочуй. Не додавай нічого від себе.\n"
            "НЕ генеруй додаткові питання або відповіді. Давай тільки одну конкретну відповідь.\n"
            "ВАЖЛИВО: Використовуй ВСЮ доступну інформацію з контексту. Не пропускай деталі.\n"
            "Контекст:\n{context}\n"
            "Питання: {question}\n"
            "Відповідь:"
        )
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa

def process_date_query(query):
    """Process date-related queries"""
    today = datetime.now().strftime("%d.%m.%Y")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d.%m.%Y")
    query = query.replace("сьогодні", today).replace("завтра", tomorrow)
    return query

def main():
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
        result = qa.invoke({"query": query})
        answer = result["result"].strip()
        if len(answer) < 10 or "не знаю" in answer.lower() or "не можу" in answer.lower():
            answer = "Питаю менеджера"
        st.session_state.history.append((user_input, answer))

    for q, a in st.session_state.history[::-1]:
        st.markdown(f"**Ви:** {q}")
        st.markdown(f"**Бот:** {a}")

if __name__ == "__main__":
    main()