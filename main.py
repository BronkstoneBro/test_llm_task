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
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
FAQ_URL = "https://ti.ua/ua/faq/"
STORES_URL = "https://ti.ua/ua/nashi-magazini/"

FAQ_KEYWORDS = ["кредит", "розстрочка", "повернення", "доставка", "оплата", "гарантія"]

FAQ_CLASSES = ["faq-accordeon", "text-content"]
FAQ_ATTRIBUTES = ["data-toggle", "role", "aria-expanded", "data-bs-toggle", "data-target", "data-bs-target"]


def setup_chrome_driver():
    """Setup Chrome driver for web scraping"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


def extract_text_from_elements(soup, selectors):
    """Extract text from elements using selectors"""
    texts = []
    for selector in selectors:
        elements = soup.find_all(**selector)
        for element in elements:
            text = element.get_text(separator="\n", strip=True)
            if len(text) > 20:
                texts.append(text)
    return texts


def extract_faq_content(soup):
    """Extract FAQ content from page"""
    all_texts = []
    
    for class_name in FAQ_CLASSES:
        elements = soup.find_all(class_=class_name)
        for element in elements:
            all_texts.append(element.get_text(separator="\n", strip=True))
    
    for attr in FAQ_ATTRIBUTES:
        elements = soup.find_all(attrs={attr: True})
        for element in elements:
            text = element.get_text(separator="\n", strip=True)
            if len(text) > 20:
                all_texts.append(text)
    
    for element in soup.find_all(id=True):
        element_id = element.get("id", "").lower()
        if any(keyword in element_id for keyword in ["faq", "accord", "collapse"]):
            text = element.get_text(separator="\n", strip=True)
            if len(text) > 20:
                all_texts.append(text)
    
    for element in soup.find_all():
        text = element.get_text(strip=True)
        if len(text) > 50 and any(keyword in text.lower() for keyword in FAQ_KEYWORDS):
            all_texts.append(text)
    
    return list(set(all_texts))


def fetch_page_text(url):
    """Load and parse text from web page"""
    driver = None
    try:
        driver = setup_chrome_driver()
        driver.get(url)
        time.sleep(3)
        
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        if "faq" in url.lower():
            faq_texts = extract_faq_content(soup)
            if faq_texts:
                return "\n\n".join(faq_texts)
        
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
        
    except Exception as e:
        st.error(f"Error parsing {url}: {str(e)}")
        return ""
    finally:
        if driver:
            driver.quit()


def prepare_corpus():
    """Prepare corpus data from web pages"""
    faq_text = fetch_page_text(FAQ_URL)
    stores_text = fetch_page_text(STORES_URL)
    
    return [
        {"source": "FAQ", "text": faq_text},
        {"source": "Stores", "text": stores_text},
    ]


def build_vectorstore(docs):
    """Create vector store from documents"""
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


def create_qa_prompt():
    """Create prompt for QA system"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Відповідай українською мовою ТІЛЬКИ на задане питання, використовуючи надану інформацію з FAQ TI.UA.\n"
            "НЕ генеруй додаткові питання або відповіді. Давай тільки один конкретний відповідь.\n"
            "ВАЖЛИВО: Використовуй ВСЮ доступну інформацію з контексту. Не пропускай деталі.\n"
            "Якщо в контексті є детальна інформація про кредити, розстрочку, сервіси - обов'язково включи їх у відповідь.\n"
            "НЕ кажи 'інформація відсутня' або 'зверніться до інших джерел'. Якщо точної інформації немає, скажи 'Питаю менеджера'.\n"
            "Контекст:\n{context}\n"
            "Питання: {question}\n"
            "Відповідь:"
        )
    )


def get_qa_chain(vectorstore):
    """Create QA chain"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    prompt = create_qa_prompt()
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa


def process_date_query(query):
    """Process date queries"""
    today = datetime.now().strftime("%d.%m.%Y")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d.%m.%Y")
    return query.replace("сьогодні", today).replace("завтра", tomorrow)


def initialize_app():
    """Initialize application"""
    if "vectorstore" not in st.session_state:
        with st.spinner("Завантаження даних..."):
            docs = prepare_corpus()
            st.session_state.vectorstore = build_vectorstore(docs)
            st.session_state.qa = get_qa_chain(st.session_state.vectorstore)
    
    if "history" not in st.session_state:
        st.session_state.history = []


def main():
    """Main application function"""
    st.set_page_config(page_title="Чат підтримки TI.UA", page_icon="💬")
    st.title("💬 Чат підтримки TI.UA")
    
    initialize_app()
    
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
