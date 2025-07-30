import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st
from config import FAQ_URL

def setup_chrome_driver():
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