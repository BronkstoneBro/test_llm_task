import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
FAQ_URL = "https://ti.ua/ua/faq/"
STORES_URL = "https://ti.ua/ua/nashi-magazini/"
