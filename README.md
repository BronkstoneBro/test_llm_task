# TI.UA Chatbot

This project is a convenient support chatbot for the TI.UA website. It answers user questions in Ukrainian, using up-to-date information parsed directly from the site.

## Features
- Automatically parses FAQ and store information from the TI.UA website
- Answers user questions in Ukrainian
- Uses OpenAI to generate answers based on real site content
- User-friendly web interface built with Streamlit

## Quick Start

1. **Clone the repository and navigate to the project folder:**
   ```bash
   git clone https://github.com/BronkstoneBro/test_llm_task/
   cd test_llm_task
   ```

2. **Create your environment file:**
   ```bash
   cp .env.sample .env
   # Open .env and add your OpenAI API key
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser** and go to the address shown by Streamlit (usually http://localhost:8501)

---

## Project Structure

- `main.py` — main entry point, handles only the UI and connects all modules
- `config.py` — all settings and constants (API keys, URLs, etc.)
- `scraper.py` — functions for parsing the website and extracting data
- `data_processor.py` — prepares the data corpus and builds the vector store
- `qa_system.py` — logic for generating answers using the LLM
- `utils.py` — helper functions (e.g., date processing)
- `.env.sample` — example environment variables file
- `requirements.txt` — list of dependencies

---

## Environment Variables

To run the project, you need a `.env` file with the following content:

```
OPENAI_API_KEY=sk-... # Your OpenAI API key
```

---

## Notes
- The project uses Selenium and ChromeDriver for web scraping. Everything is installed automatically, but make sure you have permissions to install drivers on your system.
- If you have any questions or need help, feel free to ask!

---

**Good luck and enjoy working with the TI.UA Chatbot!**
