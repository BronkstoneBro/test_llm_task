import streamlit as st
from core.data_processor import prepare_corpus, build_vectorstore
from core.qa_system import get_qa_chain
from core.utils import process_date_query

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