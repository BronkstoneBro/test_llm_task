from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY, OPENAI_MODEL

def get_qa_chain(vectorstore):
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