from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.
Use the context to answer as completely and accurately as possible.
If the answer is not in the context, say so clearly.
When listing items, always list ALL items you find in the context — never truncate.

Context:
{context}"""

def build_qa_chain(persist_dir: str = "../faiss_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        persist_dir, embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(input_dict):
        question = input_dict["question"]
        docs = retriever.invoke(question)
        return format_docs(docs)

    chain = (
        {
            "context": get_context,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain