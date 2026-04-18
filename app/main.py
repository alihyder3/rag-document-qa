import streamlit as st
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from ingest import ingest_documents
from qa_chain import build_qa_chain

st.set_page_config(page_title="RAG Document QA", page_icon="📄")
st.title("Document QA System")
st.caption("Upload a PDF and ask questions about its contents")

if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.chain is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name
    with st.spinner("Processing document..."):
        ingest_documents(tmp_path, persist_dir="../faiss_db")
        st.session_state.chain = build_qa_chain(persist_dir="../faiss_db")
    st.success("Document ready! Ask your questions below.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.chain:
    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
                st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload a PDF to get started.")