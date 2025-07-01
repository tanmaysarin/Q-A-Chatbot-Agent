import os
import streamlit as st
import time
from dotenv import load_dotenv

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

st.title("Conveyor Manual Assistant 🛠️")
st.sidebar.title("PDF Manual Loader")

# PDF path (hardcoded or use uploader)
pdf_path = "docs/mce_conveyor_manual_may2015.pdf"
process_pdf_clicked = st.sidebar.button("Process Equipment Manual")

file_path = "faiss_store_conveyor_manual"
main_placefolder = st.empty()

llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

if process_pdf_clicked:
    # Load PDF content
    loader = PyMuPDFLoader(pdf_path)
    main_placefolder.text("Loading PDF... ✅")
    data = loader.load()

    # Split into chunks with metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    main_placefolder.text("Splitting text into chunks... ✅")
    docs = text_splitter.split_documents(data)

    # Embed and store in FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(file_path)
    main_placefolder.text("Embeddings stored in vector DB! ✅")
    time.sleep(1)

# User input for query
query = main_placefolder.text_input("Ask a question from the manual:")

if query:
    if os.path.exists(file_path):
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        # Sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources")
            for src in sources.split("\n"):
                st.write(src)
