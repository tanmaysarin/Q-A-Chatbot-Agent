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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# PDF path (hardcoded or use uploader)
pdf_path = "docs/mce_conveyor_manual_may2015.pdf"
process_pdf_clicked = st.sidebar.button("Process Equipment Manual")

# Add clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

file_path = "faiss_store_conveyor_manual"

# Setup LLM and embeddings
llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

# PDF Processing (same as before)
if process_pdf_clicked:
    # Load PDF content
    loader = PyMuPDFLoader(pdf_path)
    with st.spinner("Loading PDF..."):
        data = loader.load()

    # Split into chunks with metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    with st.spinner("Splitting text into chunks..."):
        docs = text_splitter.split_documents(data)

    # Embed and store in FAISS
    with st.spinner("Creating embeddings and storing in vector DB..."):
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(file_path)

    st.sidebar.success("PDF processed successfully! ✅")

# Display chat history
st.subheader("Chat History")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                st.write(message["sources"])

# Chat input
if prompt := st.chat_input("Ask a question about the manual..."):
    # Check if vector store exists
    if not os.path.exists(file_path):
        st.error("Please process the PDF manual first using the sidebar button.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Load vector store and get answer
                    vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                    result = chain({"question": prompt}, return_only_outputs=True)

                    # Display answer
                    answer = result["answer"]
                    st.write(answer)

                    # Display sources
                    sources = result.get("sources", "")
                    if sources:
                        with st.expander("Sources"):
                            for src in sources.split("\n"):
                                if src.strip():
                                    st.write(src)

                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": answer
                    }
                    if sources:
                        assistant_message["sources"] = sources

                    st.session_state.messages.append(assistant_message)

                except Exception as e:
                    error_msg = f"Error processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})