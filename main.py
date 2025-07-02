import os
import streamlit as st
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

# PDF Processing with enhanced metadata
if process_pdf_clicked:
    # Load PDF content
    loader = PyMuPDFLoader(pdf_path)
    with st.spinner("Loading PDF..."):
        data = loader.load()

    # Add custom metadata to each document
    pdf_name = os.path.basename(pdf_path)
    for doc in data:
        doc.metadata["pdf_name"] = pdf_name
        # If page not id doc, then assign it a value of 0. PyMuPDFLoader should do this already
        if 'page' not in doc.metadata:
            doc.metadata['page'] = 0

    # Split into chunks with metadata preserved
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


def get_source_documents_with_pages(chain_result, retriever, query):
    """Get source documents with page information"""
    try:
        # Get the relevant documents for this query
        relevant_docs = retriever.get_relevant_documents(query)

        sources_info = []
        seen_pages = set()  # To avoid duplicate pages

        for doc in relevant_docs[:3]:  # Limit to top 3 sources
            pdf_name = doc.metadata.get('pdf_name', os.path.basename(pdf_path))
            page_num = doc.metadata.get('page', 'Unknown')

            # Create a unique identifier for this page
            page_id = f"{pdf_name}_page_{page_num}"

            if page_id not in seen_pages:
                sources_info.append({
                    'pdf_name': pdf_name,
                    'page': page_num + 1,
                    'content_preview': doc.page_content[:150] + "..." if len(
                        doc.page_content) > 150 else doc.page_content
                })
                seen_pages.add(page_id)

        return sources_info

    except Exception as e:
        return [{'pdf_name': os.path.basename(pdf_path), 'page': 'Unknown',
                 'content_preview': 'Source details unavailable'}]


# Display chat history
st.subheader("Chat History")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                # source expander for chat in history
                for source in message["sources"]:
                    if isinstance(source, dict):
                        st.write(f"**📄 {source['pdf_name']} - Page {source['page']}**")
                        st.write(f"*Preview:* {source['content_preview']}")
                        st.write("---")
                    else:
                        st.write(source)

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
                    retriever = vectorstore.as_retriever()
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
                    result = chain({"question": prompt}, return_only_outputs=True)

                    # Display answer
                    answer = result["answer"]
                    st.write(answer)

                    # Get detailed source information with pages
                    sources_info = get_source_documents_with_pages(result, retriever, prompt)

                    # expanding section to display source
                    if sources_info:
                        with st.expander("📚 Sources"):
                            for source in sources_info:
                                st.write(f"**📄 {source['pdf_name']} - Page {source['page']}**")
                                st.write(f"*Preview:* {source['content_preview']}")
                                st.write("---")

                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_info
                    }

                    st.session_state.messages.append(assistant_message)

                except Exception as e:
                    error_msg = f"Error processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})