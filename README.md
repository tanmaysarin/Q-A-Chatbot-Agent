# Equipment Manual Q&A Agent 🛠️

An intelligent conversational assistant that allows you to chat with your PDF documents using AI. Ask questions about your manual, documentation, or any PDF content and get instant answers with precise source citations including page numbers.

## ✨ Features

- **Interactive Chat Interface**: Have natural conversations with your PDF documents
- **Precise Source Citations**: Get exact page numbers and content previews for all answers
- **Persistent Chat History**: Maintain conversation context throughout your session
- **Vector Search**: Uses FAISS for fast and accurate document retrieval
- **Easy PDF Processing**: Simple one-click processing of PDF documents
- **Source Previews**: See snippets of the actual text used to generate answers

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

### 1. Clone the repository
   ```bash
   git clone <https://github.com/tanmaysarin/Q-A-Chatbot-Agent.git>
   cd Q-A-Chatbot-Agent
   ```

### 2. Set Up Environment
   Make sure you have Pipenv installed:
```bash
  pip install pipenv
```

   Install dependencies:
```bash
  pipenv install
```

### 3. Set up environment variables
   Create a `.env` file in the project root as seen in `.env.example` file:
```env
  OPENAI_API_KEY=your-openai-api-key-here
```

### 4. Prepare your PDF
   - Place your PDF file as `docs/mce_conveyor_manual_may2015.pdf` in the `docs` folder
   - Or modify the `pdf_path` variable in the code to point to your PDF

### 5. Run the application
```bash
  pipenv shell
  streamlit run app.py
```
Then open the URL shown in the terminal (usually `http://localhost:8501`) to use the app.

## 🛠️ How to Use

### Step 1: Process Your PDF
1. Launch the application
2. Click "Process Equipment Manual" in the sidebar
3. Wait for the processing to complete (this creates embeddings and vector database)

### Step 2: Start Chatting
1. Type your question in the chat input box
2. Press Enter to send
3. The AI will respond with an answer and show relevant sources

### Step 3: Explore Sources
Click on the "📚 Sources" expander to see:
- Exact page numbers where information was found
- PDF filename
- Preview of the actual text used
- Multiple sources if the answer draws from different pages

## 🏗️ Architecture

The application follows this workflow:

1. **PDF Processing**:
   - Loads PDF using PyMuPDF
   - Splits into chunks with metadata preservation
   - Creates embeddings using OpenAI
   - Stores in FAISS vector database

2. **Query Processing**:
   - User question is embedded
   - Similar chunks are retrieved from vector database
   - LangChain's RetrievalQA generates answer with sources
   - Results are formatted with page numbers and previews

3. **Chat Management**:
   - Streamlit session state maintains conversation history
   - Each message includes role, content, and source information

## 📁 Project Structure

```
pdf-chat-assistant/
├── main.py                 # Main application file
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── docs/                 # PDF storage folder
│   └── mce_conveyor_manual_may2015.pdf
└── faiss_store_conveyor_manual/  # Vector database (auto-created)
    ├── index.faiss
    └── index.pkl
```