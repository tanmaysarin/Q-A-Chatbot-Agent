# Conveyor Manual Q&A Chatbot

A simple Streamlit app that allows you to upload and query an equipment manual (PDF) using a Retrieval-Augmented Generation (RAG) pipeline powered by LangChain and OpenAI.

---

## 🚀 Features
- preload a conveyor equipment manual (PDF)
- Ask technician-style questions (e.g., safety, maintenance, troubleshooting)
- Returns answers using OpenAI's GPT model with context

---

## 🛠️ Tech Stack
- Python + Streamlit
- LangChain
- FAISS for vector search
- OpenAI LLM (e.g., `gpt-3.5-turbo`)
- Pipenv for environment management

---

## 📦 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/tanmaysarin/Q-A-Chatbot-Agent.git
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

### 3. Set Up OpenAI API Key
Create a `.env` file from the example:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_key_here
```

### 4. Place PDF in Project Folder
Ensure the equipment manual PDF is placed in the root folder or upload via Streamlit. You can preload it as `mce_conveyor_manual_may2015.pdf`.

---

## ▶️ Running the App
```bash
pipenv shell
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) to use the app.

---

## 📁 Files
- `app.py` — Streamlit app
- `.env.example` — Example environment file
- `.gitignore` — Files excluded from Git
- `Pipfile`, `Pipfile.lock` — Dependency management

---

## 📌 Notes
- Vector embeddings are stored locally in the `faiss_store_conveyor_manual/` folder.
- For privacy/security, never commit your `.env` file to Git.

---

## 🙋‍♀️ Questions?
Feel free to open an issue or reach out.

---

© 2025 Tanmay Sarin
