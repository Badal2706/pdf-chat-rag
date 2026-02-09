# DocuChat AI - Intelligent PDF Conversational Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLM-green.svg)](https://groq.com)

A production-ready RAG (Retrieval-Augmented Generation) application that enables intelligent conversations with multiple PDF documents using advanced NLP techniques and Groq's high-performance LLM inference.

## 🚀 Features

- **Multi-Document Processing**: Upload and analyze multiple PDFs simultaneously
- **Intelligent Chunking**: Smart text segmentation with TF-IDF vectorization for optimal retrieval
- **Semantic Search**: Cosine similarity-based context retrieval for precise answers
- **Conversational Memory**: Maintains context across multi-turn conversations
- **Streaming Responses**: Real-time token generation for enhanced UX
- **Source Attribution**: Tracks and displays which documents provided the answer

## 🛠️ Technical Architecture
    ┌─────────────┐      ┌──────────────┐     ┌─────────────┐
    │  PDF Upload │────▶│ Text Extract │────▶│   Chunking  │
    └─────────────┘      └──────────────┘     └──────┬──────┘
    │
    ┌─────────────┐      ┌──────────────┐     ┌──────▼──────┐
    │   Groq LLM  │◀────│ Context Build│◀────│ TF-IDF +    │
    │   Response  │      │   + Prompt   │     │ Similarity  │
    └─────────────┘      └──────────────┘     └─────────────┘



## 📦 Installation

1. **Download the code**
   - Click the green "Code" button above
   - Click "Download ZIP"
   - Extract the ZIP file


2. **Install Python** (3.8 or higher) from [python.org](https://python.org)


3. **Open Command Prompt in the folder**
   - Navigate to the extracted folder
   - Type `cmd` in the address bar and press Enter


4. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   
5. **Install dependencies**
   ```bash 
   pip install -r requirements.txt
   
6. **Set up API Key**
   - Create a file named .env 
   - Add: GROQ_API_KEY=your_api_key_here


7. **Run the application**
    ```bash
   streamlit run Main.py
   
**🔑 Get API Key**

    1. Go to console.groq.com
    2. Create free account
    3. Generate API key
    4. Paste in .env file

**💡 How It Works**

    Upload PDFs: Drag and drop multiple documents
    Process: Click "Process PDFs" to index content
    Chat: Ask questions in natural language
    Review: Check source references for transparency

**🧠 Technologies Used**

    Streamlit: Web interface framework
    Groq: Ultra-fast LLM inference (Llama 3.1, Mixtral)
    PyPDF2: PDF text extraction
    scikit-learn: TF-IDF vectorization and similarity search
    python-dotenv: Environment variable management

**🛡️ Privacy & Security**

    Documents processed locally in memory only
    No persistent storage of uploaded files
    API keys managed via environment variables

**📄 License**

    MIT License - see LICENSE file for details

**🙏 Acknowledgments**

    Groq for high-speed LLM inference
    Streamlit for intuitive UI framework
    scikit-learn for ML utilities

**Built with ❤️ by Badal Patel**