# 🤖 PDF Bot: AI-Powered Document Analysis

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-1CB0F5.svg)](https://streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PDF bot is an advanced document analysis tool that leverages the power of AI to extract insights from your PDF documents. Using Retrieval-Augmented Generation (RAG) and Google's Gemini AI, this application offers intelligent question-answering, summarization, and semantic search capabilities.

## 🌟 Features

- 📤 **PDF Upload**: Easily upload any PDF document for analysis.
- 🔍 **Intelligent Q&A**: Ask questions about your document and get AI-powered answers.
- 📊 **Smart Summarization**: Generate concise summaries of your PDF content.
- 🔎 **Semantic Search**: Find relevant information using natural language queries.
- 📜 **Search History**: Keep track of your previous queries and summaries.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- PyPDF2
- Google Generative AI
- Sentence Transformers
- FAISS

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pdf-genius.git
   cd pdf-genius
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## 💡 How It Works

1. **PDF Processing**: The app extracts text from uploaded PDFs and splits it into manageable chunks.
2. **Embedding Creation**: Text chunks are converted into dense vector embeddings using SentenceTransformers.
3. **Indexing**: FAISS is used to create an efficient index for similarity search.
4. **Query Processing**: User queries are embedded and matched against the document chunks.
5. **AI-Powered Responses**: Relevant chunks are sent to the Gemini AI model to generate accurate answers and summaries.

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/): For the interactive web interface
- [PyPDF2](https://pypdf2.readthedocs.io/): For PDF text extraction
- [Google Generative AI](https://ai.google/): For natural language processing and generation
- [Sentence Transformers](https://www.sbert.net/): For creating text embeddings
- [FAISS](https://github.com/facebookresearch/faiss): For efficient similarity search

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/pdf-genius/issues).

## 👨‍💻 Author

**Your Name**

- GitHub: [(https://github.com/Fathima1123)]
  


## 🙏 Acknowledgements

- Google for the Gemini AI API
- The Streamlit team for their amazing framework
- The open-source community for the various libraries used in this project

---

Give PDF Genius a star ⭐️ if you find it useful!
