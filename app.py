import streamlit as st
import PyPDF2
import google.generativeai as genai
from io import BytesIO
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'pdf_chunks' not in st.session_state:
    st.session_state.pdf_chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None

# Load the sentence transformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    end = chunk_size
    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        end = start + chunk_size
    return chunks

def create_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_similar_chunks(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    D, I = st.session_state.index.search(query_embedding, top_k)
    return [st.session_state.pdf_chunks[i] for i in I[0]]

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("📚 PDF Bot with RAG")
    st.write("Upload a PDF, ask questions, and get summaries!")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(BytesIO(uploaded_file.read()))
        st.session_state.pdf_chunks = chunk_text(pdf_text)
        embeddings = create_embeddings(st.session_state.pdf_chunks)
        st.session_state.embeddings = embeddings
        st.session_state.index = create_faiss_index(embeddings)

        st.sidebar.header("Options")
        option = st.sidebar.selectbox("Choose a feature", ["QnA", "Summarize", "Search History"])

        if option == "QnA":
            st.subheader("Ask a question about the PDF")
            question = st.text_input("Your question")
            if st.button("Ask"):
                relevant_chunks = search_similar_chunks(question)
                context = "\n".join(relevant_chunks)
                prompt = f"Based on the following context from a PDF, answer this question: {question}\n\nContext: {context}"
                response = get_gemini_response(prompt)
                st.write("Answer:", response)
                st.session_state.chat_history.append(("Q: " + question, "A: " + response))
                st.session_state.search_history.append(question)

        elif option == "Summarize":
            st.subheader("Summarize the PDF")
            if st.button("Generate Summary"):
                prompt = f"Summarize the following text from a PDF in a concise manner:\n\n{pdf_text[:4000]}"
                summary = get_gemini_response(prompt)
                st.write("Summary:", summary)
                st.session_state.chat_history.append(("Summary Request", summary))
                st.session_state.search_history.append("PDF Summary")

        elif option == "Search History":
            st.subheader("Search History")
            for item in st.session_state.search_history:
                st.write("- " + item)

    st.sidebar.header("Chat History")
    for q, a in st.session_state.chat_history[-5:]:
        st.sidebar.text(q)
        st.sidebar.text(a)
        st.sidebar.text("---")

if __name__ == "__main__":
    main()