from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,      # IMPORTANT (not 1000)
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local("vectorstore")


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    print("📄 Loading PDF...")
    text = load_pdf('C:/Users/ADMIN/OneDrive/folder1/OneDrive/Apps/qabot/data/sample.pdf')

    print("✂️ Chunking...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("🧠 Creating embeddings...")
    create_vector_store(chunks)

    print("✅ Vector store rebuilt successfully!")
