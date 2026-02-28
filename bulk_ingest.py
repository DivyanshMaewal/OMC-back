import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# --- CONFIGURATION ---
SOURCE_DIR = "./source_documents"
DB_DIR = "./data_store"
EMBED_MODEL = "nomic-embed-text"

def run_bulk_ingest():
    # 1. Ensure the source directory exists
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        print(f"Created {SOURCE_DIR}. Please drop your PDFs there and run again.")
        return

    # 2. Load all PDFs from the directory
    print(f"📂 Loading documents from {SOURCE_DIR}...")
    loader = DirectoryLoader(SOURCE_DIR, glob="./*.pdf", loader_cls=PyPDFLoader)
    
    try:
        raw_documents = loader.load()
        if not raw_documents:
            print("❌ No PDFs found in the source directory.")
            return
        print(f"✅ Loaded {len(raw_documents)} pages from documents.")
    except Exception as e:
        print(f"❌ Error loading PDFs: {e}")
        return

    # 3. Chunking Logic
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(chunks)} chunks.")

    # 4. Local Embedding & Vector Store
    print(f"🧠 Generating embeddings using {EMBED_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"💾 Database saved to {DB_DIR}")
    print("✨ Bulk ingestion complete. OMC Document Intelligence knowledge base ready.")

if __name__ == "__main__":
    run_bulk_ingest()