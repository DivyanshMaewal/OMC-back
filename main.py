import os
import shutil
import tempfile
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# LCEL Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="OMC Document Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data_store")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are the Odisha Mining Corporation (OMC) Document Intelligence Assistant.
You assist senior executives and officials by answering questions based strictly on
ingested internal documents, reports, circulars, and policy files.

CRITICAL RULES:
1. Be formal, precise, and professional in every response.
2. Cite specific document sections, page numbers, or clause references when possible.
3. If the answer is NOT found in the provided context, state that clearly — do NOT fabricate information.
4. Structure your responses with clear headings and bullet points for readability.
5. When referencing numerical data (production figures, financial data, targets), present them accurately.

Context: {context}"""

# --- API ENDPOINTS ---

@app.get("/status")
async def get_status():
    """Returns the current status of the knowledge base."""
    try:
        if not os.path.exists(DB_DIR):
            return {"document_count": 0, "status": "empty"}
        
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        count = vectorstore._collection.count()
        return {"document_count": count, "status": "ready" if count > 0 else "empty"}
    except Exception as e:
        return {"document_count": 0, "status": "error", "detail": str(e)}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Accepts a PDF upload, chunks it, and adds to the ChromaDB vector store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load and chunk the PDF
        print(f"📄 Ingesting: {file.filename}")
        loader = PyPDFLoader(tmp_path)
        raw_documents = loader.load()

        if not raw_documents:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail="PDF appears to be empty or unreadable.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(raw_documents)
        print(f"✅ Created {len(chunks)} chunks from {file.filename}")

        # Add to vector store
        if os.path.exists(DB_DIR):
            vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            vectorstore.add_documents(chunks)
        else:
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=DB_DIR,
            )

        # Cleanup temp file
        os.unlink(tmp_path)

        return {
            "message": f"Successfully ingested '{file.filename}' — {len(chunks)} chunks added to knowledge base.",
            "chunks": len(chunks),
            "pages": len(raw_documents),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/chat")
async def chat(query: str):
    """Answers a question using RAG over the ingested document knowledge base."""
    try:
        if not os.path.exists(DB_DIR):
            return {"response": "No documents have been ingested yet. Please upload a PDF to get started."}

        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        # Check if there are any documents
        if vectorstore._collection.count() == 0:
            return {"response": "The knowledge base is empty. Please ingest documents before querying."}

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = await rag_chain.ainvoke(query)

        return {"response": answer}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)