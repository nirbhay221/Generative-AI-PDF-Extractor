from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import io
from app.chat.vector_stores.pinecone import vector_stores

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding = 'utf-8')
def create_embeddings_for_pdf(pdf_id: str, pdf_path: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap = 100
    )
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split(text_splitter)
    for doc in docs:
        doc.metadata = {
            "page": doc.metadata["page"],
            "text": doc.page_content,
            "pdf_id": pdf_id
        }
    vector_stores.add_documents(docs)