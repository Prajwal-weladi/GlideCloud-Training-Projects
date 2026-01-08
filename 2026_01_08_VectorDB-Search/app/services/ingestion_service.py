from app.db.mongodb import chunks_collection
from app.core.ollam_client import get_embedding
from app.utils.text_splitter import split_text
from fastapi import UploadFile, File
from app.services.pdf_ingestion_service import ingest_pdf
import shutil
import uuid



def ingest_document(text: str):
    doc_id = str(uuid.uuid4())
    chunks = split_text(text)

    documents = []

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        documents.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk,
            "embedding": embedding
        })

    chunks_collection.insert_many(documents)

    return {
        "message": "Document stored and indexed in MongoDB Atlas",
        "doc_id": doc_id,
        "chunks": len(documents)
    }
