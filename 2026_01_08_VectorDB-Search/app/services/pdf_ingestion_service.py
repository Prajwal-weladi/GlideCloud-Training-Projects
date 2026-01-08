import uuid
import os
from app.utils.pdf_reader import extract_text_from_pdf
from app.utils.text_splitter import split_text
from app.core.ollam_client import get_embedding
from app.db.mongodb import chunks_collection


def ingest_pdf(file_path: str):
    extracted_text = extract_text_from_pdf(file_path)

    if not extracted_text:
        return {"message": "No readable text found in PDF"}

    doc_id = str(uuid.uuid4())
    chunks = split_text(extracted_text)

    documents = []

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        documents.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk,
            "embedding": embedding
        })

    if documents:
        chunks_collection.insert_many(documents)

    # Cleanup temp file
    os.remove(file_path)

    return {
        "message": "PDF processed and stored in MongoDB Atlas",
        "doc_id": doc_id,
        "chunks": len(documents)
    }
