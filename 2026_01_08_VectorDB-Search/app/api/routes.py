from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from app.services.ingestion_service import ingest_document
from app.services.query_service import query_document
from app.services.pdf_ingestion_service import ingest_pdf, check_upload_pdf

router = APIRouter()


class DocumentRequest(BaseModel):
    text: str


@router.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    check_upload_pdf()
    return ingest_pdf(temp_path)

@router.post("/documents")
def upload_document(request: DocumentRequest):
    return ingest_document(request.text)


@router.get("/query")
def ask_question(q: str):
    return query_document(q)
