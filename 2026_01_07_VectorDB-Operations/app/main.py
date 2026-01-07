from fastapi import FastAPI
from app.routes import documents

app = FastAPI(title="Vector Search API")

app.include_router(documents.router)
