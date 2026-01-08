from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Mongo + Ollama RAG")

app.include_router(router)