from fastapi import FastAPI
from app.routers import items

app = FastAPI(title="Production CRUD API")

app.include_router(items.router)

@app.get("/")
def health_check():
    return {"status": "ok"}
