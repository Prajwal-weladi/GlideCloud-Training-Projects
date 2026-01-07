from pymongo import MongoClient
from app.core.config import settings

client = MongoClient(settings.MONGODB_URI)
db = client[settings.DB_NAME]
collection = db[settings.COLLECTION_NAME]
