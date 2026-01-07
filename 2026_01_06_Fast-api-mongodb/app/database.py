from pymongo import MongoClient
from app.config import settings

client = MongoClient(settings.mongo_url)
db = client[settings.db_name]

item_collection = db["items"]
