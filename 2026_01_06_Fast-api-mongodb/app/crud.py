from bson import ObjectId
from app.database import item_collection

def create_item(data: dict):
    result = item_collection.insert_one(data)
    return str(result.inserted_id)

def get_all_items():
    items = []
    for item in item_collection.find():
        item["_id"] = str(item["_id"])
        items.append(item)
    return items

def get_item(item_id: str):
    return item_collection.find_one({"_id": ObjectId(item_id)})

def update_item(item_id: str, data: dict):
    return item_collection.update_one(
        {"_id": ObjectId(item_id)},
        {"$set": data}
    )

def delete_item(item_id: str):
    return item_collection.delete_one({"_id": ObjectId(item_id)})
