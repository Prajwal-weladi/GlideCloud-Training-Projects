from fastapi import APIRouter, HTTPException
from app import crud
from app.schemas import ItemCreate, ItemUpdate

router = APIRouter(prefix="/items", tags=["Items"])

@router.post("/")
def create(item: ItemCreate):
    item_id = crud.create_item(item.model_dump())
    return {"id": item_id}

@router.get("/")
def read_all():
    return crud.get_all_items()

@router.get("/{item_id}")
def read_one(item_id: str):
    item = crud.get_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    item["_id"] = str(item["_id"])
    return item

@router.put("/{item_id}")
def update(item_id: str, item: ItemUpdate):
    data = {k: v for k, v in item.model_dump().items() if v is not None}
    result = crud.update_item(item_id, data)
    if result.matched_count == 0:
        raise HTTPException(404, "Item not found")
    return {"message": "Item updated"}

@router.delete("/{item_id}")
def delete(item_id: str):
    result = crud.delete_item(item_id)
    if result.deleted_count == 0:
        raise HTTPException(404, "Item not found")
    return {"message": "Item deleted"}
