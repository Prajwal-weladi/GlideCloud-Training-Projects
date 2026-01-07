from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_item():
    response = client.post("/items/", json={
        "name": "Laptop",
        "description": "Gaming laptop",
        "price": 75000
    })
    assert response.status_code == 200
    assert "id" in response.json()

def test_get_items():
    response = client.get("/items/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_update_item():
    create = client.post("/items/", json={
        "name": "Phone",
        "price": 20000
    })
    item_id = create.json()["id"]

    update = client.put(f"/items/{item_id}", json={
        "price": 18000
    })
    assert update.status_code == 200

def test_delete_item():
    create = client.post("/items/", json={
        "name": "Tablet",
        "price": 15000
    })
    item_id = create.json()["id"]

    delete = client.delete(f"/items/{item_id}")
    assert delete.status_code == 200
