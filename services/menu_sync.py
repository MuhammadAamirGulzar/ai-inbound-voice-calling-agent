"""Menu pipeline: published RAG menus -> Postgres -> Chroma index."""

import re

import httpx
from sqlalchemy import func
from sqlalchemy.orm import Session

from sql import models

RAG_UPSTREAM = "http://127.0.0.1:8001"


def parse_price(price_str) -> float:
    if price_str is None:
        return 0.0
    if isinstance(price_str, (int, float)):
        return float(price_str)
    # Remove commas and non-numeric characters except dots
    cleaned = str(price_str).replace(",", "")
    match = re.search(r'\d+(?:\.\d+)?', cleaned)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0
    return 0.0

def map_category_to_db(category_str: str) -> str:
    cat_str = str(category_str).strip()
    cat_lower = cat_str.lower()
    
    # Check beverages
    beverage_keywords = ["drink", "beverage", "soda", "shake", "tea", "coffee", "juice", "water", "smoothie", "lassi", "mojito", "cocktail", "mocktail", "chai"]
    if any(kw in cat_lower for kw in beverage_keywords):
        return "beverages"
        
    # Check appetizers
    appetizer_keywords = ["appetizer", "starter", "side", "soup", "salad", "fries", "wing", "nacho", "bread", "nugget", "roll", "bite", "sauce", "dip"]
    if any(kw in cat_lower for kw in appetizer_keywords):
        return "appetizers"
        
    # Check desserts
    dessert_keywords = ["dessert", "sweet", "cake", "ice cream", "waffle", "crepe", "pudding", "pastry", "brownie", "shake", "cookie", "donut"]
    if any(kw in cat_lower for kw in dessert_keywords):
        return "desserts"
        
    # Fallback to the cleaned raw category name instead of forcing 'main-course'
    return cat_str

def save_published_menu_to_postgres(db: Session, restaurant_id: int, menu_items: list):
    for item in menu_items:
        raw_category = (item.get("category") or "General").strip()
        category = map_category_to_db(raw_category)
        base_name = (item.get("item") or "").strip()
        if not base_name:
            continue
        description = item.get("description", "")
        variants = item.get("variants") or []
        
        if variants:
            for var in variants:
                var_name = f"{base_name} ({var.get('name', '').strip()})" if var.get("name") else base_name
                price = parse_price(var.get("price"))
                
                db_item = db.query(models.MenuItem).filter(
                    models.MenuItem.restaurant_id == restaurant_id,
                    func.lower(models.MenuItem.name) == var_name.lower()
                ).first()
                
                if db_item:
                    db_item.category = category
                    db_item.description = description
                    db_item.price = price
                else:
                    new_db_item = models.MenuItem(
                        restaurant_id=restaurant_id,
                        name=var_name,
                        category=category,
                        description=description,
                        price=price
                    )
                    db.add(new_db_item)
        else:
            price = parse_price(item.get("price"))
            db_item = db.query(models.MenuItem).filter(
                models.MenuItem.restaurant_id == restaurant_id,
                func.lower(models.MenuItem.name) == base_name.lower()
            ).first()
            
            if db_item:
                db_item.category = category
                db_item.description = description
                db_item.price = price
            else:
                new_db_item = models.MenuItem(
                    restaurant_id=restaurant_id,
                    name=base_name,
                    category=category,
                    description=description,
                    price=price
                )
                db.add(new_db_item)
                
    db.commit()

async def sync_postgres_menu_to_chroma(db: Session, restaurant_id: int):
    # Query all menu items for the restaurant
    items = db.query(models.MenuItem).filter(models.MenuItem.restaurant_id == restaurant_id).all()
    
    # Construct the payload
    menu_items_payload = []
    for it in items:
        menu_items_payload.append({
            "name": it.name,
            "category": it.category,
            "description": it.description or "",
            "price": float(it.price),
            "variants": [],
            "customizations": []
        })
        
    # Call the sync endpoint on RAG_UPSTREAM
    url = f"{RAG_UPSTREAM}/businesses/{restaurant_id}/menu/sync"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, json={"menu_items": menu_items_payload})
            if r.status_code == 200:
                print(f"[sync] Successfully synced {len(items)} menu items from Postgres to Chroma.")
            else:
                print(f"[sync] Failed to sync menu items to Chroma. Status: {r.status_code}, Resp: {r.text}")
    except Exception as e:
        print(f"[sync] Error syncing menu items to Chroma: {e}")
