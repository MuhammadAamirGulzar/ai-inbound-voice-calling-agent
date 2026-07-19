import os
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
import uvicorn
from fastapi import Depends, FastAPI, Form, Request, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session

from sql import crud, models, schemas
from sql.database import engine, SessionLocal
from twilio_routes import router as twilio_router
from utils.authentication import create_access_token
from utils.logger import print_info, print_error
from utils.cookies import validate_admin_cookies, validate_superadmin_cookies, validate_cookies
from send_email import send_email

# ── NOTE: openvoicechat TTS/STT/LLM imports are intentionally NOT here. ──
# The Twilio pipeline (Mouth_piper, Ear_faster_whisper, Chatbot_LLM, etc.)
# is fully self-contained inside twilio_routes.py and instantiated per call.
# These imports belonged to the old /chatws browser WebSocket, which is deleted.

load_dotenv()

# ── Ensure cache dirs are inside the project (avoids C: drive space issues) ──
project_root = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(project_root, ".cache")
os.makedirs(cache_dir, exist_ok=True)

os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"]
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")

temp_dir = os.path.join(cache_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir

# ── Startup guards ──
SESSION_MIDDLEWARE_SECRET_KEY = os.getenv("SESSION_MIDDLEWARE_SECRET_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not SESSION_MIDDLEWARE_SECRET_KEY or not JWT_SECRET_KEY:
    raise RuntimeError(
        "Missing required environment variables: SESSION_MIDDLEWARE_SECRET_KEY and JWT_SECRET_KEY"
    )

# ── App setup ──
models.Base.metadata.create_all(bind=engine)

# Additive migration for pre-existing databases (voice engine columns).
from sql.database import ensure_columns
try:
    ensure_columns()
except Exception as _mig_err:
    print(f"[db] column migration warning: {_mig_err}")

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_MIDDLEWARE_SECRET_KEY)
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def startup_event():
    import subprocess
    import sys
    import asyncio
    if os.getenv("RAG_AUTOSTART", "true").lower() in ("0", "false", "no"):
        print_info("RAG_AUTOSTART disabled — skipping knowledge_rag sidecar.")
        return
    proj_root = os.path.dirname(os.path.abspath(__file__))
    rag_dir = os.path.join(proj_root, "knowledge_rag")

    rag_python = os.path.join(rag_dir, "venv", "Scripts", "python.exe")
    if not os.path.exists(rag_python):
        rag_python = sys.executable

    cmd = [rag_python, "-m", "uvicorn", "main:app", "--port", "8001"]

    try:
        print_info(f"Auto-starting knowledge_rag backend on port 8001...")
        # No CREATE_NO_WINDOW — inherit parent's stdout/stderr so RAG logs
        # appear in the same terminal as the main uvicorn server.
        process = subprocess.Popen(
            cmd,
            cwd=rag_dir,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        app.state.rag_process = process

        # Wait until the RAG server is healthy (up to 60s) before accepting traffic
        health_url = "http://127.0.0.1:8001/health"
        async with httpx.AsyncClient(timeout=3.0) as client:
            for attempt in range(20):
                await asyncio.sleep(3)
                try:
                    r = await client.get(health_url)
                    if r.status_code == 200:
                        print_info("knowledge_rag backend is ready on port 8001.")
                        # Sync existing postgres menus to Chroma
                        try:
                            db_session = SessionLocal()
                            restaurants = db_session.query(models.Restaurant).all()
                            for rest in restaurants:
                                await sync_postgres_menu_to_chroma(db_session, rest.id)
                            db_session.close()
                        except Exception as sync_err:
                            print_error(f"Failed to run initial startup menu sync: {sync_err}")
                        break
                except Exception:
                    pass
            else:
                print_error("knowledge_rag backend did not become ready within 60s.")
    except Exception as e:
        print_error(f"Failed to auto-start knowledge_rag backend: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    process = getattr(app.state, "rag_process", None)
    if process:
        print_info("Stopping knowledge_rag backend...")
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


def pk_time_filter(dt, fmt="%I:%M %p"):
    if not dt:
        return ""
    from datetime import timedelta
    # Add 5 hours to offset UTC to PKT (Pakistan Standard Time)
    pkt_dt = dt + timedelta(hours=5)
    return pkt_dt.strftime(fmt)

templates.env.filters["pk_time"] = pk_time_filter
templates.env.globals["pk_time"] = pk_time_filter

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(twilio_router)

# Browser-mic test console needs the local ML stack (torch/whisper);
# skip it gracefully on cloud-streaming-only deployments.
try:
    from test_routes import router as test_router
    app.include_router(test_router)
except ImportError as _test_err:
    print(f"Local test console disabled (missing local ML deps): {_test_err}")

# Include SIP router dynamically if enabled
SIP_ENABLED = os.getenv("SIP_ENABLED", "false").lower() == "true"
if SIP_ENABLED:
    try:
        from sip_routes import router as sip_router
        app.include_router(sip_router)
        print("SIP routes registered successfully.")
    except Exception as e:
        print(f"Failed to register SIP routes: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── RAG Reverse Proxy ──────────────────────────────────────────────────────────
# Forwards /rag/* to the knowledge_rag microservice on port 8001 so menu.html
# can call /rag/ingest etc. on the same origin (port 8000) — no CORS needed.
import httpx
from fastapi import Response as FastAPIResponse

RAG_UPSTREAM = "http://127.0.0.1:8001"

import re
from sqlalchemy import func

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

@app.api_route("/rag/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def rag_proxy(path: str, request: Request, db: Session = Depends(get_db)):
    url = f"{RAG_UPSTREAM}/{path}"
    # No timeout limit — image ingestion through the ngrok tunnel (vision model
    # + structuring) can take several minutes on slow connections.
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            body = await request.body()
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
            resp = await client.request(
                method=request.method,
                url=url,
                params=dict(request.query_params),
                headers=headers,
                content=body,
            )
        
        # Intercept post-publish response to save to Postgres & sync to Chroma
        if resp.status_code == 200 and request.method == "POST" and "reviews" in path and path.endswith("/publish"):
            try:
                publish_match = re.match(r"businesses/([^/]+)/reviews/([^/]+)/publish", path)
                if publish_match:
                    restaurant_id = int(publish_match.group(1))
                    published_data = resp.json()
                    menu_items = published_data.get("data", {}).get("menu") or []
                    save_published_menu_to_postgres(db, restaurant_id, menu_items)
                    await sync_postgres_menu_to_chroma(db, restaurant_id)
            except Exception as e:
                print(f"Error handling post-publish sync to Postgres/Chroma: {e}")

        return FastAPIResponse(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type"),
        )
    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={"detail": "Knowledge RAG service is not available. It may still be starting up — please wait a moment and try again."},
        )
    except httpx.ReadTimeout:
        return JSONResponse(
            status_code=504,
            content={"detail": "The RAG service took too long to respond. The ingestion may still be processing in the background — check Pending Reviews in a moment."},
        )


def check_and_send_due_date_reminders(db: Session):
    import calendar
    from datetime import datetime, timedelta
    from sql import models
    from send_email import send_email as _send_email

    restaurants = db.query(models.Restaurant).filter(models.Restaurant.is_suspended == False).all()
    now = datetime.now()

    for restaurant in restaurants:
        due_date = None
        if restaurant.is_free_plan:
            if restaurant.trial_expiration_date:
                due_date = restaurant.trial_expiration_date + timedelta(days=5)
        elif restaurant.created_at:
            start_date = restaurant.created_at
            months_passed = (now.year - start_date.year) * 12 + (now.month - start_date.month)
            if now.day < start_date.day:
                months_passed -= 1
            
            plan = restaurant.subscription_plan
            if plan == "six_monthly":
                months_passed = (months_passed // 6) * 6
                months_to_add = 6
            elif plan == "annually":
                months_passed = (months_passed // 12) * 12
                months_to_add = 12
            else:
                months_to_add = 1
                
            total_months = months_passed + months_to_add
            next_year = start_date.year + (start_date.month - 1 + total_months) // 12
            next_month = (start_date.month - 1 + total_months) % 12 + 1
            
            try:
                _, last_day = calendar.monthrange(next_year, next_month)
                next_day = min(start_date.day, last_day)
                next_cycle = datetime(next_year, next_month, next_day, start_date.hour, start_date.minute)
                due_date = next_cycle + timedelta(days=5)
            except Exception:
                due_date = None

        if due_date:
            days_left = (due_date.date() - now.date()).days
            if days_left == 5:
                # Find admin user for this restaurant
                admin_user = db.query(models.User).filter(models.User.restaurant_id == restaurant.id, models.User.role == "admin").first()
                if admin_user and admin_user.email:
                    email_body = f"""
<html><body style="font-family:Inter,sans-serif;color:#111827;">
<div style="max-width:560px;margin:0 auto;padding:32px 24px;">
  <h2 style="color:#eab308;">&#9888; Subscription Payment Due in 5 Days</h2>
  <p>Hello {admin_user.name},</p>
  <p>This is a friendly reminder that the subscription payment for <strong>{restaurant.name}</strong> is due in <strong>5 days</strong> on <strong>{due_date.strftime("%B %d, %Y")}</strong>.</p>
  <p>Please ensure your payment is processed to avoid any interruption in service.</p>
  <table style="width:100%;border-collapse:collapse;margin:20px 0;">
    <tr><td style="padding:8px;background:#f3f4f6;font-weight:600;width:40%;">Restaurant</td><td style="padding:8px;">{restaurant.name}</td></tr>
    <tr><td style="padding:8px;background:#f3f4f6;font-weight:600;">Plan</td><td style="padding:8px;text-transform:capitalize;">{restaurant.subscription_plan.replace('_', ' ')}</td></tr>
    <tr><td style="padding:8px;background:#f3f4f6;font-weight:600;">Due Date</td><td style="padding:8px;color:#eab308;"><strong>{due_date.strftime("%B %d, %Y")}</strong></td></tr>
  </table>
  <a href="https://ordersaathi.com/admin/billing" style="display:inline-block;padding:12px 24px;background:#eab308;color:black;border-radius:6px;text-decoration:none;font-weight:600;margin-top:8px;">View Invoice & Payment Details</a>
  <p style="margin-top:24px;font-size:12px;color:#6b7280;">OrderSaathi Support Team</p>
</div></body></html>
"""
                    import threading
                    threading.Thread(
                        target=_send_email,
                        args=(admin_user.email, f"[OrderSaathi] Subscription Payment Reminder — {restaurant.name}", email_body, True),
                        daemon=True
                    ).start()


def start_due_date_reminder_cron():
    import time
    import threading

    def run_cron():
        # Wait a bit on startup
        time.sleep(10)
        while True:
            db = SessionLocal()
            try:
                check_and_send_due_date_reminders(db)
            except Exception as e:
                print(f"Error in due date reminder cron: {e}")
            finally:
                db.close()
            # Sleep 24 hours
            time.sleep(86400)

    t = threading.Thread(target=run_cron, daemon=True)
    t.start()


@app.on_event("startup")
async def startup_event():
    start_due_date_reminder_cron()


# ─────────────────────────────────────────────
# Auth (temporary — kept until platform_users login is built)
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, db: Session = Depends(get_db)):
    from utils.cookies import get_cookies_from_request
    from utils.authentication import verify_access_token
    
    cookies = get_cookies_from_request(request)
    access_token = cookies.get("access_cookie")
    if access_token:
        username = verify_access_token(access_token)
        if username:
            user = crud.get_user(db, username)
            if user:
                if user.must_change_password:
                    return RedirectResponse(url="/change-password", status_code=302)
                elif user.role == "super_admin":
                    return RedirectResponse(url="/superadmin/dashboard", status_code=302)
                elif user.role == "admin":
                    return RedirectResponse(url="/admin/dashboard", status_code=302)
                    
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> JSONResponse:
    user = crud.authenticate_user(db, username, password)
    if user:
        if user.role in ["super_admin", "admin"]:
            if user.role == "admin":
                restaurant = db.query(models.Restaurant).filter(models.Restaurant.admin_user_id == user.id).first()
                if restaurant:
                    from datetime import datetime
                    is_expired_trial = (
                        restaurant.is_free_plan and 
                        restaurant.trial_expiration_date is not None and 
                        restaurant.trial_expiration_date < datetime.utcnow()
                    )
                    if restaurant.is_suspended or is_expired_trial:
                        if is_expired_trial and not restaurant.is_suspended:
                            restaurant.is_suspended = True
                            db.commit()
                        print_error(f"Login rejected: restaurant '{restaurant.name}' is suspended/expired.")
                        return JSONResponse(
                            status_code=403,
                            content={"error": "Your account/restaurant has been suspended. Please contact support."},
                        )
            token = create_access_token(username)
            print_info(f"User Logged In: {username} (Role: {user.role})")
            
            # Check if password change is forced
            if user.must_change_password:
                redirect_url = "/change-password"
            else:
                if user.role == "super_admin":
                    redirect_url = "/superadmin/dashboard"
                else:
                    redirect_url = "/admin/dashboard"
            
            response = JSONResponse(
                status_code=200,
                content={
                    "response": "Success",
                    "redirect_url": redirect_url,
                    "token": token,
                },
            )
            response.set_cookie(key="access_cookie", value=token, httponly=True)
            response.set_cookie(key="current_username", value=username, httponly=True)
            return response
        else:
            print_error(f"Login failed: user '{username}' has role '{user.role}' which is not supported")
            return JSONResponse(
                status_code=403,
                content={"error": "Account role not supported"},
            )
    print_error("Login failed: invalid credentials")
    return JSONResponse(
        status_code=401,
        content={"error": "Invalid credentials. Please check your username and/or password."},
    )


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="access_cookie")
    response.delete_cookie(key="current_username")
    return response


@app.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request, db: Session = Depends(get_db)):
    auth = validate_cookies(db, request, allow_must_change_password=True)
    if not auth["success"]:
        return auth["response"]
    return templates.TemplateResponse("change_password.html", {"request": request, "username": auth["user"].username})


@app.post("/change-password")
async def change_password_post(
    request: Request,
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db)
):
    auth = validate_cookies(db, request, allow_must_change_password=True)
    if not auth["success"]:
        return auth["response"]
        
    user = auth["user"]
    import re
    if (len(new_password) < 8 or
        not re.search(r'[A-Z]', new_password) or
        not re.search(r'[a-z]', new_password) or
        not re.search(r'[0-9]', new_password) or
        not re.search(r'[^A-Za-z0-9]', new_password)):
        return JSONResponse(
            status_code=400,
            content={"error": "Password does not meet complexity requirements. It must be at least 8 characters and include uppercase, lowercase, numeric, and special characters."}
        )
        
    if new_password != confirm_password:
        return JSONResponse(
            status_code=400,
            content={"error": "Passwords do not match."}
        )
        
    from werkzeug.security import generate_password_hash
    user.password = generate_password_hash(new_password)
    user.must_change_password = False
    db.commit()
    
    if user.role == "super_admin":
        redirect_url = "/superadmin/dashboard"
    else:
        redirect_url = "/admin/dashboard"
        
    return JSONResponse(
        status_code=200,
        content={
            "response": "Success",
            "redirect_url": redirect_url
        }
    )



# ─────────────────────────────────────────────
# Admin Dashboard (formerly v2)
# ─────────────────────────────────────────────

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    date_filter: str = "today",
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    from datetime import date, datetime, time as dtime, timedelta
    today = date.today()
    
    if date_filter == "today":
        start_dt = datetime.combine(today, dtime.min)
        end_dt = datetime.combine(today, dtime.max)
        range_display_label = f"Today, {today.strftime('%B %d, %Y')}"
    elif date_filter == "yesterday":
        yesterday = today - timedelta(days=1)
        start_dt = datetime.combine(yesterday, dtime.min)
        end_dt = datetime.combine(yesterday, dtime.max)
        range_display_label = f"Yesterday, {yesterday.strftime('%B %d, %Y')}"
    elif date_filter == "7days":
        start_dt = datetime.combine(today - timedelta(days=6), dtime.min)
        end_dt = datetime.combine(today, dtime.max)
        range_display_label = f"Last 7 Days ({start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d, %Y')})"
    elif date_filter == "30days":
        start_dt = datetime.combine(today - timedelta(days=29), dtime.min)
        end_dt = datetime.combine(today, dtime.max)
        range_display_label = f"Last 30 Days ({start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d, %Y')})"
    else:  # all
        start_dt = datetime.min
        end_dt = datetime.max
        range_display_label = "All Time"

    restaurant_id = auth["user"].restaurant_id
    
    calls_query = db.query(models.ChatHistory).filter(models.ChatHistory.restaurant_id == restaurant_id)
    orders_query = db.query(models.Order).filter(models.Order.restaurant_id == restaurant_id)
    
    if start_dt != datetime.min:
        calls_query = calls_query.filter(models.ChatHistory.timestamp >= start_dt)
        orders_query = orders_query.filter(models.Order.created_at >= start_dt)
    if end_dt != datetime.max:
        calls_query = calls_query.filter(models.ChatHistory.timestamp <= end_dt)
        orders_query = orders_query.filter(models.Order.created_at <= end_dt)
        
    calls = calls_query.all()
    orders = orders_query.all()
    
    total_calls = len(calls)
    completed_calls = sum(1 for c in calls if c.status == "completed")
    missed_calls = sum(1 for c in calls if c.status == "missed")
    
    total_orders = len(orders)
    
    # Call-to-order ratio
    if completed_calls > 0:
        conversion_pct = int((total_orders / completed_calls) * 100)
    else:
        conversion_pct = 0
        
    # Avg Call Duration
    valid_durations = [c.duration_seconds for c in calls if c.duration_seconds is not None]
    if valid_durations:
        avg_seconds = int(sum(valid_durations) / len(valid_durations))
        avg_duration_str = f"{avg_seconds // 60}m {avg_seconds % 60}s"
    else:
        avg_duration_str = "0m 0s"
        
    # Call Volume by Hour (Today)
    today_start = datetime.combine(today, dtime.min)
    today_end = datetime.combine(today, dtime.max)
    today_calls = db.query(models.ChatHistory).filter(
        models.ChatHistory.restaurant_id == restaurant_id,
        models.ChatHistory.timestamp >= today_start,
        models.ChatHistory.timestamp <= today_end
    ).all()
    
    hourly_counts = [0] * 12
    for c in today_calls:
        if c.timestamp:
            h = c.timestamp.hour
            if 9 <= h <= 20:
                hourly_counts[h - 9] += 1
                
    max_hourly_val = max(1, max(hourly_counts))
    hourly_percentages = [int((val / max_hourly_val) * 100) for val in hourly_counts]
    
    # Recent Inbound Calls (last 5)
    recent_calls = db.query(models.ChatHistory).filter(
        models.ChatHistory.restaurant_id == restaurant_id
    ).order_by(models.ChatHistory.timestamp.desc()).limit(5).all()
    
    recent_calls_data = []
    for c in recent_calls:
        linked_order = db.query(models.Order).filter(models.Order.call_id == c.id).first()
        order_info = {
            "exists": "yes" if linked_order else "no",
            "text": f"Yes ({linked_order.status})" if linked_order else "No"
        }
        
        dur_str = "-"
        if c.duration_seconds is not None:
            dur_str = f"{c.duration_seconds // 60}m {c.duration_seconds % 60}s"
            
        recent_calls_data.append({
            "id": c.id,
            "caller_number": c.caller_number or "Unknown",
            "time_str": (c.timestamp + timedelta(hours=5)).strftime("%I:%M %p") if c.timestamp else "-",
            "duration_str": dur_str,
            "status": c.status.title(),
            "order_result": order_info
        })
        
    # Recent Orders (last 5)
    recent_orders = db.query(models.Order).filter(
        models.Order.restaurant_id == restaurant_id
    ).order_by(models.Order.created_at.desc()).limit(5).all()
    
    recent_orders_data = []
    for o in recent_orders:
        time_diff = datetime.utcnow() - o.created_at
        if time_diff.total_seconds() < 60:
            time_str = "Just now"
        elif time_diff.total_seconds() < 3600:
            time_str = f"{int(time_diff.total_seconds() // 60)}m ago"
        else:
            time_str = (o.created_at + timedelta(hours=5)).strftime("%I:%M %p")
            
        items_str = ""
        if isinstance(o.items_summary, list):
            items_str = ", ".join(f"{item.get('qty', 1)}x {item.get('item', '')}" for item in o.items_summary)
        elif isinstance(o.items_summary, dict):
            items_str = ", ".join(f"{v}x {k}" for k, v in o.items_summary.items())
        else:
            items_str = str(o.items_summary)
            
        recent_orders_data.append({
            "id": o.id,
            "items_str": items_str,
            "total_price": f"PKR {o.total_price:,.0f}",
            "time_str": time_str,
            "status": o.status
        })
        
    return templates.TemplateResponse("admin/dashboard.html", {
        "request": request,
        "range_display_label": range_display_label,
        "date_filter": date_filter,
        "total_calls": total_calls,
        "total_orders": total_orders,
        "missed_calls": missed_calls,
        "avg_duration": avg_duration_str,
        "conversion_pct": conversion_pct,
        "hourly_counts": hourly_counts,
        "hourly_percentages": hourly_percentages,
        "recent_calls": recent_calls_data,
        "recent_orders": recent_orders_data,
        "is_agent_active": auth["user"].restaurant.agent_configuration.is_active if auth["user"].restaurant and auth["user"].restaurant.agent_configuration else False,
        "current_user": auth["user"]
    })

@app.get("/admin/call-history", response_class=HTMLResponse)
async def admin_call_history(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    restaurant_id = auth["user"].restaurant_id
    calls = db.query(models.ChatHistory).filter(
        models.ChatHistory.restaurant_id == restaurant_id
    ).order_by(models.ChatHistory.timestamp.desc()).all()
    
    formatted_calls = []
    from datetime import date, timedelta
    today = date.today()
    for c in calls:
        linked_order = db.query(models.Order).filter(models.Order.call_id == c.id).first()
        order_info = {
            "exists": "yes" if linked_order else "no",
            "text": f"Yes ({linked_order.status})" if linked_order else "No"
        }
        
        dur_str = "-"
        if c.duration_seconds is not None:
            dur_str = f"{c.duration_seconds // 60}m {c.duration_seconds % 60}s"
            
        time_str = (c.timestamp + timedelta(hours=5)).strftime("%b %d, %I:%M %p") if c.timestamp else "-"
        
        call_date = c.timestamp.date() if c.timestamp else today
        if call_date == today:
            date_cat = "today"
        elif call_date == today - timedelta(days=1):
            date_cat = "yesterday"
        elif today - timedelta(days=7) <= call_date <= today:
            date_cat = "7days"
        elif today - timedelta(days=30) <= call_date <= today:
            date_cat = "30days"
        else:
            date_cat = "older"
            
        formatted_calls.append({
            "id": c.id,
            "caller_number": c.caller_number or "Unknown",
            "time_str": time_str,
            "duration_str": dur_str,
            "status": c.status.title(),
            "status_lower": c.status.lower(),
            "order": order_info,
            "date_cat": date_cat
        })
        
    return templates.TemplateResponse("admin/call-history.html", {
        "request": request,
        "calls": formatted_calls,
        "current_user": auth["user"]
    })

@app.get("/admin/call-detail", response_class=HTMLResponse)
async def admin_call_detail(request: Request, id: Optional[int] = None, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    if not id:
        return RedirectResponse(url="/admin/call-history", status_code=302)
        
    call = db.query(models.ChatHistory).filter(
        models.ChatHistory.id == id,
        models.ChatHistory.restaurant_id == auth["user"].restaurant_id
    ).first()
    
    if not call:
        return HTMLResponse(content="Call record not found or access denied.", status_code=404)
        
    linked_order = db.query(models.Order).filter(models.Order.call_id == call.id).first()
    
    dur_str = "-"
    if call.duration_seconds is not None:
        dur_str = f"{call.duration_seconds // 60}m {call.duration_seconds % 60}s"
        
    return templates.TemplateResponse("admin/call-detail.html", {
        "request": request,
        "call": call,
        "duration_str": dur_str,
        "order": linked_order,
        "current_user": auth["user"]
    })

@app.get("/admin/orders", response_class=HTMLResponse)
async def admin_orders(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    restaurant_id = auth["user"].restaurant_id
    orders = db.query(models.Order).filter(
        models.Order.restaurant_id == restaurant_id
    ).order_by(models.Order.created_at.desc()).all()
    
    formatted_orders = []
    for o in orders:
        time_str = (o.created_at + timedelta(hours=5)).strftime("%b %d, %I:%M %p")
        
        items_str = ""
        if isinstance(o.items_summary, list):
            items_str = ", ".join(f"{item.get('qty', 1)}x {item.get('item', '')}" for item in o.items_summary)
        elif isinstance(o.items_summary, dict):
            items_str = ", ".join(f"{v}x {k}" for k, v in o.items_summary.items())
        else:
            items_str = str(o.items_summary)
            
        formatted_orders.append({
            "id": o.id,
            "customer_phone": o.customer_phone,
            "items_str": items_str,
            "total_price": f"PKR {o.total_price:,.0f}",
            "time_str": time_str,
            "status": o.status.title(),
            "status_lower": o.status.lower(),
            "dispatch_attempts": o.dispatch_attempts,
            "dispatched_to_external_system": o.dispatched_to_external_system,
            "last_dispatch_error": o.last_dispatch_error
        })
        
    return templates.TemplateResponse("admin/orders.html", {
        "request": request,
        "orders": formatted_orders,
        "current_user": auth["user"]
    })

@app.get("/admin/order-detail", response_class=HTMLResponse)
async def admin_order_detail(request: Request, id: Optional[int] = None, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    if not id:
        return RedirectResponse(url="/admin/orders", status_code=302)
        
    order = db.query(models.Order).filter(
        models.Order.id == id,
        models.Order.restaurant_id == auth["user"].restaurant_id
    ).first()
    
    if not order:
        return HTMLResponse(content="Order not found or access denied.", status_code=404)
        
    subtotal = sum(item.get("price", 0) * item.get("qty", 1) for item in order.items_summary) if isinstance(order.items_summary, list) else order.total_price
    
    return templates.TemplateResponse("admin/order-detail.html", {
        "request": request,
        "order": order,
        "subtotal": subtotal,
        "current_user": auth["user"]
    })

@app.post("/admin/orders/{order_id}/status")
async def admin_update_order_status(
    order_id: int,
    req_body: schemas.OrderStatusUpdate,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    status_title = req_body.status.strip().title()
    if status_title not in ["Pending", "Confirmed", "Cancelled"]:
        return JSONResponse(status_code=400, content={"error": "Invalid status. Only Pending, Confirmed, Cancelled are allowed."})

    order = db.query(models.Order).filter(
        models.Order.id == order_id,
        models.Order.restaurant_id == auth["user"].restaurant_id
    ).first()
    
    if not order:
        return JSONResponse(status_code=404, content={"error": "Order not found or access denied."})
        
    order.status = status_title
    db.commit()
    return JSONResponse(status_code=200, content={"success": True})


@app.post("/admin/orders/{order_id}/dispatch")
async def admin_dispatch_order(
    order_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    order = db.query(models.Order).filter(
        models.Order.id == order_id,
        models.Order.restaurant_id == auth["user"].restaurant_id
    ).first()
    
    if not order:
        return JSONResponse(status_code=404, content={"error": "Order not found or access denied."})
        
    if order.status.lower() not in ["confirmed", "pending"]:
        return JSONResponse(status_code=400, content={"error": "Order must be in Pending or Confirmed status to be dispatched."})
        
    from integrations import get_adapter
    restaurant = order.restaurant
    
    if not restaurant.fulfillment_target_url:
        return JSONResponse(status_code=400, content={"error": "No integration URL configured. Please update your Fulfillment settings."})
        
    adapter = get_adapter(restaurant.fulfillment_integration_type)
    
    try:
        res = adapter.dispatch(order, restaurant)
        order.dispatch_attempts = (order.dispatch_attempts or 0) + 1
        if res.success:
            order.dispatched_at = datetime.utcnow()
            order.dispatched_to_external_system = True
            order.status = "Confirmed"
            db.add(models.Notification(
                restaurant_id=restaurant.id,
                type="dispatch_success",
                message=f"Order #{order.id} successfully dispatched to your system.",
                related_order_id=order.id,
                is_read=False
            ))
            db.commit()
            return JSONResponse(status_code=200, content={
                "success": True,
                "redirect_url": res.redirect_url,
                "message": res.message
            })
        else:
            order.last_dispatch_error = res.message
            db.add(models.Notification(
                restaurant_id=restaurant.id,
                type="failed_dispatch",
                message=f"Order #{order.id} failed to dispatch: {res.message}",
                related_order_id=order.id,
                is_read=False
            ))
            db.commit()
            return JSONResponse(status_code=400, content={"error": res.message})
    except Exception as e:
        import traceback
        print("Dispatch error:", traceback.format_exc())
        order.dispatch_attempts = (order.dispatch_attempts or 0) + 1
        order.last_dispatch_error = str(e)
        db.add(models.Notification(
            restaurant_id=restaurant.id,
            type="failed_dispatch",
            message=f"Order #{order.id} encountered an error during dispatch.",
            related_order_id=order.id,
            is_read=False
        ))
        db.commit()
        return JSONResponse(status_code=500, content={"error": f"Internal dispatch error: {str(e)}"})


@app.get("/admin/notifications")
async def get_admin_notifications(
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    restaurant_id = auth["user"].restaurant_id
    notifications = db.query(models.Notification).filter(
        models.Notification.restaurant_id == restaurant_id
    ).order_by(models.Notification.created_at.desc()).limit(20).all()
    
    # helper for relative time
    now = datetime.utcnow()
    def rel_time(dt):
        if not dt:
            return ""
        diff = now - dt
        seconds = diff.total_seconds()
        if seconds < 0:
            return "Just now"
        if seconds < 60:
            return "Just now"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        else:
            return f"{diff.days}d ago"

    results = []
    for n in notifications:
        results.append({
            "id": n.id,
            "type": n.type,
            "message": n.message,
            "related_order_id": n.related_order_id,
            "is_read": n.is_read,
            "time_str": rel_time(n.created_at)
        })
        
    unread_count = db.query(models.Notification).filter(
        models.Notification.restaurant_id == restaurant_id,
        models.Notification.is_read == False
    ).count()
    
    return JSONResponse(status_code=200, content={
        "notifications": results,
        "unread_count": unread_count
    })


@app.post("/admin/notifications/{notif_id}/read")
async def mark_notification_read(
    notif_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    notif = db.query(models.Notification).filter(
        models.Notification.id == notif_id,
        models.Notification.restaurant_id == auth["user"].restaurant_id
    ).first()
    
    if notif:
        db.delete(notif)
        db.commit()
    
    return JSONResponse(status_code=200, content={"success": True})


@app.post("/admin/notifications/read-all")
async def mark_all_notifications_read(
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    restaurant_id = auth["user"].restaurant_id
    db.query(models.Notification).filter(
        models.Notification.restaurant_id == restaurant_id
    ).delete(synchronize_session=False)
    db.commit()
    return JSONResponse(status_code=200, content={"success": True})


@app.get("/admin/menu", response_class=HTMLResponse)
async def admin_menu(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    menu_items = db.query(models.MenuItem).filter(
        models.MenuItem.restaurant_id == auth["user"].restaurant_id
    ).all()
    
    restaurant = auth["user"].restaurant
    business_id = str(restaurant.id) if restaurant else "default"

    return templates.TemplateResponse("admin/menu.html", {
        "request": request, 
        "current_user": auth["user"],
        "menu_items": menu_items,
        "business_id": business_id
    })

@app.post("/admin/menu/sync")
async def manual_sync_menu(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        await sync_postgres_menu_to_chroma(db, auth["user"].restaurant_id)
        return JSONResponse(status_code=200, content={"success": True, "message": "Menu synchronized to RAG successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to sync menu: {str(e)}"})

@app.post("/admin/menu/add")
async def add_menu_item(req_body: schemas.MenuItemCreate, request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    new_item = models.MenuItem(
        restaurant_id=auth["user"].restaurant_id,
        name=req_body.name,
        category=req_body.category,
        description=req_body.description,
        price=req_body.price
    )
    db.add(new_item)
    db.commit()
    await sync_postgres_menu_to_chroma(db, auth["user"].restaurant_id)
    return JSONResponse(status_code=200, content={"success": True})

@app.put("/admin/menu/{item_id}")
async def update_menu_item(item_id: int, req_body: schemas.MenuItemUpdate, request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    item = db.query(models.MenuItem).filter(
        models.MenuItem.id == item_id,
        models.MenuItem.restaurant_id == auth["user"].restaurant_id
    ).first()
    
    if not item:
        return JSONResponse(status_code=404, content={"error": "Item not found"})
        
    if req_body.name is not None:
        item.name = req_body.name
    if req_body.category is not None:
        item.category = req_body.category
    if req_body.description is not None:
        item.description = req_body.description
    if req_body.price is not None:
        item.price = req_body.price
        
    db.commit()
    await sync_postgres_menu_to_chroma(db, auth["user"].restaurant_id)
    return JSONResponse(status_code=200, content={"success": True})

@app.delete("/admin/menu/{item_id}")
async def delete_menu_item(item_id: int, request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    item = db.query(models.MenuItem).filter(
        models.MenuItem.id == item_id,
        models.MenuItem.restaurant_id == auth["user"].restaurant_id
    ).first()
    
    if not item:
        return JSONResponse(status_code=404, content={"error": "Item not found"})
        
    db.delete(item)
    db.commit()
    await sync_postgres_menu_to_chroma(db, auth["user"].restaurant_id)
    return JSONResponse(status_code=200, content={"success": True})

@app.post("/admin/menu/bulk-delete")
async def bulk_delete_menu_items(req_body: dict, request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    item_ids = req_body.get("ids", [])
    if not item_ids:
        return JSONResponse(status_code=400, content={"error": "No IDs provided"})
        
    db.query(models.MenuItem).filter(
        models.MenuItem.id.in_(item_ids),
        models.MenuItem.restaurant_id == auth["user"].restaurant_id
    ).delete(synchronize_session=False)
    
    db.commit()
    await sync_postgres_menu_to_chroma(db, auth["user"].restaurant_id)
    return JSONResponse(status_code=200, content={"success": True, "message": f"Successfully deleted {len(item_ids)} items."})

@app.get("/admin/live-calls", response_class=HTMLResponse)
async def admin_live_calls(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    active_calls = db.query(models.ChatHistory).filter(
        models.ChatHistory.restaurant_id == auth["user"].restaurant_id,
        models.ChatHistory.status == "in_progress"
    ).order_by(models.ChatHistory.timestamp.desc()).all()
    
    return templates.TemplateResponse("admin/live-calls.html", {
        "request": request, 
        "current_user": auth["user"],
        "active_calls": active_calls
    })

@app.post("/admin/hangup")
async def admin_hangup(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    try:
        body = await request.json()
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse(status_code=400, content={"error": "Missing session_id"})
            
        from websocket_registry import active_connections
        websocket = active_connections.get(session_id)
        if websocket:
            print(f"Hanging up active call with session_id: {session_id}")
            await websocket.close()
            # Double check status update
            log = db.query(models.ChatHistory).filter(models.ChatHistory.session_id == session_id).first()
            if log and log.status == "in_progress":
                log.status = "completed"
                db.commit()
            return JSONResponse(status_code=200, content={"success": True, "message": "Call hung up successfully"})
        else:
            # Fallback DB state update
            log = db.query(models.ChatHistory).filter(models.ChatHistory.session_id == session_id).first()
            if log and log.status == "in_progress":
                log.status = "completed"
                db.commit()
                return JSONResponse(status_code=200, content={"success": True, "message": "Call updated to completed in DB"})
            return JSONResponse(status_code=404, content={"error": "Active call session not found"})
    except Exception as e:
        print(f"Error hanging up call: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/admin/analytics", response_class=HTMLResponse)
async def admin_analytics(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    restaurant_id = auth["user"].restaurant_id
    
    total_calls = db.query(models.ChatHistory).filter(models.ChatHistory.restaurant_id == restaurant_id).count()
    orders = db.query(models.Order).filter(models.Order.restaurant_id == restaurant_id, models.Order.status != "Cancelled").all()
    total_orders = len(orders)
    
    missed_calls = db.query(models.ChatHistory).filter(models.ChatHistory.restaurant_id == restaurant_id, models.ChatHistory.duration_seconds <= 15).count()
    
    total_revenue = sum(o.total_price for o in orders)
    aov = total_revenue / total_orders if total_orders > 0 else 0
    
    calls = db.query(models.ChatHistory).filter(models.ChatHistory.restaurant_id == restaurant_id).all()
    total_seconds = sum((c.duration_seconds or 0) for c in calls)
    total_hours = total_seconds / 3600.0
    
    conversion_rate = (total_orders / total_calls * 100) if total_calls > 0 else 0
    missed_rate = (missed_calls / total_calls * 100) if total_calls > 0 else 0
    
    # Calculate vs last week metrics
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)
    fourteen_days_ago = now - timedelta(days=14)
    
    def get_metrics(start_time, end_time):
        c_q = [c for c in calls if c.timestamp and start_time <= c.timestamp.replace(tzinfo=None) < end_time]
        o_q = [o for o in orders if o.created_at and start_time <= o.created_at.replace(tzinfo=None) < end_time]
        
        c_count = len(c_q)
        o_count = len(o_q)
        m_count = sum(1 for c in c_q if (c.duration_seconds or 0) <= 15)
        rev = sum(o.total_price for o in o_q)
        
        c_rate = (o_count / c_count * 100) if c_count > 0 else 0
        m_rate = (m_count / c_count * 100) if c_count > 0 else 0
        a_v = rev / o_count if o_count > 0 else 0
        h_s = sum((c.duration_seconds or 0) for c in c_q) / 3600.0
        return c_rate, m_rate, a_v, h_s

    c_rate_this, m_rate_this, aov_this, hours_this = get_metrics(seven_days_ago, now)
    c_rate_last, m_rate_last, aov_last, hours_last = get_metrics(fourteen_days_ago, seven_days_ago)

    conversion_rate_diff = c_rate_this - c_rate_last
    missed_rate_diff = m_rate_this - m_rate_last
    aov_diff = aov_this - aov_last
    hours_diff = hours_this - hours_last
    
    densities = [0] * 24
    for c in calls:
        if c.timestamp:
            densities[c.timestamp.hour] += 1
    max_d = max(densities) if densities and max(densities) > 0 else 1
    densities = [int((d / max_d) * 4) for d in densities]
    
    return templates.TemplateResponse("admin/analytics.html", {
        "request": request, 
        "current_user": auth["user"],
        "conversion_rate": conversion_rate,
        "missed_rate": missed_rate,
        "aov": aov,
        "total_hours": total_hours,
        "densities": densities,
        "conversion_rate_diff": conversion_rate_diff,
        "missed_rate_diff": missed_rate_diff,
        "aov_diff": aov_diff,
        "hours_diff": hours_diff
    })

@app.get("/admin/api/analytics/charts")
async def api_analytics_charts(
    request: Request,
    report_range: str = Query("7days", alias="range"),
    grouping: str = "daily",
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    restaurant_id = auth["user"].restaurant_id
    
    days_to_fetch = 7 if report_range == "7days" else 30
    days = [(datetime.now() - timedelta(days=i)).date() for i in range(days_to_fetch - 1, -1, -1)]
    
    calls = db.query(models.ChatHistory).filter(models.ChatHistory.restaurant_id == restaurant_id).all()
    orders = db.query(models.Order).filter(models.Order.restaurant_id == restaurant_id, models.Order.status != "Cancelled").all()
    
    call_volume = [0] * days_to_fetch
    order_volume = [0] * days_to_fetch
    
    for c in calls:
        if c.timestamp:
            d = c.timestamp.date()
            if d in days:
                call_volume[days.index(d)] += 1
                
    for o in orders:
        if o.created_at:
            d = o.created_at.date()
            if d in days:
                order_volume[days.index(d)] += 1
                
    if grouping == "weekly":
        new_labels = []
        new_cv = []
        new_ov = []
        for i in range(0, days_to_fetch, 7):
            chunk_days = days[i:i+7]
            if chunk_days:
                label = f"{chunk_days[0].strftime('%b %d')} - {chunk_days[-1].strftime('%b %d')}"
                new_labels.append(label)
                new_cv.append(sum(call_volume[i:i+7]))
                new_ov.append(sum(order_volume[i:i+7]))
        labels = new_labels
        call_volume = new_cv
        order_volume = new_ov
    else:
        labels = [d.strftime("%a" if days_to_fetch <= 7 else "%b %d") for d in days]
        
    return JSONResponse(status_code=200, content={
        "labels": labels,
        "call_volume": call_volume,
        "order_volume": order_volume
    })

@app.get("/admin/billing", response_class=HTMLResponse)
async def admin_billing(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    from sql.models import Restaurant
    user = auth["user"]
    restaurant = db.query(Restaurant).filter(Restaurant.id == user.restaurant_id).first()
    
    import calendar
    reset_date_str = ""
    if restaurant:
        if restaurant.is_free_plan and restaurant.trial_expiration_date:
            reset_date_str = (restaurant.trial_expiration_date + timedelta(days=5)).strftime("%B %d, %Y")
        elif restaurant.created_at:
            start_date = restaurant.created_at
            now = datetime.now()
            
            months_passed = (now.year - start_date.year) * 12 + (now.month - start_date.month)
            if now.day < start_date.day:
                months_passed -= 1
            
            plan = restaurant.subscription_plan
            if plan == "six_monthly":
                months_passed = (months_passed // 6) * 6
                months_to_add = 6
            elif plan == "annually":
                months_passed = (months_passed // 12) * 12
                months_to_add = 12
            else:
                months_to_add = 1
                
            total_months = months_passed + months_to_add
            next_year = start_date.year + (start_date.month - 1 + total_months) // 12
            next_month = (start_date.month - 1 + total_months) % 12 + 1
            
            _, last_day = calendar.monthrange(next_year, next_month)
            next_day = min(start_date.day, last_day)
            
            next_cycle = datetime(next_year, next_month, next_day, start_date.hour, start_date.minute)
            reset_date_str = (next_cycle + timedelta(days=5)).strftime("%B %d, %Y")
            
    if not reset_date_str:
        reset_date_str = datetime.now().strftime("%B %d, %Y")
    
    used_pct = 0
    if restaurant:
        used_pct = int((restaurant.used_minutes / max(1, restaurant.assigned_minutes)) * 100)
        
    from dotenv import load_dotenv
    load_dotenv(override=True)
    monthly_price = float(os.getenv("MONTHLY_PRICE", "5000"))
    
    plan = restaurant.subscription_plan if restaurant else "free_trial"
    base_price = 0
    if plan == "monthly":
        base_price = monthly_price
    elif plan == "six_monthly":
        base_price = monthly_price * 6 * 0.95
    elif plan == "annually":
        base_price = monthly_price * 12 * 0.90
        
    used_minutes = restaurant.used_minutes if restaurant else 0
    assigned_minutes = restaurant.assigned_minutes if restaurant else 100
    
    overage_minutes = max(0, used_minutes - assigned_minutes)
    overage_cost = overage_minutes * 10
    total_due = base_price + overage_cost
    
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    payment_holder = os.getenv("PAYMENT_ACCOUNT_NAME", "OrderSaathi Technologies")
    payment_account = os.getenv("PAYMENT_ACCOUNT_NUMBER", "0000-0000-0000-0000")
        
    monthly_price = float(os.getenv("MONTHLY_PRICE", "5000"))
    six_monthly_price = monthly_price * 6 * 0.95
    annually_price = monthly_price * 12 * 0.90

    return templates.TemplateResponse("admin/billing.html", {
        "request": request, 
        "reset_date": reset_date_str,
        "restaurant": restaurant,
        "used_percentage": used_pct,
        "current_user": user,
        "base_price": base_price,
        "overage_minutes": overage_minutes,
        "overage_cost": overage_cost,
        "total_due": total_due,
        "payment_holder": payment_holder,
        "payment_account": payment_account,
        "monthly_price": monthly_price,
        "six_monthly_price": six_monthly_price,
        "annually_price": annually_price
    })

@app.get("/admin/agent-config", response_class=HTMLResponse)
async def admin_agent_config(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    user = auth["user"]
    restaurant = user.restaurant
    if restaurant:
        agent_config = restaurant.agent_configuration
        if not agent_config:
            agent_config = models.AgentConfiguration(
                restaurant_id=restaurant.id,
                voice_engine="urdu-female",
                system_prompt=f"You are the friendly AI voice-agent taking orders for {restaurant.name}. Speak in Roman Urdu/Urdu-English mix. Reference the menu database for pricing. Do not offer discounts exceeding PKR 100. Be extremely brief, using under 30 words per conversation bubble. Make sure to collect the customer's delivery address before concluding the order."
            )
            db.add(agent_config)
            db.commit()
            db.refresh(restaurant)

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    elevenlabs_api_present = bool(api_key and api_key != "replace-with-provider-key")

    # Fetch merged RAG profile for this business
    rag_profile = {}
    restaurant_name = restaurant.name if restaurant else ""
    business_id = str(restaurant.id) if restaurant else None
    if business_id:
        try:
            import httpx as _httpx
            rag_url = os.getenv("RAG_UPSTREAM", "http://127.0.0.1:8001")
            async with _httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(f"{rag_url}/businesses/{business_id}/profile")
                if r.status_code == 200:
                    rag_profile = r.json()
        except Exception as _e:
            print(f"[agent-config] Could not fetch RAG profile: {_e}")

    # Build sensible defaults when the RAG has no data yet
    persona = rag_profile.get("persona") or {}
    details = rag_profile.get("details") or {}
    if not details.get("phone") and restaurant:
        details["phone"] = restaurant.order_phone_number or ""
    policies = rag_profile.get("policies") or {}
    rag_business_name = rag_profile.get("business_name") or restaurant_name

    default_greeting = f"Assalam-o-Alaikum! {restaurant_name} se AI assistant bol rahi hoon. Aaj aap kya order karna pasand farmayenge?"
    default_system_prompt = f"You are the friendly AI voice-agent taking orders for {restaurant_name}. Speak in Roman Urdu/Urdu-English mix. Reference the menu database for pricing. Do not offer discounts exceeding PKR 100. Be extremely brief, using under 30 words per conversation bubble. Make sure to collect the customer's delivery address before concluding the order."

    return templates.TemplateResponse(
        "admin/agent-config.html", 
        {
            "request": request, 
            "current_user": user,
            "elevenlabs_api_present": elevenlabs_api_present,
            "rag_business_name": rag_business_name,
            "rag_persona": persona,
            "rag_details": details,
            "rag_policies": policies,
            "default_greeting": persona.get("greeting_script") or default_greeting,
            "default_system_prompt": agent_config.system_prompt if agent_config else default_system_prompt,
        }
    )

@app.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    user = auth["user"]
    restaurant = user.restaurant
    if restaurant:
        agent_config = restaurant.agent_configuration
        if not agent_config:
            agent_config = models.AgentConfiguration(
                restaurant_id=restaurant.id,
                voice_engine="urdu-female",
                system_prompt="You are the friendly AI voice-agent taking orders for Khyber Shinwari Restaurant. Speak in Roman Urdu/Urdu-English mix. Reference the menu database for pricing. Do not offer discounts exceeding PKR 100. Be extremely brief, using under 30 words per conversation bubble. Make sure to collect the customer's delivery address before concluding the order."
            )
            db.add(agent_config)
            db.commit()
            db.refresh(restaurant)

    # Fetch synced values from ChromaDB (RAG profile) to display on settings page
    rag_business_name = restaurant.name if restaurant else ""
    rag_greeting = ""
    rag_system_prompt = agent_config.system_prompt if (restaurant and agent_config) else ""
    if restaurant:
        try:
            import httpx as _httpx
            rag_url = os.getenv("RAG_UPSTREAM", "http://127.0.0.1:8001")
            async with _httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{rag_url}/businesses/{restaurant.id}/profile")
                if r.status_code == 200:
                    profile = r.json()
                    if profile.get("business_name"):
                        rag_business_name = profile["business_name"]
                    # If we have greeting_script or system_prompt in ChromaDB, sync them here too
                    persona = profile.get("persona") or {}
                    if persona.get("greeting_script"):
                        rag_greeting = persona["greeting_script"]
                    if profile.get("system_prompt"):
                        rag_system_prompt = profile["system_prompt"]
        except Exception as _e:
            print(f"[settings] Could not fetch RAG profile: {_e}")

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    elevenlabs_api_present = bool(api_key and api_key != "replace-with-provider-key")

    webhook_url = f"https://{request.headers.get('host', 'localhost:8000')}/voice"
    return templates.TemplateResponse(
        "admin/settings.html", 
        {
            "request": request, 
            "current_user": user, 
            "webhook_url": webhook_url,
            "restaurant_name": rag_business_name,

            "system_prompt": rag_system_prompt,
            "elevenlabs_api_present": elevenlabs_api_present
        }
    )


@app.post("/admin/settings/profile")
async def update_profile_settings(
    req: schemas.RestaurantProfileUpdate,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    restaurant = auth["user"].restaurant
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found."})
        
    restaurant.name = req.name
    restaurant.mobile_number = req.mobile_number
    db.commit()

    agent_config = restaurant.agent_configuration
    if not agent_config:
        agent_config = models.AgentConfiguration(restaurant_id=restaurant.id)
        db.add(agent_config)


    if req.system_prompt is not None:
        agent_config.system_prompt = req.system_prompt
    db.commit()

    # Sync to RAG upstream
    try:
        rag_url = os.getenv("RAG_UPSTREAM", "http://127.0.0.1:8001")
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=8.0) as client:
            patch = {
                "business_name": req.name,
                "persona": {}
            }

            if req.system_prompt is not None:
                patch["system_prompt"] = req.system_prompt
            
            await client.patch(
                f"{rag_url}/businesses/{restaurant.id}/profile",
                json=patch,
                headers={"Content-Type": "application/json"}
            )
    except Exception as _e:
        print(f"[settings/profile] Could not sync to RAG: {_e}")
        
    return JSONResponse(content={"success": True, "message": "Profile updated successfully!"})


@app.post("/admin/settings/agent")
async def update_agent_settings(
    req: schemas.AgentConfigUpdate,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    restaurant = auth["user"].restaurant
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found."})
        
    agent_config = restaurant.agent_configuration
    if not agent_config:
        agent_config = models.AgentConfiguration(restaurant_id=restaurant.id)
        db.add(agent_config)
        
    agent_config.voice_engine = req.voice_engine
    agent_config.system_prompt = req.system_prompt
    
    db.commit()
    return JSONResponse(content={"success": True, "message": "Agent configuration updated successfully!"})


@app.post("/admin/agent-config/profile")
async def update_agent_rag_profile(
    request: Request,
    db: Session = Depends(get_db)
):
    """Proxy for updating persona/details/policies in the RAG knowledge base,
    plus voice/greeting/system_prompt in PostgreSQL — all from a single form submit."""
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    restaurant = auth["user"].restaurant
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found."})

    body = await request.json()

    # ── 1. Update PostgreSQL agent_configuration and Restaurant name ──────────
    if body.get("business_name"):
        restaurant.name = body["business_name"]
    if body.get("details") and body["details"].get("phone"):
        restaurant.order_phone_number = body["details"]["phone"]
    agent_config = restaurant.agent_configuration
    if not agent_config:
        agent_config = models.AgentConfiguration(restaurant_id=restaurant.id)
        db.add(agent_config)
    if body.get("voice_engine"):
        agent_config.voice_engine = body["voice_engine"]

    if body.get("system_prompt"):
        agent_config.system_prompt = body["system_prompt"]
    db.commit()

    # ── 2. Proxy persona/details/policies patch to RAG service ────────────────
    patch: dict = {}
    if body.get("business_name"):
        patch["business_name"] = body["business_name"]
    if body.get("system_prompt"):
        patch["system_prompt"] = body["system_prompt"]
    if body.get("persona"):
        patch["persona"] = body["persona"]
    if body.get("details"):
        patch["details"] = body["details"]
    if body.get("policies"):
        patch["policies"] = body["policies"]

    rag_result = {}
    if patch:
        try:
            rag_url = os.getenv("RAG_UPSTREAM", "http://127.0.0.1:8001")
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=30.0) as client:
                r = await client.patch(
                    f"{rag_url}/businesses/{restaurant.id}/profile",
                    json=patch,
                    headers={"Content-Type": "application/json"},
                )
                if r.status_code == 200:
                    rag_result = r.json()
                else:
                    print(f"[agent-config/profile] RAG patch returned {r.status_code}: {r.text}")
        except Exception as _e:
            print(f"[agent-config/profile] RAG patch failed: {_e}")

    return JSONResponse(content={"success": True, "rag": rag_result})

from pydantic import BaseModel
class AgentToggleReq(BaseModel):
    is_active: bool

@app.post("/admin/settings/agent/toggle")
async def toggle_agent_settings(
    req: AgentToggleReq,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    restaurant = auth["user"].restaurant
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found."})
        
    agent_config = restaurant.agent_configuration
    if not agent_config:
        agent_config = models.AgentConfiguration(restaurant_id=restaurant.id)
        db.add(agent_config)
        
    agent_config.is_active = req.is_active
    db.commit()
    return JSONResponse(content={"success": True, "is_active": agent_config.is_active})


@app.post("/admin/settings/telephony")
async def update_telephony_settings(
    req: schemas.TelephonySettingsUpdate,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    restaurant = auth["user"].restaurant
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found."})
        
    restaurant.order_phone_number = req.order_phone_number
    
    db.commit()

    # Sync to RAG upstream so agent can answer questions about the phone number
    try:
        rag_url = os.getenv("RAG_UPSTREAM", "http://127.0.0.1:8001")
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=8.0) as client:
            patch = {
                "details": {
                    "phone": req.order_phone_number
                }
            }
            await client.patch(
                f"{rag_url}/businesses/{restaurant.id}/profile",
                json=patch,
                headers={"Content-Type": "application/json"}
            )
    except Exception as _e:
        print(f"[settings/telephony] Could not sync phone to RAG: {_e}")

    return JSONResponse(content={"success": True, "message": "Telephony settings updated successfully!"})


# ─────────────────────────────────────────────
# Super Admin Dashboard
# ─────────────────────────────────────────────

@app.get("/superadmin/dashboard", response_class=HTMLResponse)
async def super_dashboard(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    from datetime import date
    from sqlalchemy import cast, Date
    from sql.models import Restaurant, ChatHistory, Order
    
    total_restaurants = db.query(Restaurant).count()
    active_subscriptions = db.query(Restaurant).filter(Restaurant.is_suspended == False).count()
    total_calls_today = db.query(ChatHistory).filter(cast(ChatHistory.timestamp, Date) == date.today()).count()
    
    # Fetch restaurants added in the last 30 days
    from datetime import timedelta
    last_month_date = date.today() - timedelta(days=30)
    recent_restaurants = db.query(Restaurant).filter(cast(Restaurant.created_at, Date) >= last_month_date).order_by(Restaurant.created_at.desc()).all()
    
    total_orders = db.query(Order).count()
    
    return templates.TemplateResponse("superadmin/dashboard.html", {
        "request": request, 
        "current_user": auth["user"],
        "total_restaurants": total_restaurants,
        "active_subscriptions": active_subscriptions,
        "total_calls_today": total_calls_today,
        "total_orders": total_orders,
        "recent_restaurants": recent_restaurants
    })

@app.get("/superadmin/restaurants", response_class=HTMLResponse)
async def super_restaurants(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    from sql.models import Restaurant
    restaurants = db.query(Restaurant).order_by(Restaurant.id.desc()).all()
    return templates.TemplateResponse("superadmin/restaurants.html", {"request": request, "restaurants": restaurants, "current_user": auth["user"]})

@app.get("/superadmin/restaurant-detail", response_class=HTMLResponse)
async def super_restaurant_detail(request: Request, id: Optional[int] = None, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    from sql.models import Restaurant
    restaurant = None
    if id:
        restaurant = db.query(Restaurant).filter(Restaurant.id == id).first()
    if not restaurant:
        return RedirectResponse(url="/superadmin/restaurants", status_code=302)
        
    used_pct = int((restaurant.used_minutes / max(1, restaurant.assigned_minutes)) * 100)
    return templates.TemplateResponse("superadmin/restaurant-detail.html", {
        "request": request,
        "restaurant": restaurant,
        "used_percentage": used_pct,
        "current_user": auth["user"]
    })

@app.get("/superadmin/create-restaurant", response_class=HTMLResponse)
async def super_create_restaurant_page(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    default_order_phone = os.getenv("DEFAULT_ORDER_PHONE_NUMBER", "")
    return templates.TemplateResponse("superadmin/create-restaurant.html", {
        "request": request,
        "default_order_phone": default_order_phone,
        "current_user": auth["user"]
    })

@app.post("/superadmin/create-restaurant")
async def super_create_restaurant_post(
    request: Request,
    restaurant_in: schemas.RestaurantCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    try:
        restaurant, user, plaintext_password = crud.create_restaurant_with_admin(db, restaurant_in)
        
        # Send professional email in background
        login_url = str(request.base_url) + "login"
        email_body = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333333; line-height: 1.6; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .header {{ background-color: #5d5fef; color: white; padding: 20px; text-align: center; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
        .content {{ padding: 20px; }}
        .button {{ display: inline-block; padding: 12px 24px; background-color: #5d5fef; color: white !important; text-decoration: none; border-radius: 4px; font-weight: bold; margin-top: 20px; }}
        .footer {{ font-size: 12px; color: #777777; margin-top: 20px; text-align: center; }}
        .credentials {{ background-color: #f7fafc; padding: 15px; border-radius: 6px; border: 1px solid #edf2f7; font-family: monospace; font-size: 14px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Welcome to OrderSaathi!</h2>
        </div>
        <div class="content">
            <p>Hello <strong>{user.name}</strong>,</p>
            <p>Your restaurant <strong>{restaurant.name}</strong> has been successfully registered on the OrderSaathi platform.</p>
            <p>Here are your admin dashboard credentials. For security reasons, you will be prompted to change your password on your first login.</p>
            
            <div class="credentials">
                <strong>Username:</strong> {user.username}<br>
                <strong>Temporary Password:</strong> {plaintext_password}
            </div>

            <p><strong>Plan Details:</strong> {restaurant.subscription_plan.replace('_', ' ').title()}</p>
            
            <p>Click below to log in and configure your restaurant's call agent settings:</p>
            <center>
                <a href="{login_url}" class="button">Log In to Dashboard</a>
            </center>
        </div>
        <div class="footer">
            <p>&copy; 2026 OrderSaathi. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""
        background_tasks.add_task(
            send_email,
            receiver_email=user.email,
            subject="Welcome to OrderSaathi! Your Admin Account is Ready",
            body=email_body,
            is_html=True
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "restaurant_name": restaurant.name,
                "subscription_plan": restaurant.subscription_plan,
                "order_phone_number": restaurant.order_phone_number,
                "username": user.username,
                "password": plaintext_password
            }
        )
    except Exception as e:
        print_error(f"Error creating restaurant: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.post("/superadmin/restaurants/{restaurant_id}/reset-password")
async def super_restaurant_reset_password(
    restaurant_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    restaurant = db.query(models.Restaurant).filter(models.Restaurant.id == restaurant_id).first()
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found"})
    
    user = db.query(models.User).filter(models.User.id == restaurant.admin_user_id).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "Admin user not found"})
    
    # Generate random password
    from sql.crud import generate_random_password
    from werkzeug.security import generate_password_hash
    plaintext_password = generate_random_password(12)
    user.password = generate_password_hash(plaintext_password)
    user.must_change_password = True
    db.commit()
    
    print_info(f"[DEBUG] Password Reset for Restaurant '{restaurant.name}' Admin '{user.username}': {plaintext_password}")
    
    # Send password reset email in background
    login_url = str(request.base_url) + "login"
    email_body = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333333; line-height: 1.6; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .header {{ background-color: #dc2626; color: white; padding: 20px; text-align: center; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
        .content {{ padding: 20px; }}
        .button {{ display: inline-block; padding: 12px 24px; background-color: #dc2626; color: white !important; text-decoration: none; border-radius: 4px; font-weight: bold; margin-top: 20px; }}
        .footer {{ font-size: 12px; color: #777777; margin-top: 20px; text-align: center; }}
        .credentials {{ background-color: #f7fafc; padding: 15px; border-radius: 6px; border: 1px solid #edf2f7; font-family: monospace; font-size: 14px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Password Reset Request</h2>
        </div>
        <div class="content">
            <p>Hello <strong>{user.name}</strong>,</p>
            <p>An administrator has reset the password for your OrderSaathi account linked with <strong>{restaurant.name}</strong>.</p>
            <p>Your temporary credentials are listed below. You will be forced to change this password on your next login attempt.</p>
            
            <div class="credentials">
                <strong>Username:</strong> {user.username}<br>
                <strong>Temporary Password:</strong> {plaintext_password}
            </div>

            <p>Click below to log in and set your new password:</p>
            <center>
                <a href="{login_url}" class="button">Log In & Reset Password</a>
            </center>
        </div>
        <div class="footer">
            <p>&copy; 2026 OrderSaathi. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""
    background_tasks.add_task(
        send_email,
        receiver_email=user.email,
        subject="Temporary Password Reset - OrderSaathi",
        body=email_body,
        is_html=True
    )
    
    return JSONResponse(status_code=200, content={"success": True, "password": plaintext_password})


@app.post("/superadmin/restaurants/{restaurant_id}/change-plan")
async def super_restaurant_change_plan(
    restaurant_id: int,
    req_body: schemas.PlanChangeRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    restaurant = db.query(models.Restaurant).filter(models.Restaurant.id == restaurant_id).first()
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found"})
    
    restaurant.is_free_plan = req_body.is_free_plan
    
    from datetime import datetime, timedelta
    if req_body.is_free_plan:
        restaurant.subscription_plan = "free_trial"
        # Default to 2 weeks trial
        restaurant.trial_expiration_date = datetime.utcnow() + timedelta(days=14)
    else:
        restaurant.subscription_plan = req_body.subscription_plan
        restaurant.trial_expiration_date = None
        
    db.commit()
    return JSONResponse(status_code=200, content={"success": True})


@app.post("/superadmin/restaurants/{restaurant_id}/extend-trial")
async def super_restaurant_extend_trial(
    restaurant_id: int,
    req_body: schemas.ExtendTrialRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    restaurant = db.query(models.Restaurant).filter(models.Restaurant.id == restaurant_id).first()
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found"})
    
    try:
        from datetime import datetime
        exp_date = datetime.strptime(req_body.trial_expiration_date, "%Y-%m-%d")
        restaurant.trial_expiration_date = exp_date
        restaurant.is_free_plan = True
        
        # If extended date is in the future, auto-unsuspend
        if exp_date > datetime.utcnow():
            restaurant.is_suspended = False
            
        db.commit()
        return JSONResponse(status_code=200, content={"success": True})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid date format: {str(e)}"})


@app.post("/superadmin/restaurants/{restaurant_id}/toggle-suspension")
async def super_restaurant_toggle_suspension(
    restaurant_id: int,
    req_body: schemas.SuspensionRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    restaurant = db.query(models.Restaurant).filter(models.Restaurant.id == restaurant_id).first()
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found"})
    
    restaurant.is_suspended = req_body.is_suspended
    db.commit()
    return JSONResponse(status_code=200, content={"success": True})


@app.post("/superadmin/restaurants/{restaurant_id}/adjust-quota")
async def super_restaurant_adjust_quota(
    restaurant_id: int,
    req_body: schemas.QuotaAdjustmentRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    restaurant = db.query(models.Restaurant).filter(models.Restaurant.id == restaurant_id).first()
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found"})
        
    restaurant.assigned_minutes = req_body.assigned_minutes
    restaurant.used_minutes = req_body.used_minutes
    
    # Auto suspend or unsuspend based on new limits
    if restaurant.used_minutes >= restaurant.assigned_minutes:
        restaurant.is_suspended = True
    else:
        restaurant.is_suspended = False
        
    db.commit()
    return JSONResponse(status_code=200, content={"success": True})



@app.get("/superadmin/billing", response_class=HTMLResponse)
async def super_billing(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    from sql.models import Restaurant
    restaurants = db.query(Restaurant).order_by(Restaurant.id.desc()).all()
    
    trial_quota = int(os.getenv("TRIAL_QUOTA", "100"))
    pro_quota = int(os.getenv("PRO_QUOTA", "1000"))
    
    trial_count = 0
    paid_count = 0
    mrr = 0
    
    for r in restaurants:
        r.computed_limit = r.assigned_minutes
        r.computed_used = round(r.used_minutes, 1)
        r.computed_percentage = int((r.used_minutes / max(1, r.assigned_minutes)) * 100)
        
        # Calculate expiry date with 5 days grace period
        if r.is_free_plan and r.trial_expiration_date:
            expiry_dt = r.trial_expiration_date + timedelta(days=5)
            r.computed_expiry = expiry_dt.strftime('%b %d, %Y')
        elif r.created_at:
            start_date = r.created_at
            now = datetime.now()
            
            months_passed = (now.year - start_date.year) * 12 + (now.month - start_date.month)
            if now.day < start_date.day:
                months_passed -= 1
            
            plan = r.subscription_plan
            if plan == "six_monthly":
                months_passed = (months_passed // 6) * 6
                months_to_add = 6
            elif plan == "annually":
                months_passed = (months_passed // 12) * 12
                months_to_add = 12
            else:
                months_to_add = 1
                
            total_months = months_passed + months_to_add
            next_year = start_date.year + (start_date.month - 1 + total_months) // 12
            next_month = (start_date.month - 1 + total_months) % 12 + 1
            
            import calendar
            _, last_day = calendar.monthrange(next_year, next_month)
            next_day = min(start_date.day, last_day)
            
            next_cycle = datetime(next_year, next_month, next_day, start_date.hour, start_date.minute)
            r.computed_expiry = (next_cycle + timedelta(days=5)).strftime('%b %d, %Y')
        else:
            r.computed_expiry = (datetime.now() + timedelta(days=35)).strftime('%b %d, %Y')
        
        if r.is_free_plan:
            trial_count += 1
            r.computed_status = "Trial Active" if not r.is_suspended else "Suspended (Expired)"
        else:
            if not r.is_suspended:
                paid_count += 1
                from dotenv import load_dotenv
                load_dotenv(override=True)
                monthly_price = float(os.getenv("MONTHLY_PRICE", "5000"))
                six_monthly_price = monthly_price * 6 * 0.95
                annually_price = monthly_price * 12 * 0.90
                
                if r.subscription_plan == "monthly":
                    mrr += monthly_price
                elif r.subscription_plan == "six_monthly":
                    mrr += (six_monthly_price / 6)
                elif r.subscription_plan == "annually":
                    mrr += (annually_price / 12)
            
            if r.subscription_plan == "monthly":
                r.computed_status = "Paid" if not r.is_suspended else "Suspended"
            elif r.subscription_plan == "six_monthly":
                r.computed_status = "Paid (6-Month)" if not r.is_suspended else "Suspended"
            elif r.subscription_plan == "annually":
                r.computed_status = "Paid (Annual)" if not r.is_suspended else "Suspended"
            else:
                r.computed_status = "Paid (Enterprise)" if not r.is_suspended else "Suspended"
                
    return templates.TemplateResponse(
        "superadmin/billing.html",
        {
            "request": request,
            "current_user": auth["user"],
            "restaurants": restaurants,
            "mrr": mrr,
            "paid_count": paid_count,
            "trial_count": trial_count,
            "monthly_price": float(os.getenv("MONTHLY_PRICE", "5000")),
            "six_monthly_price": float(os.getenv("MONTHLY_PRICE", "5000")) * 6 * 0.95,
            "annually_price": float(os.getenv("MONTHLY_PRICE", "5000")) * 12 * 0.90
        }
    )

@app.get("/superadmin/analytics", response_class=HTMLResponse)
async def super_analytics(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]

    from datetime import date as dt_date
    all_calls = db.query(models.ChatHistory).count()
    this_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month_start = (this_month_start - timedelta(days=1)).replace(day=1)

    calls_this_month = db.query(models.ChatHistory).filter(
        models.ChatHistory.timestamp >= this_month_start
    ).count()
    calls_last_month = db.query(models.ChatHistory).filter(
        models.ChatHistory.timestamp >= last_month_start,
        models.ChatHistory.timestamp < this_month_start
    ).count()

    if calls_last_month > 0:
        mom_growth = round(((calls_this_month - calls_last_month) / calls_last_month) * 100, 1)
    else:
        mom_growth = 0.0

    restaurants = db.query(models.Restaurant).all()
    this_month_onboards = sum(1 for r in restaurants if r.created_at and r.created_at >= this_month_start)
    last_month_onboards = sum(1 for r in restaurants if r.created_at and last_month_start <= r.created_at < this_month_start)

    suspended_count = sum(1 for r in restaurants if r.is_suspended)
    total_count = len(restaurants)
    churn_rate = round((suspended_count / total_count * 100), 1) if total_count > 0 else 0.0

    return templates.TemplateResponse("superadmin/analytics.html", {
        "request": request,
        "current_user": auth["user"],
        "all_calls": all_calls,
        "mom_growth": mom_growth,
        "calls_this_month": calls_this_month,
        "this_month_onboards": this_month_onboards,
        "last_month_onboards": last_month_onboards,
        "churn_rate": churn_rate,
        "suspended_count": suspended_count
    })

@app.get("/superadmin/accounts", response_class=HTMLResponse)
async def super_accounts(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    from collections import Counter
    users = db.query(models.User).order_by(models.User.id.desc()).all()
    restaurants = db.query(models.Restaurant).order_by(models.Restaurant.name.asc()).all()
    admin_counts = dict(Counter(u.restaurant_id for u in users if u.restaurant_id is not None and u.role == "admin"))
    
    return templates.TemplateResponse(
        "superadmin/accounts.html",
        {
            "request": request, 
            "users": users, 
            "restaurants": restaurants,
            "current_user": auth["user"],
            "admin_counts": admin_counts
        }
    )


@app.get("/superadmin/api/health")
async def super_api_health(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    suspended_count = db.query(models.Restaurant).filter(models.Restaurant.is_suspended == True).count()
    return JSONResponse(status_code=200, content={"suspended_count": suspended_count})


@app.post("/superadmin/accounts")
async def super_create_account(
    req_body: schemas.AdminAccountCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    # Check email uniqueness
    existing_user = db.query(models.User).filter(models.User.email == req_body.email).first()
    if existing_user:
        return JSONResponse(status_code=400, content={"error": "Email is already registered."})
        
    from sql.crud import generate_unique_username, generate_random_password
    from werkzeug.security import generate_password_hash
    
    username = generate_unique_username(db, req_body.name)
    plaintext_password = generate_random_password(12)
    hashed_password = generate_password_hash(plaintext_password)
    
    role_db = "super_admin" if req_body.role == "Super Admin" else "admin"
    restaurant_id = None if role_db == "super_admin" else req_body.restaurant_id
    
    new_user = models.User(
        name=req_body.name,
        email=req_body.email,
        phone_number=req_body.phone_number,
        username=username,
        password=hashed_password,
        role=role_db,
        must_change_password=True,
        restaurant_id=restaurant_id
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Send email notification in background
    login_url = str(request.base_url) + "login"
    role_display = "Super Admin" if role_db == "super_admin" else "Restaurant Admin"
    
    email_body = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333333; line-height: 1.6; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .header {{ background-color: #5d5fef; color: white; padding: 20px; text-align: center; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
        .content {{ padding: 20px; }}
        .button {{ display: inline-block; padding: 12px 24px; background-color: #5d5fef; color: white !important; text-decoration: none; border-radius: 4px; font-weight: bold; margin-top: 20px; }}
        .footer {{ font-size: 12px; color: #777777; margin-top: 20px; text-align: center; }}
        .credentials {{ background-color: #f7fafc; padding: 15px; border-radius: 6px; border: 1px solid #edf2f7; font-family: monospace; font-size: 14px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Welcome to OrderSaathi!</h2>
        </div>
        <div class="content">
            <p>Hello <strong>{new_user.name}</strong>,</p>
            <p>A new admin account has been created for you on the OrderSaathi platform.</p>
            
            <div class="credentials">
                <strong>Role:</strong> {role_display}<br>
                <strong>Username:</strong> {new_user.username}<br>
                <strong>Temporary Password:</strong> {plaintext_password}
            </div>
            
            <p>For security reasons, you will be required to change this temporary password on your first login.</p>
            
            <p>Click below to log in and access the dashboard:</p>
            <center>
                <a href="{login_url}" class="button">Log In to Dashboard</a>
            </center>
        </div>
        <div class="footer">
            <p>&copy; 2026 OrderSaathi. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""
    
    background_tasks.add_task(
        send_email,
        receiver_email=new_user.email,
        subject="Your OrderSaathi Admin Account has been Created",
        body=email_body,
        is_html=True
    )
    
    return JSONResponse(status_code=200, content={"success": True, "username": new_user.username, "password": plaintext_password})


@app.post("/superadmin/accounts/{user_id}/edit")
async def super_edit_account(
    user_id: int,
    req_body: schemas.AdminAccountUpdate,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found."})
        
    # Check if email is already taken by another user
    existing_user = db.query(models.User).filter(models.User.email == req_body.email, models.User.id != user_id).first()
    if existing_user:
        return JSONResponse(status_code=400, content={"error": "Email is already taken."})
        
    # Check if changing role from admin to super_admin or vice versa
    role_db = "super_admin" if req_body.role == "Super Admin" else "admin"
    restaurant_id = None if role_db == "super_admin" else req_body.restaurant_id
    
    # Save old email for notifications
    old_email = user.email
    
    user.name = req_body.name
    user.email = req_body.email
    user.phone_number = req_body.phone_number
    user.role = role_db
    user.restaurant_id = restaurant_id
    db.commit()
    
    # Send email notification in background
    login_url = str(request.base_url) + "login"
    role_display = "Super Admin" if role_db == "super_admin" else "Restaurant Admin"
    
    email_body = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333333; line-height: 1.6; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .header {{ background-color: #5d5fef; color: white; padding: 20px; text-align: center; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
        .content {{ padding: 20px; }}
        .button {{ display: inline-block; padding: 12px 24px; background-color: #5d5fef; color: white !important; text-decoration: none; border-radius: 4px; font-weight: bold; margin-top: 20px; }}
        .footer {{ font-size: 12px; color: #777777; margin-top: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Account Details Updated</h2>
        </div>
        <div class="content">
            <p>Hello <strong>{user.name}</strong>,</p>
            <p>Your OrderSaathi account details have been updated by an administrator.</p>
            
            <p><strong>Updated Details:</strong></p>
            <ul>
                <li><strong>Name:</strong> {user.name}</li>
                <li><strong>Email:</strong> {user.email}</li>
                <li><strong>Phone:</strong> {user.phone_number or 'N/A'}</li>
                <li><strong>Role:</strong> {role_display}</li>
            </ul>
            
            <center>
                <a href="{login_url}" class="button">Log In to Dashboard</a>
            </center>
        </div>
        <div class="footer">
            <p>&copy; 2026 OrderSaathi. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""
    
    # Send to both old and new email if they changed, otherwise just new email
    background_tasks.add_task(
        send_email,
        receiver_email=user.email,
        subject="Your OrderSaathi Account has been Updated",
        body=email_body,
        is_html=True
    )
    if old_email != user.email:
        background_tasks.add_task(
            send_email,
            receiver_email=old_email,
            subject="Your OrderSaathi Account Email has been Changed",
            body=email_body,
            is_html=True
        )
        
    return JSONResponse(status_code=200, content={"success": True})


@app.post("/superadmin/accounts/{user_id}/delete")
async def super_delete_account(
    user_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    current_user = auth["user"]
    if user_id == current_user.id:
        return JSONResponse(status_code=400, content={"error": "Super Admin cannot delete their own account."})
        
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found."})
        
    # Check if this user is a restaurant admin and the last admin for the restaurant
    if user.role == "admin" and user.restaurant_id is not None:
        admin_count = db.query(models.User).filter(models.User.restaurant_id == user.restaurant_id).count()
        if admin_count <= 1:
            return JSONResponse(
                status_code=400,
                content={"error": "Cannot delete the last admin account for this restaurant."}
            )
            
        # If the deleted user is the primary admin (admin_user_id) of the restaurant, assign to someone else
        restaurant = db.query(models.Restaurant).filter(models.Restaurant.id == user.restaurant_id).first()
        if restaurant and restaurant.admin_user_id == user.id:
            next_admin = db.query(models.User).filter(
                models.User.restaurant_id == restaurant.id,
                models.User.id != user.id
            ).first()
            if next_admin:
                restaurant.admin_user_id = next_admin.id
                db.flush()
                
    # Save fields for email notification before deleting
    deleted_user_email = user.email
    deleted_user_name = user.name
    
    db.delete(user)
    db.commit()
    
    # Send email notification in background
    email_body = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333333; line-height: 1.6; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .header {{ background-color: #dc2626; color: white; padding: 20px; text-align: center; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
        .content {{ padding: 20px; }}
        .footer {{ font-size: 12px; color: #777777; margin-top: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Account Deactivated</h2>
        </div>
        <div class="content">
            <p>Hello <strong>{deleted_user_name}</strong>,</p>
            <p>Your OrderSaathi administrator account has been deleted by a system administrator. You will no longer be able to log in to the dashboard.</p>
            <p>If you believe this was an error, please contact your platform support team.</p>
        </div>
        <div class="footer">
            <p>&copy; 2026 OrderSaathi. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""

    background_tasks.add_task(
        send_email,
        receiver_email=deleted_user_email,
        subject="Your OrderSaathi Account has been Deleted",
        body=email_body,
        is_html=True
    )
    
    return JSONResponse(status_code=200, content={"success": True})

@app.get("/superadmin/settings", response_class=HTMLResponse)
async def super_settings(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    
    from dotenv import dotenv_values
    env_vals = dotenv_values(".env")
    
    def mask_key(val: Optional[str]) -> str:
        if not val or val.startswith("replace-with-"):
            return ""
        if len(val) <= 8:
            return "••••••••"
        return f"{val[:4]}••••••••{val[-4:]}"
        
    masked_settings = {
        "groq_model": env_vals.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        "groq_api_key": mask_key(env_vals.get("GROQ_API_KEY")),
        "elevenlabs_api_key": mask_key(env_vals.get("ELEVENLABS_API_KEY")),
        "deepgram_api_key": mask_key(env_vals.get("DEEPGRAM_API_KEY")),
        "twilio_account_sid": env_vals.get("TWILIO_ACCOUNT_SID", ""),
        "twilio_auth_token": mask_key(env_vals.get("TWILIO_AUTH_TOKEN")),
        "default_order_phone_number": env_vals.get("DEFAULT_ORDER_PHONE_NUMBER", ""),
        "trial_quota": int(env_vals.get("TRIAL_QUOTA", "100")),
        "pro_quota": int(env_vals.get("PRO_QUOTA", "1000")),
        "payment_account_name": env_vals.get("PAYMENT_ACCOUNT_NAME", ""),
        "payment_account_number": env_vals.get("PAYMENT_ACCOUNT_NUMBER", ""),
        "monthly_price": float(env_vals.get("MONTHLY_PRICE", "5000")),
    }
    
    return templates.TemplateResponse(
        "superadmin/settings.html",
        {
            "request": request,
            "current_user": auth["user"],
            "settings": masked_settings
        }
    )


@app.post("/superadmin/settings")
async def super_update_settings(
    req_body: schemas.SystemSettingsUpdate,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        
    env_path = ".env"
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("")
            
    with open(env_path, "r") as f:
        lines = f.readlines()
        
    key_mapping = {
        "groq_model": "GROQ_MODEL",
        "groq_api_key": "GROQ_API_KEY",
        "elevenlabs_api_key": "ELEVENLABS_API_KEY",
        "deepgram_api_key": "DEEPGRAM_API_KEY",
        "twilio_account_sid": "TWILIO_ACCOUNT_SID",
        "twilio_auth_token": "TWILIO_AUTH_TOKEN",
        "default_order_phone_number": "DEFAULT_ORDER_PHONE_NUMBER",
        "trial_quota": "TRIAL_QUOTA",
        "pro_quota": "PRO_QUOTA",
        "payment_account_name": "PAYMENT_ACCOUNT_NAME",
        "payment_account_number": "PAYMENT_ACCOUNT_NUMBER",
        "monthly_price": "MONTHLY_PRICE"
    }
    
    new_values = {}
    from dotenv import dotenv_values
    current_env = dotenv_values(env_path)
    
    for req_key, env_key in key_mapping.items():
        val = getattr(req_body, req_key)
        if val is not None:
            str_val = str(val)
            if "••" in str_val:
                continue
            new_values[env_key] = str_val
            if env_key == "GROQ_API_KEY":
                new_values["LLM_EC2_KEY"] = str_val
            
    updated_keys = set()
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, v = stripped.split("=", 1)
            k = k.strip()
            if k in new_values:
                new_lines.append(f"{k}={new_values[k]}\n")
                updated_keys.add(k)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    for k, v in new_values.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}\n")
            
    with open(env_path, "w") as f:
        f.writelines(new_lines)
        
    for k, v in new_values.items():
        os.environ[k] = v
        
    load_dotenv(override=True)
    
    return JSONResponse(status_code=200, content={"success": True})



# ─────────────────────────────────────────────
# API — Call logs (powers v2 call-history UI)
# ─────────────────────────────────────────────

@app.get("/api/call-logs")
async def api_call_logs(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Return recent Twilio call logs for the v2 call-history page."""
    logs = crud.get_chat_histories(db, skip=skip, limit=limit)
    return [
        {
            "id": log.id,
            "session_id": log.session_id,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "response_time": log.response_time,
            "message_count": len(log.chat_data) if log.chat_data else 0,
        }
        for log in logs
    ]


@app.get("/api/call-logs/{log_id}")
async def api_call_log_detail(log_id: int, db: Session = Depends(get_db)):
    """Return full chat transcript for a single call log."""
    log = crud.get_chat_history(db, log_id)
    if not log:
        return JSONResponse(status_code=404, content={"error": "Call log not found."})
    return {
        "id": log.id,
        "session_id": log.session_id,
        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
        "response_time": log.response_time,
        "chat_data": log.chat_data,
    }


@app.post("/admin/settings/fulfillment")
async def update_fulfillment_settings(
    req: schemas.FulfillmentSettingsUpdate,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    restaurant = auth["user"].restaurant
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found."})
        
    restaurant.fulfillment_integration_type = req.integration_type
    restaurant.fulfillment_target_url = req.target_url
    
    # Auto-dispatch Pending orders if URL is provided
    dispatched_count = 0
    failed_count = 0
    if req.target_url:
        from sqlalchemy import or_
        pending_orders = db.query(models.Order).filter(
            models.Order.restaurant_id == restaurant.id,
            or_(
                models.Order.status == "Pending",
                models.Order.dispatched_to_external_system == False
            )
        ).all()
        
        if pending_orders:
            from integrations import get_adapter
            adapter = get_adapter(req.integration_type)
            for order in pending_orders:
                try:
                    res = adapter.dispatch(order, restaurant)
                    order.dispatch_attempts = (order.dispatch_attempts or 0) + 1
                    if res.success:
                        order.dispatched_at = datetime.utcnow()
                        order.dispatched_to_external_system = True
                        order.status = "Confirmed"
                        dispatched_count += 1
                        db.add(models.Notification(
                            restaurant_id=restaurant.id,
                            type="dispatch_success",
                            message=f"Order #{order.id} successfully dispatched to your system.",
                            related_order_id=order.id,
                            is_read=False
                        ))
                    else:
                        order.last_dispatch_error = res.message
                        failed_count += 1
                        db.add(models.Notification(
                            restaurant_id=restaurant.id,
                            type="failed_dispatch",
                            message=f"Order #{order.id} failed to dispatch: {res.message}",
                            related_order_id=order.id,
                            is_read=False
                        ))
                except Exception as e:
                    order.dispatch_attempts = (order.dispatch_attempts or 0) + 1
                    order.last_dispatch_error = str(e)
                    failed_count += 1
                    db.add(models.Notification(
                        restaurant_id=restaurant.id,
                        type="failed_dispatch",
                        message=f"Order #{order.id} encountered an error during dispatch.",
                        related_order_id=order.id,
                        is_read=False
                    ))
                    
    db.commit()
    
    return JSONResponse(content={
        "success": True,
        "dispatched_count": dispatched_count,
        "failed_count": failed_count
    })


# ─────────────────────────────────────────────
# Plan Change Request (Admin → SuperAdmin workflow)
# ─────────────────────────────────────────────

@app.post("/admin/billing/request-plan-change")
async def admin_request_plan_change(
    req_body: schemas.PlanChangeRequestCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    user = auth["user"]
    restaurant = db.query(models.Restaurant).filter(models.Restaurant.id == user.restaurant_id).first()
    if not restaurant:
        return JSONResponse(status_code=404, content={"error": "Restaurant not found"})

    # Prevent downgrade to free plan
    if req_body.requested_plan in ("free_trial",):
        return JSONResponse(status_code=400, content={"error": "Cannot request a downgrade to Free Trial."})

    # Prevent same plan
    if not restaurant.is_free_plan and restaurant.subscription_plan == req_body.requested_plan:
        return JSONResponse(status_code=400, content={"error": "You are already on this plan."})

    # Cancel any existing pending requests for this restaurant
    db.query(models.PlanChangeRequest).filter(
        models.PlanChangeRequest.restaurant_id == restaurant.id,
        models.PlanChangeRequest.status == "pending"
    ).delete(synchronize_session=False)

    current_plan = "free_trial" if restaurant.is_free_plan else restaurant.subscription_plan

    pcr = models.PlanChangeRequest(
        restaurant_id=restaurant.id,
        requested_plan=req_body.requested_plan,
        current_plan=current_plan,
        admin_note=req_body.admin_note,
        status="pending"
    )
    db.add(pcr)
    db.flush()

    # Create superadmin notification (use restaurant_id = 0 to target superadmin — stored as -1 sentinel)
    # We'll store them under restaurant_id=None-workaround: use a special notification type
    # Actually: create a Notification for superadmin by querying the superadmin user's restaurant setup
    # We'll use a special restaurant_id = 0 (will be filtered server-side for superadmin)
    # Better approach: create notification linked to the requesting restaurant with type "plan_change_request"
    # SuperAdmin will query all plan_change_requests with status=pending

    plan_labels = {"monthly": "Monthly", "six_monthly": "6-Month", "annually": "Annual"}
    db.add(models.Notification(
        restaurant_id=restaurant.id,
        type="plan_change_request_sent",
        message=f"Your plan change request to {plan_labels.get(req_body.requested_plan, req_body.requested_plan)} has been submitted and is pending approval.",
        is_read=False
    ))
    db.commit()

    # Email superadmin
    superadmin = db.query(models.User).filter(models.User.role == "super_admin").first()
    superadmin_email = superadmin.email if superadmin else None
    if superadmin_email:
        requested_label = plan_labels.get(req_body.requested_plan, req_body.requested_plan)
        current_label = plan_labels.get(current_plan, current_plan)
        note_html = f"<p><strong>Admin Note:</strong> {req_body.admin_note}</p>" if req_body.admin_note else ""
        email_body = f"""
<html><body style="font-family:Inter,sans-serif;color:#111827;">
<div style="max-width:560px;margin:0 auto;padding:32px 24px;">
  <h2 style="color:#4f46e5;">New Plan Change Request</h2>
  <p>A restaurant admin has submitted a subscription plan change request on <strong>OrderSaathi</strong>.</p>
  <table style="width:100%;border-collapse:collapse;margin:20px 0;">
    <tr><td style="padding:8px;background:#f3f4f6;font-weight:600;width:40%;">Restaurant</td><td style="padding:8px;">{restaurant.name}</td></tr>
    <tr><td style="padding:8px;background:#f3f4f6;font-weight:600;">Current Plan</td><td style="padding:8px;">{current_label}</td></tr>
    <tr><td style="padding:8px;background:#f3f4f6;font-weight:600;">Requested Plan</td><td style="padding:8px;color:#4f46e5;"><strong>{requested_label}</strong></td></tr>
  </table>
  {note_html}
  <a href="{str(request.base_url).rstrip('/')}/superadmin/plan-requests" style="display:inline-block;padding:12px 24px;background:#4f46e5;color:white;border-radius:6px;text-decoration:none;font-weight:600;margin-top:8px;">Review Request</a>
  <p style="margin-top:24px;font-size:12px;color:#6b7280;">OrderSaathi Super Admin Portal</p>
</div></body></html>
"""
        from send_email import send_email as _send_email
        import threading
        threading.Thread(target=_send_email, args=(superadmin_email, f"[OrderSaathi] Plan Change Request — {restaurant.name}", email_body, True), daemon=True).start()

    return JSONResponse(status_code=200, content={"success": True})


@app.get("/superadmin/plan-requests", response_class=HTMLResponse)
async def super_plan_requests(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]

    requests_list = db.query(models.PlanChangeRequest).order_by(
        models.PlanChangeRequest.created_at.desc()
    ).all()

    pending_count = sum(1 for r in requests_list if r.status == "pending")

    return templates.TemplateResponse("superadmin/plan-requests.html", {
        "request": request,
        "current_user": auth["user"],
        "plan_requests": requests_list,
        "pending_count": pending_count
    })


@app.post("/superadmin/plan-requests/{req_id}/review")
async def super_review_plan_request(
    req_id: int,
    req_body: schemas.PlanChangeRequestReview,
    request: Request,
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    pcr = db.query(models.PlanChangeRequest).filter(models.PlanChangeRequest.id == req_id).first()
    if not pcr:
        return JSONResponse(status_code=404, content={"error": "Request not found"})

    if pcr.status != "pending":
        return JSONResponse(status_code=400, content={"error": "Request already resolved"})

    from datetime import datetime
    pcr.status = req_body.status
    pcr.superadmin_note = req_body.superadmin_note
    pcr.resolved_at = datetime.utcnow()

    restaurant = db.query(models.Restaurant).filter(models.Restaurant.id == pcr.restaurant_id).first()
    plan_labels = {"monthly": "Monthly", "six_monthly": "6-Month", "annually": "Annual"}

    admin_user = db.query(models.User).filter(models.User.restaurant_id == pcr.restaurant_id, models.User.role == "admin").first()
    admin_email = admin_user.email if admin_user and hasattr(admin_user, 'email') else None

    if req_body.status == "approved" and restaurant:
        restaurant.subscription_plan = pcr.requested_plan
        restaurant.is_free_plan = False
        restaurant.trial_expiration_date = None
        if restaurant.is_suspended:
            restaurant.is_suspended = False

        msg = f"Your plan change request to {plan_labels.get(pcr.requested_plan, pcr.requested_plan)} Plan has been APPROVED."
        if req_body.superadmin_note:
            msg += f" Note: {req_body.superadmin_note}"
        db.add(models.Notification(
            restaurant_id=restaurant.id,
            type="plan_change_approved",
            message=msg,
            is_read=False
        ))
        # Email admin
        if admin_email:
            note_html = f"<p><strong>Note from Super Admin:</strong> {req_body.superadmin_note}</p>" if req_body.superadmin_note else ""
            email_body = f"""
<html><body style="font-family:Inter,sans-serif;color:#111827;">
<div style="max-width:560px;margin:0 auto;padding:32px 24px;">
  <h2 style="color:#16a34a;">&#10003; Plan Change Approved</h2>
  <p>Great news! Your subscription plan change request has been <strong>approved</strong> by the OrderSaathi team.</p>
  <table style="width:100%;border-collapse:collapse;margin:20px 0;">
    <tr><td style="padding:8px;background:#f3f4f6;font-weight:600;width:40%;">Restaurant</td><td style="padding:8px;">{restaurant.name}</td></tr>
    <tr><td style="padding:8px;background:#f3f4f6;font-weight:600;">New Plan</td><td style="padding:8px;color:#16a34a;"><strong>{plan_labels.get(pcr.requested_plan, pcr.requested_plan)}</strong></td></tr>
  </table>
  {note_html}
  <a href="{str(request.base_url).rstrip('/')}/admin/billing" style="display:inline-block;padding:12px 24px;background:#16a34a;color:white;border-radius:6px;text-decoration:none;font-weight:600;margin-top:8px;">View Billing</a>
  <p style="margin-top:24px;font-size:12px;color:#6b7280;">OrderSaathi Admin Portal</p>
</div></body></html>
"""
            import threading
            from send_email import send_email as _send_email
            threading.Thread(target=_send_email, args=(admin_email, "[OrderSaathi] Your Plan Change Request Has Been Approved", email_body, True), daemon=True).start()

    elif req_body.status == "rejected" and restaurant:
        msg = f"Your plan change request to {plan_labels.get(pcr.requested_plan, pcr.requested_plan)} Plan has been REJECTED."
        if req_body.superadmin_note:
            msg += f" Reason: {req_body.superadmin_note}"
        db.add(models.Notification(
            restaurant_id=restaurant.id,
            type="plan_change_rejected",
            message=msg,
            is_read=False
        ))
        # Email admin
        if admin_email:
            note_html = f"<p><strong>Reason:</strong> {req_body.superadmin_note}</p>" if req_body.superadmin_note else ""
            email_body = f"""
<html><body style="font-family:Inter,sans-serif;color:#111827;">
<div style="max-width:560px;margin:0 auto;padding:32px 24px;">
  <h2 style="color:#dc2626;">Plan Change Request Rejected</h2>
  <p>Unfortunately your subscription plan change request for <strong>{restaurant.name}</strong> has been <strong>rejected</strong>.</p>
  <table style="width:100%;border-collapse:collapse;margin:20px 0;">
    <tr><td style="padding:8px;background:#f3f4f6;font-weight:600;width:40%;">Requested Plan</td><td style="padding:8px;color:#dc2626;">{plan_labels.get(pcr.requested_plan, pcr.requested_plan)}</td></tr>
  </table>
  {note_html}
  <p>Please contact the OrderSaathi team if you have any questions or would like to discuss further.</p>
  <p style="margin-top:24px;font-size:12px;color:#6b7280;">OrderSaathi Admin Portal</p>
</div></body></html>
"""
            import threading
            from send_email import send_email as _send_email
            threading.Thread(target=_send_email, args=(admin_email, "[OrderSaathi] Your Plan Change Request Has Been Rejected", email_body, True), daemon=True).start()

    db.commit()
    return JSONResponse(status_code=200, content={"success": True})


@app.get("/superadmin/api/plan-requests/pending-count")
async def super_plan_requests_pending_count(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    count = db.query(models.PlanChangeRequest).filter(models.PlanChangeRequest.status == "pending").count()
    return JSONResponse(status_code=200, content={"pending_count": count})


@app.get("/superadmin/api/notifications")
async def super_notifications_api(request: Request, db: Session = Depends(get_db)):
    """Returns pending plan change requests for superadmin notification bell."""
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    pending = db.query(models.PlanChangeRequest).filter(
        models.PlanChangeRequest.status == "pending"
    ).order_by(models.PlanChangeRequest.created_at.desc()).limit(20).all()

    def rel_time(dt):
        if not dt:
            return ""
        diff = (datetime.utcnow() - dt).total_seconds()
        if diff < 60:    return "Just now"
        if diff < 3600:  return f"{int(diff/60)}m ago"
        if diff < 86400: return f"{int(diff/3600)}h ago"
        return f"{int(diff/86400)}d ago"

    results = []
    for p in pending:
        rest = db.query(models.Restaurant).filter(models.Restaurant.id == p.restaurant_id).first()
        results.append({
            "id": p.id,
            "restaurant_name": rest.name if rest else f"Restaurant #{p.restaurant_id}",
            "current_plan": p.current_plan or "",
            "requested_plan": p.requested_plan or "",
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "time_str": rel_time(p.created_at)
        })

    return JSONResponse(status_code=200, content={
        "pending_count": len(results),
        "pending_requests": results
    })


# ─────────────────────────────────────────────
# Superadmin Analytics API (real data)
# ─────────────────────────────────────────────

@app.get("/superadmin/api/analytics")
async def super_analytics_api(
    request: Request,
    days: int = Query(7),
    db: Session = Depends(get_db)
):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    from datetime import date as dt_date
    days_list = [(datetime.now() - timedelta(days=i)).date() for i in range(days - 1, -1, -1)]

    calls = db.query(models.ChatHistory).all()
    orders = db.query(models.Order).all()

    call_volume = [0] * days
    order_volume = [0] * days

    for c in calls:
        if c.timestamp:
            d = c.timestamp.date()
            if d in days_list:
                call_volume[days_list.index(d)] += 1

    for o in orders:
        if o.created_at:
            d = o.created_at.date()
            if d in days_list:
                order_volume[days_list.index(d)] += 1

    fmt = "%b %d" if days > 7 else "%a"
    labels = [d.strftime(fmt) for d in days_list]

    # Monthly breakdown for bar chart (last 6 months)
    monthly_labels = []
    monthly_calls = []
    for i in range(5, -1, -1):
        ref = datetime.now() - timedelta(days=30 * i)
        m_label = ref.strftime("%b")
        m_start = ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if i > 0:
            m_end = (datetime.now() - timedelta(days=30 * (i - 1))).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            m_end = datetime.now()
        count = sum(1 for c in calls if c.timestamp and m_start <= c.timestamp.replace(tzinfo=None) < m_end)
        monthly_labels.append(m_label)
        monthly_calls.append(count)

    # Restaurant leaderboard
    restaurants = db.query(models.Restaurant).all()
    leaderboard = []
    for r in restaurants:
        r_calls = [c for c in calls if c.restaurant_id == r.id]
        r_orders = [o for o in orders if o.restaurant_id == r.id]
        r_call_count = len(r_calls)
        r_order_count = len(r_orders)
        avg_dur = 0
        valid_durs = [c.duration_seconds for c in r_calls if c.duration_seconds]
        if valid_durs:
            avg_dur = int(sum(valid_durs) / len(valid_durs))
        conv_rate = round((r_order_count / r_call_count * 100), 1) if r_call_count > 0 else 0
        leaderboard.append({
            "name": r.name,
            "total_calls": r_call_count,
            "total_orders": r_order_count,
            "avg_duration": f"{avg_dur // 60}m {avg_dur % 60}s" if avg_dur else "-",
            "conversion_rate": conv_rate
        })
    leaderboard.sort(key=lambda x: x["total_calls"], reverse=True)

    # Onboarding trend (restaurants created per month over last 6 months)
    onboarding_data = []
    for i in range(5, -1, -1):
        ref = datetime.now() - timedelta(days=30 * i)
        m_start = ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if i > 0:
            m_end = (datetime.now() - timedelta(days=30 * (i - 1))).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            m_end = datetime.now()
        count = sum(1 for r in restaurants if r.created_at and m_start <= r.created_at.replace(tzinfo=None) < m_end)
        onboarding_data.append(count)

    total_calls_all = len(calls)
    return JSONResponse(status_code=200, content={
        "labels": labels,
        "call_volume": call_volume,
        "order_volume": order_volume,
        "monthly_labels": monthly_labels,
        "monthly_calls": monthly_calls,
        "leaderboard": leaderboard[:10],
        "onboarding_data": onboarding_data,
        "total_calls": total_calls_all
    })


# ─────────────────────────────────────────────
# Superadmin Dashboard Chart API (real data)
# ─────────────────────────────────────────────

@app.get("/superadmin/api/dashboard-chart")
async def super_dashboard_chart(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    days_list = [(datetime.now() - timedelta(days=i)).date() for i in range(6, -1, -1)]
    calls = db.query(models.ChatHistory).all()

    call_volume = [0] * 7
    for c in calls:
        if c.timestamp:
            d = c.timestamp.date()
            if d in days_list:
                call_volume[days_list.index(d)] += 1

    labels = [d.strftime("%a") for d in days_list]
    return JSONResponse(status_code=200, content={"labels": labels, "call_volume": call_volume})


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile="keys/key.pem",
        ssl_certfile="keys/cert.pem",
    )
