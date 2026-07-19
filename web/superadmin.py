"""Platform (superadmin) dashboard: tenants, accounts, billing, analytics."""

import os
import re
from datetime import datetime, timedelta
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from sql import crud, models, schemas
from services.menu_sync import RAG_UPSTREAM, sync_postgres_menu_to_chroma
from utils.cookies import validate_superadmin_cookies, validate_cookies
from utils.logger import print_info, print_error
from utils.mailer import send_email
from web.core import templates, get_db

router = APIRouter()

@router.get("/superadmin/dashboard", response_class=HTMLResponse)
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

@router.get("/superadmin/restaurants", response_class=HTMLResponse)
async def super_restaurants(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
    from sql.models import Restaurant
    restaurants = db.query(Restaurant).order_by(Restaurant.id.desc()).all()
    return templates.TemplateResponse("superadmin/restaurants.html", {"request": request, "restaurants": restaurants, "current_user": auth["user"]})

@router.get("/superadmin/restaurant-detail", response_class=HTMLResponse)
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

@router.get("/superadmin/create-restaurant", response_class=HTMLResponse)
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

@router.post("/superadmin/create-restaurant")
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

@router.post("/superadmin/restaurants/{restaurant_id}/reset-password")
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


@router.post("/superadmin/restaurants/{restaurant_id}/change-plan")
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


@router.post("/superadmin/restaurants/{restaurant_id}/extend-trial")
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


@router.post("/superadmin/restaurants/{restaurant_id}/toggle-suspension")
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


@router.post("/superadmin/restaurants/{restaurant_id}/adjust-quota")
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



@router.get("/superadmin/billing", response_class=HTMLResponse)
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

@router.get("/superadmin/analytics", response_class=HTMLResponse)
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

@router.get("/superadmin/accounts", response_class=HTMLResponse)
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


@router.get("/superadmin/api/health")
async def super_api_health(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    suspended_count = db.query(models.Restaurant).filter(models.Restaurant.is_suspended == True).count()
    return JSONResponse(status_code=200, content={"suspended_count": suspended_count})


@router.post("/superadmin/accounts")
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


@router.post("/superadmin/accounts/{user_id}/edit")
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


@router.post("/superadmin/accounts/{user_id}/delete")
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

@router.get("/superadmin/settings", response_class=HTMLResponse)
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


@router.post("/superadmin/settings")
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



@router.get("/superadmin/plan-requests", response_class=HTMLResponse)
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


@router.post("/superadmin/plan-requests/{req_id}/review")
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
            from utils.mailer import send_email as _send_email
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
            from utils.mailer import send_email as _send_email
            threading.Thread(target=_send_email, args=(admin_email, "[OrderSaathi] Your Plan Change Request Has Been Rejected", email_body, True), daemon=True).start()

    db.commit()
    return JSONResponse(status_code=200, content={"success": True})


@router.get("/superadmin/api/plan-requests/pending-count")
async def super_plan_requests_pending_count(request: Request, db: Session = Depends(get_db)):
    auth = validate_superadmin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    count = db.query(models.PlanChangeRequest).filter(models.PlanChangeRequest.status == "pending").count()
    return JSONResponse(status_code=200, content={"pending_count": count})


@router.get("/superadmin/api/notifications")
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

@router.get("/superadmin/api/analytics")
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

@router.get("/superadmin/api/dashboard-chart")
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

