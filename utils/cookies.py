from fastapi import Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sql import crud, models
from utils.logger import print_error
from utils.authentication import verify_access_token
from datetime import datetime


def get_cookies_from_request(request: Request) -> dict:
    cookie_header = request.headers.get("cookie")
    if not cookie_header:
        return {}
    
    cookies = {}
    for cookie in cookie_header.split(";"):
        parts = cookie.strip().split("=", 1)
        if len(parts) == 2:
            cookies[parts[0]] = parts[1]
    return cookies


def validate_admin_cookies(db: Session, request: Request) -> dict:
    cookies = get_cookies_from_request(request)
    access_token = cookies.get("access_cookie")
    
    if not access_token:
        print_error("No access token cookie found.")
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    username = verify_access_token(access_token)
    if not username:
        print_error("Invalid or expired access token.")
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    user = crud.get_user(db, username)
    if not user:
        print_error("User not found.")
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    if user.role != "admin":
        print_error(f"Unauthorized access: User role '{user.role}' is not admin.")
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    if user.must_change_password:
        print_error(f"User '{user.username}' must change password before accessing dashboard routes.")
        return {
            "success": False,
            "response": RedirectResponse(url="/change-password", status_code=302),
        }
        
    # Check suspension or expired free trial
    restaurant = db.query(models.Restaurant).filter(models.Restaurant.admin_user_id == user.id).first()
    if restaurant:
        is_expired_trial = (
            restaurant.is_free_plan and 
            restaurant.trial_expiration_date is not None and 
            restaurant.trial_expiration_date < datetime.utcnow()
        )
        if restaurant.is_suspended or is_expired_trial:
            if is_expired_trial and not restaurant.is_suspended:
                restaurant.is_suspended = True
                db.commit()
            print_error(f"Unauthorized access: Restaurant '{restaurant.name}' is suspended or trial expired.")
            response = RedirectResponse(url="/login?error=suspended", status_code=302)
            response.delete_cookie("access_cookie")
            response.delete_cookie("current_username")
            return {
                "success": False,
                "response": response,
            }
        
    return {"success": True, "user": user, "response": None}


def validate_superadmin_cookies(db: Session, request: Request) -> dict:
    cookies = get_cookies_from_request(request)
    access_token = cookies.get("access_cookie")
    
    if not access_token:
        print_error("No access token cookie found.")
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    username = verify_access_token(access_token)
    if not username:
        print_error("Invalid or expired access token.")
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    user = crud.get_user(db, username)
    if not user:
        print_error("User not found.")
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    if user.role != "super_admin":
        print_error(f"Unauthorized access: User role '{user.role}' is not super_admin.")
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    if user.must_change_password:
        print_error(f"User '{user.username}' must change password before accessing dashboard routes.")
        return {
            "success": False,
            "response": RedirectResponse(url="/change-password", status_code=302),
        }
        
    return {"success": True, "user": user, "response": None}


def validate_cookies(
    db: Session, request: Request, keys: list = ["current_username"], allow_must_change_password: bool = False
) -> dict:
    cookies = get_cookies_from_request(request)
    access_token = cookies.get("access_cookie")
    
    if not access_token:
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    username = verify_access_token(access_token)
    if not username:
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    user = crud.get_user(db, username)
    if not user:
        return {
            "success": False,
            "response": RedirectResponse(url="/login", status_code=302),
        }
        
    if user.must_change_password and not allow_must_change_password:
        return {
            "success": False,
            "response": RedirectResponse(url="/change-password", status_code=302),
        }
        
    # Check suspension or expired free trial for admin role
    if user.role == "admin":
        restaurant = db.query(models.Restaurant).filter(models.Restaurant.admin_user_id == user.id).first()
        if restaurant:
            is_expired_trial = (
                restaurant.is_free_plan and 
                restaurant.trial_expiration_date is not None and 
                restaurant.trial_expiration_date < datetime.utcnow()
            )
            if restaurant.is_suspended or is_expired_trial:
                if is_expired_trial and not restaurant.is_suspended:
                    restaurant.is_suspended = True
                    db.commit()
                print_error(f"Unauthorized access: Restaurant '{restaurant.name}' is suspended or trial expired.")
                response = RedirectResponse(url="/login?error=suspended", status_code=302)
                response.delete_cookie("access_cookie")
                response.delete_cookie("current_username")
                return {
                    "success": False,
                    "response": response,
                }
        
    ret_cookies = {}
    for key in keys:
        if key in cookies:
            ret_cookies[key] = cookies[key].strip('"')
            
    return {"success": True, "cookies": ret_cookies, "user": user}
