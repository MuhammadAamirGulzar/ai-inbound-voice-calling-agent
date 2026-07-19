"""Login, logout and password-change routes."""

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from sql import crud, models, schemas
from utils.authentication import create_access_token
from utils.cookies import validate_admin_cookies, validate_superadmin_cookies, validate_cookies
from utils.logger import print_info, print_error
from web.core import templates, get_db

router = APIRouter()

# ─────────────────────────────────────────────
# Auth (temporary — kept until platform_users login is built)
# ─────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
@router.get("/login", response_class=HTMLResponse)
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


@router.post("/login")
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


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="access_cookie")
    response.delete_cookie(key="current_username")
    return response


@router.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request, db: Session = Depends(get_db)):
    auth = validate_cookies(db, request, allow_must_change_password=True)
    if not auth["success"]:
        return auth["response"]
    return templates.TemplateResponse("change_password.html", {"request": request, "username": auth["user"].username})


@router.post("/change-password")
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

