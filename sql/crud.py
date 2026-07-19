import os
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from . import models, schemas
from werkzeug.security import check_password_hash, generate_password_hash


# ── User CRUD ──
# Temporary – kept for utils/cookies.py + /login until platform_users replaces this.

def create_user(db: Session, user: schemas.UserCreate):
    new_user = models.User(
        name=user.name,
        email=user.email,
        phone_number=user.phone_number,
        username=user.username,
        password=generate_password_hash(user.password),
        role=user.role if user.role is not None else "admin",
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


def get_user(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def update_user(db: Session, username: str, user: schemas.UserUpdate):
    db_user = db.query(models.User).filter(models.User.username == username).first()
    if db_user:
        if user.name is not None:
            db_user.name = user.name
        if user.email is not None:
            db_user.email = user.email
        if user.phone_number is not None:
            db_user.phone_number = user.phone_number
        if user.username is not None:
            db_user.username = user.username
        if user.password is not None:
            db_user.password = generate_password_hash(user.password)
        if user.role is not None:
            db_user.role = user.role
        db.commit()
        db.refresh(db_user)
        return db_user
    return None


def delete_user(db: Session, username: str):
    user = db.query(models.User).filter(models.User.username == username).first()
    if user:
        db.delete(user)
        db.commit()
        return user
    return None


def authenticate_user(db: Session, username: str, password: str):
    try:
        user = db.query(models.User).filter(models.User.username == username).one()
        if user and check_password_hash(user.password, password):
            return user
    except NoResultFound:
        return None
    return None


# ── Call Log CRUD ──
# Used by Twilio recording (twilio_routes.py) and surfaced in the v2 call-history UI.

def create_chat_history(db: Session, chat_history: schemas.ChatHistoryCreate):
    new_log = models.ChatHistory(
        session_id=chat_history.session_id,
        chat_data=chat_history.chat_data,
        response_time=chat_history.response_time,
        restaurant_id=chat_history.restaurant_id,
        caller_number=chat_history.caller_number,
        duration_seconds=chat_history.duration_seconds,
        status=chat_history.status,
        recording_url=chat_history.recording_url,
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    return new_log


def get_chat_history(db: Session, chat_history_id: int):
    return (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.id == chat_history_id)
        .first()
    )


def get_chat_histories(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.ChatHistory).offset(skip).limit(limit).all()


def delete_chat_history(db: Session, chat_history_id: int):
    log = (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.id == chat_history_id)
        .first()
    )
    if log:
        db.delete(log)
        db.commit()
        return log
    return None


def generate_unique_username(db: Session, admin_name: str) -> str:
    parts = [p.strip() for p in admin_name.split() if p.strip()]
    if not parts:
        base = "admin"
    elif len(parts) == 1:
        base = parts[0].lower()
    else:
        # First initial + last name
        first_initial = parts[0][0].lower()
        last_name = parts[-1].lower()
        base = f"{first_initial}{last_name}"
    
    # Remove non-alphanumeric characters
    base = "".join(c for c in base if c.isalnum())
    if not base:
        base = "admin"
        
    username = base
    counter = 2
    while db.query(models.User).filter(models.User.username == username).first() is not None:
        username = f"{base}{counter}"
        counter += 1
    return username


def generate_random_password(length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def create_restaurant_with_admin(db: Session, restaurant_in: schemas.RestaurantCreate) -> tuple[models.Restaurant, models.User, str]:
    username = generate_unique_username(db, restaurant_in.admin_name)
    plaintext_password = generate_random_password(12)
    hashed_password = generate_password_hash(plaintext_password)
    
    try:
        new_user = models.User(
            name=restaurant_in.admin_name,
            email=restaurant_in.email,
            phone_number=restaurant_in.mobile_number,
            username=username,
            password=hashed_password,
            role="admin",
            must_change_password=True
        )
        db.add(new_user)
        db.flush()
        
        trial_quota = int(os.getenv("TRIAL_QUOTA", "100"))
        pro_quota = int(os.getenv("PRO_QUOTA", "1000"))
        assigned = trial_quota if restaurant_in.is_free_plan else pro_quota

        new_restaurant = models.Restaurant(
            name=restaurant_in.name,
            description=restaurant_in.description,
            subscription_plan="free_trial" if restaurant_in.is_free_plan else restaurant_in.subscription_plan,
            mobile_number=restaurant_in.mobile_number,
            order_phone_number=restaurant_in.order_phone_number,
            admin_user_id=new_user.id,
            is_suspended=restaurant_in.is_suspended if restaurant_in.is_suspended is not None else False,
            is_free_plan=restaurant_in.is_free_plan if restaurant_in.is_free_plan is not None else False,
            trial_expiration_date=(datetime.utcnow() + timedelta(days=14)) if restaurant_in.is_free_plan else None,
            assigned_minutes=assigned,
            used_minutes=0.0
        )
        db.add(new_restaurant)
        db.flush()
        
        # Link user to the newly created restaurant
        new_user.restaurant_id = new_restaurant.id
        
        db.commit()
        db.refresh(new_restaurant)
        db.refresh(new_user)
        return new_restaurant, new_user, plaintext_password
    except Exception as e:
        db.rollback()
        raise e


def create_order(db: Session, restaurant_id: int, customer_phone: str, items_summary: list, total_price: float, call_id: Optional[int] = None):
    # 1. Fetch restaurant
    restaurant = db.query(models.Restaurant).filter(models.Restaurant.id == restaurant_id).first()
    if not restaurant:
        raise ValueError("Restaurant not found")
        
    # Check the restaurant's fulfillment_target_url
    if not restaurant.fulfillment_target_url:
        status = "Pending"
    else:
        status = "Confirmed"

    new_order = models.Order(
        restaurant_id=restaurant_id,
        call_id=call_id,
        customer_phone=customer_phone,
        items_summary=items_summary,
        total_price=total_price,
        status=status,
    )
    db.add(new_order)
    db.flush()  # Get the ID of the new order
    
    if not restaurant.fulfillment_target_url:
        # Create a notification row
        new_notif = models.Notification(
            restaurant_id=restaurant_id,
            type="missing_fulfillment_url",
            message=f"New order #{new_order.id} booked, but no order-system URL is configured. Set one in Settings to enable automatic sending.",
            related_order_id=new_order.id,
            is_read=False
        )
        db.add(new_notif)
    else:
        # Attempt auto-dispatch immediately
        from integrations import get_adapter
        adapter = get_adapter(restaurant.fulfillment_integration_type)
        try:
            res = adapter.dispatch(new_order, restaurant)
            new_order.dispatch_attempts = 1
            if res.success:
                new_order.dispatched_at = datetime.utcnow()
                new_order.dispatched_to_external_system = True
                db.add(models.Notification(
                    restaurant_id=restaurant_id,
                    type="dispatch_success",
                    message=f"Order #{new_order.id} successfully dispatched to your system.",
                    related_order_id=new_order.id,
                    is_read=False
                ))
            else:
                new_order.last_dispatch_error = res.message
                db.add(models.Notification(
                    restaurant_id=restaurant_id,
                    type="failed_dispatch",
                    message=f"Order #{new_order.id} failed to dispatch: {res.message}",
                    related_order_id=new_order.id,
                    is_read=False
                ))
        except Exception as e:
            new_order.dispatch_attempts = 1
            new_order.last_dispatch_error = str(e)
            db.add(models.Notification(
                restaurant_id=restaurant_id,
                type="failed_dispatch",
                message=f"Order #{new_order.id} encountered an error during dispatch.",
                related_order_id=new_order.id,
                is_read=False
            ))
            
    db.commit()
    db.refresh(new_order)
    return new_order
