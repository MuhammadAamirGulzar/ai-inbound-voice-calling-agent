import os
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base


# ── Temporary: User kept for utils/cookies.py until platform_users table is built ──
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone_number = Column(String(20), nullable=True)
    username = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="admin")
    must_change_password = Column(Boolean, nullable=False, default=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    restaurant = relationship("Restaurant", foreign_keys=[restaurant_id], back_populates="users")


class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String(500), nullable=True)
    subscription_plan = Column(String(50), nullable=False)  # "monthly", "six_monthly", "annually", "free_trial"
    mobile_number = Column(String(20), nullable=False)
    order_phone_number = Column(String(20), nullable=False, default=lambda: os.getenv("DEFAULT_ORDER_PHONE_NUMBER", ""))
    admin_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_suspended = Column(Boolean, nullable=False, default=False)
    is_free_plan = Column(Boolean, nullable=False, default=False)
    trial_expiration_date = Column(DateTime, nullable=True)
    assigned_minutes = Column(Integer, nullable=False, default=100)
    used_minutes = Column(Float, nullable=False, default=0.0)
    fulfillment_integration_type = Column(String(50), nullable=False, default="manual_redirect")
    fulfillment_target_url = Column(String(500), nullable=True)
    fulfillment_config = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    admin = relationship("User", foreign_keys=[admin_user_id])
    users = relationship("User", foreign_keys=[User.restaurant_id], back_populates="restaurant")
    orders = relationship("Order", back_populates="restaurant")
    agent_configuration = relationship("AgentConfiguration", uselist=False, back_populates="restaurant", cascade="all, delete-orphan")


class AgentConfiguration(Base):
    __tablename__ = "agent_configurations"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), unique=True, nullable=False)
    voice_engine = Column(String(50), nullable=False, default="urdu-female")
    is_active = Column(Boolean, nullable=False, default=True)

    system_prompt = Column(String(2000), nullable=False, default="You are the friendly AI voice-agent taking orders for Restaurant. Speak in Roman Urdu/Urdu-English mix. Reference the menu database for pricing. Do not offer discounts exceeding PKR 100. Do not repeat greetings (like Assalam-o-Alaikum) in subsequent turns. Be extremely brief, using under 30 words per conversation bubble. Make sure to collect the customer's delivery address before concluding the order.")

    restaurant = relationship("Restaurant", back_populates="agent_configuration")


# ── Call Logs: Twilio call transcripts surfaced in the v2 call-history UI ──
# NOTE: renamed table to "call_logs" to avoid schema conflicts with the old
#       "chat_history" table (which had NOT NULL FKs to now-deleted Org/Team/Agent).
class ChatHistory(Base):
    __tablename__ = "call_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=True)   # Twilio streamSid
    chat_data = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)
    response_time = Column(Float, nullable=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=True)
    caller_number = Column(String(50), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    status = Column(String(50), nullable=False, default="in_progress")
    recording_url = Column(String(255), nullable=True)
    transport = Column(String(50), nullable=False, default="twilio")

    restaurant = relationship("Restaurant")
    orders = relationship("Order", back_populates="call_log")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False)
    call_id = Column(Integer, ForeignKey("call_logs.id"), nullable=True)
    customer_phone = Column(String(50), nullable=False)
    items_summary = Column(JSON, nullable=False)
    total_price = Column(Float, nullable=False)
    status = Column(String(50), nullable=False, default="Pending")
    dispatched_at = Column(DateTime, nullable=True)
    dispatched_to_external_system = Column(Boolean, nullable=False, default=False)
    dispatch_attempts = Column(Integer, nullable=False, default=0)
    last_dispatch_error = Column(String(500), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    restaurant = relationship("Restaurant", back_populates="orders")
    call_log = relationship("ChatHistory", back_populates="orders")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False)
    type = Column(String(100), nullable=False)
    message = Column(String(500), nullable=False)
    related_order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)
    is_read = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, server_default=func.now())

    restaurant = relationship("Restaurant")
    related_order = relationship("Order")


class MenuItem(Base):
    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    description = Column(String(500), nullable=True)
    price = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    restaurant = relationship("Restaurant")


class PlanChangeRequest(Base):
    __tablename__ = "plan_change_requests"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False)
    requested_plan = Column(String(50), nullable=False)  # "monthly", "six_monthly", "annually"
    current_plan = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # "pending", "approved", "rejected"
    admin_note = Column(String(500), nullable=True)
    superadmin_note = Column(String(500), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    resolved_at = Column(DateTime, nullable=True)

    restaurant = relationship("Restaurant")
