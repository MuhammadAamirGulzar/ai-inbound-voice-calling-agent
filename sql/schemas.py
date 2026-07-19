from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import List, Optional


# ── User Schemas ──
# Temporary – kept for utils/cookies.py until platform_users table is built.
class UserBase(BaseModel):
    name: str
    email: EmailStr
    phone_number: Optional[str] = None
    username: str
    password: str
    role: Optional[str] = "admin"
    restaurant_id: Optional[int] = None


class UserCreate(UserBase):
    pass


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    restaurant_id: Optional[int] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class User(UserBase):
    id: int

    class Config:
        from_attributes = True


# ── Call Log Schemas ──
# Used by Twilio recording (twilio_routes.py) and surfaced in the v2 call-history UI.
class ChatHistoryBase(BaseModel):
    session_id: Optional[str] = None       # Twilio streamSid
    chat_data: Optional[List[dict]] = None
    response_time: Optional[float] = None
    restaurant_id: Optional[int] = None
    caller_number: Optional[str] = None
    duration_seconds: Optional[int] = None
    status: str = "in_progress"
    recording_url: Optional[str] = None



class ChatHistoryCreate(ChatHistoryBase):
    pass


class ChatHistory(ChatHistoryBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


# ── Restaurant Schemas ──
class RestaurantBase(BaseModel):
    name: str
    description: Optional[str] = None
    subscription_plan: str  # "monthly", "six_monthly", "annually", "free_trial"
    mobile_number: str
    order_phone_number: str
    is_suspended: Optional[bool] = False
    is_free_plan: Optional[bool] = False
    trial_expiration_date: Optional[datetime] = None
    assigned_minutes: Optional[int] = 100
    used_minutes: Optional[float] = 0.0




class RestaurantCreate(RestaurantBase):
    admin_name: str
    email: EmailStr


class Restaurant(RestaurantBase):
    id: int
    admin_user_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PlanChangeRequest(BaseModel):
    subscription_plan: str
    is_free_plan: bool


class ExtendTrialRequest(BaseModel):
    trial_expiration_date: str  # YYYY-MM-DD


class SuspensionRequest(BaseModel):
    is_suspended: bool


class QuotaAdjustmentRequest(BaseModel):
    assigned_minutes: int
    used_minutes: float


class PlanChangeRequestCreate(BaseModel):
    requested_plan: str  # "monthly", "six_monthly", "annually"
    admin_note: Optional[str] = None


class PlanChangeRequestReview(BaseModel):
    status: str  # "approved" or "rejected"
    superadmin_note: Optional[str] = None


class AdminAccountCreate(BaseModel):
    name: str
    email: EmailStr
    phone_number: Optional[str] = None
    role: str
    restaurant_id: Optional[int] = None


class AdminAccountUpdate(BaseModel):
    name: str
    email: EmailStr
    phone_number: Optional[str] = None
    role: str
    restaurant_id: Optional[int] = None


class SystemSettingsUpdate(BaseModel):
    groq_model: Optional[str] = None
    groq_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    default_order_phone_number: Optional[str] = None
    trial_quota: Optional[int] = None
    pro_quota: Optional[int] = None
    payment_account_name: Optional[str] = None
    payment_account_number: Optional[str] = None
    monthly_price: Optional[float] = None


class OrderStatusUpdate(BaseModel):
    status: str

class FulfillmentSettingsUpdate(BaseModel):
    integration_type: str
    target_url: Optional[str] = None


class MenuItemCreate(BaseModel):
    name: str
    category: str
    description: Optional[str] = None
    price: float

class MenuItemUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None


class RestaurantProfileUpdate(BaseModel):
    name: str
    mobile_number: str

    system_prompt: Optional[str] = None


class AgentConfigUpdate(BaseModel):
    voice_engine: str

    system_prompt: str


class TelephonySettingsUpdate(BaseModel):
    order_phone_number: str

