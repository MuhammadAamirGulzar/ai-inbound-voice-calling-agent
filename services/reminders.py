"""Daily cron: subscription due-date reminder emails to tenant admins."""

from sqlalchemy.orm import Session

from sql.database import SessionLocal


def check_and_send_due_date_reminders(db: Session):
    import calendar
    from datetime import datetime, timedelta
    from sql import models
    from utils.mailer import send_email as _send_email

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
