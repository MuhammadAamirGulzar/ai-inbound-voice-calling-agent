"""Shared web-layer plumbing: template environment and DB session dep."""

from fastapi.templating import Jinja2Templates

from sql.database import SessionLocal

templates = Jinja2Templates(directory="templates")


def pk_time_filter(dt, fmt="%I:%M %p"):
    if not dt:
        return ""
    from datetime import timedelta
    # Add 5 hours to offset UTC to PKT (Pakistan Standard Time)
    pkt_dt = dt + timedelta(hours=5)
    return pkt_dt.strftime(fmt)


templates.env.filters["pk_time"] = pk_time_filter
templates.env.globals["pk_time"] = pk_time_filter


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
