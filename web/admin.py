"""Restaurant admin dashboard: orders, menu, calls, analytics, settings."""

import os
import re
from datetime import datetime, timedelta
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from sql import crud, models, schemas
from services.menu_sync import (RAG_UPSTREAM, save_published_menu_to_postgres,
                                sync_postgres_menu_to_chroma)
from utils.cookies import validate_admin_cookies, validate_cookies
from utils.logger import print_info, print_error
from utils.mailer import send_email
from web.core import templates, get_db

router = APIRouter()

@router.get("/admin/dashboard", response_class=HTMLResponse)
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

@router.get("/admin/call-history", response_class=HTMLResponse)
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

@router.get("/admin/call-detail", response_class=HTMLResponse)
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

@router.get("/admin/orders", response_class=HTMLResponse)
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

@router.get("/admin/order-detail", response_class=HTMLResponse)
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

@router.post("/admin/orders/{order_id}/status")
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


@router.post("/admin/orders/{order_id}/dispatch")
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


@router.get("/admin/notifications")
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


@router.post("/admin/notifications/{notif_id}/read")
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


@router.post("/admin/notifications/read-all")
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


@router.get("/admin/menu", response_class=HTMLResponse)
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

@router.post("/admin/menu/sync")
async def manual_sync_menu(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        await sync_postgres_menu_to_chroma(db, auth["user"].restaurant_id)
        return JSONResponse(status_code=200, content={"success": True, "message": "Menu synchronized to RAG successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to sync menu: {str(e)}"})

@router.post("/admin/menu/add")
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

@router.put("/admin/menu/{item_id}")
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

@router.delete("/admin/menu/{item_id}")
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

@router.post("/admin/menu/bulk-delete")
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

@router.get("/admin/live-calls", response_class=HTMLResponse)
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

@router.post("/admin/hangup")
async def admin_hangup(request: Request, db: Session = Depends(get_db)):
    auth = validate_admin_cookies(db, request)
    if not auth["success"]:
        return auth["response"]
        
    try:
        body = await request.json()
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse(status_code=400, content={"error": "Missing session_id"})
            
        from telephony.registry import active_connections
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

@router.get("/admin/analytics", response_class=HTMLResponse)
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

@router.get("/admin/api/analytics/charts")
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

@router.get("/admin/billing", response_class=HTMLResponse)
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

@router.get("/admin/agent-config", response_class=HTMLResponse)
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

@router.get("/admin/settings", response_class=HTMLResponse)
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


@router.post("/admin/settings/profile")
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


@router.post("/admin/settings/agent")
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


@router.post("/admin/agent-config/profile")
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

@router.post("/admin/settings/agent/toggle")
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


@router.post("/admin/settings/telephony")
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



@router.post("/admin/settings/fulfillment")
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

@router.post("/admin/billing/request-plan-change")
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
        from utils.mailer import send_email as _send_email
        import threading
        threading.Thread(target=_send_email, args=(superadmin_email, f"[OrderSaathi] Plan Change Request — {restaurant.name}", email_body, True), daemon=True).start()

    return JSONResponse(status_code=200, content={"success": True})


