import os
import json
import logging
from typing import Optional
from sqlalchemy.orm import Session
from openai import OpenAI
from sql import crud, models

logger = logging.getLogger(__name__)

def extract_order_from_transcript(chat_history: models.ChatHistory, db: Session) -> Optional[models.Order]:
    """
    Parses the conversation transcript of a completed call using the configured LLM,
    extracts the items ordered, pricing, and customer details, and persists them into the DB.
    """
    if not chat_history or not chat_history.chat_data:
        logger.info(f"No chat data found for call session {chat_history.session_id}")
        return None

    # Format transcript for the LLM
    transcript_lines = []
    for msg in chat_history.chat_data:
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or msg.get("text", "")
        if role == "user":
            transcript_lines.append(f"Customer: {content}")
        elif role == "assistant":
            transcript_lines.append(f"AI Agent: {content}")

    transcript_text = "\n".join(transcript_lines)

    prompt = f"""
You are an AI order extractor for a restaurant voice calling platform.
Your task is to analyze the following transcript of a phone conversation between a Customer and an AI Agent.
Determine if the customer successfully confirmed/placed an order.

CRITICAL RULES:
1. ONLY extract an order if the customer explicitly stated they wanted to order/buy the items, AND the order was confirmed.
2. If the customer ONLY asked about prices, menu items, or hung up without confirming an order, you MUST reply exactly with: no_order
3. Do NOT assume or hallucinate that an order was placed just because a food item (like cheese naan) was mentioned or priced in the conversation.

EXAMPLES:

Example 1 (No order placed):
Transcript:
Customer: What is the price of cheese naan?
AI Agent: Cheese naan is PKR 995. Would you like to order?
Customer: No thank you.
Output:
no_order

Example 2 (Order confirmed):
Transcript:
Customer: I want to order 2 cheese naans.
AI Agent: 2 cheese naans added. Please provide delivery address.
Customer: First University, Islamabad.
AI Agent: Confirmed, delivered to First University.
Output:
{{
  "items": [
    {{"item": "Cheese Naan", "qty": 2}}
  ],
  "total_price": 1990.0,
  "delivery_address": "First University, Islamabad"
}}

Analyze this transcript:
\"\"\"
{transcript_text}
\"\"\"

Output:
"""

    endpoint = os.getenv("MODEL_ENDPOINT_URL", "http://localhost:11434/v1")
    api_key = os.getenv("MODEL_API_KEY", "ollama")
    model_name = os.getenv("MODEL_NAME", "qwen2.5vl:7b")
    timeout = float(os.getenv("MODEL_REQUEST_TIMEOUT", "180"))

    try:
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
            timeout=timeout,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        result_text = (response.choices[0].message.content or "").strip()
        logger.info(f"Order extraction raw LLM output: {result_text}")

        # Clean JSON markdown if model wrapped it
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            cleaned_lines = [l for l in lines if not l.strip().startswith("```")]
            result_text = "\n".join(cleaned_lines).strip()

        if "no_order" in result_text.lower():
            logger.info("LLM determined no order was placed in this call.")
            return None

        # Parse JSON
        order_data = json.loads(result_text)
        items = order_data.get("items", [])
        total_price = float(order_data.get("total_price", 0))
        customer_phone = chat_history.caller_number or "Unknown"

        if not items:
            logger.info("Parsed order contains no items. Skipping.")
            return None

        # Save order to DB
        new_order = crud.create_order(
            db=db,
            restaurant_id=chat_history.restaurant_id,
            customer_phone=customer_phone,
            items_summary=items,
            total_price=total_price,
            call_id=chat_history.id
        )
        logger.info(f"Order successfully created: ID {new_order.id}")
        return new_order

    except Exception as e:
        logger.error(f"Failed to extract/create order: {e}")
        return None
