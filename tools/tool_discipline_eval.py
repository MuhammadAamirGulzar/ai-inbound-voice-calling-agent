"""
Tool-discipline eval: does the agent transfer/hang up when it should —
and ONLY when it should?

Found in live testing: with tools available, LLMs sporadically route
ordinary questions ("do you deliver to Gulberg?") to transfer_call.
Because the failure is nondeterministic, each case runs R times and the
report shows a rate, not a boolean.

    python tools/tool_discipline_eval.py                # default model
    python tools/tool_discipline_eval.py --models llama-3.3-70b-versatile openai/gpt-oss-120b
    python tools/tool_discipline_eval.py --reps 5 --report discipline.json

Case sets: routine utterances must produce a spoken answer and no tool;
escalations must call transfer_call; farewells must call end_call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice.config import VoiceConfig
from voice.llm import StreamingLLM

SYSTEM_PROMPT = (
    "You are Sara, the phone assistant for Clove Cafe, taking delivery "
    "orders. Menu: Chicken Burger $5.99, Beef Burger $6.99, Fries $2.49, "
    "Soft Drink $1.49. We deliver within 5 km, fee $1, 30-40 minutes. "
    "Open 11am-11pm daily. Be warm but extremely brief: under 25 words. "
    "Answer routine questions (menu, prices, delivery area, hours) "
    "yourself — never transfer for those. Use transfer_call ONLY if the "
    "caller explicitly asks for a human or is complaining; use end_call "
    "only once the caller has clearly finished."
)
GREETING = "Hello! Thank you for calling Clove Cafe. What would you like to order today?"

ROUTINE = [  # expected: text reply, NO tool
    "Do you deliver to Gulberg?",
    "One chicken burger and fries please.",
    "How much is the beef burger?",
    "What time do you close?",
    "Deliver to house 12, street 4, Gulberg.",
    "Can I get two soft drinks with that?",
    "How long will delivery take?",
    "Do you have anything spicy?",
]
ESCALATE = [  # expected: transfer_call
    "I want to talk to a real person right now.",
    "This is the third time my order was wrong, get me your manager.",
]
FAREWELL = [  # expected: end_call
    "That's everything, thank you, bye!",
    "No that's all, goodbye.",
]


async def run_case(llm, user_text: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": GREETING},
        {"role": "user", "content": user_text},
    ]
    text, tools = "", []
    async for event in llm.stream_reply(messages):
        if event.kind == "token":
            text += event.text
        elif event.kind == "tool_call":
            tools.append(event.tool_name)
        elif event.kind == "error":
            return "error", event.text
    if "transfer_call" in tools:
        return "transfer", text
    if "end_call" in tools:
        return "end", text
    return ("answer", text) if text.strip() else ("silent", "")


async def eval_model(model: str, reps: int) -> dict:
    cfg = VoiceConfig.from_env()
    cfg.llm_model = model
    llm = StreamingLLM(cfg)

    async def score(cases, expected):
        hits = failures = 0
        for case in cases:
            for _ in range(reps):
                outcome, detail = await run_case(llm, case)
                ok = (outcome == expected)
                hits += ok
                if not ok:
                    failures += 1
                    print(f"    MISS [{model}] {case!r} -> {outcome} ({str(detail)[:50]!r})")
        total = len(cases) * reps
        return {"correct": hits, "total": total,
                "rate": round(hits / total, 3) if total else None}

    print(f"\n=== {model} (x{reps} reps) ===")
    result = {
        "routine_no_tool": await score(ROUTINE, "answer"),
        "escalate_transfers": await score(ESCALATE, "transfer"),
        "farewell_ends": await score(FAREWELL, "end"),
    }
    for name, r in result.items():
        print(f"  {name}: {r['correct']}/{r['total']} ({r['rate']})")
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=[os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--report")
    args = parser.parse_args()

    results = {}
    for model in args.models:
        results[model] = await eval_model(model, args.reps)
    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=2))
        print(f"\nreport written to {args.report}")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    asyncio.run(main())
