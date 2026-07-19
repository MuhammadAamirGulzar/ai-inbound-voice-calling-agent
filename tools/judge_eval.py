"""
LLM-as-judge evaluation of call transcripts.

Scores a conversation against the behaviors that matter for a phone
agent, using a stronger model as the judge. Runs on:

  - a transcript JSON file (list of {"role", "content"} messages):
      python tools/judge_eval.py --file call.json
  - recent calls straight from the database:
      python tools/judge_eval.py --from-db 5

Rubric (1-5 each, with rationale):
  task_completion   did the agent accomplish what the caller wanted
  brevity           replies short enough for voice (no monologues)
  language          stayed in the caller's language/register
  groundedness      no invented menu items, prices, or policies
  call_control      transferred/ended when it should have, and not when
                    it shouldn't

Exit code is non-zero when any dimension scores below --fail-under
(default 3), so this can gate a CI job on regression sets:
    python tools/judge_eval.py --file evals/regression/*.json --fail-under 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUBRIC_KEYS = ["task_completion", "brevity", "language",
               "groundedness", "call_control"]

JUDGE_PROMPT = """You are evaluating a phone call transcript between a \
restaurant's AI voice agent (assistant) and a caller (user).

Score each dimension 1-5 (5 = excellent) and give a one-sentence rationale:
- task_completion: did the agent accomplish what the caller wanted \
(order taken with address, question answered, or correctly escalated)?
- brevity: were replies short and phone-appropriate (under ~30 words)?
- language: did the agent consistently match the caller's language and \
register (including Urdu/English code-switching)?
- groundedness: did the agent avoid inventing menu items, prices, \
policies, or delivery claims not present in the conversation context?
- call_control: did the agent transfer or end the call at appropriate \
moments, and avoid doing so when unnecessary?

Reply with ONLY a JSON object:
{"task_completion": {"score": n, "why": "..."}, "brevity": {...}, \
"language": {...}, "groundedness": {...}, "call_control": {...}, \
"summary": "one-sentence overall assessment"}"""


def judge(transcript: list[dict], model: str, base_url: str,
          api_key: str) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
    convo = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in transcript if m.get("role") in ("user", "assistant"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": JUDGE_PROMPT},
                  {"role": "user", "content": convo}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def load_from_db(limit: int) -> list[tuple[str, list[dict]]]:
    from sql.database import SessionLocal
    from sql import models
    db = SessionLocal()
    try:
        rows = (db.query(models.ChatHistory)
                .filter(models.ChatHistory.chat_data.isnot(None))
                .order_by(models.ChatHistory.id.desc())
                .limit(limit).all())
        return [(f"call#{r.id}", r.chat_data) for r in rows if r.chat_data]
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge call scoring")
    parser.add_argument("--file", nargs="+", help="transcript JSON file(s), globs ok")
    parser.add_argument("--from-db", type=int, metavar="N",
                        help="judge the N most recent calls from the database")
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL")
                        or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    parser.add_argument("--fail-under", type=float, default=3.0)
    parser.add_argument("--report", help="write JSON report here")
    args = parser.parse_args()

    api_key = (os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY", ""))
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    if not api_key:
        print("GROQ_API_KEY (or LLM_API_KEY) required.")
        sys.exit(1)

    cases: list[tuple[str, list[dict]]] = []
    if args.file:
        for pattern in args.file:
            for path in (glob.glob(pattern) or [pattern]):
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                cases.append((path, data))
    elif args.from_db:
        cases = load_from_db(args.from_db)
    else:
        parser.error("provide --file or --from-db")

    all_reports = []
    worst = 5.0
    for name, transcript in cases:
        try:
            verdict = judge(transcript, args.model, base_url, api_key)
        except Exception as e:
            print(f"{name}: judge call failed: {e}")
            worst = 0.0
            continue
        scores = {k: verdict.get(k, {}).get("score") for k in RUBRIC_KEYS}
        valid = [s for s in scores.values() if isinstance(s, (int, float))]
        low = min(valid) if valid else 0
        worst = min(worst, low)
        print(f"\n{name}  " + "  ".join(
            f"{k}={scores[k]}" for k in RUBRIC_KEYS))
        print(f"  {verdict.get('summary', '')}")
        for key in RUBRIC_KEYS:
            entry = verdict.get(key, {})
            if isinstance(entry.get("score"), (int, float)) and entry["score"] < args.fail_under:
                print(f"  LOW {key}: {entry.get('why', '')}")
        all_reports.append({"case": name, "scores": scores, "verdict": verdict})

    if args.report:
        Path(args.report).write_text(
            json.dumps(all_reports, indent=2, ensure_ascii=False))
        print(f"\nreport written to {args.report}")

    if worst < args.fail_under:
        print(f"\nFAIL: lowest score {worst} < threshold {args.fail_under}")
        sys.exit(1)
    print(f"\nPASS: all dimensions >= {args.fail_under}")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    main()
