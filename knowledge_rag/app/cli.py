"""
Command-line interface — the fastest way to test the pipeline without
running the API server.

Usage:
    python -m app.cli ingest <business_id> <file_path>
    python -m app.cli ingest-text <business_id> "<raw text>" [--name label]
    python -m app.cli review-list <business_id>
    python -m app.cli review-show <business_id> <ingestion_id>
    python -m app.cli review-publish <business_id> <ingestion_id> [--file corrected.json]
    python -m app.cli review-reject <business_id> <ingestion_id> [--reason "..."]
    python -m app.cli ask <business_id> "<question>"
    python -m app.cli chat <business_id>
    python -m app.cli profile <business_id>
    python -m app.cli menu-search <business_id> "<phrase>"
    python -m app.cli delete <business_id>
"""
import argparse
import json

from . import ingest, rag, vectorstore


def main():
    parser = argparse.ArgumentParser(description="Text RAG CLI — ingest brand documents, review, and ask questions.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest a text/PDF/image file for a business")
    ingest_p.add_argument("business_id", help="Short unique id, e.g. 'clove_cafe'")
    ingest_p.add_argument("file_path", help="Path to a .txt, .pdf, .png, .jpg, or .jpeg file")

    ingest_text_p = sub.add_parser("ingest-text", help="Ingest raw pasted/typed text for a business (no file)")
    ingest_text_p.add_argument("business_id", help="Short unique id, e.g. 'clove_cafe'")
    ingest_text_p.add_argument("text", help="The raw text to ingest (wrap in quotes)")
    ingest_text_p.add_argument("--name", default="pasted_text", help="Optional label for this text (used in the saved filename)")

    review_list_p = sub.add_parser("review-list", help="List ingestions awaiting human review")
    review_list_p.add_argument("business_id")

    review_show_p = sub.add_parser("review-show", help="Show the full structured data for one ingestion")
    review_show_p.add_argument("business_id")
    review_show_p.add_argument("ingestion_id")

    review_publish_p = sub.add_parser(
        "review-publish",
        help="Approve an ingestion — makes its menu items live for order-taking",
    )
    review_publish_p.add_argument("business_id")
    review_publish_p.add_argument("ingestion_id")
    review_publish_p.add_argument(
        "--file", help="Optional path to a corrected structured-data JSON file to publish instead of the original extraction",
    )

    review_reject_p = sub.add_parser("review-reject", help="Reject an ingestion — its menu never goes live")
    review_reject_p.add_argument("business_id")
    review_reject_p.add_argument("ingestion_id")
    review_reject_p.add_argument("--reason", default="")

    ask_p = sub.add_parser("ask", help="Ask a single question about a business's ingested data")
    ask_p.add_argument("business_id")
    ask_p.add_argument("question")

    chat_p = sub.add_parser("chat", help="Interactive Q&A loop for a business")
    chat_p.add_argument("business_id")

    profile_p = sub.add_parser("profile", help="Print the merged (published-only) structured profile for a business")
    profile_p.add_argument("business_id")

    menu_search_p = sub.add_parser(
        "menu-search",
        help="Resolve a spoken/typed phrase to real, published menu items (what the voice agent will call)",
    )
    menu_search_p.add_argument("business_id")
    menu_search_p.add_argument("phrase", help="e.g. 'the spicy chicken thing' or 'large cappuccino'")

    delete_p = sub.add_parser("delete", help="Permanently delete ALL data for a business")
    delete_p.add_argument("business_id")
    delete_p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.command == "ingest":
        record = ingest.ingest_file(args.business_id, args.file_path)
        print(f"\nIngestion id: {record['ingestion_id']}  |  status: {record['status']}")
        if record["status"] == "pending_review":
            print("Run 'review-show' to inspect it, then 'review-publish' to make its menu live.")

    elif args.command == "ingest-text":
        record = ingest.ingest_text(args.business_id, args.text, source_name=args.name)
        print(f"\nIngestion id: {record['ingestion_id']}  |  status: {record['status']}")
        if record["status"] == "pending_review":
            print("Run 'review-show' to inspect it, then 'review-publish' to make its menu live.")

    elif args.command == "review-list":
        pending = ingest.list_pending_reviews(args.business_id)
        if not pending:
            print("Nothing pending review.")
        for r in pending:
            n_items = len(r["data"].get("menu") or [])
            print(f"  {r['ingestion_id']}  ({r['source_file']}, {n_items} menu item(s))")

    elif args.command == "review-show":
        record = ingest.get_review(args.business_id, args.ingestion_id)
        if record is None:
            print("Not found.")
        else:
            print(json.dumps(record, indent=2, ensure_ascii=False))

    elif args.command == "review-publish":
        corrected = None
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                corrected = json.load(f)
        record = ingest.publish_review(args.business_id, args.ingestion_id, corrected_data=corrected)
        print(f"Published. {len(record['data'].get('menu') or [])} menu item(s) now live.")

    elif args.command == "review-reject":
        ingest.reject_review(args.business_id, args.ingestion_id, reason=args.reason)
        print("Rejected.")

    elif args.command == "ask":
        result = rag.answer_question(args.business_id, args.question)
        print("\n" + result["answer"])
        if result["sources"]:
            print("\n(sources: " + ", ".join(result["sources"]) + ")")

    elif args.command == "chat":
        print(f"Chatting with {args.business_id}'s assistant. Type 'exit' to quit.\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            result = rag.answer_question(args.business_id, q)
            print(f"Assistant: {result['answer']}\n")

    elif args.command == "profile":
        print(json.dumps(ingest.get_merged_business_profile(args.business_id), indent=2, ensure_ascii=False))

    elif args.command == "menu-search":
        matches = vectorstore.query_menu_items(args.business_id, args.phrase)
        if not matches:
            print("No published menu items found — ingest and publish a menu for this business first.")
        else:
            print(json.dumps(matches, indent=2, ensure_ascii=False))

    elif args.command == "delete":
        if not args.yes:
            confirm = input(f"Type '{args.business_id}' to permanently delete all its data: ").strip()
            if confirm != args.business_id:
                print("Aborted.")
                return
        ingest.delete_business(args.business_id)
        print("Deleted.")


if __name__ == "__main__":
    main()
