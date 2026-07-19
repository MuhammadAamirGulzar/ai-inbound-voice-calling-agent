# Business Knowledge RAG

A retrieval-augmented generation (RAG) pipeline that turns a business's raw
material — a menu, a persona script, an "about us" page, delivery/hours
info — into a structured, queryable knowledge base. Provide the source as
**typed/pasted text, a PDF, or a photo**; the system extracts it, structures
it, and can then answer customer questions about that business, grounded
only in what was actually ingested.

This module is the knowledge-ingestion and Q&A layer for a larger **voice
call agent** project: it owns extraction, structuring, human review, and
retrieval, and exposes a clean HTTP API (`main.py`) for the voice agent to
call during a live call.

Every AI step — reading images/scanned PDFs, structuring text into JSON,
and generating answers — runs on a single **Qwen2.5-VL** model served
locally by **Ollama**, talked to over its OpenAI-compatible `/v1` API. No
cloud account, no API token, and no per-request cost — just Ollama running
on your machine (or a separate GPU machine reached through a tunnel, if
your own machine doesn't have enough VRAM). See [Setup](#setup) below for a
step-by-step walkthrough.

---

## How It Works

```
  typed/pasted text  /  .pdf  /  .png  /  .jpg
            │
            ▼
   ┌─────────────────┐
   │  1. EXTRACTION   │  app/extraction.py
   │  native PDF text │  → PyMuPDF (no model call needed)
   │  scanned PDF/img │  → Ollama-served Qwen2.5-VL (vision)
   │  pasted text     │  → used as-is, no extraction needed
   └────────┬─────────┘
            ▼
   ┌─────────────────┐
   │  2. STRUCTURING  │  app/structuring.py
   │  raw text → JSON │  → Ollama-served Qwen2.5-VL: persona, policies, details, menu
   │                  │  → detects any pre-structured JSON config already
   │                  │    embedded in the source and treats it as
   │                  │    authoritative (exact scripts aren't paraphrased)
   │                  │  → long documents are split into overlapping
   │                  │    sections and merged, never silently truncated
   └────────┬─────────┘
            ▼
   ┌─────────────────┐
   │ 3. HUMAN REVIEW  │  app/ingest.py — list_pending_reviews / get_review /
   │  (menu only)     │  publish_review / reject_review
   │                  │  Extracted menu items are held as "pending_review"
   │                  │  and are NOT searchable until a human approves
   │                  │  (optionally with corrections) via publish_review.
   │                  │  General Q&A text chunks (step 4) are not gated —
   │                  │  they're lower stakes and already hedge with
   │                  │  "I don't have that information" instead of
   │                  │  inventing answers. The menu is what a caller's
   │                  │  bill is built from, so it gets the stricter gate.
   └────────┬─────────┘
            ▼
   ┌─────────────────┐
   │  4. CHUNK+EMBED  │  app/chunking.py + app/vectorstore.py
   │  local, no API   │  → sentence-transformers (local) + ChromaDB (local)
   │                  │  → two collections per business: raw text chunks
   │                  │    (general Q&A, embedded immediately at ingest)
   │                  │    and individual menu items (embedded only once
   │                  │    published — see step 3)
   └────────┬─────────┘
            ▼
   ┌─────────────────┐
   │ 5a. RAG QUERY    │  app/rag.py — open-ended Q&A ("what are your
   │  general Q&A     │  delivery timings?") → chunk retrieval + Ollama text model
   ├─────────────────┤
   │ 5b. MENU SEARCH  │  vectorstore.query_menu_items() — resolves a
   │  order-taking    │  spoken phrase to an exact, priced, PUBLISHED
   │                  │  menu item. This is what the voice agent calls,
   │                  │  not 5a — and it will never return an
   │                  │  unreviewed item.
   └─────────────────┘
```

Each business gets its own isolated Chroma collections (`business_<id>` for
text chunks, `business_<id>_menu` for menu items) and its own folder under
`data/raw/` and `data/processed/` — a query about Business A can never
retrieve Business B's data. `ingest.delete_business()` / `DELETE
/businesses/{id}` wipes all of it for a given business, for resets or
mistaken onboardings.

**Why two retrieval paths instead of one:** general questions ("do you
deliver?") are well served by fuzzy chunk retrieval. Order-taking is not —
a caller saying "the spicy chicken thing" needs to resolve to one exact
menu row with a correct price, not a fragment of nearby text. Keeping menu
items in their own embedded, structured index means the voice agent gets a
precise `{name, price, variants, customizations}` result, not a guess.

**Why the menu specifically needs a human-review gate:** a wrong price or a
hallucinated item extracted from a messy scanned menu photo is the costliest
mistake this system can make — it's what a customer gets charged. So menu
items sit in a `pending_review` state after extraction and are invisible to
`menu-search`/`/menu-search` until a human calls `publish_review` (optionally
passing corrected data). Persona/details/policies text and general Q&A
chunks aren't gated the same way, since they're lower-stakes and the answer
layer is already prompted to say "I don't have that information" rather
than invent one.

**Why embeddings stay local instead of also going through Ollama/the
model API:** `query_menu_items()` is the function the voice agent calls
*live, mid-call*, to resolve what a caller just said to a real menu item.
That's the one place in this system where extra latency (or a slow/busy
tunnelled GPU machine) is actually costly. The embedding model
(`all-MiniLM-L6-v2`, ~80MB) is small enough to run on CPU in milliseconds,
so it's kept local and separate from Ollama entirely — it's a
`sentence-transformers` model, downloaded once from Hugging Face's model
hub (a one-time file download, not an API call) and run locally from then
on.

---

## Structured Data Schema

Every ingestion produces this shape (see `app/structuring.py` for the exact
prompt):

```json
{
  "business_name": "Clove Cafe",
  "persona": {
    "agent_name": "Zara",
    "tone": "warm, polite, energetic",
    "language": "English with Roman Urdu (Minglish)",
    "greeting_style": "",
    "greeting_script": "Hi, thank you for calling Clove Cafe! ...",
    "closing_script": "Thank you for choosing Clove Cafe! ...",
    "notes": "Never robotic or overly formal."
  },
  "details": {
    "address": "", "phone": "", "hours": "",
    "delivery_info": "", "delivery_areas": ["Bahria Town", "DHA"],
    "avg_preparation_time": "25-35 minutes",
    "other": ""
  },
  "policies": {
    "upsell_strategy": "Gently suggest a drink with a main meal, never pushy.",
    "out_of_stock_protocol": "Apologize warmly, suggest the closest alternative.",
    "escalation_protocol": "Stay calm, offer to transfer to a manager if complex."
  },
  "menu": [
    {
      "category": "Hot Coffee", "item": "Cappuccino", "price": "",
      "description": "",
      "variants": [
        {"name": "Regular", "price": "RS. 745"},
        {"name": "Large", "price": "RS. 825"}
      ],
      "customizations": []
    }
  ]
}
```

Two things worth knowing since they're easy to get wrong against a real
menu (e.g. one priced as `RS. 745/825`):

- **`greeting_script` / `closing_script` vs `greeting_style`**: scripts are
  exact lines to be spoken verbatim, only populated when the source text
  actually gives one (e.g. a line labelled "Standard Greeting:"). If the
  source only describes a vibe ("keep it warm and casual"), that goes in
  `greeting_style` instead — the model is instructed not to paraphrase a
  script into existence or invent one from a style description.
- **`variants` carry their own price**, not a shared item-level price. A
  line like `Cappuccino RS. 745/825` becomes one item with two priced
  variants — not a single item with an ambiguous combined price string.

**Pre-structured input**: if a business hands over a document that already
contains a JSON config block, `structuring.py` detects it via
brace-matching, repairs the line-wrapping that PDF/OCR text extraction
introduces into embedded JSON strings, and passes it to the model as
authoritative context — so an exact scripted greeting doesn't get silently
reworded by a summarization pass.

---

## Tools Used

| Purpose | Tool | Notes |
|---|---|---|
| PDF text extraction | PyMuPDF (`fitz`) | Local, no model call |
| Image / scanned-PDF reading | Ollama — Qwen2.5-VL (default: `qwen2.5vl:7b`) | Local (or tunnelled remote GPU) |
| Structuring text → JSON | Ollama — same Qwen2.5-VL model | Local |
| RAG answer generation | Ollama — same Qwen2.5-VL model | Local |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, CPU — see note above |
| Vector store | ChromaDB (persistent, on-disk) | Local |
| API layer (voice agent integration point) | FastAPI + Uvicorn | Local |
| Operator / testing UI | Streamlit | Local |

---

## Setup

### 1. Install Ollama and pull the model

This project runs entirely against a local model server — no cloud account
or API token needed.

1. Install Ollama: **https://ollama.com/download** (Windows/Mac/Linux).
2. Pull the model:
   ```bash
   ollama pull qwen2.5vl:7b
   ```
   (Use `qwen2.5vl:3b` instead if your machine is short on VRAM/RAM, or
   `qwen2.5vl:32b` if you have a beefier GPU. Whichever tag you pull must
   match `MODEL_NAME` in `.env` — see step 3.)
3. That's it — once installed, Ollama serves an OpenAI-compatible API at
   `http://localhost:11434/v1` automatically. You don't need to manually
   start a server; `ollama serve` runs as a background service after
   install (run it manually only if `ollama pull` fails saying it isn't
   running).

### 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

The defaults in `.env.example` already point at local Ollama
(`MODEL_ENDPOINT_URL=http://localhost:11434/v1`,
`MODEL_NAME=qwen2.5vl:7b`) — you don't need to change anything to get
started, as long as the tag you pulled in step 1 matches `MODEL_NAME`.

If you later move inference to a separate GPU machine (e.g. reached
through an ngrok tunnel), just change `MODEL_ENDPOINT_URL` to that
machine's tunnel URL (ending in `/v1`) — no code changes needed, see
[Changing models](#changing-models) below.

### 4. Launch the knowledge console

```bash
streamlit run scripts/knowledge_console.py
```

Paste text or upload a document, review exactly what was extracted, and
confirm before any menu data is published.

---

## Usage

### Knowledge console (recommended interface)

```bash
streamlit run scripts/knowledge_console.py
```

Choose **"Paste text"** to type or paste a menu/persona/policy document
directly, or **"Upload a file"** for a `.txt`, `.pdf`, or photo. Either way
the console extracts and structures it, shows the extracted profile
(business name, persona, details, policies, and menu items) for review, and
keeps menu data in a pending state until explicitly confirmed — nothing
reaches the voice agent's searchable index without that confirmation. Once
published, ask it questions directly from the same interface to see the
grounded answers a caller would receive.

### CLI

```bash
# Ingest a file for a business (repeat for each file: menu, persona, PDF, photo, etc.)
python -m app.cli ingest clove_cafe path/to/menu.pdf
python -m app.cli ingest clove_cafe path/to/persona.txt

# Or ingest raw text directly, no file needed
python -m app.cli ingest-text clove_cafe "Clove Cafe menu: Cappuccino - Rs. 450 ..." --name menu

# See what's waiting for human review after ingesting something with a menu
python -m app.cli review-list clove_cafe
python -m app.cli review-show clove_cafe <ingestion_id>

# Approve as-extracted...
python -m app.cli review-publish clove_cafe <ingestion_id>

# ...or approve with corrections: edit the JSON printed by review-show,
# save it, then publish that corrected version instead
python -m app.cli review-publish clove_cafe <ingestion_id> --file corrected.json

# Reject an ingestion (its menu never goes live, but the record is kept for audit)
python -m app.cli review-reject clove_cafe <ingestion_id> --reason "wrong file uploaded"

# Ask a one-off general question (general Q&A chunks are live immediately, no review needed)
python -m app.cli ask clove_cafe "What are your delivery timings?"

# Interactive chat loop
python -m app.cli chat clove_cafe

# See the merged, PUBLISHED-only structured profile (persona + policies + details + menu)
python -m app.cli profile clove_cafe

# Resolve a spoken/typed phrase to real, PUBLISHED menu items — what the
# voice agent will call during order-taking (not the general "ask" command)
python -m app.cli menu-search clove_cafe "the spicy chicken thing"

# Permanently delete all of a business's data (raw files, JSON, embeddings)
python -m app.cli delete clove_cafe
```

### API (for the voice call agent to integrate against)

```bash
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs for interactive Swagger docs, or:

```bash
# Ingest a file
curl -X POST http://127.0.0.1:8000/ingest \
  -F "business_id=clove_cafe" \
  -F "file=@/path/to/menu.pdf"

# Or ingest raw text directly, no file needed
curl -X POST http://127.0.0.1:8000/ingest-text \
  -H "Content-Type: application/json" \
  -d '{"business_id": "clove_cafe", "text": "Clove Cafe menu: Cappuccino - Rs. 450 ...", "source_name": "menu"}'

curl http://127.0.0.1:8000/businesses/clove_cafe/reviews

curl -X POST http://127.0.0.1:8000/businesses/clove_cafe/reviews/<ingestion_id>/publish \
  -H "Content-Type: application/json" -d '{}'

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"business_id": "clove_cafe", "question": "What are your delivery timings?"}'

curl -X POST http://127.0.0.1:8000/menu-search \
  -H "Content-Type: application/json" \
  -d '{"business_id": "clove_cafe", "phrase": "large cappuccino"}'

curl -X DELETE http://127.0.0.1:8000/businesses/clove_cafe
```

`/menu-search` is the endpoint the voice agent should call during
order-taking; `/query` is for general open-ended questions.

---

## Project Structure

```
business-knowledge-rag/
├── app/
│   ├── config.py         # model names, timeouts, data paths (reads .env)
│   ├── llm_client.py     # the ONLY module that talks to Ollama — chat_text() / chat_vision()
│   ├── extraction.py     # text / .pdf / image → raw text
│   ├── structuring.py    # raw text → structured JSON (persona/policies/details/menu),
│   │                     #   embedded-JSON detection, long-doc section splitting
│   ├── chunking.py       # raw text → chunks for embedding
│   ├── vectorstore.py    # local embeddings + ChromaDB storage/retrieval + delete
│   ├── ingest.py         # orchestrates extraction/text → structuring → review → publish
│   ├── rag.py            # retrieval + Ollama-generated answer
│   ├── retry.py          # retry-with-backoff for model API calls
│   └── cli.py            # command-line interface (ingest, ingest-text, review-*, ask, delete, ...)
├── scripts/
│   └── knowledge_console.py  # Streamlit operator interface (intake, review, Q&A)
├── data/
│   ├── raw/              # original uploaded files / saved pasted text, per business (gitignored)
│   ├── processed/        # structured JSON records, per business (gitignored)
│   └── chroma_db/        # vector store, persisted on disk (gitignored)
├── .streamlit/
│   └── config.toml       # disables the file watcher, sets upload limits (see Troubleshooting)
├── main.py               # FastAPI app — the integration point for the voice call agent
├── requirements.txt
├── .env.example
└── README.md
```

`scripts/inspect_db.py` and `scripts/debug_retrieval.py` from earlier
versions of this project have been removed — their functionality is
already covered by `python -m app.cli profile` / `review-show` /
`menu-search`, so keeping both was redundant for a project meant to ship
alongside a production voice agent.

---

## Changing Models

`MODEL_NAME` in `.env` is the only knob you should ever need — it's used
for both vision and text calls. To switch models:

```bash
ollama pull qwen2.5vl:3b        # or any other tag
```

then update `.env`:

```
MODEL_NAME=qwen2.5vl:3b
```

and restart. No code changes needed. `MODEL_NAME` must exactly match a tag
shown by `ollama list` on whichever machine `MODEL_ENDPOINT_URL` points at.

**Moving inference to a separate GPU machine:** if your own machine is too
slow or short on VRAM, run Ollama on a GPU machine instead, expose it with
a tunnel (e.g. `ngrok http 11434`), and point `MODEL_ENDPOINT_URL` in
`.env` at that tunnel URL with a trailing `/v1` — e.g.
`https://your-tunnel.ngrok-free.dev/v1`. Nothing else in this project
needs to change; `llm_client.py` doesn't care whether Ollama is local or
remote.

**Using a non-vision model for text-only steps:** this project intentionally
uses one Qwen2.5-VL model for everything (simpler setup, one thing to
pull/manage). If you'd rather use a smaller/faster plain-text model for
structuring and RAG answers (steps 2 and 5a) and reserve the VL model only
for image reading (step 1), that would require a small code change —
splitting `MODEL_NAME` back into two env vars in `app/config.py` and
picking the right one in `app/llm_client.py`'s two call functions — outside
the scope of the current single-model setup.

---

## Troubleshooting

### Streamlit shows "Connection error" / "server is not responding" on upload

This is a known Streamlit behavior: if the script is blocked on one very
long call, the browser tab's periodic connectivity check gets no response
and shows this popup. It's the exact symptom you'd hit with a big scanned
PDF and a slow local model.

Two things in this project already address it:

1. **Ingestion runs on a background thread**, and progress is shown by an
   `@st.fragment(run_every="1s")` (`scripts/knowledge_console.py`) that
   reruns only the small progress section — not the whole page — while a
   document is being processed. Even a large multi-page scanned PDF keeps
   the UI responsive and shows live "reading page X of Y" progress instead
   of one long silent wait.
2. **Ollama keeps the model loaded in memory between calls** (it doesn't
   reload it fresh each request), so after the first call in a session,
   later calls are noticeably faster.

If you still see it (e.g. on a very large scanned PDF, a large model on
CPU-only hardware, or a slow tunnel to a remote GPU machine):

- A local 7B vision model on CPU-only hardware can take a while per page —
  if this is consistently too slow, try `qwen2.5vl:3b` (see
  [Changing models](#changing-models)), or move inference to a GPU machine.
- Increase `MODEL_REQUEST_TIMEOUT` in `.env` if requests are timing out
  rather than completing slowly.
- Split a very large document (dozens of scanned pages) into a few smaller
  uploads rather than one huge one.
- Make sure `.streamlit/config.toml` is present (it ships with this repo)
  — it disables Streamlit's file watcher, which would otherwise restart
  the whole app mid-ingestion every time a file is written under `data/`.

### Console shows a red `NotFoundError: Failed to execute 'removeChild' on 'Node'` box

This is a Streamlit/React frontend bug, not an ingestion failure — it
happens when the page's DOM is rerun very rapidly and the browser's virtual
DOM diff gets out of sync (Streamlit's own docs note containers/fragments
as the fix for this class of issue, as opposed to a raw `st.rerun()` loop).
This project's progress polling already uses `st.fragment(run_every=...)`
specifically to avoid it — if you still hit it, click **Dismiss**, refresh
the browser tab, and check whether your ingestion actually completed (open
"Pending review" in the console or run `python -m app.cli review-list
<business_id>`) before re-submitting, since the underlying job on the
background thread is unaffected by a frontend rendering glitch.

### "Could not reach the model endpoint..." error

This means `llm_client.py` couldn't connect to Ollama. Work through the
checklist in the error message itself:

1. **Is Ollama running?** It normally runs as a background service after
   install; if not, run `ollama serve` in a terminal and leave it running.
2. **Have you pulled the model?** Run `ollama pull qwen2.5vl:7b` (or
   whichever tag you're using).
3. **Does the tag match exactly?** Run `ollama list` and compare against
   `MODEL_NAME` in `.env` — these must match character-for-character
   (e.g. `qwen2.5vl:7b`, not `qwen2.5-vl:7b`).
4. **Using a remote/tunnelled machine?** Confirm the tunnel is still up
   and that `MODEL_ENDPOINT_URL` in `.env` ends in `/v1` (Ollama's
   OpenAI-compatible path).

If you deleted `.env`, run `cp .env.example .env` to restore the local
defaults and restart.

### The error box just says "Ingestion failed:" with nothing after the colon

Fixed — some exceptions the underlying HTTP client raises (particularly
timeouts) come with no message attached, which used to show up as a blank
error. Errors now always show at least the exception type and a
plain-English explanation (see `app/errors.py`), and the full traceback is
always printed to the terminal running Streamlit/the API/the CLI, so you
can see exactly what happened even when the on-screen message is generic.
If you hit this specifically as a timeout, it usually just means the model
is slow on your hardware for that particular page/prompt — the retry logic
already retries this automatically a few times; if it still fails, try
again, raise `MODEL_REQUEST_TIMEOUT` in `.env`, or switch to a smaller
model tag (see [Changing models](#changing-models)).

### `404` / "model not found" error

This means `MODEL_NAME` in `.env` doesn't match any model Ollama actually
has pulled. Run `ollama list` to see the exact tags available, and make
sure `MODEL_NAME` matches one of them character-for-character (including
the `:7b`/`:3b`/`:32b` suffix — `qwen2.5vl` alone is not the same tag as
`qwen2.5vl:7b`). If it's missing entirely, pull it: `ollama pull
qwen2.5vl:7b`.

### A model call is very slow / seems stuck

Local inference speed depends entirely on your hardware. On CPU-only
machines, a 7B vision-language model can take a while per request,
especially on image-heavy pages. Options:

- Switch to a smaller tag: `ollama pull qwen2.5vl:3b`, then set
  `MODEL_NAME=qwen2.5vl:3b` in `.env`.
- Move inference to a machine with a GPU (see
  [Changing models](#changing-models) for the remote/tunnel setup).
- Raise `MODEL_REQUEST_TIMEOUT` in `.env` so slow-but-working requests
  don't get cut off and retried unnecessarily.

---

## Notes & Known Limits

- **Requires Ollama running and the model pulled.** If Ollama isn't
  running, or `MODEL_NAME` doesn't match a pulled tag, extraction,
  structuring, and answer calls will fail with a clear checklist error —
  `app/retry.py` retries a few times on transient errors before giving up.
  A remote/tunnelled setup additionally requires the tunnel to be up and
  `MODEL_ENDPOINT_URL` to be reachable.
- **Grounded answers only**: the system is prompted to say "I don't have
  that information" rather than guess — intentional, so wrong menu prices
  never get invented.
- **Menu items require review before they're orderable**: ingesting
  something extracts and structures it, but menu items sit in
  `pending_review` and are invisible to `menu-search`/`/menu-search` until
  a human calls `publish_review` — see "How It Works" above for why.
- **Multiple sources per business**: you can ingest a menu PDF, an "about
  us" text file, a pasted persona description, and a photo separately for
  the same `business_id` — once each is published, they all merge into one
  profile and one searchable knowledge base. Menu items are deduped by
  (category, item name); the most-recently-published version of a given
  item wins if it appears in more than one source.
- **First run is slower**: the local embedding model (~80MB) downloads
  once on first use, then runs fully offline/local after that.
- **Long documents aren't truncated**: text over ~12,000 characters is split
  into overlapping sections, each structured separately, then merged — so a
  big multi-page menu plus a separate persona doc won't silently lose
  content the way a hard character-limit truncation would.
- **Re-ingesting/re-publishing a corrected source** cleanly overwrites that
  source's previous menu-item embeddings — ids are deterministic per source
  file, so there are no duplicate/stale entries left behind after a
  correction is re-published.
- **`delete-business` / `DELETE /businesses/{id}`** removes a business's raw
  files, structured/review JSON, and both vector-store collections entirely
  — irreversible, meant for resets or fixing a mistaken onboarding.
- **Mixed text+image PDFs**: a PDF with a real text layer that also has
  embedded images carrying their own text (a photographed specials board,
  a handwritten insert, a stamped phone number) gets both — native text
  extraction runs as before, and any embedded image above a small
  size-filter is additionally read via the vision model and appended. Only
  fully-scanned PDFs (no usable text layer at all) use the
  whole-page-as-image fallback.
