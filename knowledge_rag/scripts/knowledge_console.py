"""
Knowledge Console — the operator interface for this module.

Lets an operator submit a business's source material (typed/pasted text, a
file upload, a PDF, or a photo), review exactly what was extracted and
structured — including the complete menu — and explicitly confirm before
any of it goes live. This is the same ingest -> review -> publish pipeline
the voice call agent's API sits on top of (see app/ingest.py, main.py) —
this console is a human-facing front end for it, not a separate code path.

Run:
    streamlit run scripts/knowledge_console.py

--------------------------------------------------------------------------
Why ingestion runs in a background thread + st.fragment here
--------------------------------------------------------------------------
Streamlit shows a "Connection error / server is not responding" popup when
the script is blocked on a single long-running call for too long — the
browser tab's periodic health-check can't get a response. Extraction now
calls Qwen2.5-VL over the network (via Ollama on the GPU machine) instead of a purely local call, so most documents
finish in seconds, but a large multi-page scanned PDF can still take a
while. To keep the UI responsive no matter the document size, ingestion
runs on a background thread, and progress is polled by an `@st.fragment`
(see `_poll_ingestion_job` below) instead of a full-page `st.rerun()` loop.

An earlier version polled with a plain `time.sleep(1); st.rerun()` loop in
the main script body. That works, but reran the *entire* page every
second — including widgets whose `disabled` state was changing — which
can trigger a Streamlit/React frontend bug
(`NotFoundError: Failed to execute 'removeChild' on 'Node'`) from rapid
full-page DOM diffs. `st.fragment(run_every=...)` reruns only the small
polling section instead of the whole page, which avoids that class of bug
entirely and is the pattern Streamlit's own docs recommend for this exact
use case (auto-refreshing status/progress without a full app rerun).
"""
import html
import json
import os
import sys
import tempfile
import threading
import uuid

import pandas as pd

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import errors, ingest, rag, vectorstore  # noqa: E402

st.set_page_config(page_title="Business Knowledge RAG", layout="wide", page_icon="◆")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.block-container { padding-top: 2rem; max-width: 1180px; }

/* --- Header --- */
.app-header {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding-bottom: 1.1rem;
    margin-bottom: 1.75rem;
    border-bottom: 1px solid #E2E5EA;
}
.app-mark {
    width: 34px; height: 34px;
    border-radius: 8px;
    background: linear-gradient(135deg, #2F5EE0, #1B3E9E);
    color: white;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1.05rem;
    flex-shrink: 0;
}
.app-header h1 { font-size: 1.5rem; font-weight: 700; color: #14171F; margin: 0; letter-spacing: -0.01em; }
.app-header p { color: #6B7280; font-size: 0.88rem; margin: 0.15rem 0 0 0; }

/* --- Cards --- */
.card {
    background: #FFFFFF;
    border: 1px solid #E2E5EA;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}
.section-label {
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-size: 0.72rem;
    font-weight: 600;
    color: #8890A0;
    margin: 0;
}
.section-sub { color: #9AA2AF; font-size: 0.82rem; margin-top: 0.2rem; }

/* --- Status pills --- */
.status-pill {
    display: inline-block;
    padding: 0.24rem 0.75rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
}
.status-pending   { background: #FDF0DA; color: #96600D; }
.status-published { background: #E1F3E8; color: #1B7A44; }
.status-rejected  { background: #FBE4E4; color: #A5312F; }

/* --- Profile fields grid --- */
.field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.9rem 2rem; margin-bottom: 0.4rem; }
.field-label { color: #8890A0; font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.15rem; }
.field-value { color: #1B1F27; font-size: 0.93rem; line-height: 1.4; }
.field-value.empty { color: #B7BCC6; font-style: italic; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* --- Menu table --- */
.menu-table-wrap { border: 1px solid #EBEDF0; border-radius: 8px; overflow: hidden; }
table.menu-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
table.menu-table thead th {
    text-align: left;
    background: #F7F8FA;
    color: #6B7280;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.6rem 0.9rem;
    border-bottom: 1px solid #E2E5EA;
}
table.menu-table td {
    padding: 0.6rem 0.9rem;
    border-bottom: 1px solid #F0F1F3;
    color: #262B34;
    vertical-align: top;
}
table.menu-table tr:last-child td { border-bottom: none; }
table.menu-table tr:hover td { background: #FAFBFC; }
table.menu-table td.item-name { font-weight: 600; color: #14171F; }
table.menu-table td.price { font-family: 'JetBrains Mono', monospace; font-size: 0.83rem; white-space: nowrap; }
.muted { color: #B7BCC6; font-style: italic; }

/* --- Q&A --- */
.qa-block { border-left: 3px solid #2F5EE0; padding: 0.1rem 0 0.1rem 1rem; margin-bottom: 1.5rem; }
.qa-question { font-weight: 600; color: #14171F; margin-bottom: 0.4rem; font-size: 0.98rem; }
.qa-answer { color: #262B34; font-size: 0.95rem; line-height: 1.55; }
.qa-sources { color: #8890A0; font-size: 0.78rem; margin-top: 0.5rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

STATUS_LABELS = {
    "pending_review": ("Pending confirmation", "status-pending"),
    "published": ("Published", "status-published"),
    "rejected": ("Rejected", "status-rejected"),
}


def status_pill(status: str) -> str:
    label, css_class = STATUS_LABELS.get(status, (status, "status-pending"))
    return f'<span class="status-pill {css_class}">{label}</span>'


def esc(value) -> str:
    return html.escape(str(value)) if value else ""


def field_html(label: str, value) -> str:
    if value:
        return f'<div><div class="field-label">{esc(label)}</div><div class="field-value">{esc(value)}</div></div>'
    return f'<div><div class="field-label">{esc(label)}</div><div class="field-value empty">Not found in source</div></div>'


def render_menu_table(menu_items: list) -> str:
    rows = []
    for item in menu_items:
        variants = item.get("variants") or []
        if variants:
            price = "<br>".join(
                f"{esc(v.get('name', ''))}: {esc(v.get('price', '') or '—')}" for v in variants
            )
        else:
            price = esc(item.get("price")) or '<span class="muted">—</span>'
        description = esc(item.get("description")) or '<span class="muted">—</span>'
        customizations = ", ".join(esc(c) for c in (item.get("customizations") or [])) or '<span class="muted">—</span>'
        rows.append(f"""
            <tr>
                <td>{esc(item.get('category')) or '<span class="muted">—</span>'}</td>
                <td class="item-name">{esc(item.get('item')) or '<span class="muted">—</span>'}</td>
                <td class="price">{price}</td>
                <td>{description}</td>
                <td>{customizations}</td>
            </tr>
        """)
    return f"""
    <div class="menu-table-wrap">
        <table class="menu-table">
            <thead>
                <tr><th>Category</th><th>Item</th><th>Price</th><th>Description</th><th>Customizations</th></tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """


def _start_file_ingestion(business_id: str, tmp_path: str) -> None:
    """Kick off ingest.ingest_file on a background thread and store a
    live-updating job dict in session_state. See the module docstring for
    why this runs in a thread instead of blocking the script directly."""
    job = {"id": uuid.uuid4().hex, "status": "running", "progress": None, "result": None, "error": None}
    st.session_state.ingestion_job = job

    def progress_cb(current_page: int, total_pages: int) -> None:
        job["progress"] = (current_page, total_pages)

    def worker():
        try:
            job["result"] = ingest.ingest_file(business_id, tmp_path, progress_cb=progress_cb)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001 — surface any failure to the UI rather than crashing the thread silently
            errors.log_exception("file ingestion", e)
            job["error"] = errors.describe_exception(e)
            job["status"] = "error"
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    threading.Thread(target=worker, daemon=True).start()


def _start_text_ingestion(business_id: str, text: str, source_name: str) -> None:
    """Same as _start_file_ingestion, for pasted/typed raw text instead of
    a file. Text ingestion skips OCR entirely so it's normally fast, but it
    still goes through the same background-job path for a consistent UI
    and so a large paste can't block the app either."""
    job = {"id": uuid.uuid4().hex, "status": "running", "progress": None, "result": None, "error": None}
    st.session_state.ingestion_job = job

    def worker():
        try:
            job["result"] = ingest.ingest_text(business_id, text, source_name=source_name)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            errors.log_exception("text ingestion", e)
            job["error"] = errors.describe_exception(e)
            job["status"] = "error"

    threading.Thread(target=worker, daemon=True).start()


st.markdown(
    """
    <div class="app-header">
        <div class="app-mark">◆</div>
        <div>
            <h1>Business Knowledge RAG</h1>
            <p>Extraction, structuring, and grounded Q&amp;A for the voice call agent's knowledge base.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "pending_record" not in st.session_state:
    st.session_state.pending_record = None
if "last_upload_key" not in st.session_state:
    st.session_state.last_upload_key = None
if "ingestion_job" not in st.session_state:
    st.session_state.ingestion_job = None

business_id = st.text_input(
    "Business ID",
    value="demo_business",
    help="A short unique identifier, e.g. 'clove_cafe'. Each business is kept in its own isolated knowledge base.",
)

if st.session_state.get("_last_business_id") != business_id:
    st.session_state.qa_history = []
    st.session_state.pending_record = None
    st.session_state.last_upload_key = None
    st.session_state.ingestion_job = None
    st.session_state._last_business_id = business_id

# ---------------------------------------------------------------------------
# Document intake — either upload a file, or paste/type text directly
# ---------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Document intake</p>', unsafe_allow_html=True)

job = st.session_state.ingestion_job
job_running = bool(job and job["status"] == "running")

intake_mode = st.radio(
    "Intake method",
    ["Upload a file", "Paste text"],
    horizontal=True,
    label_visibility="collapsed",
    disabled=job_running,
)

if intake_mode == "Upload a file":
    uploaded = st.file_uploader(
        "Menu, persona, hours, or policy document",
        type=["txt", "pdf", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        disabled=job_running,
    )
    st.markdown(
        '<div class="section-sub">Accepts a text file, a PDF, or a photo of a printed document. '
        'Scanned pages and photos are read automatically.</div>',
        unsafe_allow_html=True,
    )

    if uploaded is not None and not job_running:
        upload_key = f"{business_id}:{uploaded.name}:{uploaded.size}"
        if st.session_state.last_upload_key != upload_key:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            st.session_state.last_upload_key = upload_key
            _start_file_ingestion(business_id, tmp_path)
            st.rerun()

else:  # Paste text
    pasted_name = st.text_input(
        "Label for this text (optional)",
        value="pasted_menu",
        help="Just used to name the saved record — e.g. 'menu', 'persona_script'.",
        disabled=job_running,
    )
    pasted_text = st.text_area(
        "Paste or type the business's menu, persona script, hours, or policies",
        height=220,
        placeholder="e.g.\nClove Cafe\nAddress: ...\nHours: ...\n\nMenu:\nCappuccino - Rs. 450\n...",
        label_visibility="collapsed",
        disabled=job_running,
    )
    submit_text = st.button("Ingest this text", type="primary", disabled=job_running or not pasted_text.strip())
    if submit_text:
        _start_text_ingestion(business_id, pasted_text, pasted_name or "pasted_text")
        st.rerun()

# ---------------------------------------------------------------------------
# Background ingestion progress — polls the job started above via a
# fragment (reruns only this small section, not the whole page) instead of
# a full-page sleep+rerun loop. See the module docstring for why: a
# full-page rerun loop can trigger a Streamlit/React DOM-diffing bug
# ("NotFoundError: removeChild"), and a fragment avoids it by construction.
#
# The `st.empty()` placeholder + explicit widget `key=` below are an extra
# safety net for the same class of bug: without them, a widget that only
# appears in one branch (the "Dismiss" button, only shown on error) can
# occasionally leave a stale copy on screen across fragment reruns if the
# browser doesn't fully unmount it. Drawing into a fresh `.container()`
# each run guarantees the previous run's content is cleared first.
# ---------------------------------------------------------------------------
@st.fragment(run_every="1s")
def _poll_ingestion_job():
    job = st.session_state.ingestion_job
    placeholder = st.empty()
    if job is None:
        placeholder.empty()
        return
    with placeholder.container():
        if job["status"] == "running":
            progress = job["progress"]
            if progress:
                current, total = progress
                st.info(f"Reading page {current} of {total} with the Qwen2.5-VL vision model...")
            else:
                st.info("Extracting and structuring — this calls the Qwen2.5-VL model over your tunnel, so it's normally quick.")
        elif job["status"] == "error":
            st.error(f"Ingestion failed: {job['error']}")
            if st.button("Dismiss", key=f"dismiss_{job['id']}"):
                st.session_state.ingestion_job = None
                st.rerun()  # full-page rerun: re-enables the intake widgets above
        elif job["status"] == "done":
            st.session_state.pending_record = job["result"]
            st.session_state.ingestion_job = None
            st.rerun()  # full-page rerun: shows the review card below with the new record


if st.session_state.ingestion_job is not None:
    _poll_ingestion_job()

st.markdown("</div>", unsafe_allow_html=True)

def render_profile_preview(data: dict) -> None:
    """Read-only, presentable view of everything extracted for one
    ingestion: free-text fields as labelled paragraphs (grouped under
    Persona / Details / Policies headers), the menu as a real table. This
    is what an operator sees first — 'Edit' switches to the form fields
    further down; nothing here is editable."""
    persona = data.get("persona", {}) or {}
    details = data.get("details", {}) or {}
    policies = data.get("policies", {}) or {}
    menu = data.get("menu", []) or []

    st.markdown(f'<div class="field-grid" style="margin-bottom:1.1rem;">{field_html("Business name", data.get("business_name"))}</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label" style="margin-bottom:0.6rem;">Persona</p>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="field-grid">
            {field_html("Agent name", persona.get("agent_name"))}
            {field_html("Tone", persona.get("tone"))}
            {field_html("Language", persona.get("language"))}
            {field_html("Greeting style", persona.get("greeting_style"))}
        </div>
        <div class="field-grid" style="margin-top:0.9rem;">
            {field_html("Greeting script (exact)", persona.get("greeting_script"))}
            {field_html("Closing script (exact)", persona.get("closing_script"))}
        </div>
        <div style="margin-top:0.9rem;">{field_html("Notes", persona.get("notes"))}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-label" style="margin:1.3rem 0 0.6rem;">Details</p>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="field-grid">
            {field_html("Address", details.get("address"))}
            {field_html("Phone", details.get("phone"))}
            {field_html("Hours", details.get("hours"))}
            {field_html("Average preparation time", details.get("avg_preparation_time"))}
            {field_html("Delivery info", details.get("delivery_info"))}
            {field_html("Delivery areas", ", ".join(details.get("delivery_areas") or []))}
        </div>
        <div style="margin-top:0.9rem;">{field_html("Other", details.get("other"))}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-label" style="margin:1.3rem 0 0.6rem;">Policies</p>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="field-grid">
            {field_html("Upsell strategy", policies.get("upsell_strategy"))}
            {field_html("Out-of-stock protocol", policies.get("out_of_stock_protocol"))}
            {field_html("Escalation protocol", policies.get("escalation_protocol"))}
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-label" style="margin:1.3rem 0 0.6rem;">Menu</p>', unsafe_allow_html=True)
    if menu:
        st.markdown(render_menu_table(menu), unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-sub">No menu items were found in this document.</div>', unsafe_allow_html=True)


record = st.session_state.pending_record
if record is not None:
    data = record.get("data", {})
    persona = data.get("persona", {})
    details = data.get("details", {})
    policies = data.get("policies", {})
    menu = data.get("menu", [])
    ingestion_id = record["ingestion_id"]
    k = lambda name: f"{ingestion_id}__{name}"  # noqa: E731 — short-lived local helper
    mode_key = f"view_mode__{ingestion_id}"
    if mode_key not in st.session_state:
        # If the structuring model's response didn't fully parse, some
        # fields may be missing/wrong — open straight into the editable
        # form (pre-filled with whatever was recovered) instead of a
        # preview screen that has nothing useful to show for those fields.
        st.session_state[mode_key] = "edit" if data.get("_structuring_failed_raw_response") else "preview"
    view_mode = st.session_state[mode_key]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card-header">
            <p class="section-label">Extracted profile — {esc(record['source_file'])}</p>
            {status_pill(record['status'])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if record.get("warning"):
        if data.get("_structuring_failed_raw_response"):
            st.info(
                "We couldn't fully auto-extract this document — some fields below may be blank "
                "or incomplete. Please check them over and fill in anything missing before "
                "confirming."
            )
            # The model's raw response is never shown in the UI — only ever
            # useful for debugging, never for an operator, and it can
            # contain raw source markup (e.g. HTML table tags from the
            # original document). Goes to the terminal log instead.
            print(
                f"[knowledge_console] Structuring fallback triggered for "
                f"'{record['source_file']}' — raw model output:\n"
                f"{data['_structuring_failed_raw_response']}"
            )
        else:
            st.warning(record["warning"])

    if view_mode == "preview":
        st.markdown(
            '<div class="section-sub">This is what was extracted, formatted for review — free-text fields as paragraphs, '
            'the menu as a table. Click "Edit" to correct anything before confirming.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-sub">Editing — correct anything wrong below, then save. Nothing changes until you click Save / Confirm.</div>', unsafe_allow_html=True)
    st.write("")

    if view_mode == "preview":
        render_profile_preview(data)
        st.write("")
        preview_edit_col, preview_publish_col, preview_reject_col, _ = st.columns([1, 1.4, 1, 2.6])
        if preview_edit_col.button("Edit", key=k("go_edit")):
            st.session_state[mode_key] = "edit"
            st.rerun()
        if record["status"] == "pending_review":
            if preview_publish_col.button("Confirm and publish as-is", type="primary", key=k("publish_as_is")):
                updated = ingest.publish_review(business_id, ingestion_id, corrected_data=None)
                st.session_state.pending_record = updated
                st.success("Published.")
                st.rerun()
            if preview_reject_col.button("Reject", key=k("reject_preview")):
                updated = ingest.reject_review(business_id, ingestion_id, reason="Rejected from the knowledge console")
                st.session_state.pending_record = updated
                st.rerun()
        elif record["status"] == "rejected":
            st.markdown('<div class="section-sub">This ingestion was rejected and is not part of the live knowledge base.</div>', unsafe_allow_html=True)

    if view_mode == "edit":
        if st.button("← Back to preview", key=k("back_to_preview")):
            st.session_state[mode_key] = "preview"
            st.rerun()
        st.write("")

        col1, col2 = st.columns(2)
        with col1:
            business_name = st.text_input("Business name", value=data.get("business_name") or "", key=k("business_name"))
            agent_name = st.text_input("Agent persona", value=persona.get("agent_name") or "", key=k("agent_name"))
            tone = st.text_input("Tone", value=persona.get("tone") or "", key=k("tone"))
            language = st.text_input("Language", value=persona.get("language") or "", key=k("language"))
            address = st.text_input("Address", value=details.get("address") or "", key=k("address"))
            phone = st.text_input("Phone", value=details.get("phone") or "", key=k("phone"))
            hours = st.text_input("Hours", value=details.get("hours") or "", key=k("hours"))
        with col2:
            avg_prep = st.text_input("Average preparation time", value=details.get("avg_preparation_time") or "", key=k("avg_prep"))
            delivery_areas = st.text_input(
                "Delivery areas (comma-separated)",
                value=", ".join(details.get("delivery_areas") or []),
                key=k("delivery_areas"),
            )
            delivery_info = st.text_input("Delivery info", value=details.get("delivery_info") or "", key=k("delivery_info"))
            upsell = st.text_area("Upsell strategy", value=policies.get("upsell_strategy") or "", key=k("upsell"), height=80)
            out_of_stock = st.text_area("Out-of-stock protocol", value=policies.get("out_of_stock_protocol") or "", key=k("oos"), height=80)
            escalation = st.text_area("Escalation protocol", value=policies.get("escalation_protocol") or "", key=k("escalation"), height=80)

        notes = st.text_area("Notes", value=persona.get("notes") or "", key=k("notes"), height=80)

        with st.expander("Greeting and closing scripts", expanded=bool(persona.get("greeting_script") or persona.get("closing_script"))):
            greeting_script = st.text_area("Greeting (exact words the agent should say)", value=persona.get("greeting_script") or "", key=k("greeting"))
            closing_script = st.text_area("Closing (exact words the agent should say)", value=persona.get("closing_script") or "", key=k("closing"))

        st.markdown('<div class="field-label" style="margin-top:0.6rem;">Menu — edit prices/items directly in the table, add rows at the bottom, or delete a row</div>', unsafe_allow_html=True)
        menu_rows = [
            {
                "category": it.get("category", "") or "",
                "item": it.get("item", "") or "",
                "price": it.get("price", "") or "",
                "description": it.get("description", "") or "",
                "customizations": ", ".join(it.get("customizations") or []),
                "variants_json": json.dumps(it.get("variants") or [], ensure_ascii=False),
            }
            for it in menu
        ]
        menu_df = pd.DataFrame(
            menu_rows,
            columns=["category", "item", "price", "description", "customizations", "variants_json"],
        )
        edited_menu_df = st.data_editor(
            menu_df,
            num_rows="dynamic",
            use_container_width=True,
            key=k("menu_editor"),
            column_config={
                "variants_json": st.column_config.TextColumn(
                    "Variants (JSON)", help='e.g. [{"name":"Regular","price":"Rs. 450"},{"name":"Large","price":"Rs. 550"}]'
                ),
                "customizations": st.column_config.TextColumn("Customizations (comma-separated)"),
            },
        )

        def _build_corrected_data() -> dict:
            edited_menu = []
            for _, row in edited_menu_df.iterrows():
                category = str(row.get("category") or "").strip()
                item = str(row.get("item") or "").strip()
                if not category and not item:
                    continue  # skip fully blank rows added via "+" and left empty
                try:
                    variants = json.loads(row.get("variants_json") or "[]")
                    if not isinstance(variants, list):
                        variants = []
                except (json.JSONDecodeError, TypeError):
                    variants = []
                customizations = [c.strip() for c in str(row.get("customizations") or "").split(",") if c.strip()]
                edited_menu.append({
                    "category": category,
                    "item": item,
                    "price": str(row.get("price") or "").strip(),
                    "description": str(row.get("description") or "").strip(),
                    "variants": variants,
                    "customizations": customizations,
                })
            return {
                "business_name": business_name.strip(),
                "persona": {
                    "agent_name": agent_name.strip(),
                    "tone": tone.strip(),
                    "language": language.strip(),
                    "greeting_style": persona.get("greeting_style", ""),
                    "greeting_script": greeting_script.strip(),
                    "closing_script": closing_script.strip(),
                    "notes": notes.strip(),
                },
                "details": {
                    "address": address.strip(),
                    "phone": phone.strip(),
                    "hours": hours.strip(),
                    "delivery_info": delivery_info.strip(),
                    "delivery_areas": [a.strip() for a in delivery_areas.split(",") if a.strip()],
                    "avg_preparation_time": avg_prep.strip(),
                    "other": details.get("other", ""),
                },
                "policies": {
                    "upsell_strategy": upsell.strip(),
                    "out_of_stock_protocol": out_of_stock.strip(),
                    "escalation_protocol": escalation.strip(),
                },
                "menu": edited_menu,
            }

        st.write("")
        if record["status"] == "pending_review":
            st.markdown(
                '<div class="section-sub">This entire profile — persona, details, policies, and every menu item above — '
                'is held here until confirmed. Nothing is searchable by the voice agent until you publish it.</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            confirm_col, reject_col, _ = st.columns([1, 1, 3])
            if confirm_col.button("Confirm and publish", type="primary"):
                corrected_data = _build_corrected_data()
                updated = ingest.publish_review(business_id, ingestion_id, corrected_data=corrected_data)
                st.session_state.pending_record = updated
                st.session_state[mode_key] = "preview"
                st.success("Published with your edits.")
                st.rerun()
            if reject_col.button("Reject"):
                updated = ingest.reject_review(business_id, ingestion_id, reason="Rejected from the knowledge console")
                st.session_state.pending_record = updated
                st.session_state[mode_key] = "preview"
                st.rerun()
        elif record["status"] == "published":
            st.markdown('<div class="section-sub">This document is already live. You can still edit the fields above and save changes — the live data updates immediately.</div>', unsafe_allow_html=True)
            st.write("")
            if st.button("Save changes", type="primary"):
                corrected_data = _build_corrected_data()
                updated = ingest.publish_review(business_id, ingestion_id, corrected_data=corrected_data)
                st.session_state.pending_record = updated
                st.session_state[mode_key] = "preview"
                st.success("Saved.")
                st.rerun()
        elif record["status"] == "rejected":
            st.markdown('<div class="section-sub">This ingestion was rejected and is not part of the live knowledge base.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Ask a question
# ---------------------------------------------------------------------------
collection = vectorstore.get_collection(business_id)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Ask a question</p>', unsafe_allow_html=True)

if collection.count() == 0:
    st.markdown('<div class="section-sub">No knowledge base yet for this business — add a document above first.</div>', unsafe_allow_html=True)
else:
    # The question box + history live in their own fragment so that asking a
    # question only reruns this section instead of the whole page. Mixing a
    # full-page rerun (which a plain st.button triggers) with everything
    # else on the page re-rendering at once is what was causing the
    # "NotFoundError: removeChild" crash — same class of bug as the
    # ingestion-polling fragment above, same fix.
    @st.fragment
    def _ask_a_question(business_id: str):
        processing_key = f"qa_processing__{business_id}"
        pending_key = f"qa_pending_question__{business_id}"
        if processing_key not in st.session_state:
            st.session_state[processing_key] = False

        is_processing = st.session_state[processing_key]

        # A form (rather than a bare text_input + button) submits the
        # question and clears the box in one atomic step. The submit button
        # is disabled while an answer is being generated — the LLM call
        # takes a few seconds, and without this an impatient extra click
        # (or a slow network double-firing the click) submits the SAME
        # question a second time before the box has a chance to clear,
        # which is what was producing identical answers appearing 2-3
        # times in a row.
        with st.form(key=f"ask_form__{business_id}", clear_on_submit=True):
            question = st.text_input(
                "Question",
                placeholder="e.g. What are your delivery timings?",
                label_visibility="collapsed",
                disabled=is_processing,
            )
            ask_clicked = st.form_submit_button(
                "Ask" if not is_processing else "Answering…",
                type="primary",
                disabled=is_processing,
            )

        if st.button("Clear history", key=f"clear_history__{business_id}", disabled=is_processing):
            st.session_state.qa_history = []
            st.rerun(scope="fragment")

        # Phase 1: a click was registered and nothing is already in flight —
        # lock immediately and rerun so the button visibly disables *before*
        # the (slow) LLM call starts, closing the double-click window.
        if ask_clicked and question and not is_processing:
            st.session_state[processing_key] = True
            st.session_state[pending_key] = question
            st.rerun(scope="fragment")

        # Phase 2: the lock is on and there's a pending question — this is
        # the only place the actual LLM call happens, and it only ever runs
        # once per lock/unlock cycle.
        if is_processing and st.session_state.get(pending_key):
            pending_question = st.session_state[pending_key]
            with st.spinner("Retrieving context and generating an answer..."):
                result = rag.answer_question(business_id, pending_question)
            # Defensive backstop: never append an exact repeat of the turn
            # that's already on top of the history.
            history = st.session_state.qa_history
            if not history or history[-1]["question"] != pending_question or history[-1]["answer"] != result["answer"]:
                history.append({
                    "question": pending_question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                })
            st.session_state[processing_key] = False
            st.session_state[pending_key] = None
            st.rerun(scope="fragment")

        if st.session_state.qa_history:
            st.write("")
        for turn in reversed(st.session_state.qa_history):
            st.markdown('<div class="qa-block">', unsafe_allow_html=True)
            st.markdown(f'<div class="qa-question">{esc(turn["question"])}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="qa-answer">{esc(turn["answer"])}</div>', unsafe_allow_html=True)
            if turn["sources"]:
                st.markdown(f'<div class="qa-sources">Sources: {esc(", ".join(turn["sources"]))}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    _ask_a_question(business_id)

st.markdown("</div>", unsafe_allow_html=True)
