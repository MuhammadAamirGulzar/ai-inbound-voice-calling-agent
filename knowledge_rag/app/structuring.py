"""
Structuring layer: takes raw extracted text  and asks the Qwen2.5-VL model (via Ollama on the GPU machine) to
convert it into one canonical JSON shape covering persona, business
policies, contact/operational details, and menu items.

Calls go through app/llm_client.py, which talks to the Ollama endpoint —
see that module and the README for setup.

Calls go through app/llm_client.py, which talks to the Hugging Face
Inference API — see that module and the README for setup.

Two things this version adds on top of a plain "text -> JSON" call:

1. A wider schema. Real business uploads (see: Clove Cafe's PDF) don't just
   contain a menu — they contain exact scripted greeting/closing lines, an
   upsell strategy, an out-of-stock protocol, a delivery-area list, and an
   escalation rule for difficult callers. If the schema doesn't have a slot
   for these, the LLM either drops them or mashes them into a free-text
   "notes" field, which the voice agent (Module 5) can't reliably act on.

2. Detection of embedded pre-structured JSON. Some businesses (again, Clove
   Cafe) hand over a document that already contains a JSON config block
   written for exactly this purpose. Blindly re-summarizing that block
   through an LLM risks paraphrasing an exact script line. So we detect any
   valid JSON object embedded in the raw text and pass it to the model as
   "authoritative, do not paraphrase" context, rather than ignoring it or
   silently losing it in freeform text.

3. Long-document handling. Previously, raw text was hard-truncated to 15,000
   chars before structuring — anything past that point was silently dropped.
   A business with a big multi-page menu plus a separate persona doc plus an
   "about us" file can easily exceed that. This version splits long text into
   overlapping sections, structures each, and merges the results instead of
   truncating.
"""
import json
import re
from typing import List, Optional

from . import llm_client

# Above this length, raw text is split into sections and structured
# separately, then merged — instead of being silently truncated.
#
# Why these are small: this pipeline defaults to a 7B local vision-language
# model (see app/llm_client.py / README). Models this size don't just fail
# by hitting the token cap on a huge menu — they also reliably STOP EARLY
# on their own, well before any cap, when asked to enumerate a long list of
# structured JSON objects in one completion (a well-known small-model
# failure mode for long repetitive structured output: attention drifts,
# and the model emits a clean, validly-closed JSON object after 60-80
# items instead of continuing to 190). That produces syntactically valid
# JSON, so finish_reason == "stop" and nothing here can detect it after
# the fact — the only real fix is to never ask for that many items in one
# call. Keeping each section small enough for ~15-25 menu items keeps this
# well inside what a small model can reliably complete in full.
MAX_SINGLE_PASS_CHARS = 2500
SECTION_SIZE = 2200
SECTION_OVERLAP = 200

SCHEMA_DESCRIPTION = """{
  "business_name": string,
  "persona": {
    "agent_name": string,
    "tone": string,
    "language": string,
    "greeting_style": string,
    "greeting_script": string,
    "closing_script": string,
    "notes": string
  },
  "details": {
    "address": string,
    "phone": string,
    "hours": string,
    "delivery_info": string,
    "delivery_areas": [string],
    "avg_preparation_time": string,
    "other": string
  },
  "policies": {
    "upsell_strategy": string,
    "out_of_stock_protocol": string,
    "escalation_protocol": string
  },
  "menu": [
    {
      "category": string,
      "item": string,
      "price": string,
      "description": string,
      "variants": [
        {"name": string, "price": string}
      ],
      "customizations": [string]
    }
  ]
}"""

STRUCTURING_PROMPT = """You are given raw extracted text belonging to a single restaurant/business (it may include a menu, an "about us" description, contact details, delivery info, an AI voice agent persona/script, or operating rules).

Convert it into STRICT JSON matching exactly this schema, with no extra commentary, no markdown fences:

""" + SCHEMA_DESCRIPTION + """

Rules:
- If a field is not present in the text, use an empty string "" (or an empty array [] for menu/variants/customizations/delivery_areas if none are found).
- "greeting_script" and "closing_script" are EXACT lines the voice agent should speak, if the source text provides them verbatim (e.g. a line labelled "Standard Greeting:"). Copy them exactly, word for word — do not paraphrase or summarize a script into a description. If the source only describes a tone/style rather than giving an exact line, leave these empty and put the description in "greeting_style" instead.
- "variants" means distinct sizes/versions of the SAME item that a customer must pick one of (e.g. "Regular", "Large"). Each variant needs its OWN price. This matters especially for lines like "Cappuccino RS. 745/825" — that is ONE item with TWO priced variants (e.g. Regular/Large), not one price. Infer reasonable variant names (Regular/Large, Half/Full, Small/Medium/Large) ONLY when a single item line clearly lists multiple prices; do not invent variants when only one price is given.
- "customizations" means optional add-ons/removals a customer can request for that item (e.g. "Extra cheese (+150)", "No onion"). Only include these if the source text actually lists them for that item.
- "policies.upsell_strategy", "policies.out_of_stock_protocol", and "policies.escalation_protocol" should capture any business rule about what the voice agent should DO in that situation (e.g. suggest a drink with a burger, suggest a close alternative when out of stock, offer to transfer a difficult caller) — paraphrase these into clear instructions if the source describes them narratively.
- "details.delivery_areas" should be a list of individual area names, not one combined string (e.g. ["Bahria Town", "DHA", "Chaklala"], not "Bahria Town, DHA, Chaklala").
- Do not invent prices, items, or details that are not present in the source text.
- Preserve prices exactly as written (currency symbol, formatting) for the top-level "price" and each variant's "price".
- Return ONLY the JSON object, nothing else — no markdown fences, no explanation.

{authoritative_block}

SOURCE TEXT:
---
{raw_text}
---
"""

AUTHORITATIVE_BLOCK_TEMPLATE = """
The following JSON block was found embedded verbatim in the source text. It is an AUTHORITATIVE, already-structured configuration written by the business for this exact purpose. Map its fields faithfully into the output schema above — especially exact scripted strings (greetings, closings) — WITHOUT paraphrasing them. Use it as the primary source of truth for any field it covers; use the surrounding source text only to fill in what it doesn't cover.

AUTHORITATIVE JSON BLOCK(S):
{blocks}
"""

_EMPTY_STRUCTURE = {
    "business_name": "",
    "persona": {
        "agent_name": "", "tone": "", "language": "", "greeting_style": "",
        "greeting_script": "", "closing_script": "", "notes": "",
    },
    "details": {
        "address": "", "phone": "", "hours": "", "delivery_info": "",
        "delivery_areas": [], "avg_preparation_time": "", "other": "",
    },
    "policies": {
        "upsell_strategy": "", "out_of_stock_protocol": "", "escalation_protocol": "",
    },
    "menu": [],
}


def _empty_structure() -> dict:
    return json.loads(json.dumps(_EMPTY_STRUCTURE))


def _repair_wrapped_json(candidate: str) -> str:
    """
    PDF/OCR text extraction preserves the document's visual line wrapping,
    which means a JSON string value that happened to wrap across two lines
    in the original layout (e.g. a long "greeting" sentence) comes out with
    a literal newline inside the quotes — which is invalid JSON (raw control
    characters aren't allowed unescaped inside a JSON string). A strict
    json.loads on a PDF-extracted JSON block will fail even though the
    block is "logically" valid.

    Fix: walk the candidate character by character, and while inside a
    (non-escaped) string, replace raw newlines/tabs with a single space
    instead of leaving them as literal control characters. Outside of
    strings, whitespace is left alone (it's insignificant to JSON anyway).
    """
    out = []
    in_string = False
    escaped = False
    for ch in candidate:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                out.append(ch)
                in_string = False
            elif ch in ("\n", "\r", "\t"):
                out.append(" ")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _find_embedded_json_blocks(raw_text: str) -> List[dict]:
    """
    Scan raw text for embedded, syntactically-valid JSON objects (e.g. a
    "Technical System Prompt (For RAG / Database Integration)" block a
    business pasted straight into their document). Uses brace-matching
    rather than a naive greedy regex, since documents can contain multiple
    such blocks or text that merely contains stray braces. Falls back to
    `_repair_wrapped_json` for candidates that fail to parse as-is, since
    PDF-extracted text commonly has line-wrapped strings inside the block.
    """
    blocks = []
    n = len(raw_text)
    i = 0
    while i < n:
        if raw_text[i] == "{":
            depth = 0
            j = i
            while j < n:
                if raw_text[j] == "{":
                    depth += 1
                elif raw_text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw_text[i:j + 1]
                        # Only treat "substantial" objects as candidates —
                        # skip trivial matches like "{}" or stray fragments.
                        if len(candidate) > 40:
                            parsed = None
                            try:
                                parsed = json.loads(candidate)
                            except json.JSONDecodeError:
                                try:
                                    parsed = json.loads(_repair_wrapped_json(candidate))
                                except json.JSONDecodeError:
                                    parsed = None
                            if isinstance(parsed, dict) and parsed:
                                blocks.append(parsed)
                        i = j
                        break
                j += 1
        i += 1
    return blocks


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _autoclose_json(candidate: str) -> str:
    """Repair JSON cut off mid-object (e.g. the model hit max_tokens before
    finishing). Tracks bracket depth outside of strings, closes an
    unterminated string, drops a dangling trailing comma, then appends
    whatever closing brace/bracket characters are needed to balance what
    is still open."""
    stack = []
    in_string = False
    escaped = False
    for ch in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
    repaired = candidate
    if in_string:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired.rstrip())
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _parse_json_loose(text: str) -> Optional[dict]:
    """Try increasingly permissive strategies before giving up on a text
    model response: a bare strict json.loads(), then the same after fixing
    literal newlines inside string values, then after slicing to the outer
    {...} span (in case of stray prose around the JSON), then after
    treating it as truncated output and auto-closing open brackets/strings.
    Real model output commonly fails a strict parse for reasons that don't
    mean the content is unusable, so each of these is worth trying before
    concluding the response is genuinely unparseable."""
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    first, last = stripped.find("{"), stripped.rfind("}")
    if first != -1 and last > first:
        candidates.append(stripped[first:last + 1])

    for cand in candidates:
        for variant in (cand, _repair_wrapped_json(cand)):
            try:
                parsed = json.loads(variant)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    for cand in candidates:
        repaired = _autoclose_json(_repair_wrapped_json(cand))
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


_FLAT_FIELD_PATHS = {
    "business_name": ("business_name",),
    "agent_name": ("persona", "agent_name"),
    "tone": ("persona", "tone"),
    "language": ("persona", "language"),
    "greeting_style": ("persona", "greeting_style"),
    "greeting_script": ("persona", "greeting_script"),
    "closing_script": ("persona", "closing_script"),
    "notes": ("persona", "notes"),
    "address": ("details", "address"),
    "phone": ("details", "phone"),
    "hours": ("details", "hours"),
    "delivery_info": ("details", "delivery_info"),
    "avg_preparation_time": ("details", "avg_preparation_time"),
    "other": ("details", "other"),
    "upsell_strategy": ("policies", "upsell_strategy"),
    "out_of_stock_protocol": ("policies", "out_of_stock_protocol"),
    "escalation_protocol": ("policies", "escalation_protocol"),
}


def _unescape_json_string_value(raw: str) -> str:
    try:
        return json.loads('"' + raw.replace("\n", "\\n").replace("\r", "") + '"')
    except json.JSONDecodeError:
        return raw.replace("\n", " ").replace("\r", "").strip()


def _best_effort_field_extract(text: str) -> dict:
    """Last-resort salvage when the response can't be coaxed into valid
    JSON at all. Regexes out any individual "key": "value" pairs
    recognized from the schema - these are almost always still intact even
    when the document as a whole doesn't parse - so the review/edit screen
    opens pre-filled with whatever the model did manage to produce,
    instead of a blank form."""
    result = _empty_structure()
    for key, path in _FLAT_FIELD_PATHS.items():
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if not match:
            continue
        value = _unescape_json_string_value(match.group(1))
        if len(path) == 1:
            result[path[0]] = value
        else:
            result[path[0]][path[1]] = value

    areas_match = re.search(r'"delivery_areas"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if areas_match:
        areas = re.findall(r'"((?:[^"\\]|\\.)*)"', areas_match.group(1))
        result["details"]["delivery_areas"] = [
            _unescape_json_string_value(a) for a in areas if a.strip()
        ]

    result["menu"] = _best_effort_menu_extract(text)
    return result


_MENU_ITEM_FIELD_KEYS = ("category", "item", "price", "description")


def _best_effort_menu_extract(text: str) -> list:
    """Salvage individual menu item objects out of a response that doesn't
    parse as JSON overall (e.g. broken partway through by a stray
    unescaped quote in one item, well before any token cap was hit).
    Without this, _best_effort_field_extract used to return an empty menu
    for the WHOLE response even when most items were written out fine -
    silently dropping every item instead of just the one that broke
    parsing. Finds the "menu": [ ... ] array via bracket-matching (so it
    isn't thrown off by nested [] in e.g. "variants"), then brace-matches
    each top-level object inside it and regexes out whatever flat fields
    that individual object has - one bad item just means that one item is
    skipped, not the whole array."""
    menu_start = re.search(r'"menu"\s*:\s*\[', text)
    if not menu_start:
        return []
    i = menu_start.end()
    depth = 1
    array_end = len(text)
    j = i
    while j < len(text) and depth > 0:
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                array_end = j
        j += 1
    array_body = text[i:array_end]

    items = []
    k = 0
    n = len(array_body)
    while k < n:
        if array_body[k] == "{":
            depth = 0
            start = k
            while k < n:
                if array_body[k] == "{":
                    depth += 1
                elif array_body[k] == "}":
                    depth -= 1
                    if depth == 0:
                        obj_text = array_body[start:k + 1]
                        item = {}
                        for key in _MENU_ITEM_FIELD_KEYS:
                            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', obj_text, re.DOTALL)
                            item[key] = _unescape_json_string_value(m.group(1)) if m else ""
                        item["variants"] = []
                        item["customizations"] = []
                        if item.get("item") or item.get("price"):
                            items.append(item)
                        k += 1
                        break
                k += 1
        else:
            k += 1
    return items


# Output token budget for one structuring call. Menu items are the field
# most likely to blow through this on a big menu (each item costs roughly
# 30-60 output tokens once you include the JSON punctuation/keys), so this
# is generous - but see the truncation-retry logic below, which is the real
# safety net: no fixed number here is safe for an arbitrarily large menu.
STRUCTURING_MAX_TOKENS = 16384

# If a section is still too big for the model to finish structuring even
# after bumping max_tokens, split it and structure each half separately
# rather than accepting a silently-truncated (i.e. missing menu items)
# result. Stop splitting below this size so we don't recurse forever on a
# pathological single huge unbroken line.
_MIN_RETRY_SPLIT_CHARS = 1500


def _structure_single_pass(raw_text: str, authoritative_blocks: Optional[List[dict]] = None,
                            max_tokens: int = STRUCTURING_MAX_TOKENS, _depth: int = 0) -> dict:
    if not raw_text.strip() and not authoritative_blocks:
        return _empty_structure()

    authoritative_section = ""
    if authoritative_blocks:
        authoritative_section = AUTHORITATIVE_BLOCK_TEMPLATE.replace(
            "{blocks}", json.dumps(authoritative_blocks, indent=2, ensure_ascii=False)
        )

    # NOTE: plain string.replace, not str.format — the schema/prompt text
    # above contains literal { } braces (the JSON schema itself), which
    # would otherwise collide with str.format's placeholder syntax.
    prompt = (
        STRUCTURING_PROMPT
        .replace("{authoritative_block}", authoritative_section)
        .replace("{raw_text}", raw_text)
    )
    raw_response, finish_reason = llm_client.chat_text_ex(prompt, temperature=0.0, max_tokens=max_tokens)
    text = _strip_fences(raw_response)

    if finish_reason == "length" and len(raw_text) > _MIN_RETRY_SPLIT_CHARS:
        # The model hit the max_tokens cap before finishing - the JSON is
        # genuinely incomplete (not just messy), so _autoclose_json patching
        # it up "successfully" would silently drop whatever menu items came
        # after the cutoff. Split this section in half by line and structure
        # each half separately instead, then merge - this is what actually
        # guarantees every item gets seen by some call, no matter how big
        # the source menu is.
        print(f"[structuring] Response was truncated (hit the {max_tokens}-token output cap) on a "
              f"{len(raw_text)}-char section — splitting it and retrying so nothing gets silently dropped.")
        lines = raw_text.splitlines(keepends=True)
        mid = len(lines) // 2 or 1
        left = "".join(lines[:mid])
        right = "".join(lines[mid:])
        left_blocks = [b for b in (authoritative_blocks or []) if json.dumps(b, sort_keys=True) in left]
        right_blocks = [b for b in (authoritative_blocks or []) if json.dumps(b, sort_keys=True) in right] \
            or (authoritative_blocks if not left_blocks else None)
        left_result = _structure_single_pass(left, left_blocks or None, max_tokens=max_tokens, _depth=_depth + 1)
        right_result = _structure_single_pass(right, right_blocks, max_tokens=max_tokens, _depth=_depth + 1)
        return _merge_structures(left_result, right_result)

    parsed = _parse_json_loose(text)
    if parsed is not None:
        if finish_reason == "length":
            print(f"[structuring] WARNING: response was truncated but still under the "
                  f"{_MIN_RETRY_SPLIT_CHARS}-char re-split floor — accepting the partial "
                  f"result as-is (some trailing menu items may be missing).")
        # Defensive merge against the empty structure so a partial/odd LLM
        # response never causes a KeyError downstream.
        merged = _empty_structure()
        _deep_merge(merged, parsed)
        return merged

    # Genuinely unparseable even after repair attempts - fall back to
    # regex-salvaged fields so the review screen opens with whatever could
    # be recovered pre-filled, rather than a blank form. The raw text is
    # still kept around (under _structuring_failed_raw_response) purely for
    # debugging; the UI decides how prominently to surface it.
    fallback = _best_effort_field_extract(text)
    fallback["_structuring_failed_raw_response"] = text
    return fallback


def _deep_merge(base: dict, incoming: dict) -> None:
    for key, value in (incoming or {}).items():
        if key not in base:
            continue
        if isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _merge_structures(a: dict, b: dict) -> dict:
    """Merge two structured outputs (used when a long document was split
    into sections). Menu items concatenate; scalar/text fields keep 'a's
    value unless empty, in which case 'b's value is used."""
    merged = _empty_structure()
    all_areas = []
    for src in (a, b):
        if src.get("business_name"):
            merged["business_name"] = src["business_name"]
        for k, v in (src.get("persona") or {}).items():
            if v:
                merged["persona"][k] = v
        for k, v in (src.get("details") or {}).items():
            if k == "delivery_areas":
                # list field: accumulate across sections instead of the
                # last section's list silently replacing earlier ones.
                all_areas.extend(v or [])
                continue
            if v:
                merged["details"][k] = v
        for k, v in (src.get("policies") or {}).items():
            if v:
                merged["policies"][k] = v
        merged["menu"].extend(src.get("menu") or [])
    # Sections are built with overlap (see SECTION_OVERLAP) so nothing gets
    # truncated at a boundary — but that same overlap means an item sitting
    # right at the boundary can appear, fully intact, in both adjacent
    # sections. Dedupe by (category, item, price) so it only shows up once
    # in the final merged menu instead of twice.
    seen_items = set()
    deduped_menu = []
    for item in merged["menu"]:
        key = (
            (item.get("category") or "").strip().lower(),
            (item.get("item") or "").strip().lower(),
            (item.get("price") or "").strip().lower(),
        )
        if key in seen_items:
            continue
        seen_items.add(key)
        deduped_menu.append(item)
    merged["menu"] = deduped_menu
    # dedupe delivery_areas while preserving order
    seen = set()
    areas = []
    for area in all_areas:
        if area not in seen:
            seen.add(area)
            areas.append(area)
    merged["details"]["delivery_areas"] = areas
    return merged


def _split_into_sections(text: str, target_size: int = SECTION_SIZE, overlap: int = SECTION_OVERLAP) -> List[str]:
    """Line-aware splitter for structuring only (separate from the generic
    char-based app/chunking.py used for Q&A embedding). Menu entries here
    are always on their own line(s) - name+price on one line, description
    on the next - so splitting on a raw character offset (the old approach)
    could slice an entry in half right at the section boundary and hand the
    model a dangling fragment on both sides. This instead accumulates whole
    lines up to ~target_size, so a section boundary always falls between
    two complete lines. `overlap` lines are repeated at the start of the
    next section (rather than an overlap byte count) for the same "don't
    cut mid-entry" reason; exact-match dedup in _merge_structures already
    handles items that end up appearing in both."""
    lines = text.splitlines()
    if not lines:
        return []
    sections = []
    start = 0
    n = len(lines)
    while start < n:
        size = 0
        end = start
        while end < n and (size == 0 or size + len(lines[end]) + 1 <= target_size):
            size += len(lines[end]) + 1
            end += 1
        sections.append("\n".join(lines[start:end]))
        if end >= n:
            break
        # step back a bit for overlap, measured in lines rather than a
        # fixed count so short/long menus both get a reasonable overlap
        overlap_lines = 0
        overlap_size = 0
        j = end - 1
        while j > start and overlap_size < overlap:
            overlap_size += len(lines[j]) + 1
            overlap_lines += 1
            j -= 1
        start = max(start + 1, end - overlap_lines)
    return sections


# A rough per-item signal independent of the model entirely: this style of
# menu marks essentially every priced line with "RS." (see New-Menu.pdf).
# Counting these in the raw text and comparing against the final structured
# menu count is a cheap, model-independent way to catch under-extraction
# that would otherwise only be noticed by manually counting the PDF - it
# won't catch every menu format, but printing both numbers costs nothing
# and makes the gap immediately visible in the logs instead of silent.
_PRICE_TAG_RE = re.compile(r"\bRS\.?\s*[\d,]+", re.IGNORECASE)


def _log_completeness_check(raw_text: str, menu_item_count: int) -> None:
    price_tags = len(_PRICE_TAG_RE.findall(raw_text))
    if price_tags == 0:
        return
    print(f"[structuring] Completeness check: found {price_tags} price-like tag(s) "
          f"(e.g. 'RS. 995') in the source text vs {menu_item_count} menu item(s) structured.")
    if menu_item_count < price_tags * 0.7:
        print(f"[structuring] WARNING: structured menu item count ({menu_item_count}) is well "
              f"below the price-tag count ({price_tags}) — this strongly suggests items were "
              f"dropped during structuring. Check the console output above for section-level "
              f"warnings, or try lowering SECTION_SIZE further.")


def structure_business_data(raw_text: str) -> dict:
    raw_text = raw_text or ""
    authoritative_blocks = _find_embedded_json_blocks(raw_text)

    if len(raw_text) <= MAX_SINGLE_PASS_CHARS:
        result = _structure_single_pass(raw_text, authoritative_blocks)
        _log_completeness_check(raw_text, len(result.get("menu") or []))
        return result

    print(f"[structuring] Raw text is {len(raw_text)} chars — splitting into "
          f"sections instead of a single pass.")
    sections = _split_into_sections(raw_text)
    print(f"[structuring] Split into {len(sections)} section(s) (~{SECTION_SIZE} chars each).")
    result = _empty_structure()
    for idx, section in enumerate(sections):
        print(f"[structuring]   structuring section {idx + 1}/{len(sections)} "
              f"({len(section)} chars) ...")
        # Only pass authoritative blocks to whichever section(s) actually contain them,
        # so they aren't duplicated/re-interpreted in every section.
        section_blocks = [b for b in authoritative_blocks if json.dumps(b, sort_keys=True) in section
                           or all(str(v) in section for v in _flatten_leaf_values(b)[:1])]
        section_result = _structure_single_pass(section, section_blocks or None)
        section_count = len(section_result.get("menu") or [])
        print(f"[structuring]     -> {section_count} menu item(s) from this section.")
        result = _merge_structures(result, section_result)
    _log_completeness_check(raw_text, len(result.get("menu") or []))
    return result


def _flatten_leaf_values(d: dict) -> List[str]:
    vals = []
    for v in d.values():
        if isinstance(v, dict):
            vals.extend(_flatten_leaf_values(v))
        elif isinstance(v, str):
            vals.append(v)
    return vals
