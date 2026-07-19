"""
Extraction layer: turns any input file (.txt, .pdf, .png/.jpg/.jpeg/.webp)
into raw text, using the cheapest reliable method available:

  .txt              -> read directly
  .pdf (has text)   -> native extraction via PyMuPDF (no model call needed)
  .pdf (scanned)    -> rendered to page images, then read via the
                       Hugging Face-hosted vision model
  .png/.jpg/.jpeg   -> read via the same Hugging Face-hosted vision model

This is the single place that decides "how do I turn this file into text" —
everything downstream (structuring, chunking, embedding) just works on text.

All vision/OCR calls go through app/llm_client.py, which talks to the
Hugging Face Inference API — see that module and the README for setup.
"""
import io
import os

import fitz  # PyMuPDF
from PIL import Image

from . import llm_client

VISION_EXTRACTION_PROMPT = """You are reading a photo or scanned page belonging to a restaurant/business's menu, profile, or information sheet.

Carefully transcribe ALL visible text exactly as written, preserving item names, prices, and structure/layout as best as possible using plain text (use line breaks and simple dashes for structure, no markdown tables).

Return ONLY the transcribed text. Do not summarize, do not add commentary, do not translate unless the text is unreadable in its original script. If the image is purely decorative (a logo mark, a plain photo, a background pattern) and has no readable text at all, respond with exactly: NONE"""


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# If the shorter side of an image is below this, upscale it before sending —
# small/low-res photos of menus make small price text hard for the vision
# model to read accurately, and there's otherwise no preprocessing step.
_MIN_SHORT_SIDE = 1400


def _prepare_image_for_vision(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    short_side = min(img.size)
    if short_side < _MIN_SHORT_SIDE:
        scale = _MIN_SHORT_SIDE / short_side
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _vision_transcribe(img: Image.Image, max_tokens: int = 8192) -> str:
    prepared = _prepare_image_for_vision(img)
    return llm_client.chat_vision(VISION_EXTRACTION_PROMPT, prepared, temperature=0.0, max_tokens=max_tokens).strip()


def extract_text_from_image(file_path: str) -> str:
    """Use the Hugging Face-hosted vision model to transcribe a menu/profile photo."""
    return _vision_transcribe(Image.open(file_path))


def _native_pdf_text(file_path: str) -> str:
    """Fast path: pull text directly out of the PDF's text layer (no API call, no cost)."""
    text_parts = []
    doc = fitz.open(file_path)
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts).strip()


def _pages_with_images(file_path: str, min_dim: int = 150, min_area: int = 40000):
    """
    Identify which pages of a text-native PDF contain at least one
    "real" (non-decorative) embedded raster image — e.g. a photographed
    specials board, a scanned menu page, or a handwritten insert pasted
    into an otherwise typed document. The page's normal text layer never
    captures that image's text, which is exactly the gap this fills.

    Returns a sorted list of 0-based page indices.

    Deliberately page-level rather than per-embedded-image: earlier this
    extracted each embedded image individually via `Document.extract_image`
    and OCR'd it on its own. That approach silently dropped images whose
    encoding `extract_image` can't handle (CMYK JPEGs, JPX, some inline
    masks — it raises and the image was just skipped), and could split one
    photo's menu text across several separately-OCR'd tiles when an image
    was stored as multiple stacked XObjects, losing items at the tile
    boundary. Rendering the WHOLE PAGE once instead sidesteps both classes
    of bug — it reads the page exactly as it visually looks, the same
    proven method already used for fully-scanned PDFs below — at the cost
    of a couple of extra (but reliable) vision calls per document.

    `min_dim`/`min_area` still filter out pages whose only images are tiny
    decorative elements (icons, bullets, divider lines, small logos) not
    worth a vision call.
    """
    doc = fitz.open(file_path)
    pages = []
    for page_index, page in enumerate(doc):
        for img_info in page.get_images(full=True):
            width, height = img_info[2], img_info[3]
            if width >= min_dim and height >= min_dim and (width * height) >= min_area:
                pages.append(page_index)
                break
    doc.close()
    return pages


def _render_page(file_path: str, page_index: int, zoom: float = 2.0) -> Image.Image:
    doc = fitz.open(file_path)
    try:
        page = doc[page_index]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()


def extract_text_from_pdf(file_path: str, native_text_min_chars: int = 200, progress_cb=None, stats: dict = None) -> str:
    """
    Try native text extraction first (fast, free, no API call).

    If the PDF looks scanned/image-based (little to no extractable text),
    fall back to rendering each whole page as an image and running it
    through the Hugging Face vision model — this is what makes messy
    scanned menu PDFs work.

    If the PDF DOES have a real text layer, it can still contain pages with
    embedded images that carry their own text (a photographed specials
    board, a handwritten insert, a stamped phone number, a page that's
    mostly a photo of a menu board) — the text layer alone never captures
    that. So in that case we additionally render each such page (whole page,
    not just the individual embedded image — see `_pages_with_images` for
    why) and OCR it via the vision model, then append any text found to the
    native text. This is the fix for PDFs that mix typed text with
    photos/scans of text.

    Every page read via vision is wrapped in its own try/except: one page
    failing (a transient tunnel/model error even after retries) no longer
    silently drops that page's content OR aborts the whole document —
    it's logged clearly and extraction continues with the remaining pages,
    and the final counts are printed so a partial result is easy to spot.

    `progress_cb`, if given, is called as progress_cb(current_page, total_pages)
    after each page during the vision fallback — used by the Streamlit
    console to show live progress on multi-page scans instead of a single
    long silent wait.

    `stats`, if given (a plain dict), is filled in with counts describing
    exactly what happened during extraction — pages found/read/blank/failed
    — so the caller (see app/ingest.py) can show the operator something
    concrete like "read 17 of 17 image pages" instead of that information
    only existing in the server's terminal log.
    """
    if stats is None:
        stats = {}
    native_text = _native_pdf_text(file_path)

    if len(native_text) < native_text_min_chars:
        print("[extraction] PDF has no usable text layer — falling back to vision extraction per page.")
        doc = fitz.open(file_path)
        total_pages = doc.page_count
        doc.close()
        page_texts = []
        failed_pages = []
        for i in range(total_pages):
            print(f"[extraction]   reading page {i + 1}/{total_pages} via vision model...")
            try:
                img = _render_page(file_path, i)
                page_texts.append(_vision_transcribe(img))
            except Exception as e:
                print(f"[extraction]   WARNING: page {i + 1}/{total_pages} failed to read ({e!r}) — "
                      f"continuing with the remaining pages.")
                failed_pages.append(i + 1)
            if progress_cb:
                progress_cb(i + 1, total_pages)
        if failed_pages:
            print(f"[extraction] WARNING: {len(failed_pages)} of {total_pages} page(s) could not be read: "
                  f"{failed_pages}. Re-uploading this file (or retrying) may pick these up.")
        stats.update({
            "mode": "scanned",
            "pages_total": total_pages,
            "pages_read": total_pages - len(failed_pages),
            "pages_failed": len(failed_pages),
        })
        return "\n\n".join(page_texts)

    image_pages = _pages_with_images(file_path)
    if not image_pages:
        stats.update({"mode": "native_text_only", "image_pages_found": 0})
        return native_text

    print(f"[extraction] PDF has a text layer AND {len(image_pages)} page(s) with embedded image(s) "
          f"— rendering and OCR-ing each such page too, in case it carries its own text.")
    image_text_parts = []
    failed_pages = []
    none_pages = []
    for page_index in image_pages:
        page_num = page_index + 1
        try:
            img = _render_page(file_path, page_index)
            text = _vision_transcribe(img)
        except Exception as e:
            print(f"[extraction]   WARNING: image page {page_num} failed to read ({e!r}) — "
                  f"continuing with the remaining pages.")
            failed_pages.append(page_num)
            continue
        if not text or text.strip().upper() == "NONE" or len(text) <= 3:
            none_pages.append(page_num)
            continue
        image_text_parts.append(f"[Text found in an image on page {page_num}]\n{text}")

    print(f"[extraction] Image pages: {len(image_pages)} found, {len(image_text_parts)} yielded text, "
          f"{len(none_pages)} had no readable text, {len(failed_pages)} failed to read.")
    if failed_pages:
        print(f"[extraction] WARNING: pages {failed_pages} could not be read — their content is missing "
              f"from this extraction. Re-uploading (or retrying) may pick these up.")

    stats.update({
        "mode": "native_text_plus_images",
        "image_pages_found": len(image_pages),
        "image_pages_with_text": len(image_text_parts),
        "image_pages_blank": len(none_pages),
        "image_pages_failed": len(failed_pages),
    })

    if not image_text_parts:
        return native_text

    return native_text + "\n\n" + "\n\n".join(image_text_parts)


def extract_text(file_path: str, progress_cb=None, stats: dict = None) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".txt", ".md"):
        return extract_text_from_txt(file_path)
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path, progress_cb=progress_cb, stats=stats)
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (supported: .txt, .pdf, .png, .jpg, .jpeg, .webp)")
