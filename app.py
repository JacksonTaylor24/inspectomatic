# app.py — FULL DROP-IN REPLACEMENT
# Change in this version: ✅ Stable, deterministic sort of normalized_items BEFORE pricing
# so item indexes (and therefore pricing attachment) are consistent across runs.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple, Any, Callable, Awaitable
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import io, re, os, json
from pathlib import Path
import requests
import asyncio
import time
import httpx
from uuid import uuid4

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
print("DEBUG[config]: GOOGLE_MAPS_API_KEY present:", bool(GOOGLE_MAPS_API_KEY))

# ---------------- Claude-only AI client setup ----------------
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "200000"))
CLAUDE_RESPONSE_MAX_TOKENS = int(os.getenv("CLAUDE_RESPONSE_MAX_TOKENS", "8192"))
PRICING_CONCURRENCY = max(1, int(os.getenv("PRICING_CONCURRENCY", "3")))
PROVIDER_CONCURRENCY = max(1, int(os.getenv("PROVIDER_CONCURRENCY", "3")))
GOOGLE_HTTP_TIMEOUT = float(os.getenv("GOOGLE_HTTP_TIMEOUT", "5"))
UPLOAD_JOB_TTL_SECONDS = int(os.getenv("UPLOAD_JOB_TTL_SECONDS", "3600"))

print("DEBUG[config]: CLAUDE_MODEL =", CLAUDE_MODEL)
print("DEBUG[config]: ANTHROPIC_API_KEY present:", bool(ANTHROPIC_API_KEY))
print("DEBUG[config]: CLAUDE_MAX_TOKENS =", CLAUDE_MAX_TOKENS, "CLAUDE_RESPONSE_MAX_TOKENS =", CLAUDE_RESPONSE_MAX_TOKENS)
print("DEBUG[config]: PRICING_CONCURRENCY =", PRICING_CONCURRENCY, "PROVIDER_CONCURRENCY =", PROVIDER_CONCURRENCY)

_upload_jobs: Dict[str, Dict[str, Any]] = {}
_upload_jobs_lock = asyncio.Lock()

_anthropic_client = None
try:
    import anthropic
    _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
    print("DEBUG[config]: Anthropic client initialized:", bool(_anthropic_client))
except Exception as e:
    print("DEBUG[config]: Error initializing Anthropic client:", e)
    _anthropic_client = None

# ---------------- App setup ----------------
app = FastAPI(title="Inspectomatic – AI-Powered Extraction")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "web"))

# ---------------- Categories ----------------
CATEGORIES = [
    "Plumbing",
    "Electrical",
    "HVAC",
    "Carpentry & Trim",
    "Painting & Finishes (Cosmetic)",
    "Roofing",
    "Siding & Exterior Envelope",
    "Windows/Glass",
    "Gutters & Downspouts",
    "Drywall & Plaster",
    "Flooring & Tile",
    "Insulation & Air Sealing",
    "Masonry & Concrete",
    "Garage Door Systems",
    "Chimney/Fireplace",
    "Waterproofing & Mold",
    "Pest Control",
    "Landscaping & Drainage",
    "Ventilation & Appliances",
    "General Contractor (Multi-Trade)",
    "Minor Handyman Repairs",
    "Septic & Well Systems",
]

FREQUENCY_ORDER = [
    "Plumbing",
    "Electrical",
    "HVAC",
    "Minor Handyman Repairs",
    "Septic & Well Systems",
    "Painting & Finishes (Cosmetic)",
    "Carpentry & Trim",
    "Windows/Glass",
    "Siding & Exterior Envelope",
    "Roofing",
    "Drywall & Plaster",
    "Flooring & Tile",
    "Gutters & Downspouts",
    "Insulation & Air Sealing",
    "Garage Door Systems",
    "Ventilation & Appliances",
    "Landscaping & Drainage",
    "Pest Control",
    "Masonry & Concrete",
    "Chimney/Fireplace",
    "Waterproofing & Mold",
    "General Contractor (Multi-Trade)",
]
FREQ_TEXT = ", ".join(FREQUENCY_ORDER)

CATEGORY_PROVIDER_MAP: Dict[str, Dict[str, list]] = {
    "Plumbing": {"providers": ["Plumber", "Plumbing Contractor"]},
    "Electrical": {"providers": ["Electrician", "Electrical Contractor"]},
    "HVAC": {"providers": ["HVAC Technician", "Heating & Cooling Contractor"]},
    "Carpentry & Trim": {"providers": ["Carpenter", "Finish Carpenter"]},
    "Painting & Finishes (Cosmetic)": {"providers": ["Painter", "Painting Contractor"]},
    "Roofing": {"providers": ["Roofer", "Roofing Contractor"]},
    "Siding & Exterior Envelope": {"providers": ["Siding Contractor", "Exterior Contractor"]},
    "Windows/Glass": {"providers": ["Window Contractor", "Glazier"]},
    "Gutters & Downspouts": {"providers": ["Gutter Contractor", "Roofing Contractor"]},
    "Drywall & Plaster": {"providers": ["Drywall Contractor"]},
    "Flooring & Tile": {"providers": ["Flooring Contractor", "Tile Setter"]},
    "Insulation & Air Sealing": {"providers": ["Insulation Contractor", "Weatherization Contractor"]},
    "Masonry & Concrete": {"providers": ["Masonry Contractor", "Concrete Contractor"]},
    "Garage Door Systems": {"providers": ["Garage Door Company"]},
    "Chimney/Fireplace": {"providers": ["Chimney Sweep/Service", "Fireplace/Gas Log Technician"]},
    "Waterproofing & Mold": {"providers": ["Mold Remediation", "Basement/Crawl Waterproofing"]},
    "Pest Control": {"providers": ["Pest Control", "Exterminator"]},
    "Landscaping & Drainage": {"providers": ["Landscaper", "Drainage Contractor"]},
    "Ventilation & Appliances": {"providers": ["Appliance Technician", "Ventilation Contractor", "HVAC (for ducted fans)"]},
    "General Contractor (Multi-Trade)": {"providers": ["General Contractor"]},
    "Minor Handyman Repairs": {"providers": ["Handyman"]},
    "Septic & Well Systems": {"providers": ["Septic Service / Pumping", "Onsite Wastewater Contractor", "Well & Pump Contractor"]},
}

DEFAULT_EXPLANATION_BY_CATEGORY = {
    "Plumbing": "Requires plumbing knowledge for proper water/drain connections, venting, and code compliance.",
    "Electrical": "Requires electrical licensing, safe de-energizing, and code-compliant wiring/GFCI/AFCI work.",
    "HVAC": "Requires HVAC diagnostics, gauges/refrigerant handling, and system balancing expertise.",
    "Roofing": "Requires roofing safety/equipment and correct flashing/shingle/underlayment techniques.",
    "Siding & Exterior Envelope": "Requires building envelope expertise to prevent water intrusion with correct flashing/sealants.",
    "Windows/Glass": "Requires window/glazing expertise for sash balance, IGU replacement, and weather sealing.",
    "Gutters & Downspouts": "Requires correct slope/joins/outlet placement to prevent water damage.",
    "Drywall & Plaster": "Requires correct substrate repair, taping/mudding, and finishing.",
    "Flooring & Tile": "Requires substrate prep, layout, cutting, and proper adhesives/grouts.",
    "Insulation & Air Sealing": "Requires knowledge of R-values, vapor/air barriers, and safe installation.",
    "Masonry & Concrete": "Requires specialty materials, mixing/placement, and curing/crack control.",
    "Garage Door Systems": "Requires torsion spring/safety handling and opener configuration.",
    "Chimney/Fireplace": "Requires chimney service expertise for combustion safety and venting.",
    "Waterproofing & Mold": "Requires remediation protocols, containment, and moisture control.",
    "Pest Control": "Requires licensed pesticide use and exclusion methods.",
    "Landscaping & Drainage": "Requires grading/drainage design to direct water away from structure.",
    "Ventilation & Appliances": "Requires ducting, make-up air, and safe appliance install/venting.",
    "General Contractor (Multi-Trade)": "Requires permitted, multi-trade coordination and scheduling.",
    "Carpentry & Trim": "Requires advanced carpentry/structural skills beyond basic handyman work.",
    "Septic & Well Systems": "Requires onsite wastewater/well licensing and specialized equipment.",
    "Minor Handyman Repairs": "Basic repair/installation within typical handyman scope."
}

# ---------------- Models ----------------
class PriceEstimate(BaseModel):
    low: float
    high: float
    currency: str = "USD"
    basis: str = "per job"
    confidence: str = "medium"
    notes: Optional[str] = None

class LineItem(BaseModel):
    category: str
    item_text: str
    notes: Optional[str] = None
    qty: Optional[float] = None
    priority: Optional[str] = None
    cost_low: Optional[float] = None
    cost_high: Optional[float] = None
    currency: Optional[str] = None

class NormalizedLineItem(BaseModel):
    category: str
    item: str
    verbatim: Optional[str] = None
    location: Optional[str] = None
    qty: Optional[float] = None
    units: Optional[str] = None
    severity: Optional[str] = None
    explanation: Optional[str] = None
    price: Optional[PriceEstimate] = None

class IgnoredExample(BaseModel):
    verbatim: str
    why: Optional[str] = None

class ParsedResponse(BaseModel):
    address: Optional[str] = None
    notes: Optional[str] = None
    items: List[LineItem]
    normalized_items: Optional[List[NormalizedLineItem]] = None
    ignored_examples: Optional[List[IgnoredExample]] = None
    meta: Optional[Dict] = None

class UploadStartResponse(BaseModel):
    job_id: str

class UploadStatusResponse(BaseModel):
    job_id: str
    status: str
    percent: float
    stage: str
    message: str
    done: bool
    error: Optional[str] = None

# ---------------- Helpers ----------------
def _coerce_json(text: str):
    if not text:
        return {}
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```[\w-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return {}
    return {}

def _num(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("$", "").replace(",", "")
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def _elapsed_s(start: float) -> float:
    return round(time.perf_counter() - start, 3)

def _print_upload_timing_summary(timings: Dict[str, float]) -> None:
    print(
        "TIMING[/upload]:\n"
        f"  extract_text={timings.get('extract_text', 0.0):.3f}s\n"
        f"  doc_gate={timings.get('doc_gate', 0.0):.3f}s\n"
        f"  extraction={timings.get('extraction', 0.0):.3f}s\n"
        f"  stable_sort={timings.get('stable_sort', 0.0):.3f}s\n"
        f"  pricing={timings.get('pricing', 0.0):.3f}s\n"
        f"  providers={timings.get('providers', 0.0):.3f}s\n"
        f"  total={timings.get('total', 0.0):.3f}s"
    )

async def _set_job_state(job_id: str, **updates: Any) -> None:
    async with _upload_jobs_lock:
        job = _upload_jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()

async def _cleanup_old_jobs() -> None:
    now = time.time()
    async with _upload_jobs_lock:
        stale = [jid for jid, job in _upload_jobs.items() if now - float(job.get("updated_at", now)) > UPLOAD_JOB_TTL_SECONDS]
        for jid in stale:
            _upload_jobs.pop(jid, None)

def chunk_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return [text]
    sections = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""
    for section in sections:
        if len(current) + len(section) + 2 <= max_chars:
            current += ("\n\n" + section) if current else section
        else:
            if current:
                chunks.append(current.strip())
            current = section
    if current:
        chunks.append(current.strip())
    return chunks

def _sample_snippets(text: str, n: int = 3, snippet_chars: int = 900) -> List[str]:
    if not text:
        return []
    t = text.strip()
    if len(t) <= snippet_chars:
        return [t]
    positions = []
    for k in range(n):
        base = int((k + 1) * len(t) / (n + 1))
        jitter = int((k + 1) * 37)
        pos = max(0, min(len(t) - snippet_chars, base + jitter))
        positions.append(pos)
    return [t[p:p + snippet_chars] for p in positions]

# ---------------- Canonicalization + ✅ stable sort ----------------
_STOPWORDS = {
    "please","ensure","properly","all","the","a","an","that","to","be","is","are","needs","need",
    "should","with","of","for","on","in","at","and"
}

def _canon(s: str) -> str:
    if not s:
        return ""
    t = str(s).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    # remove common filler
    t = re.sub(r"\b(" + "|".join(sorted(_STOPWORDS)) + r")\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _cat_rank(cat: str) -> int:
    try:
        return FREQUENCY_ORDER.index(cat)
    except ValueError:
        return 999

def sort_items_stably(items: List[NormalizedLineItem]) -> List[NormalizedLineItem]:
    """
    ✅ Deterministic, stable ordering so pricing indexes map consistently.
    Sort key:
      1) category rank (FREQUENCY_ORDER)
      2) canonical item text
      3) canonical location (if any)
      4) canonical verbatim (tie-break)
    """
    return sorted(
        items,
        key=lambda it: (
            _cat_rank(it.category or ""),
            _canon(it.item or ""),
            _canon(it.location or ""),
            _canon(it.verbatim or ""),
        ),
    )

# ---------------- Text extraction ----------------
def extract_pages_text_from_pdf(file_bytes: bytes) -> Tuple[str, int, str]:
    """
    Returns: (full_text, num_pages, first_pages_text)
    """
    print("DEBUG[extract_pages_text_from_pdf]: starting")
    num_pages = 0
    pages_text: List[str] = []

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        for page in reader.pages:
            pages_text.append((page.extract_text() or "").strip())
        combined = "\n\n".join(pages_text).strip()
        first_pages_text = "\n\n".join(pages_text[:2]).strip()
        print("DEBUG[extract_pages_text_from_pdf]: native extracted chars:", len(combined), "pages:", num_pages)
        if len(combined) > 100:
            return combined, num_pages, first_pages_text
    except Exception as e:
        print("DEBUG[extract_pages_text_from_pdf]: native text error:", e)

    try:
        print("DEBUG[extract_pages_text_from_pdf]: falling back to OCR")
        images = convert_from_bytes(file_bytes, fmt="png", dpi=300)
        num_pages = len(images)
        ocr_pages = [pytesseract.image_to_string(img) for img in images]
        combined = "\n\n".join([p.strip() for p in ocr_pages]).strip()
        first_pages_text = "\n\n".join([p.strip() for p in ocr_pages[:2]]).strip()
        print("DEBUG[extract_pages_text_from_pdf]: OCR extracted chars:", len(combined), "pages:", num_pages)
        return combined, num_pages, first_pages_text
    except Exception as e:
        print("DEBUG[extract_pages_text_from_pdf]: OCR failure:", e)
        raise HTTPException(status_code=400, detail=f"Could not extract text from PDF: {e}")

def extract_text_from_image(file_bytes: bytes) -> str:
    print("DEBUG[extract_text_from_image]: starting")
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        print("DEBUG[extract_text_from_image]: extracted chars:", len(text))
        return text
    except Exception as e:
        print("DEBUG[extract_text_from_image]: failure:", e)
        raise HTTPException(status_code=400, detail=f"Image OCR failed: {e}")

# ---------------- Open-set doc classifier (LLM Gate) ----------------
DOC_GATE_SYSTEM = """
You are a strict document classifier for a real estate repair-scoping tool.

The tool ONLY supports analyzing:
1) Home Inspection Reports (multi-page reports describing inspection findings), OR
2) Repair/Replacement Proposals (short forms or addenda listing repairs to be performed)

If it is not clearly one of those, refuse.

Return JSON ONLY:
{
  "in_domain": true|false,
  "doc_type": "inspection_report"|"repair_proposal"|"unknown",
  "confidence": "high"|"medium"|"low",
  "doc_label": "Inspection Report"|"Repair Proposal"|"Unknown"|"Unsupported Document",
  "reason": "<one sentence explanation>"
}

Rules:
- If not clearly inspection/proposal: in_domain=false, doc_type="unknown", doc_label="Unsupported Document".
- Use page_count as a strong clue but not the only one.
- Prefer refusing over guessing when uncertain.
"""

def classify_document_llm(page_count: int, first_pages_text: str, snippets: List[str]) -> Dict:
    if not ANTHROPIC_API_KEY or not _anthropic_client:
        return {
            "in_domain": True,
            "doc_type": "unknown",
            "confidence": "low",
            "doc_label": "Unknown",
            "reason": "Classifier unavailable (Anthropic not configured)."
        }

    payload = {
        "page_count": int(page_count or 0),
        "first_pages_text": (first_pages_text or "").strip()[:8000],
        "snippets": [(s or "")[:1200] for s in (snippets or [])[:4]],
    }

    user_msg = f"""Classify this document.

INPUT (JSON):
{json.dumps(payload, ensure_ascii=False)}
"""
    try:
        resp = _anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            temperature=0,
            system=DOC_GATE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text if resp.content else "{}"
        parsed = _coerce_json(raw)
        if not isinstance(parsed, dict):
            parsed = {}

        in_domain = bool(parsed.get("in_domain", False))
        doc_type = str(parsed.get("doc_type", "unknown"))
        conf = str(parsed.get("confidence", "low"))
        label = str(parsed.get("doc_label", "Unknown"))
        reason = str(parsed.get("reason", "")).strip() or "No reason provided."

        if doc_type not in ("inspection_report", "repair_proposal", "unknown"):
            doc_type = "unknown"
        if conf not in ("high", "medium", "low"):
            conf = "low"
        if not in_domain:
            doc_type = "unknown"
            label = "Unsupported Document"
        if label not in ("Inspection Report", "Repair Proposal", "Unknown", "Unsupported Document"):
            label = "Unknown" if in_domain else "Unsupported Document"

        return {
            "in_domain": in_domain,
            "doc_type": doc_type,
            "confidence": conf,
            "doc_label": label,
            "reason": reason,
        }
    except Exception as e:
        print("DEBUG[classify_document_llm]: exception:", e)
        return {
            "in_domain": True,
            "doc_type": "unknown",
            "confidence": "low",
            "doc_label": "Unknown",
            "reason": f"Classifier error: {e}"
        }

# ---------------- Extraction prompts ----------------
NEGATIVE_EXAMPLES = """
NEVER EXTRACT these meta-instructions:
- "Seller to repair all items outlined in attached Inspection Report"
- "Buyer to complete all inspection recommendations"
- "Address items marked with blue tape"
- "Repair per inspector's recommendations"
- "Complete all items in report"
- "Seller to remedy all deficiencies noted"
- "All noted defects to be corrected"
- "Repair/replace items as needed"
- "Repair/Replacement to be made on the following items"
- "The following items require repair or replacement"
- "See following items for repairs needed"
"""

SYSTEM_PROMPT_PROPOSAL = f"""Think carefully.

Only extract specific, actionable repair tasks that a contractor can quote and complete.
Ignore meta-instructions and boilerplate.

{NEGATIVE_EXAMPLES}

This document is a repair proposal / negotiation form.
Extract ONLY concrete repair/replace tasks as contractor scope.

Use EXACTLY these category names:
{", ".join(CATEGORIES)}

When borderline, prefer earlier in:
{FREQ_TEXT}

Output JSON:
{{
  "items": [
    {{
      "category": "<category>",
      "item": "<contractor-quotable task>",
      "verbatim": "<exact source text>",
      "location": "<where if given>",
      "qty": <number or null>,
      "units": "<units or null>",
      "severity": "low|medium|high",
      "explanation": "<why this category vs. handyman>"
    }}
  ],
  "ignored_examples": [{{"verbatim":"...", "why":"..."}}],
  "property_address": "<if found>",
  "total_items_found": <number>
}}
"""

SYSTEM_PROMPT_INSPECTION = f"""Think carefully.

Only extract specific, actionable repair tasks that a contractor can quote and complete.
Skip SOP/limitations/disclaimer/educational content.

{NEGATIVE_EXAMPLES}

This document is a home inspection report.
Transform findings into contractor-quotable repair scope items.

Use EXACTLY these category names:
{", ".join(CATEGORIES)}

When borderline, prefer earlier in:
{FREQ_TEXT}

Output JSON:
{{
  "items": [
    {{
      "category": "<category>",
      "item": "<contractor-quotable task>",
      "verbatim": "<exact source text>",
      "location": "<where if given>",
      "qty": <number or null>,
      "units": "<units or null>",
      "severity": "low|medium|high",
      "explanation": "<why this category vs. handyman>"
    }}
  ],
  "ignored_examples": [{{"verbatim":"...", "why":"..."}}],
  "property_address": "<if found>",
  "total_items_found": <number>
}}
"""

def _active_system_prompt(doc_type: str) -> str:
    return SYSTEM_PROMPT_PROPOSAL if doc_type == "repair_proposal" else SYSTEM_PROMPT_INSPECTION

# ---------------- Pricing (coverage-guaranteed) ----------------
PRICING_SYSTEM_PROMPT = """
You are a home repair cost estimator for a tool used in real estate deals.

CRITICAL REQUIREMENT:
- You MUST return exactly ONE pricing object for EACH input item index.
- The output "items" array length MUST equal the number of input objects.
- Every input index MUST appear exactly once in the output.
- If you are uncertain, make reasonable assumptions and still return a best-effort range.

Pricing rules:
- Estimate a realistic price RANGE in USD (low and high).
- Keep ranges fairly tight: in normal cases high is ~1.2x to 1.4x low.
- Only widen if scope uncertainty is explicit in the item text.
- If qty + units are provided, scale the range accordingly.
- Severity guide:
  * low = minor repair/tune-up
  * medium = moderate repair/partial replacement
  * high = major repair/likely replacement

Output JSON ONLY:
{
  "items": [
    {
      "index": <int>,
      "price_low": <number>,
      "price_high": <number>,
      "currency": "USD",
      "basis": "per job" | "per unit" | "per sq ft" | "per linear ft",
      "confidence": "low" | "medium" | "high",
      "notes": "<short assumptions>"
    }
  ]
}
"""

def call_claude_extraction(text: str, address: str = "", doc_type: str = "inspection_report") -> dict:
    if not ANTHROPIC_API_KEY or not _anthropic_client:
        return {"items": [], "ignored_examples": [], "error": "Anthropic API not configured"}

    system_prompt = _active_system_prompt(doc_type)
    user_message = f"""Only extract specific actionable repair tasks (not boilerplate).

Document Type: {doc_type}
Property Address: {address}

Document Content:
{text}
"""
    try:
        resp = _anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_RESPONSE_MAX_TOKENS,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = resp.content[0].text if resp.content else "{}"
        print("\n=== DEBUG: Raw Claude Extraction (first 700 chars) ===")
        print(raw[:700])
        print("=" * 50)
        parsed = _coerce_json(raw)
        if isinstance(parsed, list):
            parsed = {"items": parsed, "ignored_examples": []}
        return parsed if isinstance(parsed, dict) else {"items": [], "ignored_examples": []}
    except Exception as e:
        print("DEBUG[call_claude_extraction]: exception:", e)
        return {"items": [], "ignored_examples": [], "error": str(e)}

def _call_claude_pricing_batch(payload: List[dict], address: str) -> dict:
    user_message = f"""
Property Address (may help you infer region): {address or "Unknown"}

INPUT ITEMS (array of objects). Output MUST include one pricing object per input index:
{json.dumps(payload, ensure_ascii=False)}
"""
    resp = _anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3500,
        temperature=0,
        system=PRICING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = resp.content[0].text if resp.content else "{}"
    print("\n=== DEBUG: Raw Claude Pricing (first 900 chars) ===")
    print(raw[:900])
    print("=" * 50)
    return _coerce_json(raw) or {}

def call_claude_pricing(
    items: List["NormalizedLineItem"],
    address: str = "",
    batch_size: int = 15,
    max_passes: int = 2
) -> Dict[int, PriceEstimate]:
    if not ANTHROPIC_API_KEY or not _anthropic_client:
        print("DEBUG[call_claude_pricing]: Anthropic not configured")
        return {}

    total = len(items)
    out: Dict[int, PriceEstimate] = {}

    def build_payload(indexes: List[int]) -> List[dict]:
        payload = []
        for idx in indexes:
            it = items[idx]
            payload.append({
                "index": idx,
                "category": it.category,
                "item": it.item,
                "location": it.location,
                "severity": it.severity,
                "qty": it.qty,
                "units": it.units,
            })
        return payload

    def ingest(parsed: dict):
        if not isinstance(parsed, dict):
            return
        arr = parsed.get("items", [])
        if not isinstance(arr, list):
            return
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            try:
                idx = int(obj.get("index"))
            except Exception:
                continue
            low = _num(obj.get("price_low"))
            high = _num(obj.get("price_high"))
            if low is None or high is None:
                continue
            try:
                out[idx] = PriceEstimate(
                    low=float(low),
                    high=float(high),
                    currency=(obj.get("currency") or "USD"),
                    basis=(obj.get("basis") or "per job"),
                    confidence=(obj.get("confidence") or "medium"),
                    notes=obj.get("notes"),
                )
            except Exception:
                continue

    all_indexes = list(range(total))

    # pass 1
    for start in range(0, total, batch_size):
        batch_idxs = all_indexes[start:start + batch_size]
        payload = build_payload(batch_idxs)
        try:
            parsed = _call_claude_pricing_batch(payload, address)
            ingest(parsed)
        except Exception as e:
            print("DEBUG[call_claude_pricing]: batch exception:", e)

    # retries
    for pass_i in range(1, max_passes + 1):
        missing = [i for i in all_indexes if i not in out]
        if not missing:
            break
        print(f"DEBUG[call_claude_pricing]: pass {pass_i} retrying missing={len(missing)}")
        for start in range(0, len(missing), batch_size):
            batch_idxs = missing[start:start + batch_size]
            payload = build_payload(batch_idxs)
            try:
                parsed = _call_claude_pricing_batch(payload, address)
                ingest(parsed)
            except Exception as e:
                print("DEBUG[call_claude_pricing]: retry batch exception:", e)

    print("DEBUG[call_claude_pricing]: built pricing map for", len(out), "of", total, "items")
    return out

async def call_claude_pricing_async(
    items: List["NormalizedLineItem"],
    address: str = "",
    batch_size: int = 15,
    max_passes: int = 2,
    concurrency: int = PRICING_CONCURRENCY,
) -> Dict[int, PriceEstimate]:
    if not ANTHROPIC_API_KEY or not _anthropic_client:
        print("DEBUG[call_claude_pricing_async]: Anthropic not configured")
        return {}

    total = len(items)
    out: Dict[int, PriceEstimate] = {}
    sem = asyncio.Semaphore(max(1, concurrency))

    def build_payload(indexes: List[int]) -> List[dict]:
        payload = []
        for idx in indexes:
            it = items[idx]
            payload.append({
                "index": idx,
                "category": it.category,
                "item": it.item,
                "location": it.location,
                "severity": it.severity,
                "qty": it.qty,
                "units": it.units,
            })
        return payload

    def ingest(parsed: dict):
        if not isinstance(parsed, dict):
            return
        arr = parsed.get("items", [])
        if not isinstance(arr, list):
            return
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            try:
                idx = int(obj.get("index"))
            except Exception:
                continue
            low = _num(obj.get("price_low"))
            high = _num(obj.get("price_high"))
            if low is None or high is None:
                continue
            try:
                out[idx] = PriceEstimate(
                    low=float(low),
                    high=float(high),
                    currency=(obj.get("currency") or "USD"),
                    basis=(obj.get("basis") or "per job"),
                    confidence=(obj.get("confidence") or "medium"),
                    notes=obj.get("notes"),
                )
            except Exception:
                continue

    async def run_batch(batch_idxs: List[int], phase: str):
        payload = build_payload(batch_idxs)
        try:
            async with sem:
                return await asyncio.to_thread(_call_claude_pricing_batch, payload, address)
        except Exception as e:
            print(f"DEBUG[call_claude_pricing_async]: {phase} batch exception:", e)
            return None

    all_indexes = list(range(total))

    # pass 1
    initial_batches = [all_indexes[start:start + batch_size] for start in range(0, total, batch_size)]
    initial_results = await asyncio.gather(*(run_batch(batch, "pass1") for batch in initial_batches))
    for parsed in initial_results:
        if parsed:
            ingest(parsed)

    # retries
    for pass_i in range(1, max_passes + 1):
        missing = [i for i in all_indexes if i not in out]
        if not missing:
            break
        print(f"DEBUG[call_claude_pricing_async]: pass {pass_i} retrying missing={len(missing)}")
        retry_batches = [missing[start:start + batch_size] for start in range(0, len(missing), batch_size)]
        retry_results = await asyncio.gather(*(run_batch(batch, f"retry{pass_i}") for batch in retry_batches))
        for parsed in retry_results:
            if parsed:
                ingest(parsed)

    print("DEBUG[call_claude_pricing_async]: built pricing map for", len(out), "of", total, "items")
    return out

# ---------------- Header/numbering filters + fallback ----------------
NUMBERED_OR_BULLET = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s*[\.)-]?\s*|[-*•]\s+)?(.+)$", re.MULTILINE)
SECTION_HEADER_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)+\s+[A-Za-z].{0,80}:\s+.+$")
GENERIC_HEADER_LIKE_PATTERN = re.compile(r"^\s*(?:[A-Z][a-z]+(?:\s*[-/]\s*[A-Z][a-z]+)*)(?:\s*[-–]\s*[A-Z][a-z]+)*\s*:\s+.+$")
REPAIR_ACTION_HINT = re.compile(
    r"\b(repair|replace|install|reinstall|secure|tighten|adjust|service|clean|seal|caulk|recaulk|patch|correct|fix|rebuild|reset|regrade|remove|add)\b",
    re.I
)
META_INSTRUCTION_PATTERN = re.compile(
    r"\b(seller|buyer|all items|complete all|per report|outlined in|attached|blue tape|remedy all|deficiencies noted|as recommended|noted defects|to be corrected)\b",
    re.I
)
SUMMARY_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s*[\.)-]?\s*)?(?:seller|buyer)\s+(?:to|shall|will|must)\s+(?:repair|complete|address|remedy)",
    re.I
)
INSPECTION_BOILERPLATE_PATTERN = re.compile(
    r"\b(standards of practice|limitations|exclusions|disclaimer|inspection agreement|scope of inspection)\b",
    re.I
)

def _looks_like_section_header(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if SECTION_HEADER_PATTERN.search(s):
        return True
    if GENERIC_HEADER_LIKE_PATTERN.search(s) and not REPAIR_ACTION_HINT.search(s):
        return True
    return False

def simple_fallback_parse(text: str) -> List[LineItem]:
    items: List[LineItem] = []
    for m in NUMBERED_OR_BULLET.finditer(text):
        line = (m.group(1) or "").strip()
        if len(line) < 6:
            continue
        if _looks_like_section_header(line):
            continue
        if META_INSTRUCTION_PATTERN.search(line):
            continue
        if SUMMARY_PATTERN.search(line):
            continue
        if INSPECTION_BOILERPLATE_PATTERN.search(line):
            continue
        if REPAIR_ACTION_HINT.search(line):
            items.append(LineItem(category="Minor Handyman Repairs", item_text=line))
    return items

# ---------------- Dedupe (same as your current lightweight) ----------------
ACTION_WORDS = {"repair","replace","install","trim","add","seal","caulk","paint","fill","balance","adjust","fasten","secure","patch","correct","clean","recaulk","remove","reinstall"}

def _jaccard(a: str, b: str) -> float:
    A = set(_canon(a).split())
    B = set(_canon(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _is_compound(text: str) -> bool:
    tokens = set(_canon(text).split())
    return len(tokens & ACTION_WORDS) >= 2

def _similar_actions(a: NormalizedLineItem, b: NormalizedLineItem) -> bool:
    ak = _canon(a.item + " " + (a.verbatim or ""))
    bk = _canon(b.item + " " + (b.verbatim or ""))
    if len(ak) >= 12 and len(bk) >= 12 and (ak in bk or bk in ak):
        return True
    base = max([
        _jaccard(a.item, b.item),
        _jaccard(a.item, b.verbatim or ""),
        _jaccard(a.verbatim or "", b.item),
        _jaccard(a.verbatim or "", b.verbatim or ""),
    ])
    return base >= 0.72

def dedupe_and_arbitrate(items: List[NormalizedLineItem]) -> List[NormalizedLineItem]:
    seen = set()
    fast = []
    for it in items:
        key = (it.category or "", _canon(it.item or ""))
        if key in seen:
            continue
        seen.add(key)
        fast.append(it)

    fixed = fast
    n = len(fixed)
    used = [False] * n

    by_v = {}
    for i, it in enumerate(fixed):
        vk = _canon(it.verbatim or it.item or "") or f"i{i}"
        by_v.setdefault(vk, []).append(i)

    for _, idxs in by_v.items():
        if len(idxs) <= 1:
            continue
        compounds = [i for i in idxs if _is_compound((fixed[i].item or "") + " " + (fixed[i].verbatim or ""))]
        for i in compounds:
            used[i] = True

    pool = [i for i in range(n) if not used[i]]
    clusters = []
    visited = set()
    for i in pool:
        if i in visited:
            continue
        visited.add(i)
        g = [i]
        for j in pool:
            if j in visited:
                continue
            if _similar_actions(fixed[i], fixed[j]):
                g.append(j)
                visited.add(j)
        clusters.append(g)

    result: List[NormalizedLineItem] = []
    for g in clusters:
        if len(g) == 1:
            result.append(fixed[g[0]])
        else:
            winner = max([fixed[k] for k in g], key=lambda x: len(x.item or ""))
            result.append(winner)
    return result

def ensure_explanations(items: List[NormalizedLineItem]) -> List[NormalizedLineItem]:
    for it in items:
        if not it.explanation or not str(it.explanation).strip():
            it.explanation = DEFAULT_EXPLANATION_BY_CATEGORY.get(it.category, "Assigned based on required trade skills and safety/codes.")
    return items

# ---------------- Core extraction flow ----------------
def extract_repairs_comprehensive(text: str, address: str = "", doc_type: str = "inspection_report") -> Tuple[List[NormalizedLineItem], List[IgnoredExample], Dict]:
    estimated_tokens = len(text) // 4
    system_prompt = _active_system_prompt(doc_type)
    system_tokens = len(system_prompt) // 4 + 200

    def _normalize_from_result(result_obj: dict) -> Tuple[List[NormalizedLineItem], List[IgnoredExample]]:
        items: List[NormalizedLineItem] = []
        ignored: List[IgnoredExample] = []
        its = result_obj.get("items", []) if isinstance(result_obj, dict) else []
        if isinstance(its, list):
            for it in its:
                if not isinstance(it, dict):
                    continue
                item_text = (it.get("item") or "").strip()
                if not item_text:
                    continue
                if _looks_like_section_header(item_text):
                    continue
                items.append(NormalizedLineItem(
                    category=it.get("category") or "Minor Handyman Repairs",
                    item=item_text,
                    verbatim=(it.get("verbatim") or item_text),
                    location=it.get("location"),
                    qty=_num(it.get("qty")),
                    units=it.get("units"),
                    severity=(it.get("severity") or "medium"),
                    explanation=it.get("explanation"),
                ))
        ign = result_obj.get("ignored_examples", []) if isinstance(result_obj, dict) else []
        if isinstance(ign, list):
            for g in ign:
                if not isinstance(g, dict):
                    continue
                v = (g.get("verbatim") or "").strip()
                if v:
                    ignored.append(IgnoredExample(verbatim=v, why=g.get("why")))
        return items, ignored

    if estimated_tokens + system_tokens <= CLAUDE_MAX_TOKENS - CLAUDE_RESPONSE_MAX_TOKENS:
        result = call_claude_extraction(text, address, doc_type=doc_type)
        norm_items, ignored = _normalize_from_result(result)
        norm_items = dedupe_and_arbitrate(norm_items)
        norm_items = ensure_explanations(norm_items)
        meta = {
            "mode": "single_request",
            "estimated_tokens": estimated_tokens,
            "items_from_llm": len(norm_items),
            "total_items_found": (result.get("total_items_found") if isinstance(result, dict) else None),
            "extracted_address": (result.get("property_address") if isinstance(result, dict) else None),
            "error": (result.get("error") if isinstance(result, dict) else None),
            "model": CLAUDE_MODEL,
            "ai_provider": "claude",
        }
        return norm_items, ignored, meta

    chunks = chunk_text_by_tokens(text, CLAUDE_MAX_TOKENS - system_tokens - CLAUDE_RESPONSE_MAX_TOKENS)
    all_norm: List[NormalizedLineItem] = []
    all_ignored: List[IgnoredExample] = []
    for i, chunk in enumerate(chunks):
        print(f"DEBUG[extract_repairs_comprehensive]: chunk {i+1}/{len(chunks)} chars={len(chunk)}")
        result = call_claude_extraction(chunk, address, doc_type=doc_type)
        n, ig = _normalize_from_result(result)
        all_norm.extend(n)
        all_ignored.extend(ig)

    deduped = dedupe_and_arbitrate(all_norm)
    deduped = ensure_explanations(deduped)
    meta = {
        "mode": "multi_chunk",
        "chunks": len(chunks),
        "estimated_tokens": estimated_tokens,
        "items_from_llm": len(deduped),
        "total_raw_items": len(all_norm),
        "model": CLAUDE_MODEL,
        "ai_provider": "claude",
    }
    return deduped, all_ignored, meta

# ---------------- Provider search helpers ----------------
def provider_score(rating: float, reviews: int, prior_mean: float = 4.3, prior_weight: int = 20) -> float:
    if rating <= 0 or reviews < 0:
        return 0.0
    return (rating * reviews + prior_mean * prior_weight) / (reviews + prior_weight)

def geocode_address(address: str):
    if not GOOGLE_MAPS_API_KEY or not address:
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": GOOGLE_MAPS_API_KEY},
            timeout=GOOGLE_HTTP_TIMEOUT,
        )
        data = resp.json()
        if data.get("status") != "OK":
            return None
        loc = data["results"][0]["geometry"]["location"]
        return (loc["lat"], loc["lng"])
    except Exception:
        return None

def _places_nearby(lat: float, lng: float, keyword: str, radius_m: int) -> list:
    if not GOOGLE_MAPS_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={"key": GOOGLE_MAPS_API_KEY, "location": f"{lat},{lng}", "radius": radius_m, "keyword": keyword},
            timeout=GOOGLE_HTTP_TIMEOUT,
        )
        data = resp.json()
        return data.get("results", []) or []
    except Exception:
        return []

def _place_details(place_id: str) -> dict:
    if not GOOGLE_MAPS_API_KEY or not place_id:
        return {}
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/details/json",
            params={"key": GOOGLE_MAPS_API_KEY, "place_id": place_id, "fields": "formatted_phone_number,website,formatted_address"},
            timeout=GOOGLE_HTTP_TIMEOUT,
        )
        data = resp.json()
        return data.get("result", {}) or {}
    except Exception:
        return {}

def find_providers_for_category(category: str, lat: float, lng: float) -> list:
    labels = CATEGORY_PROVIDER_MAP.get(category, {}).get("providers") or [category]
    keyword = " ".join(labels)
    radii_miles = [10, 20, 40]
    radii = [int(r * 1609.34) for r in radii_miles]
    seen = set()
    candidates = []

    for r_m in radii:
        results = _places_nearby(lat, lng, keyword, r_m)
        for r in results:
            pid = r.get("place_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            rating = float(r.get("rating", 0.0))
            reviews = int(r.get("user_ratings_total", 0))
            if rating < 4.0 or reviews < 5:
                continue
            score = provider_score(rating, reviews)
            candidates.append({
                "place_id": pid,
                "name": r.get("name"),
                "rating": rating,
                "review_count": reviews,
                "address": r.get("vicinity"),
                "score": score,
            })
        if len(candidates) >= 15:
            break

    candidates.sort(key=lambda x: (x["score"], x["review_count"]), reverse=True)
    top = candidates[:3]
    enriched = []
    for p in top:
        details = _place_details(p["place_id"])
        if details:
            p["phone"] = details.get("formatted_phone_number")
            p["website"] = details.get("website")
            p["address"] = details.get("formatted_address") or p.get("address")
        enriched.append(p)
    return enriched

async def geocode_address_async(client: httpx.AsyncClient, address: str):
    if not GOOGLE_MAPS_API_KEY or not address:
        return None
    try:
        resp = await client.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": GOOGLE_MAPS_API_KEY},
        )
        data = resp.json()
        if data.get("status") != "OK":
            return None
        loc = data["results"][0]["geometry"]["location"]
        return (loc["lat"], loc["lng"])
    except Exception:
        return None

async def _places_nearby_async(client: httpx.AsyncClient, lat: float, lng: float, keyword: str, radius_m: int) -> list:
    if not GOOGLE_MAPS_API_KEY:
        return []
    try:
        resp = await client.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={"key": GOOGLE_MAPS_API_KEY, "location": f"{lat},{lng}", "radius": radius_m, "keyword": keyword},
        )
        data = resp.json()
        return data.get("results", []) or []
    except Exception:
        return []

async def _place_details_async(client: httpx.AsyncClient, place_id: str) -> dict:
    if not GOOGLE_MAPS_API_KEY or not place_id:
        return {}
    try:
        resp = await client.get(
            "https://maps.googleapis.com/maps/api/place/details/json",
            params={"key": GOOGLE_MAPS_API_KEY, "place_id": place_id, "fields": "formatted_phone_number,website,formatted_address"},
        )
        data = resp.json()
        return data.get("result", {}) or {}
    except Exception:
        return {}

async def find_providers_for_category_async(
    category: str,
    lat: float,
    lng: float,
    client: httpx.AsyncClient,
    details_concurrency: int = PROVIDER_CONCURRENCY,
) -> list:
    labels = CATEGORY_PROVIDER_MAP.get(category, {}).get("providers") or [category]
    keyword = " ".join(labels)
    radii_miles = [10, 20, 40]
    radii = [int(r * 1609.34) for r in radii_miles]
    seen = set()
    candidates = []

    for r_m in radii:
        results = await _places_nearby_async(client, lat, lng, keyword, r_m)
        for r in results:
            pid = r.get("place_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            rating = float(r.get("rating", 0.0))
            reviews = int(r.get("user_ratings_total", 0))
            if rating < 4.0 or reviews < 5:
                continue
            score = provider_score(rating, reviews)
            candidates.append({
                "place_id": pid,
                "name": r.get("name"),
                "rating": rating,
                "review_count": reviews,
                "address": r.get("vicinity"),
                "score": score,
            })
        if len(candidates) >= 15:
            break

    candidates.sort(key=lambda x: (x["score"], x["review_count"]), reverse=True)
    top = candidates[:3]
    detail_sem = asyncio.Semaphore(max(1, details_concurrency))

    async def enrich_provider(p: dict) -> dict:
        provider = dict(p)
        try:
            async with detail_sem:
                details = await _place_details_async(client, provider["place_id"])
            if details:
                provider["phone"] = details.get("formatted_phone_number")
                provider["website"] = details.get("website")
                provider["address"] = details.get("formatted_address") or provider.get("address")
        except Exception:
            pass
        return provider

    enriched = await asyncio.gather(*(enrich_provider(p) for p in top))
    return list(enriched)

# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def _run_upload_pipeline(
    data: bytes,
    name: str,
    address: str = "",
    notes: str = "",
    progress_cb: Optional[Callable[[float, str, str], Awaitable[None]]] = None,
) -> ParsedResponse:
    async def emit_progress(percent: float, stage: str, message: str) -> None:
        if progress_cb:
            await progress_cb(percent, stage, message)

    req_start = time.perf_counter()
    timings: Dict[str, float] = {
        "extract_text": 0.0,
        "doc_gate": 0.0,
        "extraction": 0.0,
        "stable_sort": 0.0,
        "pricing": 0.0,
        "providers": 0.0,
        "total": 0.0,
    }

    print("\n=== DEBUG[/upload]: new request ===")
    print("DEBUG[/upload]: incoming filename:", name)

    text = ""
    first_pages_text = ""
    num_pages = 1

    await emit_progress(2.0, "extract_text", "Reading and extracting text from your document...")
    extract_text_start = time.perf_counter()
    if name.endswith(".pdf"):
        text, num_pages, first_pages_text = await asyncio.to_thread(extract_pages_text_from_pdf, data)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
        first_pages_text = text[:6000]
        num_pages = 1
    elif name.endswith((".png", ".jpg", ".jpeg")):
        text = await asyncio.to_thread(extract_text_from_image, data)
        first_pages_text = text[:6000]
        num_pages = 1
    else:
        raise HTTPException(status_code=400, detail="Supported: .pdf, .txt, .png, .jpg, .jpeg")
    timings["extract_text"] = _elapsed_s(extract_text_start)
    await emit_progress(15.0, "extract_text", "Text extraction complete.")

    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail="No meaningful text could be extracted from the file.")

    print("DEBUG[/upload]: extracted chars:", len(text), "pages:", num_pages)

    # --- OPEN-SET LLM GATE ---
    await emit_progress(18.0, "doc_gate", "Classifying document type...")
    snippets = _sample_snippets(text, n=3, snippet_chars=900)
    doc_gate_start = time.perf_counter()
    gate = await asyncio.to_thread(
        classify_document_llm,
        page_count=num_pages,
        first_pages_text=first_pages_text,
        snippets=snippets,
    )
    timings["doc_gate"] = _elapsed_s(doc_gate_start)
    await emit_progress(25.0, "doc_gate", "Document classification complete.")

    doc_type = gate.get("doc_type", "unknown")
    doc_label = gate.get("doc_label", "Unknown")
    doc_conf = gate.get("confidence", "low")
    in_domain = bool(gate.get("in_domain", False))
    doc_reason = gate.get("reason", "")

    meta: Dict = {
        "llm_used": False,
        "extraction_method": "none",
        "text_length": len(text),
        "page_count": num_pages,
        "model": CLAUDE_MODEL,
        "ai_provider": "claude",
        "doc_type": doc_type,
        "doc_type_label": doc_label,
        "doc_type_confidence": doc_conf,
        "doc_type_reason": doc_reason,
        "in_domain": in_domain,
        "pricing_used": False,
        "pricing_attempted": False,
        "pricing_items_in": 0,
        "pricing_items_priced": 0,
        "pricing_totals": {"low": 0.0, "high": 0.0, "currency": "USD"},
        "providers": {},
        "timings": dict(timings),
    }

    # Refuse if out-of-domain or too-uncertain
    if (not in_domain) or (doc_type == "unknown" and doc_conf == "low"):
        meta["user_message"] = "Unable to complete analysis. This document does not appear to be an inspection report or repair proposal. Did you upload the right document?"
        timings["total"] = _elapsed_s(req_start)
        meta["timings"] = dict(timings)
        _print_upload_timing_summary(timings)
        await emit_progress(100.0, "complete", "Analysis complete.")
        return ParsedResponse(address=address or None, notes=notes or None, items=[], normalized_items=[], ignored_examples=[], meta=meta)

    if doc_type == "unknown":
        meta["user_message"] = "Unable to confidently determine document type. Please upload the inspection report or repair proposal."
        timings["total"] = _elapsed_s(req_start)
        meta["timings"] = dict(timings)
        _print_upload_timing_summary(timings)
        await emit_progress(100.0, "complete", "Analysis complete.")
        return ParsedResponse(address=address or None, notes=notes or None, items=[], normalized_items=[], ignored_examples=[], meta=meta)

    # --- Extraction ---
    await emit_progress(30.0, "extraction", "Extracting actionable repair items...")
    normalized_items: List[NormalizedLineItem] = []
    ignored_examples: List[IgnoredExample] = []
    items_for_display: List[LineItem] = []
    extraction_start = time.perf_counter()

    if ANTHROPIC_API_KEY and _anthropic_client:
        meta["llm_used"] = True
        meta["extraction_method"] = f"claude_comprehensive_{doc_type}"
        try:
            normalized_items, ignored_examples, extraction_meta = await asyncio.to_thread(
                extract_repairs_comprehensive,
                text,
                address,
                doc_type,
            )
            meta.update(extraction_meta)
        except Exception as e:
            meta["llm_error"] = str(e)
            meta["extraction_method"] = f"claude_failed_{doc_type}"

    if not normalized_items:
        fallback = simple_fallback_parse(text)
        normalized_items = [
            NormalizedLineItem(category=i.category, item=i.item_text, verbatim=i.item_text, severity="medium")
            for i in fallback
        ]
        normalized_items = ensure_explanations(normalized_items)
        meta["extraction_method"] = f"fallback_regex_{doc_type}"

    # final cleanup + dedupe
    normalized_items = [it for it in normalized_items if it.item and not _looks_like_section_header(it.item)]
    normalized_items = dedupe_and_arbitrate(normalized_items)
    normalized_items = ensure_explanations(normalized_items)

    timings["extraction"] = _elapsed_s(extraction_start)
    await emit_progress(60.0, "extraction", "Repair extraction complete.")

    # ✅ NEW: stable sort before pricing (this is what you asked for)
    await emit_progress(61.0, "stable_sort", "Ordering repair items...")
    stable_sort_start = time.perf_counter()
    normalized_items = sort_items_stably(normalized_items)
    meta["stable_sort_before_pricing"] = True
    timings["stable_sort"] = _elapsed_s(stable_sort_start)
    await emit_progress(62.0, "stable_sort", "Item ordering complete.")

    # --- Pricing (coverage-guaranteed) ---
    meta["pricing_attempted"] = False
    meta["pricing_items_in"] = len(normalized_items)
    meta["pricing_items_priced"] = 0
    await emit_progress(66.0, "pricing", "Estimating repair pricing...")
    pricing_start = time.perf_counter()

    if normalized_items and ANTHROPIC_API_KEY and _anthropic_client:
        meta["pricing_attempted"] = True
        try:
            pricing_map = await call_claude_pricing_async(
                normalized_items,
                address,
                batch_size=15,
                max_passes=2,
                concurrency=PRICING_CONCURRENCY,
            )

            priced_count = 0
            for idx, it in enumerate(normalized_items):
                if idx in pricing_map:
                    it.price = pricing_map[idx]
                    priced_count += 1
            meta["pricing_items_priced"] = priced_count
            meta["pricing_map_size"] = len(pricing_map)

            # your rule: drop 0–0 priced items
            filtered = []
            for it in normalized_items:
                if it.price and it.price.low == 0 and it.price.high == 0:
                    continue
                filtered.append(it)
            normalized_items = filtered

            totals = {"low": 0.0, "high": 0.0, "currency": "USD"}
            for it in normalized_items:
                if it.price:
                    totals["low"] += float(it.price.low)
                    totals["high"] += float(it.price.high)
                    totals["currency"] = it.price.currency or "USD"
            meta["pricing_totals"] = totals
            meta["pricing_used"] = (totals["low"] > 0 or totals["high"] > 0)
        except Exception as e:
            meta["pricing_error"] = str(e)
    timings["pricing"] = _elapsed_s(pricing_start)
    await emit_progress(88.0, "pricing", "Pricing complete.")

    # --- Provider search ---
    await emit_progress(90.0, "providers", "Finding top-rated local providers...")
    providers_by_category: Dict[str, List[Dict]] = {}
    providers_start = time.perf_counter()
    if GOOGLE_MAPS_API_KEY and address and normalized_items:
        try:
            timeout = httpx.Timeout(GOOGLE_HTTP_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout) as client:
                coords = await geocode_address_async(client, address)
                if coords:
                    lat, lng = coords
                    used_categories = sorted({it.category for it in normalized_items})
                    cat_sem = asyncio.Semaphore(max(1, PROVIDER_CONCURRENCY))

                    async def fetch_category_providers(cat: str):
                        try:
                            async with cat_sem:
                                providers = await find_providers_for_category_async(
                                    cat,
                                    lat,
                                    lng,
                                    client,
                                    details_concurrency=PROVIDER_CONCURRENCY,
                                )
                            return cat, providers
                        except Exception as e:
                            print(f"DEBUG[/upload]: Provider search error for {cat}: {e}")
                            return cat, []

                    pairs = await asyncio.gather(*(fetch_category_providers(cat) for cat in used_categories))
                    for cat, providers in pairs:
                        providers_by_category[cat] = providers
        except Exception as e:
            print("DEBUG[/upload]: Provider search setup error:", e)
    meta["providers"] = providers_by_category
    timings["providers"] = _elapsed_s(providers_start)
    await emit_progress(98.0, "providers", "Provider matching complete.")

    # --- Build display items ---
    for it in normalized_items:
        if not it.explanation:
            it.explanation = DEFAULT_EXPLANATION_BY_CATEGORY.get(it.category, "Assigned based on required trade skills and safety/codes.")

        reason_note = f"Reason: {it.explanation}"

        price_note = None
        if it.price:
            try:
                price_note = (
                    f"Estimated cost range: ${it.price.low:,.0f}–${it.price.high:,.0f} "
                    f"{it.price.currency} ({it.price.basis}, confidence: {it.price.confidence})"
                )
            except Exception:
                price_note = f"Estimated cost range: {it.price.low}–{it.price.high} {it.price.currency}"

        full_note = reason_note if not price_note else reason_note + " | " + price_note

        items_for_display.append(LineItem(
            category=it.category,
            item_text=it.item,
            notes=full_note,
            qty=it.qty,
            priority=it.severity,
            cost_low=it.price.low if it.price else None,
            cost_high=it.price.high if it.price else None,
            currency=it.price.currency if it.price else None,
        ))

    timings["total"] = _elapsed_s(req_start)
    meta["timings"] = dict(timings)
    _print_upload_timing_summary(timings)
    await emit_progress(100.0, "complete", "Report ready.")
    return ParsedResponse(
        address=address or None,
        notes=notes or None,
        items=items_for_display,
        normalized_items=normalized_items,
        ignored_examples=ignored_examples,
        meta=meta,
    )

async def _process_upload_job(
    job_id: str,
    data: bytes,
    name: str,
    address: str,
    notes: str,
) -> None:
    async def progress(percent: float, stage: str, message: str) -> None:
        await _set_job_state(
            job_id,
            percent=max(0.0, min(100.0, float(percent))),
            stage=stage,
            message=message,
            status="running",
            done=False,
        )

    try:
        await progress(1.0, "queued", "Upload queued.")
        result = await _run_upload_pipeline(
            data=data,
            name=name,
            address=address,
            notes=notes,
            progress_cb=progress,
        )
        await _set_job_state(
            job_id,
            status="completed",
            percent=100.0,
            stage="complete",
            message="Report ready.",
            done=True,
            result=result.model_dump(),
            error=None,
        )
    except HTTPException as e:
        await _set_job_state(
            job_id,
            status="failed",
            stage="failed",
            message="Upload failed.",
            done=True,
            error=str(e.detail),
        )
    except Exception as e:
        await _set_job_state(
            job_id,
            status="failed",
            stage="failed",
            message="Upload failed.",
            done=True,
            error=str(e),
        )

@app.post("/upload/start", response_model=UploadStartResponse)
async def upload_start(
    file: UploadFile = File(...),
    address: str = Form(default=""),
    notes: str = Form(default=""),
):
    data = await file.read()
    name = (file.filename or "").lower()
    job_id = uuid4().hex
    now = time.time()

    async with _upload_jobs_lock:
        _upload_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "percent": 0.0,
            "stage": "queued",
            "message": "Upload queued.",
            "done": False,
            "error": None,
            "result": None,
            "created_at": now,
            "updated_at": now,
        }

    asyncio.create_task(_process_upload_job(job_id, data, name, address, notes))
    await _cleanup_old_jobs()
    return UploadStartResponse(job_id=job_id)

@app.get("/upload/status/{job_id}", response_model=UploadStatusResponse)
async def upload_status(job_id: str):
    await _cleanup_old_jobs()
    async with _upload_jobs_lock:
        job = _upload_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return UploadStatusResponse(
            job_id=job_id,
            status=str(job.get("status", "queued")),
            percent=float(job.get("percent", 0.0)),
            stage=str(job.get("stage", "queued")),
            message=str(job.get("message", "")),
            done=bool(job.get("done", False)),
            error=job.get("error"),
        )

@app.get("/upload/result/{job_id}", response_model=ParsedResponse)
async def upload_result(job_id: str):
    await _cleanup_old_jobs()
    async with _upload_jobs_lock:
        job = _upload_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        status = str(job.get("status", "queued"))
        if status == "failed":
            raise HTTPException(status_code=500, detail=str(job.get("error") or "Upload failed"))
        if status != "completed":
            raise HTTPException(status_code=409, detail="Result not ready")
        result = job.get("result")
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Invalid job result")
    return ParsedResponse(**result)

@app.post("/upload", response_model=ParsedResponse)
async def upload(
    file: UploadFile = File(...),
    address: str = Form(default=""),
    notes: str = Form(default=""),
):
    data = await file.read()
    name = (file.filename or "").lower()
    return await _run_upload_pipeline(data=data, name=name, address=address, notes=notes)

# Run: uvicorn app:app --reload
