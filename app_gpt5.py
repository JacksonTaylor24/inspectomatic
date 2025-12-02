from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import io, re, os, json
from pathlib import Path

# ---------------- OpenAI client ----------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_openai_client = None
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _openai_client = None

# ---------------- App setup ----------------
app = FastAPI(title="Inspectomatic – AI-Powered Extraction")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------- Config ----------------
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128000"))  # GPT-5 context window
RESPONSE_MAX_TOKENS = int(os.getenv("RESPONSE_MAX_TOKENS", "16000"))

# ---------------- Categories (expanded) ----------------
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

# Frequency order (borderline → prefer earlier)
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

# Category → providers (reference for downstream agent)
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
MAPPING_TEXT = json.dumps(CATEGORY_PROVIDER_MAP, ensure_ascii=False)
FREQ_TEXT = ", ".join(FREQUENCY_ORDER)

# ---------------- Models ----------------
class LineItem(BaseModel):
    category: str
    item_text: str
    notes: Optional[str] = None
    qty: Optional[float] = None
    priority: Optional[str] = None

class NormalizedLineItem(BaseModel):
    category: str
    item: str
    verbatim: Optional[str] = None
    location: Optional[str] = None
    qty: Optional[float] = None
    units: Optional[str] = None
    severity: Optional[str] = None
    explanation: Optional[str] = None

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

# ---------------- Text extraction ----------------
def extract_pages_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF, prefer native text; OCR as fallback."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text.strip())
        combined = "\n\n".join(pages_text).strip()
        if len(combined) > 100:
            return combined
    except Exception:
        pass

    try:
        images = convert_from_bytes(file_bytes, fmt="png", dpi=300)
        ocr_text = [pytesseract.image_to_string(img) for img in images]
        return "\n\n".join(ocr_text).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not extract text from PDF: {e}")

def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image OCR failed: {e}")

# ---------------- GPT-5 extraction schema (reference) ----------------
def get_extraction_schema():
    return {
        "name": "repair_extraction",
        "strict": False,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "category": {"type": "string", "enum": CATEGORIES},
                            "item": {"type": "string"},
                            "verbatim": {"type": "string"},
                            "location": {"type": "string"},
                            "qty": {"type": "number"},
                            "units": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low","medium","high"]},
                            "explanation": {"type": "string"},
                        },
                        "required": ["category","item","verbatim"]
                    }
                },
                "ignored_examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "verbatim": {"type": "string"},
                            "why": {"type": "string"}
                        },
                        "required": ["verbatim"]
                    }
                },
                "property_address": {"type": "string"},
                "total_items_found": {"type": "number"}
            },
            "required": ["items","ignored_examples"]
        }
    }

# ---------------- AI-powered extraction ----------------
# Keep the JSON example in a plain string so braces don't collide with f-strings
JSON_SPEC = """{
  "items": [{"category": "<one of the categories above>", "item": "<actionable task>",
  "verbatim": "<exact source text>", "location": "<where if given>", "qty": <number or omit>,
  "units": "<units or omit>", "severity": "low|medium|high", "explanation": "<why this category>"}],
  "ignored_examples": [{"verbatim": "<non-actionable>", "why": "<short reason>"}]
}"""

# Enhanced system prompt with better instruction hierarchy
NEGATIVE_EXAMPLES = """
❌ NEVER EXTRACT these meta-instructions:
- "Seller to repair all items outlined in attached Inspection Report"
- "Buyer to complete all inspection recommendations"
- "Address items marked with blue tape"
- "Repair per inspector's recommendations"
- "Complete all items in report"
- "Seller to remedy all deficiencies noted"
- "All noted defects to be corrected"
- "Repair/replace as recommended by inspector"

✅ DO EXTRACT specific actionable tasks:
- "Replace broken window pane in master bedroom"
- "Seal gap around exterior door frame with caulk"
- "Install GFCI outlet in guest bathroom"
- "Repair loose handrail on front porch steps"
- "Clean gutters and downspouts"
"""

SYSTEM_PROMPT = f"""🚨 CRITICAL FILTERING RULE: Only extract specific, actionable repair tasks that a contractor can quote and complete. IGNORE all meta-instructions, general directives, and summary statements about completing work.

{NEGATIVE_EXAMPLES}

You are an expert inspection report analyzer for Inspectomatic, a real estate tool that 
helps buyers and sellers negotiate repair costs by providing accurate categorization and cost 
estimates. Your job is to transform overwhelming inspection reports into organized, actionable 
epair lists that contractors can quote and complete.

CORE MISSION: Extract EVERY actionable repair/maintenance item and categorize it by the **service provider who would actually do the work**. Skip any organizational language, summary statements, or instructions about who should complete work.

FILTERING CRITERIA:
- ✅ EXTRACT: Specific tasks with clear actions (repair, replace, install, seal, clean, adjust)
- ❌ SKIP: General instructions, summary statements, references to "all items" or "per report"
- ❌ SKIP: Lines mentioning "seller," "buyer," "outlined in," "attached," "blue tape," "complete all"

Use EXACTLY these category names:
{", ".join(CATEGORIES)}

When borderline between categories, prefer the earlier one in this frequency list:
{FREQ_TEXT}

UNIQUENESS & SPLITTING:
- From any single source line, output **either** ONE handyman item (if a competent handyman can reasonably do the work) **or** multiple trade-specific items — **never both**.
- Do **not** duplicate the same action across multiple categories.

HANDYMAN POLICY:
- If a task can reasonably be completed by a general handyman (touch-ups, small caulk/paint, minor hardware/door/cord fixes, small patches), choose **"Minor Handyman Repairs"** over a specialist.

GENERAL CONTRACTOR POLICY:
- **Reserve "General Contractor (Multi-Trade)" for scopes that clearly require multi-trade coordination and permitting.**
- Do **not** use GC for single-trade work like **sistering rafters**, **rebuilding a deck**, **replacing siding**, **window glass**, etc.

DISAMBIGUATION HINTS:
- Wastewater systems (septic tank pumping/repairs) → **Septic & Well Systems**.
- Fogged/failed insulated glass → **Windows/Glass**. Mirror cord/hardware → **Minor Handyman Repairs**.
- Ducts/returns/airflow/drain pan → **HVAC**.
- Exterior wall boxes/panels/siding sealing → **Siding & Exterior Envelope**.
- Interior trim/baseboard caulk & paint → **Minor Handyman Repairs** (unless it's clearly a whole-house repaint handled by a painter).
- Retaining wall construction → **Masonry & Concrete**; post-install backfill/grading → **Landscaping & Drainage**.
- **Sistering rafters/roof framing** → **Roofing** (or Carpentry if explicitly framed as interior framing).
- **Deck rebuild/bring to code** → **Carpentry & Trim**.

Output JSON (strict schema):
{JSON_SPEC}
"""

def chunk_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return [text]
    sections = re.split(r'\n\s*\n', text)
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

# ---------- JSON coercion hardening ----------
def _coerce_json(text: str):
    if not text:
        return {}
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[\w-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r'(\{.*\}|\[.*\])', s, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return {}
    return {}

def call_gpt5_extraction(text: str, address: str = "") -> dict:
    if not OPENAI_API_KEY or not _openai_client:
        return {"items": [], "ignored_examples": [], "error": "OpenAI API not configured"}

    combined_input = f"""{SYSTEM_PROMPT}

Property Address: {address}

Document Content:
{text}
"""
    try:
        if MODEL_NAME.startswith("gpt-5"):
            resp = _openai_client.responses.create(
                model=MODEL_NAME,
                input=combined_input,
                reasoning={"effort": "high"},
                text={"verbosity": "medium"}
            )
            result_text = getattr(resp, "output_text", None)
            parsed = _coerce_json(result_text)
            if isinstance(parsed, list):
                parsed = {"items": parsed, "ignored_examples": []}
            return parsed if isinstance(parsed, dict) else {"items": [], "ignored_examples": []}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Property Address: {address}\n\nDocument Content:\n\n{text}"}
        ]
        resp = _openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=RESPONSE_MAX_TOKENS,
        )
        result_text = resp.choices[0].message.content
        parsed = _coerce_json(result_text)
        if isinstance(parsed, list):
            parsed = {"items": parsed, "ignored_examples": []}
        return parsed if isinstance(parsed, dict) else {"items": [], "ignored_examples": []}
    except Exception as e:
        return {"items": [], "ignored_examples": [], "error": str(e)}

# ---------- Heuristic capture ----------
NUMBERED_OR_BULLET = re.compile(r'^\s*(?:\d+\s*[\.)-]\s*|[-*•]\s+)?(.+)$', re.MULTILINE)
REPAIR_HINT = re.compile(
    r'\b(repair|replace|install|clean|seal|caulk|fix|adjust|service|test|leak|broken|missing|damaged|inoperable|not working|not cooling|paint|trim|flashing)\b',
    re.I
)

def simple_fallback_parse(text: str) -> List[LineItem]:
    items: List[LineItem] = []
    for m in NUMBERED_OR_BULLET.finditer(text):
        line = (m.group(1) or "").strip()
        if len(line) < 6:
            continue
        if REPAIR_HINT.search(line):
            items.append(LineItem(category="Minor Handyman Repairs", item_text=line))
    return items

# ---- Dedupe & arbitration (prefer Handyman when sufficient) ----
_WORD = re.compile(r"[a-z0-9]+")
ACTION_WORDS = {"repair","replace","install","trim","add","seal","caulk","paint","fill","vacuum","reseed","regrade","balance","evaluate","adjust","sister","fasten","secure","patch"}

def _canon(s: str) -> str:
    if not s: return ""
    t = s.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\b(please|ensure|properly|all|the|a|an|that|to|be|is|are|needs|need|should|with|of|for|on|in|at|and)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _vkey(s: Optional[str]) -> str:
    return _canon(s or "")

def _jaccard(a: str, b: str) -> float:
    A = set(_canon(a).split())
    B = set(_canon(b).split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

HANDY_HINT = re.compile(
    r"(minor|small|touch ?up|adjust|tighten|hardware|door|hinge|latch|mirror|cord|patch|baseboard|trim|caulk|seal|paint|screws?)",
    re.I
)
HEAVY_HINT = re.compile(
    r"(septic|pump tank|leach field|well|breaker|panel|gfci|afci|condenser|furnace|coil|duct|rafter|ridge|roof leak|mold|fungal|encapsulation|sump|grading|french drain|retaining wall|insulated glass|igu|window pane)",
    re.I
)

def _handyman_eligible(text: str) -> bool:
    return bool(HANDY_HINT.search(text)) and not bool(HEAVY_HINT.search(text))

def _is_compound(text: str) -> bool:
    tokens = set(_canon(text).split())
    return len(tokens & ACTION_WORDS) >= 2

def _post_categorization_fixups(it: NormalizedLineItem) -> NormalizedLineItem:
    txt = f"{it.item} {it.verbatim or ''}".lower()
    if re.search(r"\b(septic|pump tank|leach field|well pump|onsite wastewater)\b", txt):
        it.category = "Septic & Well Systems"
    if "mirror" in txt and ("cord" in txt or "power" in txt):
        it.category = "Minor Handyman Repairs"
    if re.search(r"\b(caulk|seal)\b", txt) and re.search(r"\b(paint|painted)\b", txt) and re.search(r"\b(trim|baseboard|baseboards)\b", txt):
        it.category = "Minor Handyman Repairs"
    return it

def _pick_preferred(a: NormalizedLineItem, b: NormalizedLineItem, context_text: str) -> NormalizedLineItem:
    if (a.category == "Minor Handyman Repairs" or b.category == "Minor Handyman Repairs") and _handyman_eligible(context_text):
        return a if a.category == "Minor Handyman Repairs" else b
    sev_rank = {"high": 3, "medium": 2, "low": 1}
    sa = sev_rank.get(str(a.severity or "").lower(), 0)
    sb = sev_rank.get(str(b.severity or "").lower(), 0)
    if sa != sb:
        return a if sa > sb else b
    ia = FREQUENCY_ORDER.index(a.category) if a.category in FREQUENCY_ORDER else 999
    ib = FREQUENCY_ORDER.index(b.category) if b.category in FREQUENCY_ORDER else 999
    return a if ia <= ib else b

def _similar_actions(a: NormalizedLineItem, b: NormalizedLineItem) -> bool:
    ak = _canon(a.item + " " + (a.verbatim or ""))
    bk = _canon(b.item + " " + (b.verbatim or ""))
    if len(ak) >= 12 and len(bk) >= 12 and (ak in bk or bk in ak):
        return True
    sims = [
        _jaccard(a.item, b.item),
        _jaccard(a.item, b.verbatim or ""),
        _jaccard(a.verbatim or "", b.item),
        _jaccard(a.verbatim or "", b.verbatim or ""),
    ]
    base = max(sims)
    if base >= 0.66:
        return True
    DOMAIN_NOUNS = {"siding","drain","pipe","elbow","flashing","panel","mirror","cord","baseboard","trim","rafter","deck","retaining","wall","insulation","filter","return","duct","septic","window","pane","glass","vapor","barrier"}
    A = set(_canon(a.item).split()) | set(_canon(a.verbatim or "").split())
    B = set(_canon(b.item).split()) | set(_canon(b.verbatim or "").split())
    if len((A & B) & DOMAIN_NOUNS) > 0 and base >= 0.52:
        return True
    return False

def dedupe_and_arbitrate(items: List[NormalizedLineItem]) -> List[NormalizedLineItem]:
    fixed = [_post_categorization_fixups(it) for it in items]

    n = len(fixed)
    used = [False] * n
    keep_indices: List[int] = []

    by_verbatim: Dict[str, List[int]] = {}
    for i, it in enumerate(fixed):
        vk = _vkey(it.verbatim)
        by_verbatim.setdefault(vk, []).append(i)

    for vk, idxs in by_verbatim.items():
        if len(idxs) == 1:
            continue
        group_text = " ".join((fixed[i].verbatim or fixed[i].item) for i in idxs)
        handyman_idxs = [i for i in idxs if fixed[i].category == "Minor Handyman Repairs"]
        if handyman_idxs and _handyman_eligible(group_text):
            best = max(handyman_idxs, key=lambda i: len(fixed[i].item))
            for i in idxs:
                used[i] = True
            used[best] = False
            keep_indices.append(best)
            continue
        compounds = [i for i in idxs if _is_compound(fixed[i].item) or _is_compound(fixed[i].verbatim or "")]
        if len(idxs) > 1 and compounds:
            for i in compounds:
                used[i] = True

    pool = [i for i in range(n) if not used[i]]

    clusters: List[List[int]] = []
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
            continue
        context = " ".join((fixed[k].verbatim or fixed[k].item) for k in g)
        winner = fixed[g[0]]
        for k in g[1:]:
            winner = _pick_preferred(winner, fixed[k], context)
        sev_rank = {"high": 3, "medium": 2, "low": 1}
        best = sev_rank.get((winner.severity or "").lower(), 0)
        exps = [winner.explanation] if winner.explanation else []
        for k in g:
            if fixed[k] is winner:
                continue
            s = sev_rank.get((fixed[k].severity or "").lower(), 0)
            if s > best:
                winner.severity = fixed[k].severity
                best = s
            if (fixed[k].location or "") and len(fixed[k].location or "") > len(winner.location or ""):
                winner.location = fixed[k].location
            if fixed[k].explanation and fixed[k].explanation not in exps:
                exps.append(fixed[k].explanation)
        if exps:
            winner.explanation = " | ".join(exps)
        result.append(winner)

    return result

# ---------- Merge heuristics ----------
def _is_substantial_overlap(text1: str, text2: str, threshold: float = 0.8) -> bool:
    if not text1 or not text2:
        return False
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 or not words2:
        return False
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    similarity = len(intersection) / len(union) if union else 0
    return similarity >= threshold

def _merge_with_heuristics(full_text: str, normalized: List[NormalizedLineItem]) -> Tuple[List[NormalizedLineItem], int]:
    heur_items = []
    for m in NUMBERED_OR_BULLET.finditer(full_text):
        line = (m.group(1) or "").strip()
        if len(line) < 6:
            continue
        if REPAIR_HINT.search(line):
            heur_items.append(NormalizedLineItem(category="Minor Handyman Repairs", item=line, verbatim=line))
    added = 0
    for h in heur_items:
        dup = False
        for ex in normalized:
            if _similar_actions(h, ex):
                dup = True
                break
        if not dup:
            normalized.append(h); added += 1
    return normalized, added

# ---------------- Core extraction flow ----------------
def extract_repairs_comprehensive(text: str, address: str = "") -> Tuple[List[NormalizedLineItem], List[IgnoredExample], Dict]:
    estimated_tokens = len(text) // 4
    system_tokens = len(SYSTEM_PROMPT) // 4 + 200

    def _normalize_from_result(result_obj: dict) -> Tuple[List[NormalizedLineItem], List[IgnoredExample]]:
        items: List[NormalizedLineItem] = []
        ignored: List[IgnoredExample] = []
        its = result_obj.get("items", [])
        if isinstance(its, list):
            for it in its:
                item_text = (it.get("item") or "").strip() if isinstance(it, dict) else ""
                if not item_text:
                    continue
                items.append(NormalizedLineItem(
                    category=it.get("category") or "Minor Handyman Repairs",
                    item=item_text,
                    verbatim=it.get("verbatim") or item_text,
                    location=it.get("location"),
                    qty=it.get("qty"),
                    units=it.get("units"),
                    severity=it.get("severity"),
                    explanation=it.get("explanation")
                ))
        ign = result_obj.get("ignored_examples", [])
        if isinstance(ign, list):
            for g in ign:
                v = (g.get("verbatim") or "").strip() if isinstance(g, dict) else ""
                if v:
                    ignored.append(IgnoredExample(verbatim=v, why=g.get("why")))
        return items, ignored

    # Single-shot path
    if estimated_tokens + system_tokens <= MAX_TOKENS - RESPONSE_MAX_TOKENS:
        result = call_gpt5_extraction(text, address)
        norm_items, ignored = _normalize_from_result(result)
        norm_items = dedupe_and_arbitrate(norm_items)
        meta = {
            "mode": "single_request",
            "estimated_tokens": estimated_tokens,
            "items_from_llm": len(norm_items),
            "total_found": result.get("total_items_found"),
            "extracted_address": result.get("property_address"),
            "error": result.get("error")
        }
        merged_items, added_count = _merge_with_heuristics(text, norm_items)
        merged_items = dedupe_and_arbitrate(merged_items)
        meta["merged_heuristics_added"] = added_count
        return merged_items, ignored, meta

    # Chunked path
    chunks = chunk_text_by_tokens(text, MAX_TOKENS - system_tokens - RESPONSE_MAX_TOKENS)
    all_norm: List[NormalizedLineItem] = []
    all_ignored: List[IgnoredExample] = []

    for chunk in chunks:
        result = call_gpt5_extraction(chunk, address)
        n, ig = _normalize_from_result(result)
        all_norm.extend(n)
        all_ignored.extend(ig)

    deduped = dedupe_and_arbitrate(all_norm)
    merged_items, added_count = _merge_with_heuristics(text, deduped)
    merged_items = dedupe_and_arbitrate(merged_items)

    meta = {
        "mode": "multi_chunk",
        "chunks": len(chunks),
        "estimated_tokens": estimated_tokens,
        "items_from_llm": len(deduped),
        "merged_heuristics_added": added_count,
        "total_raw_items": len(all_norm)
    }
    return merged_items, all_ignored, meta

# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=ParsedResponse)
async def upload(
    file: UploadFile = File(...),
    address: str = Form(default=""),
    notes: str = Form(default="")
):
    data = await file.read()
    name = (file.filename or "").lower()

    if name.endswith(".pdf"):
        text = extract_pages_text_from_pdf(data)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    elif name.endswith((".png", ".jpg", ".jpeg")):
        text = extract_text_from_image(data)
    else:
        raise HTTPException(status_code=400, detail="Supported: .pdf, .txt, .png, .jpg, .jpeg")

    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail="No meaningful text could be extracted from the file.")

    normalized_items: List[NormalizedLineItem] = []
    ignored_examples: List[IgnoredExample] = []
    items_for_display: List[LineItem] = []

    meta = {
        "llm_used": False,
        "extraction_method": "none",
        "text_length": len(text)
    }

    if OPENAI_API_KEY and _openai_client:
        try:
            normalized_items, ignored_examples, extraction_meta = extract_repairs_comprehensive(text, address)
            meta.update({
                "llm_used": True,
                "extraction_method": "gpt5_comprehensive",
                **extraction_meta
            })
        except Exception as e:
            meta["llm_error"] = str(e)
            meta["extraction_method"] = "gpt5_failed"

    if not normalized_items:
        # last-resort regex fallback
        fallback = simple_fallback_parse(text)
        normalized_items = [
            NormalizedLineItem(category=i.category, item=i.item_text, verbatim=i.item_text)
            for i in fallback
        ]
        meta["extraction_method"] = "fallback_regex"
        items_for_display = fallback
    else:
        items_for_display = [LineItem(category=it.category, item_text=it.item) for it in normalized_items]

    return ParsedResponse(
        address=address or None,
        notes=notes or None,
        items=items_for_display,
        normalized_items=normalized_items,
        ignored_examples=ignored_examples,
        meta=meta
    )

# Run: uvicorn app:app --reload

