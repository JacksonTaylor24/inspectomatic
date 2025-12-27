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

# ---------------- AI client setup ----------------
AI_PROVIDER = os.getenv("AI_PROVIDER", "claude")  # "claude" or "openai"

# Claude setup
#CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MAX_TOKENS = 200000
CLAUDE_RESPONSE_MAX_TOKENS = 8192

# OpenAI setup
OPENAI_MODEL = "gpt-5"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MAX_TOKENS = 128000
OPENAI_RESPONSE_MAX_TOKENS = 16000

# Set active config based on provider
if AI_PROVIDER.lower() == "claude":
    MODEL_NAME = CLAUDE_MODEL
    API_KEY = ANTHROPIC_API_KEY
    MAX_TOKENS = CLAUDE_MAX_TOKENS
    RESPONSE_MAX_TOKENS = CLAUDE_RESPONSE_MAX_TOKENS
else:
    MODEL_NAME = OPENAI_MODEL
    API_KEY = OPENAI_API_KEY
    MAX_TOKENS = OPENAI_MAX_TOKENS
    RESPONSE_MAX_TOKENS = OPENAI_RESPONSE_MAX_TOKENS

# Initialize clients
_anthropic_client = None
_openai_client = None

try:
    import anthropic
    _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
except Exception:
    _anthropic_client = None

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

# ---------------- Default category explanations ----------------
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
    basis: str = "per job"  # "per job", "per unit", etc.
    confidence: str = "medium"  # "low" | "medium" | "high"
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

# ---------------- Enhanced System Prompt ----------------
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

DO EXTRACT specific actionable tasks:
- "Replace broken window pane in master bedroom"
- "Seal gap around exterior door frame with caulk"
- "Install GFCI outlet in guest bathroom"
- "Repair loose handrail on front porch steps"
- "Clean gutters and downspouts"
"""

SYSTEM_PROMPT = f"""Think very carefully in your answers, reason through everything step by step.

CRITICAL FILTERING RULE: Only extract specific, actionable repair tasks that a contractor can quote and complete. IGNORE all meta-instructions, general directives, and summary statements about completing work.

{NEGATIVE_EXAMPLES}

You are an expert inspection report analyzer for Inspectomatic, a real estate tool that helps buyers and sellers negotiate repair costs by providing accurate categorization and cost estimates. Your job is to transform overwhelming inspection reports into organized, actionable repair lists that contractors can quote and complete.

CORE MISSION: Extract EVERY actionable repair/maintenance item and categorize it by the **service provider who would actually do the work**. Skip any organizational language, summary statements, or instructions about who should complete work.

FILTERING CRITERIA:
- EXTRACT: Specific tasks with clear actions (repair, replace, install, seal, clean, adjust)
- SKIP: General instructions, summary statements, references to "all items" or "per report"
- SKIP: Lines that talk about "the following items", "items below", or "these items" without naming a specific repair
- SKIP: Lines mentioning "seller," "buyer," "outlined in," "attached," "blue tape," "complete all"

Use EXACTLY these category names:
{", ".join(CATEGORIES)}

When borderline between categories, prefer the earlier one in this frequency list:
{FREQ_TEXT}

UNIQUENESS & SPLITTING:
- From any single source line, output **either** ONE handyman item (if a competent handyman can reasonably do the work) **or** multiple trade-specific items — **never both**.
- Do **not** duplicate the same action across multiple categories.

HANDYMAN CONSOLIDATION PRIORITY:
- **MINIMIZE TOTAL CONTRACTORS** - When in doubt between a specialist and handyman, choose **"Minor Handyman Repairs"** to reduce the number of professionals needed
- Only use specialists for work that truly requires licensing, permits, or specialized tools/knowledge
- Prioritize practical job completion over theoretical trade boundaries

Handyman-appropriate work includes:
- Simple door installations, hardware, and basic trim work
- Access panels, hatches, and basic openings (attic access, crawl space doors)
- Basic shelving installation and simple carpentry
- Minor caulking, touch-up paint, and small patches
- Simple plumbing fixtures (toilet seats, basic hardware) - but NOT major plumbing repairs
- Basic electrical work like outlet covers, switch plates - but NOT wiring or panel work

SPECIALIST-ONLY categories (do NOT move to handyman):
- Major plumbing: pipe repairs, water heater work, drain clearing, fixture installation requiring plumbing knowledge
- Electrical: any wiring, breaker panels, GFCI installation, electrical troubleshooting
- HVAC: any system repairs, ductwork, unit repairs, damper adjustments
- Roofing: anything involving roof structure, shingles, major flashing
- Major structural work requiring permits

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
- **Simple access doors, hatches, basic hardware** → **Minor Handyman Repairs**.

Output JSON format:
{{
  "items": [
    {{
      "category": "<one of the categories above>",
      "item": "<actionable task>",
      "verbatim": "<exact source text>",
      "location": "<where if given>",
      "qty": <number or null>,
      "units": "<units or null>",
      "severity": "low|medium|high",
      "explanation": "<REQUIRED: explain why this category vs. handyman - what specialized knowledge, tools, or licensing is needed>"
    }}
  ],
  "ignored_examples": [
    {{
      "verbatim": "<non-actionable text>",
      "why": "<short reason>"
    }}
  ],
  "property_address": "<if found>",
  "total_items_found": <number>
}}

EXPLANATION REQUIREMENTS:
- For "Minor Handyman Repairs": Brief explanation like "Basic hardware installation" or "Simple repair within handyman scope"
- For specialist categories: MUST explain why a specialist is needed over a handyman
  * Plumbing: "Requires plumbing knowledge for proper connections/drainage/water pressure"
  * Electrical: "Requires electrical licensing and knowledge of wiring/safety codes" 
  * HVAC: "Requires HVAC certification and specialized tools for system diagnostics"
  * Roofing: "Requires roofing expertise and safety equipment for structural work"
  * Carpentry & Trim: "Requires advanced carpentry skills beyond basic handyman scope"
  * etc.
"""

# Pricing system prompt
PRICING_SYSTEM_PROMPT = """
You are a home repair cost estimator for a tool used in real estate deals.
You will be given a list of repair items that have already been categorized by trade.

Your job:
- For EACH item, estimate a realistic price RANGE in USD (low and high).
- These are ballpark estimates for negotiation, not binding quotes.
- Assume work is being done by properly insured, professional contractors (not unlicensed handymen),
  in a typical mid-cost U.S. metro area, unless otherwise indicated by the address.

RANGE TIGHTNESS RULES (VERY IMPORTANT):
- Your goal is to provide **usable, fairly tight negotiation ranges**, not huge uncertainty bands.
- In normal cases:
  * HIGH should typically be within about **20–40% above LOW**.
  * As a numeric guideline, in most cases use:  high ≈ low * 1.2 to low * 1.4
- Only widen the range more than this when there is clear, explicit uncertainty in scope
  (e.g. "may need full replacement", "extent of damage unknown").
- Avoid very wide bands like "$500–$2,500" unless truly unavoidable. It is better to choose
  your best judgment around a narrower range that would be reasonable for negotiations.

Other rules:
- Always return prices in USD.
- If quantity is provided (qty + units), scale your range accordingly.
- Severity:
  * low: minor repair or tune-up
  * medium: moderate repair or partial replacement
  * high: major repair or likely full replacement

When thinking about the numbers:
- Imagine what 2–3 local pros would actually quote.
- LOW should represent a solid but not "cut-rate" contractor.
- HIGH should represent the upper end of typical quotes, not extreme outliers.

Output JSON ONLY in this format:
{
  "items": [
    {
      "index": <int matching input index>,
      "price_low": <float>,
      "price_high": <float>,
      "currency": "USD",
      "basis": "per job" | "per unit" | "per sq ft" | "per linear ft",
      "confidence": "low" | "medium" | "high",
      "notes": "<short note explaining what you assumed for pricing>"
    }
  ]
}
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

# ---------------- AI extraction/pricing wrappers ----------------
def call_ai_extraction(text: str, address: str = "") -> dict:
    if AI_PROVIDER.lower() == "claude":
        return call_claude_extraction(text, address)
    else:
        return call_openai_extraction(text, address)

def call_claude_extraction(text: str, address: str = "") -> dict:
    if not ANTHROPIC_API_KEY or not _anthropic_client:
        return {"items": [], "ignored_examples": [], "error": "Anthropic API not configured"}

    try:
        user_message = f"""FILTERING REMINDER: Skip meta-instructions like "seller to repair all items" - only extract specific tasks.

CRITICAL: Output JSON format with explanations for each item explaining why that category vs handyman.

Property Address: {address}

Document Content:

{text}"""

        response = _anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_RESPONSE_MAX_TOKENS,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        result_text = response.content[0].text if response.content else "{}"
        print("\n=== DEBUG: Raw Claude Extraction (first 500 chars) ===")
        print(result_text[:500])
        print("=" * 50)

        parsed = _coerce_json(result_text)
        if isinstance(parsed, list):
            parsed = {"items": parsed, "ignored_examples": []}
        return parsed if isinstance(parsed, dict) else {"items": [], "ignored_examples": []}
    except Exception as e:
        return {"items": [], "ignored_examples": [], "error": str(e)}

def call_openai_extraction(text: str, address: str = "") -> dict:
    if not OPENAI_API_KEY or not _openai_client:
        return {"items": [], "ignored_examples": [], "error": "OpenAI API not configured"}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""FILTERING REMINDER: Skip meta-instructions like "seller to repair all items" - only extract specific tasks.

CRITICAL: Output JSON and include an "explanation" for every item explaining why that category vs. handyman.

Property Address: {address}

Document Content:

{text}"""}
    ]
    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            top_p=0.1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=OPENAI_RESPONSE_MAX_TOKENS,
        )
        result_text = resp.choices[0].message.content
        parsed = _coerce_json(result_text)
        if isinstance(parsed, list):
            parsed = {"items": parsed, "ignored_examples": []}
        return parsed if isinstance(parsed, dict) else {"items": [], "ignored_examples": []}
    except Exception as e:
        return {"items": [], "ignored_examples": [], "error": str(e)}

def call_claude_pricing(items: List[NormalizedLineItem], address: str = "") -> Dict[int, PriceEstimate]:
    if not ANTHROPIC_API_KEY or not _anthropic_client:
        return {}

    payload = []
    for idx, it in enumerate(items):
        payload.append({
            "index": idx,
            "category": it.category,
            "item": it.item,
            "location": it.location,
            "severity": it.severity,
            "qty": it.qty,
            "units": it.units,
        })

    user_message = f"""
Property Address (may help you infer region): {address or "Unknown"}

Here are the repair items to price (array of objects):
{json.dumps(payload, ensure_ascii=False)}
"""
    try:
        response = _anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            temperature=0,
            system=PRICING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        result_text = response.content[0].text if response.content else "{}"
        print("\n=== DEBUG: Raw Claude Pricing (first 500 chars) ===")
        print(result_text[:500])
        print("=" * 50)

        parsed = _coerce_json(result_text)
        if not isinstance(parsed, dict):
            return {}

        out: Dict[int, PriceEstimate] = {}
        for obj in parsed.get("items", []):
            try:
                idx = int(obj.get("index"))
            except Exception:
                continue
            try:
                pe = PriceEstimate(
                    low=float(obj.get("price_low", 0.0)),
                    high=float(obj.get("price_high", 0.0)),
                    currency=obj.get("currency", "USD"),
                    basis=obj.get("basis", "per job"),
                    confidence=obj.get("confidence", "medium"),
                    notes=obj.get("notes")
                )
                out[idx] = pe
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"Pricing error: {e}")
        return {}

def call_ai_pricing(items: List[NormalizedLineItem], address: str = "") -> Dict[int, PriceEstimate]:
    if AI_PROVIDER.lower() == "claude":
        return call_claude_pricing(items, address)
    return {}

# ---------- Heuristic capture ----------
NUMBERED_OR_BULLET = re.compile(r'^\s*(?:\d+\s*[\.)-]\s*|[-*•]\s+)?(.+)$', re.MULTILINE)
REPAIR_HINT = re.compile(
    r'\b(repair|replace|install|clean|seal|caulk|fix|adjust|service|test|leak|broken|missing|damaged|inoperable|not working|not cooling|paint|trim|flashing)\b',
    re.I
)

META_INSTRUCTION_PATTERN = re.compile(
    r'\b(seller|buyer|all items|complete all|per report|outlined in|attached|blue tape|remedy all|deficiencies noted|as recommended|noted defects|to be corrected)\b',
    re.I
)

SUMMARY_PATTERN = re.compile(
    r'^\s*(?:\d+\s*[\.)-]\s*)?(?:seller|buyer)\s+(?:to|shall|will|must)\s+(?:repair|complete|address|remedy)',
    re.I
)

def simple_fallback_parse(text: str) -> List[LineItem]:
    items: List[LineItem] = []
    for m in NUMBERED_OR_BULLET.finditer(text):
        line = (m.group(1) or "").strip()
        if len(line) < 6:
            continue
        if META_INSTRUCTION_PATTERN.search(line):
            continue
        if SUMMARY_PATTERN.search(line):
            continue
        if re.search(r'\b(report|inspection|recommendations?|items?)\b', line, re.I) and not REPAIR_HINT.search(line):
            continue
        if REPAIR_HINT.search(line):
            items.append(LineItem(category="Minor Handyman Repairs", item_text=line))
    return items

# ---- Dedupe & arbitration ----
_WORD = re.compile(r"[a-z0-9]+")
ACTION_WORDS = {"repair","replace","install","trim","add","seal","caulk","paint","fill","vacuum","reseed","regrade","balance","evaluate","adjust","sister","fasten","secure","patch"}

def _canon(s: str) -> str:
    if not s:
        return ""
    try:
        t = s.lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\b(please|ensure|properly|all|the|a|an|that|to|be|is|are|needs|need|should|with|of|for|on|in|at|and)\b", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    except Exception:
        return str(s).encode('ascii', 'ignore').decode('ascii').lower().strip()

def _vkey(s: Optional[str]) -> str:
    return _canon(s or "")

def _jaccard(a: str, b: str) -> float:
    A = set(_canon(a).split())
    B = set(_canon(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

HANDY_HINT = re.compile(
    r"(minor|small|touch ?up|adjust|tighten|hardware|door|hinge|latch|mirror|cord|patch|baseboard|trim|caulk|seal|paint|screws?)",
    re.I
)
HEAVY_HINT = re.compile(
    r"(septic|pump tank|leach field|well|breaker|panel|gfci|afci|condenser|furnace|coil|duct|rafter|ridge|roof leak|mold|fungal|encapsulation|sump|grading|french drain|retaining wall|insulation|filter|return|duct|septic|window pane|igu|insulated glass)",
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
    try:
        fixed = [_post_categorization_fixups(it) for it in items]
    except Exception as e:
        print(f"Error in post_categorization_fixups: {e}")
        fixed = items[:]

    n = len(fixed)
    used = [False] * n

    by_verbatim = {}
    for i, it in enumerate(fixed):
        try:
            verbatim_text = getattr(it, 'verbatim', None) or ""
            if verbatim_text:
                safe_verbatim = str(verbatim_text).encode('ascii', 'ignore').decode('ascii')
                vk = _vkey(safe_verbatim)
            else:
                vk = f"item_{i}"
        except Exception as e:
            print(f"Error processing verbatim for item {i}: {e}")
            vk = f"item_{i}"
        by_verbatim.setdefault(vk, []).append(i)

    for vk, idxs in by_verbatim.items():
        if len(idxs) == 1:
            continue
        try:
            group_texts = []
            for i in idxs:
                text = fixed[i].verbatim or fixed[i].item or ""
                safe_text = str(text).encode('ascii', 'ignore').decode('ascii') if text else ""
                group_texts.append(safe_text)
            group_text = " ".join(group_texts)

            handyman_idxs = [i for i in idxs if fixed[i].category == "Minor Handyman Repairs"]
            if handyman_idxs and _handyman_eligible(group_text):
                best = max(handyman_idxs, key=lambda i: len(fixed[i].item or ""))
                for i in idxs:
                    used[i] = True
                used[best] = False
                continue

            compounds = []
            for i in idxs:
                try:
                    item_text = fixed[i].item or ""
                    verbatim_text = fixed[i].verbatim or ""
                    if _is_compound(item_text) or _is_compound(verbatim_text):
                        compounds.append(i)
                except Exception as e:
                    print(f"Error checking compound for item {i}: {e}")
                    continue

            if len(idxs) > 1 and compounds:
                for i in compounds:
                    used[i] = True
        except Exception as e:
            print(f"Error processing group {vk}: {e}")
            continue

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
            try:
                if _similar_actions(fixed[i], fixed[j]):
                    g.append(j)
                    visited.add(j)
            except Exception as e:
                print(f"Error comparing items {i} and {j}: {e}")
                continue
        clusters.append(g)

    result: List[NormalizedLineItem] = []
    for g in clusters:
        try:
            if len(g) == 1:
                result.append(fixed[g[0]])
                continue

            context_parts = []
            for k in g:
                text = fixed[k].verbatim or fixed[k].item or ""
                safe_text = str(text).encode('ascii', 'ignore').decode('ascii') if text else ""
                context_parts.append(safe_text)
            context = " ".join(context_parts)

            winner = fixed[g[0]]
            sev_rank = {"high": 3, "medium": 2, "low": 1}
            best = sev_rank.get((winner.severity or "").lower(), 0)
            exps = [winner.explanation] if winner.explanation else []

            for k in g[1:]:
                try:
                    candidate = fixed[k]
                    winner = _pick_preferred(winner, candidate, context)
                    s = sev_rank.get((candidate.severity or "").lower(), 0)
                    if s > best:
                        winner.severity = candidate.severity
                        best = s
                    if (candidate.location or "") and len(candidate.location or "") > len(winner.location or ""):
                        winner.location = candidate.location
                    if candidate.explanation and candidate.explanation not in exps:
                        exps.append(candidate.explanation)
                except Exception as e:
                    print(f"Error merging attributes for item {k}: {e}")
                    continue

            if exps:
                try:
                    winner.explanation = " | ".join([e for e in exps if e])
                except Exception as e:
                    print(f"Error joining explanations: {e}")
                    winner.explanation = exps[0] if exps else None

            result.append(winner)
        except Exception as e:
            print(f"Error processing cluster: {e}")
            if g:
                result.append(fixed[g[0]])
            continue

    return result

# ---------- Merge heuristics ----------
def _merge_with_heuristics(full_text: str, normalized: List[NormalizedLineItem]) -> Tuple[List[NormalizedLineItem], int]:
    heur_items = []
    for m in NUMBERED_OR_BULLET.finditer(full_text):
        line = (m.group(1) or "").strip()
        if len(line) < 6:
            continue
        if META_INSTRUCTION_PATTERN.search(line):
            continue
        if SUMMARY_PATTERN.search(line):
            continue
        if re.search(r'\b(report|inspection|recommendations?|items?)\b', line, re.I) and not REPAIR_HINT.search(line):
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
            normalized.append(h)
            added += 1
    return normalized, added

# ---------------- Explanations backstop ----------------
def ensure_explanations(items: List[NormalizedLineItem]) -> List[NormalizedLineItem]:
    for it in items:
        if not it.explanation or not str(it.explanation).strip():
            it.explanation = DEFAULT_EXPLANATION_BY_CATEGORY.get(
                it.category,
                "Assigned based on required trade skills and safety/codes."
            )
    return items

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
        result = call_ai_extraction(text, address)
        norm_items, ignored = _normalize_from_result(result)
        norm_items = dedupe_and_arbitrate(norm_items)
        merged_items, added_count = _merge_with_heuristics(text, norm_items)
        merged_items = dedupe_and_arbitrate(merged_items)
        merged_items = ensure_explanations(merged_items)
        meta = {
            "mode": "single_request",
            "estimated_tokens": estimated_tokens,
            "items_from_llm": len(merged_items),
            "total_items_found": result.get("total_items_found"),
            "extracted_address": result.get("property_address"),
            "error": result.get("error"),
            "merged_heuristics_added": added_count,
            "model": MODEL_NAME,
            "ai_provider": AI_PROVIDER,
        }
        return merged_items, ignored, meta

    # Chunked path
    chunks = chunk_text_by_tokens(text, MAX_TOKENS - system_tokens - RESPONSE_MAX_TOKENS)
    all_norm: List[NormalizedLineItem] = []
    all_ignored: List[IgnoredExample] = []

    for chunk in chunks:
        result = call_ai_extraction(chunk, address)
        n, ig = _normalize_from_result(result)
        all_norm.extend(n)
        all_ignored.extend(ig)

    deduped = dedupe_and_arbitrate(all_norm)
    merged_items, added_count = _merge_with_heuristics(text, deduped)
    merged_items = dedupe_and_arbitrate(merged_items)
    merged_items = ensure_explanations(merged_items)

    meta = {
        "mode": "multi_chunk",
        "chunks": len(chunks),
        "estimated_tokens": estimated_tokens,
        "items_from_llm": len(deduped),
        "merged_heuristics_added": added_count,
        "total_raw_items": len(all_norm),
        "model": MODEL_NAME,
        "ai_provider": AI_PROVIDER,
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

    meta: Dict = {
        "llm_used": False,
        "extraction_method": "none",
        "text_length": len(text),
        "model": MODEL_NAME,
        "ai_provider": AI_PROVIDER,
    }

    # Extraction
    if API_KEY and ((AI_PROVIDER.lower() == "claude" and _anthropic_client) or (AI_PROVIDER.lower() == "openai" and _openai_client)):
        try:
            normalized_items, ignored_examples, extraction_meta = extract_repairs_comprehensive(text, address)
            print("\n=== DEBUG: First 3 extracted items ===")
            for i, item in enumerate(normalized_items[:3]):
                print(f"{i+1}. Item: {item.item}")
                print(f"   Category: {item.category}")
                print(f"   Explanation: {item.explanation}")
                print("---")
            meta.update({
                "llm_used": True,
                "extraction_method": f"{AI_PROVIDER.lower()}_comprehensive",
                **extraction_meta,
            })
        except Exception as e:
            meta["llm_error"] = str(e)
            meta["extraction_method"] = f"{AI_PROVIDER.lower()}_failed"

    if not normalized_items:
        fallback = simple_fallback_parse(text)
        normalized_items = [
            NormalizedLineItem(category=i.category, item=i.item_text, verbatim=i.item_text)
            for i in fallback
        ]
        normalized_items = ensure_explanations(normalized_items)
        meta["extraction_method"] = "fallback_regex"

    # --- Pricing step (Claude-only for now) ---
    pricing_totals = {"low": 0.0, "high": 0.0, "currency": "USD"}
    meta["pricing_used"] = False

    if normalized_items and AI_PROVIDER.lower() == "claude" and _anthropic_client:
        try:
            pricing_map = call_ai_pricing(normalized_items, address)
            print(f"=== DEBUG: Pricing map size: {len(pricing_map)} ===")

            # Attach prices
            for idx, it in enumerate(normalized_items):
                if idx in pricing_map:
                    it.price = pricing_map[idx]

            # NEW: drop all items the pricing agent says are $0–$0
            filtered_items = []
            for it in normalized_items:
                if it.price and it.price.low == 0 and it.price.high == 0:
                    # treat as non-actionable / meta, drop it entirely
                    print(f"DEBUG: Dropping zero-priced item: {it.item}")
                    continue
                filtered_items.append(it)
            normalized_items = filtered_items

            # Recompute totals only from non-zero items
            for it in normalized_items:
                if it.price:
                    pricing_totals["low"] += it.price.low
                    pricing_totals["high"] += it.price.high
                    pricing_totals["currency"] = it.price.currency or "USD"

            if pricing_totals["low"] > 0 or pricing_totals["high"] > 0:
                meta["pricing_totals"] = pricing_totals
                meta["pricing_used"] = True
        except Exception as e:
            meta["pricing_error"] = str(e)


    # Build display items
    for it in normalized_items:
        if not it.explanation:
            it.explanation = DEFAULT_EXPLANATION_BY_CATEGORY.get(
                it.category, "Assigned based on required trade skills and safety/codes."
            )

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

    return ParsedResponse(
        address=address or None,
        notes=notes or None,
        items=items_for_display,
        normalized_items=normalized_items,
        ignored_examples=ignored_examples,
        meta=meta
    )

# Run: uvicorn app:app --reload
