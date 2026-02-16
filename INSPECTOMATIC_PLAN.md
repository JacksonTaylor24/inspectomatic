# Inspectomatic — Project Plan
Single Source of Truth for Architecture, Current Functionality, and Roadmap

---

## 1) What the app is
Inspectomatic is a web app that ingests uploaded documents (PDF / image / txt) and produces a clean, categorized repair scope report with:
- actionable repair items grouped by standardized categories
- pricing ranges per item and total estimated repair range
- recommended local providers (top 3 per category) based on Google Places results
- a “beautiful report” view and a client-side exported PDF

It supports:
- Home Inspection Reports
- Repair Proposals / Contractor Estimates
- Rejects unrelated documents with a clear message

---

## 2) Current stack and deployment targets
### Backend
- Framework: FastAPI
- Template/UI serving: Jinja templates + static assets mounted under `/static`
- Main API: `POST /upload` returns JSON report payload
- Hosting target: Fly.io

### Frontend
- Wizard UI in `web/` (served by backend)
- Report rendering: built in-browser from API response
- PDF export: browser-side `html2pdf` with page-break avoidance rules
- Hosting target (future): Vercel (frontend separated + calls backend API)

### Storage + DB (roadmap)
- Postgres: users, documents, reports, billing metadata
- Object storage (S3-compatible): uploaded PDFs + generated artifacts

---

## 3) Supported document types and gating rules
The system must classify each uploaded document as:
- Inspection Report
- Repair Proposal
- Unsupported / Unrelated

Gating requirements:
- Classification happens before extraction logic.
- If unsupported OR too uncertain, the app must refuse analysis and return a helpful message:
  "Unable to complete analysis. This document does not appear to be an inspection report or repair proposal. Did you upload the right document?"
- Do not generate a report for unsupported docs.

Current implementation notes:
- Extract text from PDF (native) and fall back to OCR if needed
- Use an LLM-based open-set gate with temperature 0 for determinism

---

## 4) Extraction pipeline (current behavior)
### Text extraction
- PDFs: try native extraction via PdfReader; if insufficient text, OCR via pdf2image + pytesseract
- Images: OCR via pytesseract
- TXT: decode and use directly
- Reject if extracted text is too short / meaningless

### LLM extraction (Claude)
- Two separate system prompts:
  - inspection report prompt (transform findings into contractor-quotable items)
  - repair proposal prompt (extract concrete scope tasks)
- Must ignore boilerplate/meta-instructions (e.g., “Seller to repair all items in report…”)
- Output must use EXACT standardized category names
- If document is too large, chunk and extract across chunks, then dedupe/merge

### Fallback extraction
- Regex-based fallback parser finds actionable repair lines (action verbs), filters headers/boilerplate/meta instructions

### Post-processing
- Remove header-like lines
- Dedupe and arbitrate duplicates (favor more specific items)
- Ensure each item has a category explanation fallback if missing

---

## 5) Determinism and stability requirements (critical)
Non-negotiables:
- Same input document must produce stable results as much as possible.
- Stable ordering is required before pricing so indexes map correctly.

Current mechanism:
- Deterministic stable sort of `normalized_items` BEFORE pricing attachment, keyed by:
  1) category rank (FREQUENCY_ORDER)
  2) canonical item text
  3) canonical location
  4) canonical verbatim

Additional rules:
- Temperature must be 0 for classification, extraction, and pricing calls.
- Avoid index drift (no relying on non-deterministic ordering).

---

## 6) Pricing engine (current behavior)
Pricing is produced by an LLM pricing subsystem that must:
- Return exactly one pricing object per input index (coverage guarantee)
- Use batch pricing with retries for missing indices
- Attach pricing back to items by index
- Compute totals (low/high) across all priced items
- Drop items with an exact 0–0 price range (current filtering rule)

Output requirements per item:
- price low/high (USD)
- basis (per job/per unit/per sq ft/etc)
- confidence (low/medium/high)
- notes/assumptions

---

## 7) Provider recommendations (current behavior)
The report must include recommended providers by category (target: top 3).

Current implementation:
- Requires `GOOGLE_MAPS_API_KEY`
- Geocode the user-provided address
- For each used category:
  - map category to provider keywords (CATEGORY_PROVIDER_MAP)
  - Google Places Nearby search (expanding radii 10/20/40 miles)
  - filter: rating >= 4.0 AND review_count >= 5
  - score via Bayesian-style smoothing:
    score = (rating * reviews + prior_mean * prior_weight) / (reviews + prior_weight)
  - take top 3 candidates by score + review_count
  - enrich via Place Details: phone, website, formatted_address
- Store provider results in response payload under `meta.providers[category]`

Requirements:
- Cache/rate-limit provider calls (roadmap improvement)
- Provider results must be per-report reproducible (store results with report run)

---

## 8) Report rendering (current behavior)
The system outputs two layers:
1) Structured JSON payload from backend (`items`, `normalized_items`, `meta`)
2) A “beautiful report” generated client-side from the payload

Frontend report requirements:
- Display grouped categories in CATEGORY_ORDER
- Show severity badges, location, qty/units when available
- Show per-item price ranges when present
- Show per-category “Recommended Providers” blocks (top 3 + blanks if missing)
- Show total repair range pill at the top when totals exist
- Show generated timestamp and doc type label

PDF export requirements (client-side):
- Export current report view to PDF via html2pdf
- Avoid page breaks inside each item block and provider section:
  - `.no-break` elements must not split
  - `pagebreak: { mode: ["css","legacy"], avoid: [".no-break"] }`

Important:
- PDF is currently generated on the client; backend does not render PDFs yet.

---

## 9) Configuration (current)
Environment variables in use:
- GOOGLE_MAPS_API_KEY (provider lookup + geocoding)
- ANTHROPIC_API_KEY (Claude extraction/pricing)
- CLAUDE_MODEL (default: claude-sonnet-4-5-20250929)
- CLAUDE_MAX_TOKENS, CLAUDE_RESPONSE_MAX_TOKENS

---

## 10) Roadmap: user accounts, history, billing
### User management
Users can:
- sign up / log in / log out
- view past runs
- open the original uploaded document and the generated report

### Artifact persistence
For each run, store:
- original upload (PDF/image/txt)
- normalized JSON output
- rendered report artifact(s)
- provider results snapshot
- pricing totals and pricing attachments

### Plans and billing
- Free vs paid plan tiers
- Payment provider: Stripe
- Webhooks must update entitlements
- Backend enforces usage limits (uploads/reports/pages per period)

---

## 11) Workstreams and priorities
### P0 — Reliability and correctness
- Preserve doc gate behavior (inspection/proposal/unsupported) and refusal messages
- Preserve stable sort before pricing
- Add regression tests (golden PDFs) to detect drift
- Improve deterministic behavior and logging

### P1 — Pricing completeness
- Ensure every item is priced or explicitly marked with reason code (future improvement)
- Tighten pricing mapping and retry behavior

### P2 — Provider system hardening
- Add caching, rate limits, and failure-tolerant behavior
- Persist provider snapshot per report run

### P3 — Accounts + history
- Auth + sessions
- Stored artifacts + “My Reports” dashboard
- Multi-tenant access control

### P4 — Billing
- Stripe checkout + subscription
- Webhook entitlements
- Usage limits

### P5 — Frontend/Backend split
- Deploy frontend to Vercel
- Deploy API to Fly.io
- Configure CORS + secure auth token flow

---

## 12) Definition of done (any change)
- Existing functionality is preserved (doc gate, extraction, pricing, providers, report rendering, PDF export)
- Determinism checked (same input -> stable normalized JSON ordering and pricing attachment)
- Tests added/updated
- No cross-tenant data access
- Clear logs and run IDs for debugging
- Small, reviewable diffs
