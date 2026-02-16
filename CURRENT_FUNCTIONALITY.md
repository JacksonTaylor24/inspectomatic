# Inspectomatic Current Functionality (Repo Scan)

This document reflects the current implementation found in the repository at scan time.

## Runtime Entry Points and Endpoints

Active FastAPI app is defined in `projects/inspectomatic/app.py:45`.

- `GET /` renders the main UI template (`index.html`).
Reference: `projects/inspectomatic/app.py:1012`, `projects/inspectomatic/app.py:1013`, `projects/inspectomatic/app.py:1014`
- `POST /upload` accepts multipart form data (`file`, `address`, `notes`) and returns a `ParsedResponse` JSON payload.
Reference: `projects/inspectomatic/app.py:1016`, `projects/inspectomatic/app.py:1017`, `projects/inspectomatic/app.py:1018`, `projects/inspectomatic/app.py:1019`, `projects/inspectomatic/app.py:1020`, `projects/inspectomatic/app.py:1203`
- `/static/*` is mounted for frontend assets.
Reference: `projects/inspectomatic/app.py:48`

Frontend submits uploads to `/upload` via `fetch`.
Reference: `projects/inspectomatic/web/static/script.js:197`

## Extraction Flow

### 1) File ingestion and text extraction

`/upload` branches by extension:
- PDF: native extraction with `pypdf`, OCR fallback with `pdf2image + pytesseract`
References: `projects/inspectomatic/app.py:1032`, `projects/inspectomatic/app.py:301`, `projects/inspectomatic/app.py:322`
- TXT: UTF-8 decode
Reference: `projects/inspectomatic/app.py:1034`, `projects/inspectomatic/app.py:1035`
- Image (`.png/.jpg/.jpeg`): OCR via PIL + pytesseract
References: `projects/inspectomatic/app.py:1038`, `projects/inspectomatic/app.py:335`
- Unsupported extension: HTTP 400
Reference: `projects/inspectomatic/app.py:1042`, `projects/inspectomatic/app.py:1043`
- If extracted text is too short: HTTP 422
Reference: `projects/inspectomatic/app.py:1045`, `projects/inspectomatic/app.py:1046`

### 2) LLM document gate (open-set classification)

Before extraction, text is classified as in-domain/out-of-domain and typed (`inspection_report` / `repair_proposal` / `unknown`) using Claude when configured.
References: `projects/inspectomatic/app.py:1051`, `projects/inspectomatic/app.py:1052`, `projects/inspectomatic/app.py:371`

Refusal behavior:
- Out-of-domain or low-confidence unknown => returns empty items with a user-facing refusal message in `meta.user_message`.
References: `projects/inspectomatic/app.py:1080`, `projects/inspectomatic/app.py:1082`, `projects/inspectomatic/app.py:1083`
- Unknown doc type => returns empty items with a confidence message.
References: `projects/inspectomatic/app.py:1085`, `projects/inspectomatic/app.py:1086`, `projects/inspectomatic/app.py:1087`

### 3) Repair item extraction

Primary path:
- Uses Claude extraction prompts (different system prompt by doc type).
References: `projects/inspectomatic/app.py:526`, `projects/inspectomatic/app.py:565`, `projects/inspectomatic/app.py:1094`, `projects/inspectomatic/app.py:1098`
- Handles single-request or chunked extraction based on token budget.
References: `projects/inspectomatic/app.py:839`, `projects/inspectomatic/app.py:877`, `projects/inspectomatic/app.py:894`
- Normalizes into `NormalizedLineItem`, filters section-header-like items, deduplicates/arbitrates, ensures explanations.
References: `projects/inspectomatic/app.py:844`, `projects/inspectomatic/app.py:855`, `projects/inspectomatic/app.py:781`, `projects/inspectomatic/app.py:832`, `projects/inspectomatic/app.py:1114`, `projects/inspectomatic/app.py:1115`, `projects/inspectomatic/app.py:1116`

Fallback path:
- If no normalized items were extracted, uses regex-based fallback parser and assigns default category.
References: `projects/inspectomatic/app.py:1104`, `projects/inspectomatic/app.py:736`, `projects/inspectomatic/app.py:1111`

Final pre-pricing ordering:
- Stable sort is applied before pricing to keep index-to-price mapping deterministic.
References: `projects/inspectomatic/app.py:281`, `projects/inspectomatic/app.py:1118`, `projects/inspectomatic/app.py:1120`

## Pricing Flow

Pricing runs only when there are normalized items and Anthropic is configured.
References: `projects/inspectomatic/app.py:1127`, `projects/inspectomatic/app.py:624`

Flow details:
- Batch payload built from normalized items (category/item/location/severity/qty/units).
References: `projects/inspectomatic/app.py:631`, `projects/inspectomatic/app.py:635`
- Claude pricing prompt enforces one output object per input index.
References: `projects/inspectomatic/app.py:530`, `projects/inspectomatic/app.py:533`, `projects/inspectomatic/app.py:534`
- Batch pass + retry passes for missing indexes.
References: `projects/inspectomatic/app.py:677`, `projects/inspectomatic/app.py:687`, `projects/inspectomatic/app.py:689`
- Parsed pricing is attached back to items by index.
References: `projects/inspectomatic/app.py:646`, `projects/inspectomatic/app.py:1133`, `projects/inspectomatic/app.py:1135`
- Items priced at exactly `0-0` are dropped.
References: `projects/inspectomatic/app.py:1140`, `projects/inspectomatic/app.py:1143`
- Totals (`low`, `high`, `currency`) are aggregated into `meta.pricing_totals`.
References: `projects/inspectomatic/app.py:1148`, `projects/inspectomatic/app.py:1154`, `projects/inspectomatic/app.py:1155`

Pricing metadata included in response:
- `pricing_attempted`, `pricing_items_in`, `pricing_items_priced`, `pricing_used`, `pricing_totals`, optional `pricing_error`.
References: `projects/inspectomatic/app.py:1123`, `projects/inspectomatic/app.py:1124`, `projects/inspectomatic/app.py:1125`, `projects/inspectomatic/app.py:1137`, `projects/inspectomatic/app.py:1155`, `projects/inspectomatic/app.py:1157`

## Provider Lookup Flow

Provider lookup runs only when all three are true:
- `GOOGLE_MAPS_API_KEY` present
- `address` provided
- at least one normalized item
Reference: `projects/inspectomatic/app.py:1161`

Flow details:
- Geocode address using Google Geocoding API.
References: `projects/inspectomatic/app.py:923`, `projects/inspectomatic/app.py:928`
- For each used category, map to search keywords, search Places Nearby with expanding radii (10/20/40 miles), filter (`rating >= 4.0`, `reviews >= 5`), rank, enrich with Place Details, keep top 3.
References: `projects/inspectomatic/app.py:968`, `projects/inspectomatic/app.py:969`, `projects/inspectomatic/app.py:971`, `projects/inspectomatic/app.py:977`, `projects/inspectomatic/app.py:985`, `projects/inspectomatic/app.py:999`, `projects/inspectomatic/app.py:1000`, `projects/inspectomatic/app.py:1003`
- Results are returned in `meta.providers` keyed by category.
References: `projects/inspectomatic/app.py:1165`, `projects/inspectomatic/app.py:1168`, `projects/inspectomatic/app.py:1171`

## Report Rendering and PDF Export

### Frontend rendering pipeline

- `index.html` loads UI, report container, and scripts/libraries.
References: `projects/inspectomatic/web/index.html:84`, `projects/inspectomatic/web/index.html:88`, `projects/inspectomatic/web/index.html:100`
- On submit, frontend posts form data to `/upload`, stores response, and calls `buildAndDisplayReport`.
References: `projects/inspectomatic/web/static/script.js:190`, `projects/inspectomatic/web/static/script.js:197`, `projects/inspectomatic/web/static/script.js:204`, `projects/inspectomatic/web/static/script.js:224`
- Report builder prefers `data.items`; falls back to `data.normalized_items`.
References: `projects/inspectomatic/web/static/script.js:249`, `projects/inspectomatic/web/static/script.js:267`
- Items are sorted by `CATEGORY_ORDER`, then severity, then item text.
References: `projects/inspectomatic/web/static/script.js:293`, `projects/inspectomatic/web/static/script.js:301`, `projects/inspectomatic/web/static/script.js:305`
- HTML report includes:
  - header and metadata
  - total pricing range pill when totals are non-zero
  - category sections and item cards
  - per-category recommended providers (up to 3, with placeholders if fewer)
References: `projects/inspectomatic/web/static/script.js:330`, `projects/inspectomatic/web/static/script.js:342`, `projects/inspectomatic/web/static/script.js:414`, `projects/inspectomatic/web/static/script.js:480`, `projects/inspectomatic/web/static/script.js:487`, `projects/inspectomatic/web/static/script.js:508`

### PDF export

- Download button triggers `exportToPDF()`.
References: `projects/inspectomatic/web/static/script.js:227`, `projects/inspectomatic/web/static/script.js:562`
- Uses `html2pdf` with `html2canvas`/`jsPDF` options; output filename is slugified from address.
References: `projects/inspectomatic/web/index.html:9`, `projects/inspectomatic/web/index.html:10`, `projects/inspectomatic/web/index.html:11`, `projects/inspectomatic/web/static/script.js:587`, `projects/inspectomatic/web/static/script.js:595`, `projects/inspectomatic/web/static/script.js:666`
- Adds print CSS and `pagebreak` config to avoid splitting `.no-break` sections.
References: `projects/inspectomatic/web/static/script.js:572`, `projects/inspectomatic/web/static/script.js:578`, `projects/inspectomatic/web/static/script.js:592`

## Environment Variables (All Found in Repo)

Values are intentionally not included here.

### Used by active backend (`app.py`)

- `GOOGLE_MAPS_API_KEY`
  - Used for geocoding + Places nearby/details provider lookup.
  - References: `projects/inspectomatic/app.py:21`, `projects/inspectomatic/app.py:923`, `projects/inspectomatic/app.py:940`, `projects/inspectomatic/app.py:954`
- `ANTHROPIC_API_KEY`
  - Enables Claude document gate, extraction, and pricing.
  - References: `projects/inspectomatic/app.py:26`, `projects/inspectomatic/app.py:372`, `projects/inspectomatic/app.py:566`, `projects/inspectomatic/app.py:624`
- `CLAUDE_MODEL` (default: `claude-sonnet-4-5-20250929`)
  - Claude model selection for gate/extraction/pricing.
  - Reference: `projects/inspectomatic/app.py:25`
- `CLAUDE_MAX_TOKENS` (default: `200000`)
  - Controls chunking threshold logic.
  - References: `projects/inspectomatic/app.py:28`, `projects/inspectomatic/app.py:877`, `projects/inspectomatic/app.py:894`
- `CLAUDE_RESPONSE_MAX_TOKENS` (default: `8192`)
  - Output token cap and chunk-budget calculations.
  - References: `projects/inspectomatic/app.py:29`, `projects/inspectomatic/app.py:581`, `projects/inspectomatic/app.py:877`

### Used by other repo scripts/variants

- `OPENAI_API_KEY`
  - Used in `api_test.py` and GPT-5 variant/checkpoint apps.
  - References: `projects/inspectomatic/api_test.py:8`, `projects/inspectomatic/app_gpt5.py:19`, `projects/inspectomatic/app_checkpoint_2.py:29`
- `MODEL_NAME` (default: `gpt-5`)
  - GPT-5 variant model selector.
  - Reference: `projects/inspectomatic/app_gpt5.py:18`
- `MAX_TOKENS` (default: `128000`)
  - GPT-5 variant context/token setting.
  - Reference: `projects/inspectomatic/app_gpt5.py:36`
- `RESPONSE_MAX_TOKENS` (default: `16000`)
  - GPT-5 variant response cap.
  - Reference: `projects/inspectomatic/app_gpt5.py:37`
- `AI_PROVIDER` (default: `claude`)
  - Checkpoint app provider switch (`claude` or `openai`).
  - Reference: `projects/inspectomatic/app_checkpoint_2.py:18`

### `.env` contents currently present

Repo currently includes these names in `projects/inspectomatic/.env`:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_MAPS_API_KEY`
Reference: `projects/inspectomatic/.env:1`, `projects/inspectomatic/.env:2`, `projects/inspectomatic/.env:3`

## Response Shape Returned by `/upload`

`/upload` returns `ParsedResponse` with:
- `address`, `notes`
- `items` (display-ready line items)
- `normalized_items` (structured canonical items, optionally priced)
- `ignored_examples`
- `meta` (doc classification, extraction/pricing/provider diagnostics)
References: `projects/inspectomatic/app.py:186`, `projects/inspectomatic/app.py:189`, `projects/inspectomatic/app.py:190`, `projects/inspectomatic/app.py:191`, `projects/inspectomatic/app.py:192`, `projects/inspectomatic/app.py:1203`
