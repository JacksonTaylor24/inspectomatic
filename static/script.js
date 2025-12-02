// static/script.js — pricing, clean reasons, and locations restored
(function () {
  "use strict";

  const CATEGORY_ORDER = [
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
    "General Contractor (Multi-Trade)"
  ];
  const SEV_SCORE = { high: 3, medium: 2, low: 1 };

  console.log("DEBUG: script.js is loading (clean reasons + locations)...");

  const steps = Array.from(document.querySelectorAll(".step"));
  const panes = {
    1: document.getElementById("pane-1"),
    2: document.getElementById("pane-2"),
    3: document.getElementById("pane-3"),
    4: document.getElementById("pane-4")
  };

  // Step 1 elements
  const drop = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const fileBadge = document.getElementById("fileBadge");
  const nextFrom1 = document.getElementById("nextFrom1");

  // Step 2 (Address)
  const backTo1 = document.getElementById("backTo1");
  const addressInput = document.getElementById("address");
  const addrSuggestions = document.getElementById("addrSuggestions");
  const nextFrom2 = document.getElementById("nextFrom2");

  // Step 3 (Notes)
  const backTo2 = document.getElementById("backTo2");
  const notesEl = document.getElementById("notes");
  const nextFrom3 = document.getElementById("nextFrom3");

  // Step 4 (Submit)
  const backTo3 = document.getElementById("backTo3");
  const submitBtn = document.getElementById("submitBtn");
  const reportDisplay = document.getElementById("reportDisplay");
  const loadingDisplay = document.getElementById("loadingDisplay");
  const loadingText = document.getElementById("loadingText");
  const statusEl = document.getElementById("status");
  const downloadPdfBtn = document.getElementById("downloadPdfBtn");

  let selectedFile = null;
  let lastResult = null;
  let lastNotes = "";
  let lastAddress = "";

  function go(step) {
    Object.values(panes).forEach((p) => p.classList.add("hidden"));
    panes[step].classList.remove("hidden");
    steps.forEach((s) => s.classList.remove("active"));
    const active = steps.find((el) => el.dataset.step === String(step));
    if (active) active.classList.add("active");
  }

  function badge(text) {
    fileBadge.innerHTML = text ? '<span class="pill">' + text + "</span>" : "";
  }

  function setFile(f) {
    selectedFile = f || null;
    if (selectedFile) {
      badge(
        `${selectedFile.name} • ${(selectedFile.size / 1024).toFixed(1)} KB`
      );
      nextFrom1.disabled = false;
    } else {
      badge("");
      nextFrom1.disabled = true;
    }
  }

  // Drag & drop
  if (drop) {
    ["dragenter", "dragover"].forEach((evt) =>
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.classList.add("drag");
      })
    );
    ["dragleave", "drop"].forEach((evt) =>
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.classList.remove("drag");
      })
    );
    drop.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files?.[0];
      if (f) setFile(f);
    });
  }

  // Browse
  browseBtn?.addEventListener("click", () => fileInput.click());
  fileInput?.addEventListener("change", () => setFile(fileInput.files?.[0]));

  // Navigation
  nextFrom1?.addEventListener("click", () => go(2));
  backTo1?.addEventListener("click", () => go(1));
  backTo2?.addEventListener("click", () => go(2));
  nextFrom3?.addEventListener("click", () => go(4));
  backTo3?.addEventListener("click", () => go(3));

  // --- Address autocomplete (Nominatim for dev) ---
  let acTimer = null;
  addressInput?.addEventListener("input", () => {
    const q = addressInput.value.trim();
    nextFrom2.disabled = q.length < 3;

    if (acTimer) clearTimeout(acTimer);
    if (q.length < 3) {
      hideSuggestions();
      return;
    }

    acTimer = setTimeout(async () => {
      try {
        const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&addressdetails=1&limit=5&countrycodes=us&q=${encodeURIComponent(
          q
        )}`;
        const res = await fetch(url, {
          headers: { Accept: "application/json" }
        });
        const data = await res.json();
        renderSuggestions(Array.isArray(data) ? data : []);
      } catch {
        hideSuggestions();
      }
    }, 300);
  });

  function renderSuggestions(arr) {
    if (!arr.length) {
      hideSuggestions();
      return;
    }
    addrSuggestions.innerHTML = arr
      .map((item) => {
        const label = item.display_name || "";
        return `<li data-label="${escapeHtml(
          label
        )}">${escapeHtml(label)}</li>`;
      })
      .join("");
    addrSuggestions.classList.remove("hidden");
  }

  function hideSuggestions() {
    addrSuggestions.innerHTML = "";
    addrSuggestions.classList.add("hidden");
  }

  addrSuggestions?.addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (!li) return;
    const label = li.getAttribute("data-label");
    addressInput.value = label || addressInput.value;
    nextFrom2.disabled = addressInput.value.trim().length < 3;
    hideSuggestions();
  });

  document.addEventListener("click", (e) => {
    if (
      addrSuggestions &&
      !addrSuggestions.contains(e.target) &&
      e.target !== addressInput
    ) {
      hideSuggestions();
    }
  });

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;"
    }[c]));
  }

  // Strip price & "Reason:" label out of explanation text
  function cleanExplanation(text) {
    if (!text) return "";
    let t = String(text);

    // Cut off anything after a pipe (we used "Reason: ... | Estimated cost range: ...")
    const pipeIdx = t.indexOf("|");
    if (pipeIdx !== -1) {
      t = t.slice(0, pipeIdx);
    }

    // Remove explicit "Estimated cost range: ..." fragments if any remain
    t = t.replace(/Estimated cost range:.*$/i, "");

    // Remove leading "Reason:" or "Reason -" labels
    t = t.replace(/^\s*Reason[:\-]\s*/i, "");

    // Collapse whitespace
    t = t.replace(/\s+/g, " ").trim();

    return t;
  }

  // Step 2 next
  nextFrom2?.addEventListener("click", () => go(3));

  // Submit
  submitBtn?.addEventListener("click", async () => {
    console.log("DEBUG: Submit button clicked");

    if (!selectedFile) {
      go(1);
      return;
    }

    reportDisplay.classList.add("hidden");
    loadingDisplay.style.display = "block";
    loadingText.textContent = "Analyzing your inspection report...";
    statusEl.textContent = "Processing document with AI...";
    downloadPdfBtn.disabled = true;

    lastNotes = (notesEl?.value || "").trim();
    lastAddress = (addressInput?.value || "").trim();

    const form = new FormData();
    form.append("file", selectedFile);
    form.append("address", lastAddress);
    form.append("notes", lastNotes);

    try {
      console.log("DEBUG: Making API call to /upload");
      const res = await fetch("/upload", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      lastResult = data;

      console.log("=== FULL API RESPONSE ===");
      console.log(JSON.stringify(data, null, 2));

      statusEl.textContent = "Generating beautiful report...";

      buildAndDisplayReport(lastResult, lastAddress, lastNotes);

      downloadPdfBtn.disabled = false;
      downloadPdfBtn.onclick = () => exportToPDF();

      statusEl.textContent = "Analysis complete ✓";
    } catch (e) {
      console.error("DEBUG: Error in submit:", e);
      loadingText.textContent = "Error: " + e.message;
      statusEl.textContent = "Analysis failed ✕";
    }
  });

  // ---------- Beautiful Report Builder ----------
  function buildAndDisplayReport(data, address, notes) {
    console.log("=== BUILDING BEAUTIFUL REPORT ===");

    let items = [];
    const pricingTotals = data?.meta?.pricing_totals || null;
    const normalized = Array.isArray(data?.normalized_items)
      ? data.normalized_items
      : null;

    // IMPORTANT: prefer items array (has cost_low / cost_high), but pull location/explanation from normalized when available
    if (Array.isArray(data?.items) && data.items.length > 0) {
      console.log("Using items array (with normalized_items for locations)");
      items = data.items.map((it, idx) => {
        const norm = normalized && normalized[idx] ? normalized[idx] : null;
        return {
          category: it.category || "Minor Handyman Repairs",
          item: (it.item_text || "").trim(),
          verbatim: it.item_text || "",
          location: norm && norm.location ? norm.location : null,
          qty: it.qty,
          units: norm && norm.units ? norm.units : null,
          severity: it.priority || (norm && norm.severity) || "medium",
          explanation: cleanExplanation(
            it.notes || (norm && norm.explanation) || ""
          ),
          price_low:
            typeof it.cost_low === "number" ? it.cost_low : null,
          price_high:
            typeof it.cost_high === "number" ? it.cost_high : null,
          currency: it.currency || (pricingTotals?.currency || "USD")
        };
      });
    } else if (
      Array.isArray(data?.normalized_items) &&
      data.normalized_items.length > 0
    ) {
      console.log("Using normalized_items array only");
      items = data.normalized_items.map((it) => ({
        category: it.category || "Minor Handyman Repairs",
        item: (it.item || "").trim(),
        verbatim: it.verbatim || it.item || "",
        location: it.location,
        qty: it.qty,
        units: it.units,
        severity: it.severity || "medium",
        explanation: cleanExplanation(it.explanation),
        price_low:
          it.price && typeof it.price.low === "number"
            ? it.price.low
            : null,
        price_high:
          it.price && typeof it.price.high === "number"
            ? it.price.high
            : null,
        currency:
          it.price && it.price.currency
            ? it.price.currency
            : (pricingTotals?.currency || "USD")
      }));
    }

    if (!items.length) {
      reportDisplay.innerHTML =
        '<div style="text-align: center; padding: 40px; color: #6b7280;">No repair items found in the document.</div>';
      loadingDisplay.style.display = "none";
      reportDisplay.classList.remove("hidden");
      return;
    }

    // Sort items by category order, then severity, then alphabetically
    items.sort((a, b) => {
      const ca = CATEGORY_ORDER.indexOf(a.category);
      const cb = CATEGORY_ORDER.indexOf(b.category);
      if (ca !== -1 && cb !== -1 && ca !== cb) return ca - cb;
      if (ca !== -1 && cb === -1) return -1;
      if (ca === -1 && cb !== -1) return 1;

      const sa = scoreSeverity(a.severity);
      const sb = scoreSeverity(b.severity);
      if (sa !== sb) return sb - sa;

      return String(a.item).localeCompare(String(b.item));
    });

    // Group by category
    const byCat = new Map();
    items.forEach((it) => {
      if (!byCat.has(it.category)) byCat.set(it.category, []);
      byCat.get(it.category).push(it);
    });

    const reportHtml = buildReportHTML(
      byCat,
      address,
      notes,
      items.length,
      pricingTotals
    );

    reportDisplay.innerHTML = reportHtml;
    loadingDisplay.style.display = "none";
    reportDisplay.classList.remove("hidden");
  }

  function buildReportHTML(byCat, address, notes, totalItems, pricingTotals) {
    const currentDate = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric"
    });

    const currentTime = new Date().toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit"
    });

    // Top-level total pricing pill
    const totalRangeHtml =
      pricingTotals &&
      typeof pricingTotals.low === "number" &&
      typeof pricingTotals.high === "number"
        ? `
        <div style="margin: 16px 0 24px; display: flex; justify-content: center;">
          <div style="display: inline-flex; flex-direction: column; align-items: center; padding: 12px 20px; border-radius: 999px; border: 1px solid #000; background: #f9fafb; gap: 4px;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: #4b5563;">Estimated Total Repair Range</div>
            <div style="font-size: 18px; font-weight: 600; color: #111827;">
              ${formatMoney(pricingTotals.low, pricingTotals.currency)} – ${formatMoney(pricingTotals.high, pricingTotals.currency)}
            </div>
          </div>
        </div>
      `
        : "";

    let html = `
      <div class="report-container">
        <div class="report-header">
          <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 16px;">
            <div style="display: flex; gap: 4px;">
              <div style="width: 12px; height: 12px; border-radius: 50%; background: #000;"></div>
              <div style="width: 12px; height: 12px; border-radius: 50%; background: #000; opacity: 0.3;"></div>
            </div>
            <h2 style="margin: 0; font-size: 24px; font-weight: normal; letter-spacing: 1px;">INSPECTOMATIC</h2>
          </div>
          <p style="margin: 0; font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">HOME BUYING RESOURCE TO SIMPLIFY</p>
          <p style="margin: 0; font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">THE INSPECTION PROCESS</p>
          <div style="margin: 20px 0 8px; height: 1px; background: #000; width: 120px; margin-left: auto; margin-right: auto;"></div>
          <h3 style="margin: 0; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">PROPERTY INSPECTION REPORT</h3>
        </div>

        <div style="text-align: center; margin-bottom: 8px;">
          <p style="font-size: 16px; font-weight: 600; color: #000; margin: 0 0 8px 0;">${escapeHtml(
            address || "Property Address Not Specified"
          )}</p>
        </div>

        <div class="report-meta">
          <div>Generated: ${currentDate} at ${currentTime}</div>
          <div>Total Items: ${totalItems}</div>
        </div>

        ${totalRangeHtml}
    `;

    if (notes) {
      html += `
        <div style="background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 16px; margin-bottom: 32px;">
          <h4 style="margin: 0 0 8px 0; font-size: 12px; font-weight: 600; color: #000; text-transform: uppercase; letter-spacing: 0.5px;">ADDITIONAL NOTES</h4>
          <p style="margin: 0; font-size: 14px; color: #666; line-height: 1.5;">${escapeHtml(
            notes
          )}</p>
        </div>
      `;
    }

    // Category sections
    byCat.forEach((arr, category) => {
      if (!arr || !arr.length) return;

      html += `
        <div class="report-section" style="margin-bottom: 40px;">
          <h3 style="background: #000; color: white; padding: 12px 16px; border-radius: 4px; font-weight: 600; font-size: 13px; margin: 0 0 1px 0; text-transform: uppercase; letter-spacing: 0.5px;">${escapeHtml(
            category
          )}</h3>
          <div style="border: 1px solid #e9ecef; border-top: none; border-radius: 0 0 4px 4px; background: #fff;">
      `;

      arr.forEach((item, index) => {
        const severityText =
          item.severity.charAt(0).toUpperCase() +
          item.severity.slice(1);
        const hasPrice =
          typeof item.price_low === "number" &&
          typeof item.price_high === "number";

        html += `
          <div style="padding: 20px; border-bottom: 1px solid #f1f3f4;">
            <div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 12px;">
              <div style="width: 24px; height: 24px; border: 1px solid #000; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <span style="font-size: 12px; font-weight: 600;">${index +
                  1}</span>
              </div>
              <div style="flex: 1;">
                <h4 style="margin: 0 0 8px 0; font-size: 15px; font-weight: 600; color: #000;">${escapeHtml(
                  item.item
                )}</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 8px;">
                  <span style="padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; ${getSeverityStyles(
                    item.severity
                  )}">${severityText} Priority</span>
        `;

        if (item.location) {
          html += `<span style="font-size: 12px; color: #666;">📍 ${escapeHtml(
            item.location
          )}</span>`;
        }

        if (item.qty) {
          html += `<span style="font-size: 12px; color: #666;">📦 Qty: ${
            item.qty
          }${
            item.units ? " " + escapeHtml(item.units) : ""
          }</span>`;
        }

        if (hasPrice) {
          html += `<span style="font-size: 12px; color: #111827; font-weight: 600;">💰 ${formatMoney(
            item.price_low,
            item.currency
          )} – ${formatMoney(item.price_high, item.currency)}</span>`;
        }

        html += `
                </div>
        `;

        if (item.explanation) {
          const specialistLabel =
            category === "Minor Handyman Repairs"
              ? "Why this can be handled by a general handyman:"
              : `Why this requires a ${getSpecialistLabel(category)}:`;
          html += `
            <div style="margin-top: 12px; padding: 12px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #2563eb;">
              <strong style="font-size: 13px; color: #000;">${escapeHtml(
                specialistLabel
              )}</strong>
              <span style="font-size: 13px; color: #666; margin-left: 4px;">${escapeHtml(
                item.explanation
              )}</span>
            </div>
          `;
        }

        html += `
              </div>
            </div>
          </div>
        `;
      });

      html += `
            <div style="padding: 20px; background: #f8f9fa; border-top: 1px solid #e9ecef;">
              <h4 style="margin: 0 0 12px 0; font-size: 12px; font-weight: 600; color: #000; text-transform: uppercase; letter-spacing: 0.5px;">RECOMMENDED PROVIDERS</h4>
              <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="padding: 8px 12px; background: #fff; border: 1px dashed #ccc; border-radius: 4px; font-size: 12px; color: #666;">1) Company: ___________________   Phone: ___________________   Rating: ___________________</div>
                <div style="padding: 8px 12px; background: #fff; border: 1px dashed #ccc; border-radius: 4px; font-size: 12px; color: #666;">2) Company: ___________________   Phone: ___________________   Rating: ___________________</div>
                <div style="padding: 8px 12px; background: #fff; border: 1px dashed #ccc; border-radius: 4px; font-size: 12px; color: #666;">3) Company: ___________________   Phone: ___________________   Rating: ___________________</div>
              </div>
            </div>
          </div>
        </div>
      `;
    });

    html += `
        <div style="text-align: center; padding: 20px; background: #000; color: white; border-radius: 4px; margin-top: 32px;">
          <div style="font-size: 24px; font-weight: 700; margin-bottom: 4px;">${totalItems}</div>
          <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">${
            totalItems === 1
              ? "REPAIR ITEM IDENTIFIED"
              : "REPAIR ITEMS IDENTIFIED"
          }</div>
          <div style="font-size: 10px; color: #ccc; margin-top: 8px; font-style: italic;">Generated by Inspectomatic</div>
        </div>
      </div>
    `;

    return html;
  }

  function getSeverityStyles(severity) {
    const styles = {
      high: "background: #fee2e2; color: #dc2626;",
      medium: "background: #fef3c7; color: #d97706;",
      low: "background: #d1fae5; color: #059669;"
    };
    return (
      styles[String(severity || "").toLowerCase()] ||
      "background: #f3f4f6; color: #374151;"
    );
  }

  function formatMoney(value, currency) {
    if (typeof value !== "number" || isNaN(value)) return "";
    const cur = currency || "USD";
    try {
      return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: cur,
        maximumFractionDigits: 0
      }).format(value);
    } catch {
      return `$${value.toFixed(0)}`;
    }
  }

  // ---------- PDF Export ----------
  async function exportToPDF() {
    const element = document.getElementById("reportDisplay");
  
    const opt = {
      margin:       0.5,
      filename:     slug(lastAddress || "inspection-report") + ".pdf",
      image:        { type: 'jpeg', quality: 0.98 },
      html2canvas:  { scale: 2, useCORS: true },
      jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
  
    html2pdf().from(element).set(opt).save();
  }
  

  // ---------- Helper Functions ----------
  function getSpecialistLabel(category) {
    const specialistMap = {
      Plumbing: "plumber",
      Electrical: "electrician",
      HVAC: "HVAC technician",
      "Minor Handyman Repairs": "general handyman",
      "Septic & Well Systems": "septic/well specialist",
      "Painting & Finishes (Cosmetic)": "painting contractor",
      "Carpentry & Trim": "carpenter",
      "Windows/Glass": "window specialist",
      "Siding & Exterior Envelope": "siding contractor",
      Roofing: "roofing contractor",
      "Drywall & Plaster": "drywall contractor",
      "Flooring & Tile": "flooring contractor",
      "Gutters & Downspouts": "gutter specialist",
      "Insulation & Air Sealing": "insulation contractor",
      "Garage Door Systems": "garage door technician",
      "Ventilation & Appliances": "appliance technician",
      "Landscaping & Drainage": "landscaping contractor",
      "Pest Control": "pest control specialist",
      "Masonry & Concrete": "masonry contractor",
      "Chimney/Fireplace": "chimney specialist",
      "Waterproofing & Mold": "waterproofing specialist",
      "General Contractor (Multi-Trade)": "general contractor"
    };
    return specialistMap[category] || "specialist";
  }

  function scoreSeverity(sev) {
    const key = String(sev || "").toLowerCase();
    return SEV_SCORE[key] || 0;
  }

  function slug(s) {
    return (
      String(s)
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 80) || "inspection-report"
    );
  }

  console.log("DEBUG: Starting wizard...");
  go(1);
})();
