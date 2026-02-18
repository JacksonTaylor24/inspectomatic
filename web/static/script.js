// web/static/script.js — FULL DROP-IN REPLACEMENT
// Fixes:
// 1) removes "Report Type:" label and moves doc label under "Generated" with same font
// 4) prevents item blocks from splitting across pages in PDF export (css + html2pdf pagebreak)
// Minor: removes "Report type:" in status messages, keeps clean doc label
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

  console.log("DEBUG: script.js is loading (clean doc label + improved PDF page breaks)...");

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
  const progressBar = document.getElementById("progressBar");
  const progressPercent = document.getElementById("progressPercent");
  const downloadPdfBtn = document.getElementById("downloadPdfBtn");

  let selectedFile = null;
  let lastResult = null;
  let lastNotes = "";
  let lastAddress = "";
  let lastDocTypeLabel = "";
  let lastDocType = "";

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
      badge(`${selectedFile.name} • ${(selectedFile.size / 1024).toFixed(1)} KB`);
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

  // Step 2: enable/disable "Next" based on address text
  addressInput?.addEventListener("input", () => {
    const q = addressInput.value.trim();
    nextFrom2.disabled = q.length < 3;
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

    const pipeIdx = t.indexOf("|");
    if (pipeIdx !== -1) t = t.slice(0, pipeIdx);

    t = t.replace(/Estimated cost range:.*$/i, "");
    t = t.replace(/^\s*Reason[:\-]\s*/i, "");
    t = t.replace(/\s+/g, " ").trim();

    return t;
  }

  // Step 2 next
  nextFrom2?.addEventListener("click", () => go(3));

  function setLoading(docLabelMaybe) {
    reportDisplay.classList.add("hidden");
    loadingDisplay.style.display = "block";
    loadingText.textContent = "Inspectomatic is working tirelessly for you...";
    statusEl.textContent = docLabelMaybe ? `Building your report… • ${docLabelMaybe}` : "Building your report…";
    if (progressBar) progressBar.style.width = "0%";
    if (progressPercent) progressPercent.textContent = "0%";
    downloadPdfBtn.disabled = true;
  }

  function updateProgress(percent, message, stage) {
    const p = Math.max(0, Math.min(100, Number(percent) || 0));
    if (progressBar) progressBar.style.width = `${p.toFixed(1)}%`;
    if (progressPercent) progressPercent.textContent = `${Math.round(p)}%`;
    loadingText.textContent = "Inspectomatic is working tirelessly for you...";
    if (message && stage) {
      statusEl.textContent = `${message} • ${stage}`;
    } else if (message) {
      statusEl.textContent = message;
    } else if (stage) {
      statusEl.textContent = stage;
    }
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  const LEGACY_UPLOAD_SECONDS_KEY = "inspectomatic_legacy_upload_seconds_ema";

  function getExpectedLegacySeconds() {
    const v = Number(localStorage.getItem(LEGACY_UPLOAD_SECONDS_KEY));
    if (!Number.isFinite(v) || v < 10 || v > 300) return 75;
    return v;
  }

  function updateExpectedLegacySeconds(actualSeconds) {
    if (!Number.isFinite(actualSeconds) || actualSeconds < 1 || actualSeconds > 600) return;
    const prev = getExpectedLegacySeconds();
    const next = prev * 0.7 + actualSeconds * 0.3;
    localStorage.setItem(LEGACY_UPLOAD_SECONDS_KEY, String(next));
  }

  function getLegacyEstimatedProgress(elapsedSeconds, expectedSeconds) {
    const r = Math.max(0, elapsedSeconds / Math.max(1, expectedSeconds));
    let percent = 15;
    let stage = "extract_text";
    let message = "Extracting text from your document...";

    if (r < 0.12) {
      percent = 15 + (r / 0.12) * 10; // 15 -> 25
      stage = "doc_gate";
      message = "Classifying document type...";
    } else if (r < 0.48) {
      percent = 25 + ((r - 0.12) / 0.36) * 35; // 25 -> 60
      stage = "extraction";
      message = "Extracting actionable repair items...";
    } else if (r < 0.52) {
      percent = 60 + ((r - 0.48) / 0.04) * 2; // 60 -> 62
      stage = "stable_sort";
      message = "Ordering repair items...";
    } else if (r < 0.84) {
      percent = 62 + ((r - 0.52) / 0.32) * 26; // 62 -> 88
      stage = "pricing";
      message = "Estimating repair pricing...";
    } else if (r < 0.97) {
      percent = 88 + ((r - 0.84) / 0.13) * 10; // 88 -> 98
      stage = "providers";
      message = "Finding top-rated local providers...";
    } else {
      // Keep moving gently past 98% to avoid a long visual stall.
      const tail = Math.min(1.0, (r - 0.97) / 0.5);
      percent = 98 + tail * 1.7; // 98 -> 99.7
      stage = "finalizing";
      message = "Finalizing your report...";
    }

    return {
      percent: Math.max(15, Math.min(99.7, percent)),
      stage,
      message,
    };
  }

  function buildFormData() {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("address", lastAddress);
    form.append("notes", lastNotes);
    return form;
  }

  async function runLegacyUpload() {
    console.log("DEBUG: Falling back to legacy /upload endpoint");
    updateProgress(15, "Uploading document...", "Stage: upload");

    const expectedSeconds = getExpectedLegacySeconds();
    const startedAt = Date.now();
    const ticker = setInterval(() => {
      const elapsedSeconds = (Date.now() - startedAt) / 1000;
      const est = getLegacyEstimatedProgress(elapsedSeconds, expectedSeconds);
      updateProgress(est.percent, est.message, `Stage: ${est.stage}`);
    }, 900);

    let res;
    try {
      res = await fetch("/upload", { method: "POST", body: buildFormData() });
    } catch (e) {
      console.warn("DEBUG: Legacy /upload network error, retrying once...", e);
      await sleep(350);
      res = await fetch("/upload", { method: "POST", body: buildFormData() });
    } finally {
      clearInterval(ticker);
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Upload failed");
    }

    updateExpectedLegacySeconds((Date.now() - startedAt) / 1000);
    updateProgress(100, "Report ready.", "Stage: complete");
    return await res.json();
  }

  // Submit
  submitBtn?.addEventListener("click", async () => {
    console.log("DEBUG: Submit button clicked");

    if (!selectedFile) {
      console.warn("DEBUG: No file selected, returning to step 1");
      go(1);
      return;
    }

    setLoading("");

    lastNotes = (notesEl?.value || "").trim();
    lastAddress = (addressInput?.value || "").trim();

    console.log("DEBUG: Submitting with address =", lastAddress);

    try {
      console.log("DEBUG: Starting async upload job");
      let data = null;
      let startRes = null;
      try {
        startRes = await fetch("/upload/start", { method: "POST", body: buildFormData() });
      } catch (e) {
        console.warn("DEBUG: /upload/start network error, trying legacy /upload...", e);
        data = await runLegacyUpload();
        lastResult = data;
      }
      if (!data && !startRes.ok && startRes.status === 404) {
        data = await runLegacyUpload();
        lastResult = data;
      } else if (!data && !startRes.ok) {
        const err = await startRes.json().catch(() => ({ detail: startRes.statusText }));
        console.error("DEBUG: /upload/start returned non-OK", startRes.status, err);
        throw new Error(err.detail || "Upload failed to start");
      } else if (!data) {
        const { job_id: jobId } = await startRes.json();
        if (!jobId) throw new Error("Missing upload job ID");

        let jobStatus = null;
        while (true) {
          const statusRes = await fetch(`/upload/status/${encodeURIComponent(jobId)}`);
          if (!statusRes.ok) {
            const err = await statusRes.json().catch(() => ({ detail: statusRes.statusText }));
            throw new Error(err.detail || "Status check failed");
          }
          jobStatus = await statusRes.json();
          updateProgress(
            jobStatus?.percent,
            jobStatus?.message || "Inspectomatic is working tirelessly for you...",
            jobStatus?.stage ? `Stage: ${jobStatus.stage}` : "Building your report…"
          );
          if (jobStatus?.done) break;
          await sleep(600);
        }

        if (jobStatus?.status === "failed") {
          throw new Error(jobStatus?.error || "Upload failed");
        }

        const resultRes = await fetch(`/upload/result/${encodeURIComponent(jobId)}`);
        if (!resultRes.ok) {
          const err = await resultRes.json().catch(() => ({ detail: resultRes.statusText }));
          throw new Error(err.detail || "Result retrieval failed");
        }

        data = await resultRes.json();
        lastResult = data;
      }

      lastDocType = data?.meta?.doc_type || "";
      lastDocTypeLabel = data?.meta?.doc_type_label || "";

      console.log("=== DEBUG: FULL API RESPONSE (truncated) ===");
      console.log(JSON.stringify(data, null, 2).slice(0, 4000));
      console.log("=== DEBUG: pricing meta ===");
      console.log({
        pricing_attempted: data?.meta?.pricing_attempted,
        pricing_used: data?.meta?.pricing_used,
        pricing_items_in: data?.meta?.pricing_items_in,
        pricing_items_priced: data?.meta?.pricing_items_priced,
        pricing_error: data?.meta?.pricing_error,
        pricing_totals: data?.meta?.pricing_totals
      });

      statusEl.textContent = lastDocTypeLabel ? `Generating beautiful report… • ${lastDocTypeLabel}` : "Generating beautiful report...";

      buildAndDisplayReport(lastResult, lastAddress, lastNotes);

      downloadPdfBtn.disabled = false;
      downloadPdfBtn.onclick = () => exportToPDF();

      statusEl.textContent = lastDocTypeLabel ? `Analysis complete ✓ • ${lastDocTypeLabel}` : "Analysis complete ✓";
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
    const normalized = Array.isArray(data?.normalized_items) ? data.normalized_items : null;
    const providersByCategory = data?.meta?.providers || {};

    const docTypeLabel = data?.meta?.doc_type_label || lastDocTypeLabel || "";

    // Prefer items array (has cost_low / cost_high), but pull location from normalized when available
    if (Array.isArray(data?.items) && data.items.length > 0) {
      console.log("DEBUG: Using items array. Length:", data.items.length);
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
          explanation: cleanExplanation(it.notes || (norm && norm.explanation) || ""),
          price_low: typeof it.cost_low === "number" ? it.cost_low : null,
          price_high: typeof it.cost_high === "number" ? it.cost_high : null,
          currency: it.currency || (pricingTotals?.currency || "USD")
        };
      });
    } else if (Array.isArray(data?.normalized_items) && data.normalized_items.length > 0) {
      console.log("DEBUG: Using normalized_items array only. Length:", data.normalized_items.length);
      items = data.normalized_items.map((it) => ({
        category: it.category || "Minor Handyman Repairs",
        item: (it.item || "").trim(),
        verbatim: it.verbatim || it.item || "",
        location: it.location,
        qty: it.qty,
        units: it.units,
        severity: it.severity || "medium",
        explanation: cleanExplanation(it.explanation),
        price_low: it.price && typeof it.price.low === "number" ? it.price.low : null,
        price_high: it.price && typeof it.price.high === "number" ? it.price.high : null,
        currency: it.price && it.price.currency ? it.price.currency : (pricingTotals?.currency || "USD")
      }));
    }

    if (!items.length) {
      console.warn("DEBUG: No items found to display in report");
      reportDisplay.innerHTML =
        '<div style="text-align: center; padding: 40px; color: var(--report-text-muted);">No repair items found in the document.</div>';
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
      pricingTotals,
      providersByCategory,
      docTypeLabel
    );

    reportDisplay.innerHTML = reportHtml;
    loadingDisplay.style.display = "none";
    reportDisplay.classList.remove("hidden");
  }

  function buildReportHTML(byCat, address, notes, totalItems, pricingTotals, providersByCategory, docTypeLabel) {
    const currentDate = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric"
    });

    const currentTime = new Date().toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit"
    });

    const totalRangeHtml =
      pricingTotals && typeof pricingTotals.low === "number" && typeof pricingTotals.high === "number" &&
      (pricingTotals.low > 0 || pricingTotals.high > 0)
        ? `
        <div style="margin: 16px 0 24px; display: flex; justify-content: center;">
          <div style="display: inline-flex; flex-direction: column; align-items: center; padding: 12px 20px; border-radius: 999px; border: 1px solid var(--report-border-strong); background: var(--report-surface-soft); gap: 4px;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--report-text-muted);">Estimated Total Repair Range</div>
            <div style="font-size: 18px; font-weight: 600; color: var(--report-text);">
              ${formatMoney(pricingTotals.low, pricingTotals.currency)} – ${formatMoney(pricingTotals.high, pricingTotals.currency)}
            </div>
          </div>
        </div>
      `
        : "";

    // Doc label should appear under "Generated" in SAME font (report-meta).
    const docLabelMetaHtml = docTypeLabel
      ? `<div>${escapeHtml(docTypeLabel)}</div>`
      : "";

    let html = `
      <div class="report-container">
        <div class="report-header">
          <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 16px;">
            <div style="display: flex; gap: 4px;">
              <div style="width: 12px; height: 12px; border-radius: 50%; background: var(--report-accent);"></div>
              <div style="width: 12px; height: 12px; border-radius: 50%; background: var(--report-accent); opacity: 0.3;"></div>
            </div>
            <h2 style="margin: 0; font-size: 24px; font-weight: 600; letter-spacing: 0.06em; color: var(--report-text);">INSPECTOMATIC</h2>
          </div>
          <p style="margin: 0; font-size: 11px; color: var(--report-text-muted); text-transform: uppercase; letter-spacing: 0.5px;">HOME BUYING RESOURCE TO SIMPLIFY</p>
          <p style="margin: 0; font-size: 11px; color: var(--report-text-muted); text-transform: uppercase; letter-spacing: 0.5px;">THE INSPECTION PROCESS</p>
          <div style="margin: 20px 0 8px; height: 1px; background: var(--report-border-strong); width: 120px; margin-left: auto; margin-right: auto;"></div>
          <h3 style="margin: 0; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--report-text);">PROPERTY REPORT</h3>
        </div>

        <div style="text-align: center; margin-bottom: 8px;">
          <p style="font-size: 16px; font-weight: 600; color: var(--report-text); margin: 0 0 8px 0;">${escapeHtml(
            address || "Property Address Not Specified"
          )}</p>
        </div>

        
        <div class="report-meta" style="display:flex; justify-content:space-between; align-items:flex-end; gap:16px;">
          <div style="display:flex; flex-direction:column; gap:4px;">
            <div>Generated: ${currentDate} at ${currentTime}</div>
            ${docTypeLabel ? `<div>Report Type: ${escapeHtml(docTypeLabel)}</div>` : ``}
          </div>
          <div style="display:flex; flex-direction:column; align-items:flex-end; gap:4px;">
            <div>Total Items: ${totalItems}</div>
          </div>
        </div>


        ${totalRangeHtml}
    `;

    if (notes) {
      html += `
        <div style="background: var(--report-surface-soft); border: 1px solid var(--report-border); border-radius: 6px; padding: 16px; margin-bottom: 24px;">
          <h4 style="margin: 0 0 8px 0; font-size: 12px; font-weight: 600; color: var(--report-text); text-transform: uppercase; letter-spacing: 0.5px;">ADDITIONAL NOTES</h4>
          <p style="margin: 0; font-size: 14px; color: var(--report-text-muted); line-height: 1.5;">${escapeHtml(notes)}</p>
        </div>
      `;
    }

    byCat.forEach((arr, category) => {
      if (!arr || !arr.length) return;

      const providers = providersByCategory[category] || [];

      html += `
        <div class="report-section" style="margin-bottom: 40px;">
          <h3 style="background: var(--report-accent); color: var(--report-on-accent); padding: 12px 16px; border-radius: 4px; font-weight: 600; font-size: 13px; margin: 0 0 1px 0; text-transform: uppercase; letter-spacing: 0.5px;">${escapeHtml(category)}</h3>
          <div style="border: 1px solid var(--report-border); border-top: none; border-radius: 0 0 4px 4px; background: var(--report-surface);">
      `;

      arr.forEach((item, index) => {
        const severityText =
          item.severity && item.severity.length
            ? item.severity.charAt(0).toUpperCase() + item.severity.slice(1)
            : "Medium";
        const hasPrice = typeof item.price_low === "number" && typeof item.price_high === "number";

        // KEY: prevent page breaks inside each item block
        html += `
          <div class="no-break" style="padding: 18px; border-bottom: 1px solid var(--report-accent-soft); break-inside: avoid; page-break-inside: avoid;">
            <div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 12px;">
              <div style="width: 24px; height: 24px; border: 1px solid var(--report-border-strong); border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <span style="font-size: 12px; font-weight: 600;">${index + 1}</span>
              </div>
              <div style="flex: 1;">
                <h4 style="margin: 0 0 8px 0; font-size: 15px; font-weight: 600; color: var(--report-text);">${escapeHtml(item.item)}</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 8px;">
                  <span style="padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; ${getSeverityStyles(
                    item.severity
                  )}">${severityText} Priority</span>
        `;

        if (item.location) {
          html += `<span style="font-size: 12px; color: var(--report-text-muted);">📍 ${escapeHtml(item.location)}</span>`;
        }

        if (item.qty) {
          html += `<span style="font-size: 12px; color: var(--report-text-muted);">📦 Qty: ${item.qty}${
            item.units ? " " + escapeHtml(item.units) : ""
          }</span>`;
        }

        if (hasPrice) {
          html += `<span style="font-size: 12px; color: var(--report-text); font-weight: 600;">💰 ${formatMoney(
            item.price_low,
            item.currency
          )} – ${formatMoney(item.price_high, item.currency)}</span>`;
        }

        html += `</div>`;

        if (item.explanation) {
          const specialistLabel =
            category === "Minor Handyman Repairs"
              ? "Why this can be handled by a general handyman:"
              : `Why this requires a ${getSpecialistLabel(category)}:`;
          html += `
            <div style="margin-top: 10px; padding: 12px; background: var(--report-surface-soft); border-radius: 4px; border-left: 2px solid var(--report-border-strong);">
              <strong style="font-size: 13px; color: var(--report-text);">${escapeHtml(specialistLabel)}</strong>
              <span style="font-size: 13px; color: var(--report-text-muted); margin-left: 4px;">${escapeHtml(item.explanation)}</span>
            </div>
          `;
        }

        html += `
              </div>
            </div>
          </div>
        `;
      });

      // Providers section should also avoid splitting
      html += `
            <div class="no-break" style="padding: 18px; background: var(--report-surface-soft); border-top: 1px solid var(--report-border); break-inside: avoid; page-break-inside: avoid;">
              <h4 style="margin: 0 0 12px 0; font-size: 12px; font-weight: 700; color: var(--report-text); text-transform: uppercase; letter-spacing: 0.5px;">RECOMMENDED PROVIDERS</h4>
              <div style="display: flex; flex-direction: column; gap: 8px;">
      `;

      if (providers.length) {
        providers.slice(0, 3).forEach((p, idx) => {
          html += `
            <div style="padding: 8px 12px; background: var(--report-surface); border: 1px dashed var(--report-border-strong); border-radius: 4px; font-size: 12px; color: var(--report-text);">
              <strong>${idx + 1}) ${escapeHtml(p.name || "Unknown")}</strong><br>
              ${p.phone ? `📞 ${escapeHtml(p.phone)} ` : ""}
              ${
                p.rating
                  ? `• ⭐ ${Number(p.rating).toFixed(1)} (${p.review_count || 0} reviews)`
                  : ""
              }
              ${
                p.address
                  ? `<br><span style="color: var(--report-text-muted);">📍 ${escapeHtml(p.address)}</span>`
                  : ""
              }
            </div>
          `;
        });
      }

      for (let i = providers.length + 1; i <= 3; i++) {
        html += `
          <div style="padding: 8px 12px; background: var(--report-surface); border: 1px dashed var(--report-border-strong); border-radius: 4px; font-size: 12px; color: var(--report-text-muted);">
            ${i}) Company: ___________________   Phone: ___________________   Rating: ___________________
          </div>
        `;
      }

      html += `
              </div>
            </div>
          </div>
        </div>
      `;
    });

    html += `
        <div class="no-break" style="text-align: center; padding: 20px; background: var(--report-accent); color: var(--report-on-accent); border-radius: 4px; margin-top: 24px; break-inside: avoid; page-break-inside: avoid;">
          <div style="font-size: 24px; font-weight: 700; margin-bottom: 4px;">${totalItems}</div>
          <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">${
            totalItems === 1 ? "REPAIR ITEM IDENTIFIED" : "REPAIR ITEMS IDENTIFIED"
          }</div>
          <div style="font-size: 10px; color: var(--report-summary-muted); margin-top: 8px; font-style: italic;">Generated by Inspectomatic</div>
        </div>
      </div>
    `;

    return html;
  }

  function getSeverityStyles(severity) {
    const styles = {
      high: "background: var(--sev-high-bg); color: var(--sev-high-text);",
      medium: "background: var(--sev-medium-bg); color: var(--sev-medium-text);",
      low: "background: var(--sev-low-bg); color: var(--sev-low-text);"
    };
    return styles[String(severity || "").toLowerCase()] || "background: var(--report-surface-soft); color: var(--report-text-muted);";
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

  // ---------- PDF Export via html2pdf ----------
  async function exportToPDF() {
    const element = document.getElementById("reportDisplay");
    if (!element) {
      alert("No report to export");
      return;
    }

    downloadPdfBtn.disabled = true;
    downloadPdfBtn.textContent = "Generating PDF...";

    // Inject a tiny print CSS to strengthen page-break control
    const styleId = "inspectomatic-print-style";
    if (!document.getElementById(styleId)) {
      const st = document.createElement("style");
      st.id = styleId;
      st.textContent = `
        .no-break { break-inside: avoid !important; page-break-inside: avoid !important; }
        .report-section { break-inside: auto; }
      `;
      document.head.appendChild(st);
    }

    try {
      const opt = {
        margin: 0.5,
        filename: slug(lastAddress || "inspection-report") + ".pdf",
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true },
        jsPDF: { unit: "in", format: "letter", orientation: "portrait" },
        // KEY: let css rules control, and avoid breaking inside .no-break
        pagebreak: { mode: ["css", "legacy"], avoid: [".no-break"] }
      };

      await html2pdf().from(element).set(opt).save();
    } catch (err) {
      console.error("PDF generation failed:", err);
      alert("Failed to generate PDF. Please try again.");
    } finally {
      downloadPdfBtn.disabled = false;
      downloadPdfBtn.textContent = "Download PDF Report";
    }
  }

  // ---------- Google Places Autocomplete ----------
  function initGoogleAutocomplete() {
    if (!addressInput) {
      console.warn("DEBUG: No address input found for autocomplete");
      return;
    }
    if (!window.google || !google.maps || !google.maps.places) {
      console.warn("Google Places library not available");
      return;
    }

    console.log("DEBUG: Initializing Google Autocomplete");
    const autocomplete = new google.maps.places.Autocomplete(addressInput, {
      fields: ["formatted_address", "geometry"],
      componentRestrictions: { country: "us" }
    });

    autocomplete.addListener("place_changed", () => {
      const place = autocomplete.getPlace();
      if (!place || !place.formatted_address) return;

      addressInput.value = place.formatted_address;
      const q = addressInput.value.trim();
      nextFrom2.disabled = q.length < 3;
      console.log("DEBUG: Autocomplete selected address:", addressInput.value);
    });
  }

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

  document.addEventListener("DOMContentLoaded", () => {
    console.log("DEBUG: DOMContentLoaded – starting wizard + Google autocomplete...");
    go(1);
    initGoogleAutocomplete();
  });
})();
