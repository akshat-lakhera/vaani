const $ = (id) => document.getElementById(id);

let sttReady = false;
let currentAnswerText = "";
let rec = null;
let recStream = null;
let chunks = [];
let startedAt = 0;
let recTimerInterval = null;

// Initialize Health & Backend State
async function checkHealth() {
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    sttReady = Boolean(h.stt_configured);
    const dot = $("live-dot");
    if (h.ok) {
      dot.className = "indicator-dot online";
      $("health-status").textContent = `${h.dense ? "Hybrid (Dense+BM25)" : "BM25"} Index Online`;
      $("health-details").textContent = `${h.chunks.toLocaleString()} chunks · STT: ${h.stt_provider}${sttReady ? " (ready)" : " (key missing)"} · Polish: ${h.llm_configured ? "ready" : "off"}`;
    } else {
      dot.className = "indicator-dot";
      $("health-status").textContent = "Index Not Loaded";
      $("health-details").textContent = "Run scripts/ingest.py to build index";
    }
    $("mic").disabled = !sttReady;
    $("mic").querySelector(".mic-label").textContent = sttReady ? "TAP TO TALK" : "KEY REQUIRED";
    $("stt-hint").hidden = sttReady;
  } catch (err) {
    $("live-dot").className = "indicator-dot";
    $("health-status").textContent = "API Unreachable";
    $("health-details").textContent = "Start server on port 8080";
  }
}

// Render Query Result
function renderResult(resp) {
  $("result").hidden = false;
  currentAnswerText = resp.answer || "";

  // Status Pill
  const st = $("status");
  st.textContent = resp.status.toUpperCase();
  st.className = `pill ${resp.status}`;

  // Budget Pill & Badge
  const ragMs = resp.timings?.total_rag_ms || 0;
  const budgetPill = $("budget");
  const budgetBadge = $("budget-badge");
  if (resp.within_budget) {
    budgetPill.textContent = `⚡ RAG ${ragMs.toFixed(1)}ms (<200ms)`;
    budgetPill.className = "pill grounded";
    budgetBadge.textContent = "✓ Within 200ms Target";
    budgetBadge.className = "badge-success";
  } else {
    budgetPill.textContent = `⏱ RAG ${ragMs.toFixed(1)}ms`;
    budgetPill.className = "pill abstain";
    budgetBadge.textContent = "⏱ Outside 200ms Window";
    budgetBadge.className = "badge-tag";
  }

  $("support").textContent = `Support: ${Number(resp.support || 0).toFixed(3)}`;
  $("strategy-pill").textContent = `Strategy: ${resp.strategy || "whole"}`;

  // Transcript Box
  const trBox = $("transcript-box");
  if (resp.transcript) {
    trBox.hidden = false;
    $("transcript").textContent = resp.transcript;
  } else {
    trBox.hidden = true;
  }

  // Answer & Reason
  $("answer").textContent = resp.answer || "(No answer generated)";
  $("reason").textContent = resp.reason || (resp.polished ? "✨ Grounded and polished by Grok-4.5" : "");

  // Render Waterfall Bar & Timings Grid
  renderWaterfall(resp.timings || {});

  // Render Citations
  renderCitations(resp.citations || []);

  // Scroll smoothly to results
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderWaterfall(t) {
  const segments = [
    { label: "Guard In", ms: t.guard_in_ms || 0, cls: "wf-guard-in" },
    { label: "Embed", ms: t.embed_ms || 0, cls: "wf-embed" },
    { label: "Retrieve", ms: t.retrieve_ms || 0, cls: "wf-retrieve" },
    { label: "Extract", ms: t.extract_ms || 0, cls: "wf-extract" },
    { label: "Guard Out", ms: t.guard_out_ms || 0, cls: "wf-guard-out" },
  ];

  const totalInBudget = segments.reduce((acc, s) => acc + s.ms, 0) || 1;
  const bar = $("waterfall-bar");
  bar.innerHTML = segments
    .filter((s) => s.ms > 0)
    .map((s) => {
      const pct = Math.max(2, (s.ms / totalInBudget) * 100);
      return `<div class="wf-segment ${s.cls}" style="width: ${pct}%" title="${s.label}: ${s.ms.toFixed(1)}ms"></div>`;
    })
    .join("");

  const allTimings = [
    ["STT Network", t.stt_ms],
    ["Input Guard", t.guard_in_ms],
    ["Query Embed", t.embed_ms],
    ["Search & RRF", t.retrieve_ms],
    ["Span Extract", t.extract_ms],
    ["Output Guard", t.guard_out_ms],
    ["Grok Polish", t.generate_ms],
    ["Total RAG", t.total_rag_ms],
  ];

  $("timings").innerHTML = allTimings
    .filter(([, v]) => v != null)
    .map(([k, v]) => `<div class="timing-item"><span>${k}</span><strong>${Number(v).toFixed(1)}ms</strong></div>`)
    .join("");
}

function renderCitations(cites) {
  $("cites-count").textContent = `${cites.length} source${cites.length === 1 ? "" : "s"}`;
  $("cites").innerHTML = cites
    .map(
      (c, i) => `
      <li class="cite-card">
        <div class="cite-meta">
          <span class="cite-id">[#${i + 1}] ${escapeHtml(c.passage_id)}</span>
          ${c.lang ? `<span class="cite-tag">${escapeHtml(c.lang)}</span>` : ""}
          ${c.query_type ? `<span class="cite-tag">${escapeHtml(c.query_type)}</span>` : ""}
          <span class="cite-tag">Score: ${Number(c.score || 0).toFixed(3)}</span>
        </div>
        <p class="cite-text">${escapeHtml(c.text)}</p>
      </li>
    `
    )
    .join("");
}

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setBusy(on, label) {
  const mic = $("mic");
  mic.classList.toggle("busy", on);
  if (label) mic.querySelector(".mic-label").textContent = label;
  $("btn-ask").disabled = on;
}

// Ask via Text
async function askText(text) {
  setBusy(true, "THINKING…");
  try {
    const body = new FormData();
    body.set("text", text);
    body.set("polish", $("polish").checked ? "true" : "false");
    const lang = $("lang-select").value;
    if (lang && lang !== "auto") body.set("language", lang);

    const resp = await fetch("/api/ask", { method: "POST", body }).then((r) => r.json());
    renderResult(resp);
  } catch (err) {
    renderResult({
      status: "refuse",
      answer: String(err.message || err),
      reason: "Network error calling /api/ask",
      support: 0,
      timings: { total_rag_ms: 0 },
      citations: [],
    });
  } finally {
    setBusy(false, sttReady ? "TAP TO TALK" : "KEY REQUIRED");
  }
}

// Ask via Audio
async function askAudio(blob, ext) {
  setBusy(true, "TRANSCRIBING…");
  try {
    const body = new FormData();
    body.set("audio", blob, `clip.${ext}`);
    body.set("polish", $("polish").checked ? "true" : "false");
    const lang = $("lang-select").value;
    if (lang && lang !== "auto") body.set("language", lang);

    const resp = await fetch("/api/ask", { method: "POST", body }).then((r) => r.json());
    renderResult(resp);
  } catch (err) {
    renderResult({
      status: "refuse",
      answer: String(err.message || err),
      reason: "Audio transcription / pipeline error",
      support: 0,
      timings: { total_rag_ms: 0 },
      citations: [],
    });
  } finally {
    setBusy(false, sttReady ? "TAP TO TALK" : "KEY REQUIRED");
  }
}

// Recording Controls
function pickMime() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  if (!window.MediaRecorder) return "";
  return types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
}

function extFor(mime) {
  if (mime.includes("mp4")) return "m4a";
  if (mime.includes("ogg")) return "ogg";
  return "webm";
}

async function startRec() {
  if (!sttReady) {
    alert("Set SARVAM_API_KEY or ELEVENLABS_API_KEY in .env and restart server.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    const mime = pickMime();
    rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    recStream = stream;
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size) chunks.push(e.data);
    };
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      recStream = null;
      clearInterval(recTimerInterval);
      $("rec-timer").hidden = true;
      const elapsed = Date.now() - startedAt;
      if (elapsed < 400 || !chunks.length) {
        setBusy(false, "TAP TO TALK");
        return;
      }
      const type = rec.mimeType || "audio/webm";
      askAudio(new Blob(chunks, { type }), extFor(type));
    };

    rec.start();
    startedAt = Date.now();
    $("mic").classList.add("hot");
    $("mic").querySelector(".mic-label").textContent = "LISTENING…";
    $("rec-timer").hidden = false;
    $("rec-timer").textContent = "00:00";

    recTimerInterval = setInterval(() => {
      const sec = Math.floor((Date.now() - startedAt) / 1000);
      const m = String(Math.floor(sec / 60)).padStart(2, "0");
      const s = String(sec % 60).padStart(2, "0");
      $("rec-timer").textContent = `${m}:${s}`;
    }, 500);
  } catch (err) {
    alert(`Microphone permission error: ${err.message}`);
  }
}

function stopRec() {
  if (rec && rec.state !== "inactive") rec.stop();
  $("mic").classList.remove("hot");
}

// Event Listeners
$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (q) askText(q);
});

$("mic").addEventListener("click", (e) => {
  e.preventDefault();
  if (rec && rec.state === "recording") stopRec();
  else startRec();
});

// Prompt Chips
document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const text = btn.getAttribute("data-q");
    $("q").value = text;
    askText(text);
  });
});

// Audio TTS Readout
$("btn-speak").addEventListener("click", () => {
  if (!currentAnswerText) return;
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(currentAnswerText);
    const lang = $("lang-select").value;
    u.lang = lang === "en-IN" ? "en-IN" : "hi-IN";
    window.speechSynthesis.speak(u);
  } else {
    alert("Browser does not support SpeechSynthesis.");
  }
});

// Copy Answer
$("btn-copy").addEventListener("click", async () => {
  if (!currentAnswerText) return;
  try {
    await navigator.clipboard.writeText(currentAnswerText);
    const btn = $("btn-copy");
    btn.innerHTML = `<span class="btn-icon">✓</span> Copied!`;
    setTimeout(() => {
      btn.innerHTML = `<span class="btn-icon">📋</span> Copy`;
    }, 2000);
  } catch {
    alert("Could not copy to clipboard.");
  }
});

// Benchmark Modal
$("btn-bench-modal").addEventListener("click", () => {
  $("modal-bench").hidden = false;
});

$("close-bench").addEventListener("click", () => {
  $("modal-bench").hidden = true;
});

$("btn-run-bench").addEventListener("click", async () => {
  const n = $("bench-n").value;
  $("btn-run-bench").disabled = true;
  $("bench-loading").hidden = false;
  $("bench-results").hidden = true;

  try {
    const res = await fetch(`/api/benchmark?n=${n}`).then((r) => r.json());
    $("p50").textContent = `${res.p50_ms.toFixed(1)}ms`;
    $("p70").textContent = `${res.p70_ms.toFixed(1)}ms`;
    $("p90").textContent = `${res.p90_ms.toFixed(1)}ms`;
    $("p100").textContent = `${res.p100_ms.toFixed(1)}ms`;
    $("bench-under-budget").textContent = `✓ ${res.under_200ms} / ${res.n} queries evaluated completed strictly under 200ms budget`;
    $("bench-note").textContent = res.note || "";
    $("bench-results").hidden = false;
  } catch (err) {
    alert(`Benchmark error: ${err.message}`);
  } finally {
    $("bench-loading").hidden = true;
    $("btn-run-bench").disabled = false;
  }
});

// Compare Strategies Modal
$("btn-compare-modal").addEventListener("click", () => {
  $("modal-compare").hidden = false;
});

$("close-compare").addEventListener("click", () => {
  $("modal-compare").hidden = true;
});

$("btn-run-compare").addEventListener("click", async () => {
  const q = $("compare-q").value.trim();
  if (!q) return;

  $("btn-run-compare").disabled = true;
  $("compare-loading").hidden = false;
  $("compare-results").hidden = true;

  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: q }),
    }).then((r) => r.json());

    const tbody = $("compare-table-body");
    tbody.innerHTML = (res.results || [])
      .map(
        (r) => `
        <tr>
          <td><code>${escapeHtml(r.strategy)}</code></td>
          <td>${r.chunks_created}</td>
          <td><span class="pill ${r.status}">${r.status}</span></td>
          <td><strong>${r.rag_ms.toFixed(1)}ms</strong></td>
          <td>${r.support.toFixed(3)}</td>
          <td>${escapeHtml(r.answer.slice(0, 140))}${r.answer.length > 140 ? "…" : ""}</td>
        </tr>
      `
      )
      .join("");
    $("compare-results").hidden = false;
  } catch (err) {
    alert(`Compare error: ${err.message}`);
  } finally {
    $("compare-loading").hidden = true;
    $("btn-run-compare").disabled = false;
  }
});

// Close modals when clicking backdrop
document.querySelectorAll(".modal-backdrop").forEach((m) => {
  m.addEventListener("click", (e) => {
    if (e.target === m) m.hidden = true;
  });
});

// Initial boot
checkHealth();
