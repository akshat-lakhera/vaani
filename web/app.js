const $ = (id) => document.getElementById(id);

let sttReady = false;

async function health() {
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    sttReady = Boolean(h.stt_configured);
    $("health").textContent = h.ok
      ? `${h.chunks.toLocaleString()} chunks · ${h.dense ? "hybrid" : "bm25"} · ${h.strategies.join(",")}\nSTT ${h.stt_provider}${sttReady ? "" : " (set SARVAM_API_KEY)"} · polish ${h.llm_configured ? "on" : "off"}`
      : "index not loaded — run scripts/ingest.py";
    $("mic").disabled = !sttReady;
    $("mic").querySelector(".label").textContent = sttReady ? "tap to talk" : "needs API key";
    $("stt-hint").hidden = sttReady;
  } catch {
    $("health").textContent = "api unreachable";
  }
}

function render(resp) {
  $("result").hidden = false;
  const st = $("status");
  st.textContent = resp.status;
  st.className = "pill " + resp.status;
  $("budget").textContent = resp.within_budget
    ? `rag ${resp.timings.total_rag_ms.toFixed(1)}ms < 200`
    : `rag ${resp.timings.total_rag_ms.toFixed(1)}ms`;
  $("support").textContent = `support ${Number(resp.support || 0).toFixed(3)}`;
  $("transcript").textContent = resp.transcript ? `heard: ${resp.transcript}` : "";
  $("answer").textContent = resp.answer || "";
  $("reason").textContent = resp.reason || (resp.polished ? "polished by Grok" : "");
  const t = resp.timings || {};
  $("timings").innerHTML = [
    ["stt", t.stt_ms],
    ["guard in", t.guard_in_ms],
    ["embed", t.embed_ms],
    ["retrieve", t.retrieve_ms],
    ["extract", t.extract_ms],
    ["guard out", t.guard_out_ms],
    ["polish", t.generate_ms],
    ["rag total", t.total_rag_ms],
  ]
    .filter(([, v]) => v != null)
    .map(([k, v]) => `<span>${k} ${Number(v).toFixed(1)}ms</span>`)
    .join("");
  $("cites").innerHTML = (resp.citations || [])
    .map(
      (c) =>
        `<li><strong>${c.passage_id}</strong> · ${c.lang} · ${c.query_type}<br/>${escapeHtml(c.text)}</li>`
    )
    .join("");
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setBusy(on, label) {
  $("mic").classList.toggle("busy", on);
  if (label) $("mic").querySelector(".label").textContent = label;
}

async function askText(text) {
  setBusy(true, "thinking…");
  try {
    const body = new FormData();
    body.set("text", text);
    body.set("polish", $("polish").checked ? "true" : "false");
    const resp = await fetch("/api/ask", { method: "POST", body }).then((r) => r.json());
    render(resp);
  } catch (err) {
    render({
      status: "refuse",
      answer: String(err.message || err),
      reason: "network",
      support: 0,
      timings: { total_rag_ms: 0 },
      citations: [],
    });
  } finally {
    setBusy(false, sttReady ? "tap to talk" : "needs API key");
  }
}

async function askAudio(blob, ext) {
  setBusy(true, "transcribing…");
  try {
    const body = new FormData();
    body.set("audio", blob, `clip.${ext}`);
    body.set("polish", $("polish").checked ? "true" : "false");
    const resp = await fetch("/api/ask", { method: "POST", body }).then((r) => r.json());
    render(resp);
  } catch (err) {
    render({
      status: "refuse",
      answer: String(err.message || err),
      reason: "network",
      support: 0,
      timings: { total_rag_ms: 0 },
      citations: [],
    });
  } finally {
    setBusy(false, sttReady ? "tap to talk" : "needs API key");
  }
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (q) askText(q);
});

let rec = null;
let recStream = null;
let chunks = [];
let startedAt = 0;

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
    alert("Set SARVAM_API_KEY in .env and restart the server.");
    return;
  }
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
    const elapsed = Date.now() - startedAt;
    if (elapsed < 400 || !chunks.length) {
      setBusy(false, "tap to talk");
      return;
    }
    const type = rec.mimeType || "audio/webm";
    askAudio(new Blob(chunks, { type }), extFor(type));
  };
  rec.start();
  startedAt = Date.now();
  $("mic").classList.add("hot");
  $("mic").querySelector(".label").textContent = "listening… tap to send";
}

function stopRec() {
  if (rec && rec.state !== "inactive") rec.stop();
  $("mic").classList.remove("hot");
}

$("mic").addEventListener("click", (e) => {
  e.preventDefault();
  if (rec && rec.state === "recording") stopRec();
  else startRec().catch((err) => alert(err.message));
});

health();
