"use strict";
/**
 * STRIVE dashboard client.
 *
 * Responsibilities, in order of importance:
 *   1. Render risk events streamed from the backend over WebSocket.
 *   2. Mirror the business-context policy locally so the context controls feel
 *      instant. Acoustic authenticity risk is NEVER computed here - it arrives
 *      from the engine and is only ever displayed.
 *   3. Fall back to an offline replay of baked events when this file is opened
 *      as the portable STRIVE_Demo.html (window.STRIVE_DEMO present).
 *
 * Contract with the backend: an unavailable branch reports score === null and
 * is rendered as an em dash, never as zero. Do not "helpfully" default these
 * to 0 - a zero reads as "genuine" and defeats the abstention design.
 */

/* ========================================================================== *
 * Constants
 * ========================================================================== */

const $ = (id) => document.getElementById(id);

/** True when running inside the self-contained STRIVE_Demo.html bundle. */
const portable = Boolean(window.STRIVE_DEMO);

/** Risk thresholds. Must stay in step with strive/config.py. */
const THRESHOLD = { warning: 0.5, alert: 0.75, critical: 0.9 };

/** Context-risk contributions. Mirrors strive/policy.py context_risk(). */
const CONTEXT_WEIGHT = {
  amountHigh: 0.35, // >= 10,00,000
  amountMedium: 0.15, // >= 1,00,000
  urgent: 0.2,
  newBeneficiary: 0.25,
  privileged: 0.2,
};
const AMOUNT_HIGH = 1000000;
const AMOUNT_MEDIUM = 100000;

/** Context alone forces at least VERIFY_CALLER at or above this value. */
const CONTEXT_FORCES_REVIEW = 0.7;

const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
const MAX_EVENT_ROWS = 24;
const MIC_QUEUE_LIMIT = 3;
const STATUS_POLL_MS = 5000;
const REPLAY_FRAME_MS = 100;
const DEFAULT_DURATION_S = 40;
const TARGET_SAMPLE_RATE = 16000;

/** Chart geometry and palette. Colours match the legend swatches in style.css. */
const CHART = {
  height: 250,
  padding: { left: 28, right: 12, top: 12, bottom: 24 },
  minWidth: 300,
  gridlines: [0, 0.25, 0.5, 0.75, 1],
  colour: {
    authenticity: "#e9b44c", // --trace-auth
    decision: "#8b9dff", // --trace-decision
    axis: "#5a616a", // --faint
    grid: "#1a1e23",
    gridWarning: "#4a3a12",
    gridAlert: "#5d2130",
    onset: "#eceef0", // --text
  },
};

/** Engine reason codes -> operator-facing text. */
const reasonNames = {
  LOW_EVIDENCE: "Collecting sufficient voiced evidence",
  LOW_AUDIO_ACTIVITY: "Low audio activity",
  BOOTSTRAP_REJECTED: "Opening evidence was not trusted",
  GLOBAL_REFERENCE_ANOMALY: "Artifact evidence increased",
  SESSION_INCONSISTENCY: "Voice differs from the in-call profile",
  BOUNDARY_DISCONTINUITY: "Speech continuity changed",
  GLOBAL_LANGUAGE_FALLBACK:
    "Language index unavailable; using global reference",
  SURROGATE_FEATURES_NOT_A_DEEPFAKE_VERDICT:
    "Demo DSP evidence; no neural accuracy claim",
  MODEL_OR_INDEX_ERROR: "Model or reference evidence unavailable",
  COMPUTE_EXCEEDS_STRIDE: "Inference exceeded the audio hop",
  CAPTURE_QUEUE_OVERFLOW: "Capture queue overflow; continuity reset",
  BRANCH_RESULT_REUSED: "Fresh cached branch result reused",
};

/** Per-branch availability reasons. */
const branchReasons = {
  MEASURED_UNCALIBRATED: "Measured, uncalibrated evidence",
  NO_GLOBAL_EVIDENCE: "No artifact evidence",
  NO_TRUSTED_PROFILE: "Temporary profile not established",
  NO_ADJACENT_VOICED_PAIR: "No valid adjacent voiced pair",
  QUALITY_NOT_SPOOF_RISK: "Reliability only; never spoof evidence",
};

/** Recommended actions -> display text. */
const policyNames = {
  HOLD_AND_ESCALATE: "HOLD & ESCALATE",
  HOLD_AND_VERIFY: "HOLD PENDING VERIFICATION",
  VERIFY_CALLER: "VERIFY CALLER",
  CONTINUE_MONITORING: "CONTINUE MONITORING",
  AWAIT_EVIDENCE: "AWAITING EVIDENCE",
};

const BRANCHES = ["artifact", "session", "coherence", "channel"];

/** Presentation scenario name -> engine scenario key used by the portable bundle. */
const PORTABLE_SCENARIOS = {
  genuine: "steady",
  spoof: "suspicious_start",
  mid_call: "switch",
};
const PORTABLE_ONSET_S = { genuine: null, spoof: 0, mid_call: 18 };

/* ========================================================================== *
 * Mutable state
 * ========================================================================== */

let token = "";
let callId = null;
let socket = null;
let audioCtx = null;
let media = null;
let node = null;

let events = [];
let auditEvents = [];
let last = null;
let running = false;
let attackOnset = null;
let playbackDuration = DEFAULT_DURATION_S;

/* ========================================================================== *
 * Formatting
 * ========================================================================== */

/** Render a 0-1 ratio as a whole percentage, or an em dash when absent. */
const pct = (value) => (value == null ? "—" : String(Math.round(value * 100)));

/** Render seconds as mm:ss. */
const clock = (value) =>
  `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(Math.floor(value % 60)).padStart(2, "0")}`;

/* ========================================================================== *
 * Business context (mirrors strive/policy.py)
 * ========================================================================== */

/** Read the business-context controls into the API payload shape. */
function context() {
  return {
    amount_inr: Math.max(0, Number($("amount").value) || 0),
    urgent: $("urgent").checked,
    new_beneficiary: $("beneficiary").checked,
    privileged_request: $("privileged").checked,
  };
}

/** Local mirror of policy.context_risk(). Kept in step deliberately. */
function contextScore(c) {
  const amount =
    c.amount_inr >= AMOUNT_HIGH
      ? CONTEXT_WEIGHT.amountHigh
      : c.amount_inr >= AMOUNT_MEDIUM
        ? CONTEXT_WEIGHT.amountMedium
        : 0;
  const total =
    amount +
    (c.urgent ? CONTEXT_WEIGHT.urgent : 0) +
    (c.new_beneficiary ? CONTEXT_WEIGHT.newBeneficiary : 0) +
    (c.privileged_request ? CONTEXT_WEIGHT.privileged : 0);
  return Math.min(1, total);
}

/** Map an authenticity risk to its state label. */
function riskState(risk) {
  if (risk == null) return "ANALYZING";
  if (risk >= THRESHOLD.critical) return "CRITICAL";
  if (risk >= THRESHOLD.alert) return "HIGH";
  if (risk >= THRESHOLD.warning) return "REVIEW";
  return "LOW";
}

/**
 * Local mirror of policy.decide(), used so the context controls respond without
 * a round-trip. Context can only amplify risk, never reduce it.
 */
function localPolicy(event) {
  const ctx = contextScore(context());
  const acoustic = event.s_risk;
  const decisionRisk =
    acoustic == null ? null : 1 - (1 - acoustic) * (1 - 0.5 * ctx);

  let decision;
  if (decisionRisk == null) decision = "ANALYZING";
  else if (decisionRisk >= THRESHOLD.critical) decision = "CRITICAL";
  else if (decisionRisk >= THRESHOLD.alert) decision = "HIGH";
  else if (decisionRisk >= THRESHOLD.warning || ctx >= CONTEXT_FORCES_REVIEW)
    decision = "REVIEW";
  else decision = "LOW";

  const auth = riskState(acoustic);
  const action =
    decision === "CRITICAL"
      ? "HOLD_AND_ESCALATE"
      : decision === "HIGH"
        ? "HOLD_AND_VERIFY"
        : decision === "REVIEW"
          ? "VERIFY_CALLER"
          : decision === "ANALYZING"
            ? "AWAIT_EVIDENCE"
            : "CONTINUE_MONITORING";

  return {
    ...event,
    context_risk: ctx,
    decision_risk: decisionRisk,
    state: auth,
    authenticity_state: auth,
    decision_state: decision,
    recommended_action: action,
  };
}

/* ========================================================================== *
 * API client
 * ========================================================================== */

/**
 * Fetch wrapper that attaches the bearer token, JSON-encodes plain objects and
 * surfaces the backend's `detail` message as the Error message.
 */
async function api(path, options = {}) {
  const headers = {
    ...(options.headers || {}),
    ...(token ? { Authorization: "Bearer " + token } : {}),
  };
  const body = options.body;
  const isRawBody = body instanceof Blob || body instanceof File;
  if (body && typeof body === "object" && !isRawBody) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(
      typeof detail.detail === "string"
        ? detail.detail
        : `Request failed (${response.status})`,
    );
  }
  return response.json();
}

/* ========================================================================== *
 * UI primitives
 * ========================================================================== */

/** Show a status line. `kind` is 'info' or 'error'. */
function message(text, kind = "info") {
  $("notice").textContent = text;
  $("notice").className = "notice " + kind;
}

/** Toggle the running state and the Stop button. */
function setBusy(value) {
  running = value;
  document.body.classList.toggle("busy", value);
  $("stop").disabled = !value;
}

/** Append a line to the audit trail, newest first, capped at MAX_EVENT_ROWS. */
function logEvent(text, time = last?.session_age_s || 0, kind = "info") {
  auditEvents.push({ time, text, kind });
  const row = document.createElement("p");
  row.textContent = `${clock(time)} · ${text}`;
  row.dataset.kind = kind;
  $("events").prepend(row);
  while ($("events").children.length > MAX_EVENT_ROWS)
    $("events").lastChild.remove();
}

/** Return every trusted-channel outcome button. */
const outcomeButtons = () => document.querySelectorAll(".outcomes button");

/* ========================================================================== *
 * View reset
 * ========================================================================== */

/** Return the dashboard to its pre-call state. Does not touch the backend. */
function resetView() {
  events = [];
  auditEvents = [];
  last = null;
  attackOnset = null;
  playbackDuration = DEFAULT_DURATION_S;

  document.body.classList.remove("has-data");
  $("reasons").dataset.signature = "";
  $("events").replaceChildren();
  logEvent("Ready for a new call", 0);

  $("auth").textContent = "—";
  $("decision").textContent = "—";
  $("call-state").textContent = "READY";
  $("call-state").className = "";
  $("auth-state").textContent = "ANALYZING";
  $("decision-state").textContent = "ANALYZING";
  $("bootstrap").textContent = "PENDING";
  $("entries").textContent = "0";
  $("similarity").textContent = "Similarity —";
  $("route").textContent = "—";
  $("transaction-result").textContent = "No action is currently held.";

  $("playback-card").hidden = true;
  $("alert-timing").hidden = true;
  $("onset-legend").hidden = true;
  $("hold").disabled = true;
  outcomeButtons().forEach((button) => {
    button.disabled = true;
  });

  document.querySelectorAll(".evidence-card").forEach((card) => {
    card.querySelector(".branch-state").textContent = "Unavailable";
    card.querySelector(".branch-value").textContent = "—";
    card.querySelector("progress").value = 0;
  });

  draw();
}

/* ========================================================================== *
 * Event normalisation
 * ========================================================================== */

/**
 * Older recorded events (portable bundle) carry track_scores rather than the
 * branches object. Rebuild the modern shape so one render path serves both.
 */
function normalizeEvent(event) {
  if (event.state) return event;
  const tracks = event.track_scores || {};
  const branch = (score, missingReason) => ({
    score,
    available: score != null,
    freshness: "recorded",
    reason: score == null ? missingReason : "MEASURED_UNCALIBRATED",
  });
  const branches = {
    artifact: branch(tracks.s_global, "NO_GLOBAL_EVIDENCE"),
    session: branch(tracks.s_session, "NO_TRUSTED_PROFILE"),
    coherence: branch(tracks.s_coherence, "NO_ADJACENT_VOICED_PAIR"),
    channel: {
      score: event.channel?.quality,
      available: event.channel?.quality != null,
      freshness: "recorded",
      reason: "QUALITY_NOT_SPOOF_RISK",
    },
  };
  return localPolicy({
    ...event,
    branches,
    workflow: "monitoring",
    hold_latched: false,
  });
}

/* ========================================================================== *
 * Risk timeline chart
 * ========================================================================== */

/**
 * Coalesce redraws into one per animation frame.
 *
 * Events arrive faster than the display refreshes (up to 10/s during offline
 * replay, plus a burst per resize). Painting synchronously on each one made
 * the dashboard feel sluggish, so callers schedule instead of painting.
 */
let drawQueued = false;
function draw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => {
    drawQueued = false;
    paint();
  });
}

/** Render the risk timeline. Call draw(); this is the frame-synced painter. */
function paint() {
  const canvas = $("chart");
  const ratio = window.devicePixelRatio || 1;
  const { left, right, top, bottom } = CHART.padding;
  const w = Math.max(CHART.minWidth, canvas.clientWidth);
  const h = CHART.height;
  const width = w - left - right;
  const height = h - top - bottom;

  // Assigning canvas.width reallocates the backing store, clears it and forces
  // layout. Only pay that when the size actually changed; otherwise clear.
  const pixelWidth = Math.round(w * ratio);
  const pixelHeight = Math.round(h * ratio);
  const c = canvas.getContext("2d");
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
  } else {
    c.clearRect(0, 0, w, h);
  }
  c.font = "10px ui-monospace, monospace";

  // Gridlines, with the review and alert thresholds dashed and tinted.
  for (const value of CHART.gridlines) {
    const y = top + (1 - value) * height;
    c.strokeStyle =
      value === THRESHOLD.alert
        ? CHART.colour.gridAlert
        : value === THRESHOLD.warning
          ? CHART.colour.gridWarning
          : CHART.colour.grid;
    c.setLineDash(
      value === THRESHOLD.warning || value === THRESHOLD.alert ? [4, 5] : [],
    );
    c.beginPath();
    c.moveTo(left, y);
    c.lineTo(w - right, y);
    c.stroke();
    c.fillStyle = CHART.colour.axis;
    c.fillText(String(value * 100), 0, y + 3);
  }

  const maxTime = Math.max(playbackDuration, last?.session_age_s || 0, 10);

  // Ground-truth marker for the synthetic scenarios.
  if (attackOnset != null) {
    const x = left + (attackOnset / maxTime) * width;
    c.strokeStyle = CHART.colour.onset;
    c.setLineDash([2, 4]);
    c.beginPath();
    c.moveTo(x, top);
    c.lineTo(x, top + height);
    c.stroke();
    c.fillStyle = CHART.colour.onset;
    c.fillText("VOICE CHANGE", Math.min(x + 4, w - 90), top + 10);
  }

  // A null value breaks the line rather than interpolating across an abstention.
  const line = (key, colour) => {
    c.strokeStyle = colour;
    c.lineWidth = 2.4;
    c.setLineDash([]);
    c.beginPath();
    let connected = false;
    for (const event of events) {
      const value = event[key];
      if (value == null) {
        connected = false;
        continue;
      }
      const x = left + (event.session_age_s / maxTime) * width;
      const y = top + (1 - value) * height;
      if (connected) c.lineTo(x, y);
      else c.moveTo(x, y);
      connected = true;
    }
    c.stroke();
  };
  line("authenticity_risk", CHART.colour.authenticity);
  line("decision_risk", CHART.colour.decision);

  c.fillStyle = CHART.colour.axis;
  for (let i = 0; i <= 4; i++) {
    c.fillText(clock((maxTime * i) / 4), left + (i / 4) * width - 7, h - 4);
  }
}

/* ========================================================================== *
 * Rendering a risk event
 * ========================================================================== */

/** Update the headline risk figures and call state. */
function renderHeadline(event) {
  $("auth").textContent = pct(event.authenticity_risk ?? event.s_risk);
  $("context-risk").textContent = pct(event.context_risk);
  $("decision").textContent = pct(event.decision_risk);
  $("decision-state").textContent = event.decision_state || "ANALYZING";
  $("auth-state").textContent = event.authenticity_state || "ANALYZING";
  $("clock").textContent = clock(event.session_age_s);
  $("call-state").textContent = event.state;
  $("call-state").className = event.state.toLowerCase();
  $("call-detail").textContent =
    event.s_risk == null
      ? "Insufficient current evidence"
      : event.demo_only
        ? "Observed DSP evidence · uncalibrated"
        : "Observed neural evidence · uncalibrated";
  $("bootstrap").textContent = String(
    event.bootstrap || "pending",
  ).toUpperCase();
  $("entries").textContent = event.profile_entries ?? 0;
  $("similarity").textContent =
    "Similarity " +
    (event.session_similarity == null
      ? "unavailable"
      : pct(event.session_similarity) + "%");
  $("language-result").textContent = String(
    event.language || "und",
  ).toUpperCase();
  $("route").textContent = event.retrieval?.route || "—";
}

/** Update the four evidence cards. An unavailable branch shows an em dash. */
function renderBranches(event) {
  for (const name of BRANCHES) {
    const branch = event.branches?.[name] || {};
    const card = document.querySelector(`[data-branch="${name}"]`);
    const value = branch.score;
    card.querySelector(".branch-value").textContent =
      value == null ? "—" : pct(value);
    card.querySelector("progress").value = value ?? 0;
    card.querySelector(".branch-state").textContent = branch.available
      ? `${branch.freshness || "fresh"} · rel ${pct(branch.reliability)}`
      : branch.freshness || "unavailable";
    card.querySelector(".branch-reason").textContent =
      branchReasons[branch.reason] || branch.reason || "Unavailable";
  }
}

/** Update the engineering panel: channel, latency, queue and reason codes. */
function renderEngineering(event) {
  const channel = event.channel || {};
  const bandwidth =
    channel.estimated_bandwidth_hz == null
      ? "—"
      : Math.round(channel.estimated_bandwidth_hz) + " Hz";
  const snr = channel.snr_db == null ? "—" : channel.snr_db.toFixed(1) + " dB";
  const level =
    channel.rms_dbfs == null ? "—" : channel.rms_dbfs.toFixed(1) + " dBFS";
  $("channel-detail").textContent =
    `Bandwidth ${bandwidth} · SNR ${snr} · Input ${level}`;

  $("latency").textContent =
    event.latency_ms?.end_to_end == null
      ? "—"
      : event.latency_ms.end_to_end.toFixed(1) + " ms";
  $("queue").textContent =
    `${event.capture?.current_depth ?? 0} / ${event.capture?.capacity ?? "—"}`;
  $("dropped").textContent =
    event.capture?.windows_dropped ?? event.dropped_windows ?? 0;
  $("geometry").textContent = event.geometry
    ? `${event.geometry.sample_rate / 1000} kHz · ${event.geometry.window_s}s / ${event.geometry.hop_s}s`
    : "—";

  const reasons = event.reasons?.length
    ? event.reasons
    : ["NO_CURRENT_ANOMALY"];
  // Reason codes change rarely; rebuilding these nodes on every window was
  // pure churn. Skip when the set matches what is already rendered.
  const signature = reasons.join("|");
  if ($("reasons").dataset.signature === signature) return;
  $("reasons").dataset.signature = signature;
  $("reasons").replaceChildren(
    ...reasons.map((reason) => {
      const span = document.createElement("span");
      span.textContent = reasonNames[reason] || reason;
      return span;
    }),
  );
}

/** Update the recommended action and enable the prevention controls. */
function renderPolicy(event) {
  $("policy").textContent =
    policyNames[event.recommended_action] || event.recommended_action;
  $("policy").className =
    "policy " + (event.decision_state || "analyzing").toLowerCase();

  const held =
    Boolean(event.hold_latched) ||
    ["HOLD_AND_VERIFY", "HOLD_AND_ESCALATE"].includes(event.recommended_action);
  $("hold").disabled = !callId || portable;
  outcomeButtons().forEach((button) => {
    button.disabled = !callId || !held || portable;
  });
  if (held) {
    $("transaction-result").textContent =
      "Possible voice impersonation. Sensitive action requires a trusted-channel check.";
  }
}

/** Write audit lines only on transitions, so the trail stays readable. */
function renderTransitions(event) {
  const previous = events.at(-2);
  if (!previous || previous.state !== event.state) {
    const text =
      event.state === "LOW"
        ? "Low observed risk"
        : event.state === "ANALYZING"
          ? "Analyzing audio evidence"
          : `${event.state} threshold crossed`;
    logEvent(text, event.session_age_s, event.state.toLowerCase());
  }
  if (event.bootstrap === "trusted" && previous?.bootstrap !== "trusted") {
    logEvent(
      "Temporary in-call profile established",
      event.session_age_s,
      "profile",
    );
  }
  const inconsistent = event.reasons?.includes("SESSION_INCONSISTENCY");
  if (inconsistent && !previous?.reasons?.includes("SESSION_INCONSISTENCY")) {
    logEvent(
      "Session voice consistency changed",
      event.session_age_s,
      "review",
    );
  }
}

/** Render one risk event end to end. */
function display(raw) {
  const event = normalizeEvent(raw);
  last = event;
  events.push(event);

  // Clears the loading skeletons the first time real evidence lands.
  document.body.classList.add("has-data");

  renderHeadline(event);
  renderBranches(event);
  renderEngineering(event);
  renderPolicy(event);
  renderTransitions(event);

  draw();
  $("download").disabled = false;
}

/* ========================================================================== *
 * Call lifecycle
 * ========================================================================== */

/** Close the socket and release the backend call. Safe to call repeatedly. */
async function closeCall() {
  if (socket) {
    socket.onclose = null;
    socket.close();
    socket = null;
  }
  $("websocket-state").textContent = "Disconnected";
  if (callId && !portable) {
    await api("/v1/calls/" + callId, { method: "DELETE" }).catch(() => {});
  }
  callId = null;
}

/** Tear down capture, audio graph and call. */
async function stop() {
  if (node) {
    node.disconnect();
    node = null;
  }
  if (media) {
    media.getTracks().forEach((track) => track.stop());
    media = null;
  }
  if (audioCtx) {
    await audioCtx.close().catch(() => {});
    audioCtx = null;
  }
  await closeCall();
  setBusy(false);
  message("Call stopped. Transient audio and in-call profile were deleted.");
  logEvent("Call ended", last?.session_age_s || 0);
}

/** Create a backend call carrying the current business context. */
async function createCall() {
  await closeCall();
  const result = await api("/v1/calls", {
    method: "POST",
    body: { language: "auto", context: context() },
  });
  callId = result.call_id;
  return result;
}

/** Handle one inbound WebSocket frame. */
function handleSocketMessage(data, resolve) {
  if (data.type === "ready") {
    $("websocket-state").textContent = "Connected";
    resolve();
    return;
  }
  if (data.type === "playback_started") {
    attackOnset = data.attack_onset_sec;
    playbackDuration = data.duration_s;
    $("playback-card").hidden = false;
    $("source-name").textContent = data.source;
    $("source-meta").textContent =
      `${data.mode.toUpperCase()} · 16 kHz mono${data.synthetic ? " · DEMO / SYNTHETIC" : ""}`;
    $("onset-legend").hidden = attackOnset == null;
    logEvent("Call started", 0);
    return;
  }
  if (data.type === "events") {
    data.events.forEach(display);
    if (data.playback) {
      $("playback-progress").value = data.playback.progress;
      $("playback-clock").textContent =
        `${clock(data.playback.current_s)} / ${clock(data.playback.duration_s)}`;
    }
    return;
  }
  if (data.type === "playback_complete") {
    setBusy(false);
    $("playback-progress").value = 1;
    if (data.attack_onset_sec != null) {
      const firstHigh =
        data.first_high_sec == null
          ? "not reached"
          : data.first_high_sec.toFixed(1) + " s";
      const timeToAlert =
        data.time_to_alert_sec == null
          ? "not available"
          : data.time_to_alert_sec.toFixed(1) + " s";
      $("alert-timing").hidden = false;
      $("alert-timing").textContent =
        `Attack onset: ${data.attack_onset_sec.toFixed(1)} s · First HIGH: ${firstHigh} · Time-to-alert: ${timeToAlert}`;
    }
    message("Playback complete. Use Hold & Verify to demonstrate prevention.");
    logEvent("Audio playback complete", data.duration_s);
    return;
  }
  if (data.type === "error") {
    message(data.message, "error");
    setBusy(false);
  }
}

/** Open the risk stream and resolve once the server sends `ready`. */
async function connect(path) {
  return new Promise((resolve, reject) => {
    socket = new WebSocket(location.origin.replace(/^http/, "ws") + path);
    socket.onopen = () => socket.send(JSON.stringify({ token }));
    socket.onerror = () =>
      reject(new Error("Could not connect to the real-time stream."));
    socket.onclose = () => {
      if (running)
        message("WebSocket disconnected. Start a new call.", "error");
      $("websocket-state").textContent = "Disconnected";
    };
    socket.onmessage = (messageEvent) =>
      handleSocketMessage(JSON.parse(messageEvent.data), resolve);
  });
}

/* ========================================================================== *
 * Scenario playback
 * ========================================================================== */

/** Stream a synthetic presentation scenario through the live pipeline. */
async function runScenario(name) {
  try {
    resetView();
    setBusy(true);
    const call = await createCall();
    await connect(call.ws_path);
    socket.send(
      JSON.stringify({
        type: "playback",
        scenario: name,
        mode: $("playback-mode").value,
      }),
    );
    message(
      name === "mid_call"
        ? "Building a trusted profile, then introducing the synthetic source change at 15 seconds."
        : "Streaming the synthetic scenario through the live pipeline.",
    );
  } catch (error) {
    setBusy(false);
    message(error.message, "error");
  }
}

/** Replay baked events from the portable bundle. No model runs in this path. */
async function runPortable(name) {
  resetView();
  setBusy(true);
  const record = structuredClone(window.STRIVE_DEMO[PORTABLE_SCENARIOS[name]]);
  attackOnset = PORTABLE_ONSET_S[name];
  playbackDuration = DEFAULT_DURATION_S;

  $("playback-card").hidden = false;
  $("source-name").textContent = record.description;
  $("source-meta").textContent = "OFFLINE REPLAY / FALLBACK DEMO";
  $("onset-legend").hidden = attackOnset == null;

  let i = 0;
  const timer = setInterval(() => {
    if (i >= record.events.length) {
      clearInterval(timer);
      setBusy(false);
      message("Offline replay complete. No model ran in this file.");
      return;
    }
    display(record.events[i++]);
    $("playback-progress").value = i / record.events.length;
    $("playback-clock").textContent = `${clock(i)} / 00:40`;
  }, REPLAY_FRAME_MS);
}

/* ========================================================================== *
 * Input sources
 * ========================================================================== */

/** Decode an uploaded recording in memory and stream it through the pipeline. */
async function handleUpload() {
  const file = $("file").files[0];
  if (!file) return;
  try {
    if (file.size > MAX_UPLOAD_BYTES)
      throw new Error("Maximum upload size is 20 MiB.");
    resetView();
    setBusy(true);
    const call = await createCall();
    const metadata = await api(
      "/v1/calls/" +
        callId +
        "/upload?filename=" +
        encodeURIComponent(file.name),
      {
        method: "POST",
        body: file,
        headers: { "Content-Type": "application/octet-stream" },
      },
    );
    await connect(call.ws_path);
    $("source-name").textContent = metadata.filename;
    socket.send(
      JSON.stringify({ type: "playback", mode: $("playback-mode").value }),
    );
    message(
      "Decoded in memory and streaming through the same call pipeline as microphone audio.",
    );
  } catch (error) {
    setBusy(false);
    message(error.message, "error");
    await closeCall();
  } finally {
    $("file").value = "";
  }
}

/**
 * Capture live microphone audio through an AudioWorklet and stream 16 kHz PCM.
 * Backpressure: one outstanding frame at a time; more than MIC_QUEUE_LIMIT
 * queued frames means inference has fallen behind and capture stops.
 */
async function startMicrophone() {
  try {
    if (portable) throw new Error("Microphone needs the local backend.");
    resetView();
    setBusy(true);
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone access requires localhost or HTTPS.");
    }
    media = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    const call = await createCall();
    await connect(call.ws_path);

    audioCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    await audioCtx.resume();
    await audioCtx.audioWorklet.addModule("/assets/pcm-worklet.js");

    const source = audioCtx.createMediaStreamSource(media);
    node = new AudioWorkletNode(audioCtx, "strive-pcm");
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    source.connect(node);
    node.connect(mute);
    mute.connect(audioCtx.destination);

    let sequence = 0;
    let pending = false;
    const queue = [];

    const sendNext = () => {
      if (pending || !queue.length || socket?.readyState !== WebSocket.OPEN)
        return;
      const bytes = new Uint8Array(queue.shift());
      let binary = "";
      for (let i = 0; i < bytes.length; i++)
        binary += String.fromCharCode(bytes[i]);
      pending = true;
      socket.send(
        JSON.stringify({ sequence: sequence++, pcm_s16le: btoa(binary) }),
      );
    };

    socket.onmessage = (messageEvent) => {
      const data = JSON.parse(messageEvent.data);
      if (data.type === "events") {
        pending = false;
        data.events.forEach(display);
        sendNext();
      } else if (data.type === "error") {
        message(data.message, "error");
      }
    };

    node.port.onmessage = (event) => {
      queue.push(event.data);
      if (queue.length > MIC_QUEUE_LIMIT) {
        stop();
        message("Capture stopped because inference fell behind.", "error");
        return;
      }
      sendNext();
    };

    $("playback-card").hidden = false;
    $("source-name").textContent = "Live microphone";
    $("source-meta").textContent =
      `Browser input ${audioCtx.sampleRate} Hz → 16 kHz mono`;
    $("websocket-state").textContent = "Connected";
    logEvent("Microphone capture started", 0);
    message(
      "Microphone active. Speak naturally; low observed risk is not identity verification.",
    );
  } catch (error) {
    await stop();
    message(error.message, "error");
  }
}

/* ========================================================================== *
 * Prevention workflow
 * ========================================================================== */

/** Apply business context. Acoustic authenticity risk is preserved verbatim. */
async function applyContext() {
  const before = last?.authenticity_risk;
  if (callId && !portable) {
    await api("/v1/calls/" + callId + "/context", {
      method: "PATCH",
      body: context(),
    }).catch((error) => message(error.message, "error"));
  }
  $("context-risk").textContent = pct(contextScore(context()));
  if (last) {
    const revised = localPolicy(last);
    revised.authenticity_risk = before; // context must never move acoustic risk
    display(revised);
  }
  message("Business context updated. Acoustic authenticity risk is unchanged.");
  logEvent("Business context updated", last?.session_age_s || 0);
}

/** Latch a hold on the sensitive action and unlock the outcome buttons. */
async function holdAction() {
  try {
    if (!callId) return;
    const result = await api("/v1/calls/" + callId + "/hold", {
      method: "POST",
    });
    $("policy").textContent = "HOLD PENDING VERIFICATION";
    $("policy").className = "policy held";
    $("transaction-result").textContent =
      "Sensitive action held. Select a trusted-channel outcome.";
    outcomeButtons().forEach((button) => {
      button.disabled = false;
    });
    logEvent("Sensitive action held", last?.session_age_s || 0, "high");
    message(result.recommended_action.replaceAll("_", " "));
  } catch (error) {
    message(error.message, "error");
  }
}

/** Record a simulated trusted-channel verification outcome. */
async function recordOutcome(outcome) {
  try {
    const method = $("verification").value;
    const result = await api("/v1/calls/" + callId + "/verify", {
      method: "POST",
      body: { method, outcome },
    });
    $("transaction-result").textContent =
      outcome === "verified"
        ? "Trusted verification succeeded. Action may proceed."
        : outcome === "failed"
          ? "Verification failed. Action blocked and escalated."
          : "Supervisor review requested. Action remains held.";
    $("policy").textContent =
      outcome === "verified"
        ? "VERIFIED · MAY PROCEED"
        : outcome === "failed"
          ? "BLOCKED & ESCALATED"
          : "HELD · SUPERVISOR REVIEW";
    $("policy").className =
      "policy " + (outcome === "failed" ? "blocked" : outcome);
    logEvent(
      outcome === "verified"
        ? "Verification succeeded"
        : outcome === "failed"
          ? "Verification failed; action blocked"
          : "Escalated to supervisor review",
      last?.session_age_s || 0,
      outcome,
    );
    message(result.status.replaceAll("_", " "));
  } catch (error) {
    message(error.message, "error");
  }
}

/* ========================================================================== *
 * Export and access
 * ========================================================================== */

/** Download session metadata. Raw audio is never included. */
function exportEvidence() {
  const payload = {
    schema_version: "sih-mvp-1",
    evidence_type: portable
      ? "offline_recorded_replay"
      : "local_runtime_metadata",
    raw_audio_included: false,
    attack_onset_sec: attackOnset,
    events,
    audit_events: auditEvents,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "strive-session-evidence.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Prompt for a bearer token. The token stays in page memory only. */
async function configureAccess() {
  token = prompt("API bearer token (empty for loopback demo):", "") || "";
  try {
    await api("/ready");
    message("API access verified. Token remains only in page memory.");
  } catch (error) {
    message(error.message, "error");
  }
}

/* ========================================================================== *
 * Backend status
 * ========================================================================== */

/** Poll /ready and /v1/status to keep the engineering panel current. */
async function refreshStatus() {
  if (portable) return;
  try {
    const [ready, status] = await Promise.all([
      api("/ready"),
      api("/v1/status"),
    ]);
    $("backend").textContent = "Healthy";
    $("connection").textContent = "Local engine";
    $("connection-dot").className = "online";
    $("device").textContent = ready.device.toUpperCase();
    $("model").textContent = ready.model_version;
    $("index-state").textContent = ready.reference_index_ready
      ? "READY"
      : "NOT LOADED";
    $("percentiles").textContent =
      status.latency_ms.p50 == null
        ? "—"
        : `${status.latency_ms.p50.toFixed(1)} / ${status.latency_ms.p95.toFixed(1)} ms`;
    $("geometry").textContent =
      `${status.geometry.sample_rate / 1000} kHz · ${status.geometry.window_s}s / ${status.geometry.hop_s}s`;
    if (ready.mode === "research") {
      $("mode").innerHTML =
        "RESEARCH NEURAL<small>Frozen local models · uncalibrated scores</small>";
      document.querySelectorAll(".scenario").forEach((button) => {
        button.disabled = true;
      });
    }
  } catch (error) {
    $("backend").textContent = "Unavailable";
    $("connection").textContent = "Backend unavailable";
    message(error.message, "error");
  }
}

/* ========================================================================== *
 * Wiring
 * ========================================================================== */

document.querySelectorAll(".scenario").forEach((button) => {
  button.addEventListener("click", () =>
    portable
      ? runPortable(button.dataset.scenario)
      : runScenario(button.dataset.scenario),
  );
});

outcomeButtons().forEach((button) => {
  button.addEventListener("click", () => recordOutcome(button.dataset.outcome));
});

$("upload").addEventListener("click", () =>
  portable ? message("Upload needs the local backend.") : $("file").click(),
);
$("file").addEventListener("change", handleUpload);
$("mic").addEventListener("click", startMicrophone);
$("stop").addEventListener("click", stop);
$("hold").addEventListener("click", holdAction);
$("context-apply").addEventListener("click", applyContext);
$("download").addEventListener("click", exportEvidence);
$("auth-config").addEventListener("click", configureAccess);

$("new-call").addEventListener("click", async () => {
  await stop();
  resetView();
  message("New call ready.");
});

$("reset").addEventListener("click", async () => {
  await stop();
  resetView();
  $("amount").value = AMOUNT_HIGH;
  $("urgent").checked = true;
  $("beneficiary").checked = true;
  $("privileged").checked = false;
  $("context-risk").textContent = "80";
  message("Demo reset to the recommended transaction context.");
});

$("presentation-mode").addEventListener("click", () => {
  const on = document.body.classList.toggle("presentation");
  $("presentation-mode").textContent = on
    ? "Exit Presentation Mode"
    : "Presentation Mode";
  $("presentation-mode").setAttribute("aria-pressed", String(on));
});

window.addEventListener("resize", draw);

/* ========================================================================== *
 * Boot
 * ========================================================================== */

$("context-risk").textContent = pct(contextScore(context()));

if (portable) {
  $("connection").textContent = "Offline replay";
  $("connection-dot").className = "online";
  $("backend").textContent = "Offline fallback";
  $("mic").disabled = true;
  $("upload").disabled = true;
  $("auth-config").disabled = true;
  $("mode").innerHTML =
    "OFFLINE REPLAY / FALLBACK DEMO<small>No model runs in this file</small>";
  message(
    "Offline replay of captured API events. Use the local application for live audio.",
  );
} else {
  refreshStatus();
  setInterval(refreshStatus, STATUS_POLL_MS);
}

draw();
