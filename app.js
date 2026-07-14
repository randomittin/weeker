/* ============================================================
   Weeker — vanilla no-build study UI
   fetch wrapper + hash router + per-view renderers.
   Same-origin API (base = ""). No framework, no external requests.
   ============================================================ */
"use strict";

function noop() { return undefined; }

/* ---------------- DOM helpers ---------------- */
function el(tag, attrs, children) {
  const n = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    const v = attrs[k];
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (k === "dataset") for (const d in v) n.dataset[d] = v[d];
    else if (v === true) n.setAttribute(k, "");
    else n.setAttribute(k, v);
  }
  if (children != null) {
    const arr = Array.isArray(children) ? children : [children];
    for (const c of arr) { if (c == null || c === false) continue; n.append(c.nodeType ? c : document.createTextNode(String(c))); }
  }
  return n;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
const VIEW = () => document.getElementById("view");

/* ---------------- Toasts ---------------- */
function toast(msg, kind) {
  const box = document.getElementById("toasts");
  const t = el("div", { class: "toast " + (kind || "") }, [
    el("span", { text: msg }),
    el("button", { class: "x", "aria-label": "Dismiss", onclick: () => t.remove() }, "✕"),
  ]);
  box.append(t);
  setTimeout(() => t.remove(), 6000);
}

/* ---------------- Auth / API base ---------------- */
// Base origin for all API calls. "" = same-origin (local `weeker serve`).
// Set window.WEEKER_API_BASE in config.js for hosted static deploys.
const API_BASE = String(window.WEEKER_API_BASE || "").replace(/\/+$/, "");
const TOKEN_KEY = "weeker_token";
const USER_KEY = "weeker_user";
const EXP_KEY = "weeker_token_exp";
// Paths that must NOT carry the bearer token and must NOT trigger a 401 logout.
const AUTH_EXEMPT = new Set(["/api/login", "/api/health"]);

function getToken() { try { return localStorage.getItem(TOKEN_KEY); } catch (_) { return null; } }
function getUser() { try { return localStorage.getItem(USER_KEY); } catch (_) { return null; } }
function setAuth(token, username, expiresIn) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
    if (username != null) localStorage.setItem(USER_KEY, username);
    if (expiresIn) localStorage.setItem(EXP_KEY, String(Date.now() + Number(expiresIn) * 1000));
    else localStorage.removeItem(EXP_KEY);
  } catch (_) { /* storage unavailable — token lives only for this tab session */ }
}
function clearAuth() {
  try { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY); localStorage.removeItem(EXP_KEY); } catch (_) { /* ignore */ }
}
function tokenExpired() {
  try { const e = localStorage.getItem(EXP_KEY); return e != null && Date.now() > Number(e); } catch (_) { return false; }
}
// Session-expired / no-token: drop creds and bounce to the login screen.
function handleUnauthorized() { clearAuth(); showLogin(true); }

/* ---------------- API wrapper ---------------- */
class ApiError extends Error {
  constructor(kind, message) { super(message); this.kind = kind; }
}
function networkMessage() {
  return API_BASE
    ? "Backend unreachable — check the API URL (WEEKER_API_BASE) or your connection."
    : "Backend not running — run `weeker serve`";
}
async function request(method, path, body) {
  const exempt = AUTH_EXEMPT.has(path);
  const opts = { method, headers: {} };
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  const token = getToken();
  if (token && !exempt) opts.headers["Authorization"] = "Bearer " + token;
  let res;
  try { res = await fetch(API_BASE + path, opts); }
  catch (e) { throw new ApiError("network", networkMessage()); }
  if (res.status === 401 && !exempt) { handleUnauthorized(); throw new ApiError("auth", "Session expired — please sign in again."); }
  if (res.status === 204) return null;
  let data = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) { try { data = await res.json(); } catch (_) { data = null; } }
  if (!res.ok) throw new ApiError("http", (data && (data.detail || data.error)) || ("Request failed (HTTP " + res.status + ")"));
  return data;
}
const api = {
  get: (p) => request("GET", p),
  post: (p, b) => request("POST", p, b === undefined ? {} : b),
};

// Login uses a raw fetch (no bearer, no global 401 logout) so the login
// screen can show an inline "invalid credentials" vs "unreachable" message.
async function loginRequest(username, password) {
  let res;
  try {
    res = await fetch(API_BASE + "/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
  } catch (e) { throw new ApiError("network", networkMessage()); }
  if (res.status === 401) throw new ApiError("auth", "Invalid username or password.");
  let data = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) { try { data = await res.json(); } catch (_) { data = null; } }
  if (!res.ok) throw new ApiError("http", (data && (data.detail || data.error)) || ("Login failed (HTTP " + res.status + ")"));
  return data;
}

/* ---------------- Shared render states ---------------- */
function loadingState(label) {
  return el("div", { class: "state" }, [el("div", { class: "spinner", "aria-hidden": "true" }), el("p", { class: "muted", text: label || "Loading…" })]);
}
function errorState(err, retry) {
  const down = err && err.kind === "network";
  const s = el("div", { class: "state error" }, [
    el("div", { class: "ico", "aria-hidden": "true", text: down ? "⚡" : "⚠" }),
    el("h2", { text: down ? "Backend not running" : "Something went wrong" }),
    el("p", { text: down ? "Start the API with `weeker serve`, then retry." : (err && err.message) || "Unexpected error." }),
  ]);
  if (retry) s.append(el("button", { class: "btn primary", onclick: retry }, "Retry"));
  return s;
}
function emptyState(icon, title, msg, action) {
  const s = el("div", { class: "state" }, [
    el("div", { class: "ico", "aria-hidden": "true", text: icon }),
    el("h2", { text: title }),
    msg ? el("p", { text: msg }) : null,
  ]);
  if (action) s.append(action);
  return s;
}
function pageHead(title, sub) {
  return el("header", { class: "page-head" }, [
    el("h1", { text: title }),
    sub ? el("p", { class: "sub", text: sub }) : null,
  ]);
}

/* ---------------- Utility ---------------- */
const pct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");
const num = (x, d) => (x == null ? "—" : Number(x).toFixed(d == null ? 2 : d));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function titleCase(s) { return String(s).replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()); }

// Sequential green mastery ramp (dark -> bright). Color + number => never color-alone.
const RAMP = [[11, 46, 35], [20, 83, 45], [22, 101, 52], [22, 163, 74], [74, 222, 128]];
function masteryRGB(m) {
  m = clamp(Number(m) || 0, 0, 1);
  const seg = (RAMP.length - 1) * m, i = Math.floor(seg), t = seg - i;
  const a = RAMP[i], b = RAMP[Math.min(i + 1, RAMP.length - 1)];
  return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * t));
}
function masteryFg(rgb) { return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) > 140 ? "#08130D" : "#EAFBF1"; }

const CONF = [
  { key: "1", val: "sure", label: "Sure" },
  { key: "2", val: "unsure", label: "Unsure" },
  { key: "3", val: "guessing", label: "Guessing" },
];

/* ============================================================
   Option / confidence widgets (shared by study, diagnostic, mock)
   ============================================================ */
function optionList(options, chosenKey, revealKey, onPick) {
  const wrap = el("div", { class: "options", role: "radiogroup", "aria-label": "Answer options" });
  options.forEach((o, i) => {
    const k = String(o.key);
    let cls = "opt";
    const disabled = !!revealKey;
    const isChosen = chosenKey && k.toLowerCase() === String(chosenKey).toLowerCase();
    if (revealKey) {
      if (k.toLowerCase() === String(revealKey).toLowerCase()) cls += " correct";
      else if (isChosen) cls += " wrong";
    } else if (isChosen) cls += " selected";
    const b = el("button", {
      class: cls, type: "button", role: "radio",
      "aria-checked": isChosen ? "true" : "false",
      "aria-disabled": disabled ? "true" : null,
      onclick: disabled ? null : () => onPick(k),
    }, [
      el("span", { class: "key", "aria-hidden": "true", text: (o.key || String.fromCharCode(97 + i)).toString().toUpperCase() }),
      el("span", { class: "txt", text: o.text }),
    ]);
    wrap.append(b);
  });
  return wrap;
}
function confidenceRow(current, onPick) {
  const row = el("div", { class: "conf", role: "group", "aria-label": "Confidence" });
  CONF.forEach((c) => {
    row.append(el("button", {
      class: "conf-btn" + (current === c.val ? " on" : ""), type: "button",
      "aria-pressed": current === c.val ? "true" : "false",
      onclick: () => onPick(c.val),
    }, [el("span", { class: "kbd", text: c.key }), el("span", { text: c.label })]));
  });
  return row;
}
// Map a keyboard key to an option key given the options array.
function keyToOption(e, options) {
  const k = e.key.toLowerCase();
  const byLetter = options.find((o) => String(o.key).toLowerCase() === k);
  if (byLetter) return String(byLetter.key);
  if (/^[1-9]$/.test(k)) { const idx = parseInt(k, 10) - 1; if (idx < options.length) return String(options[idx].key); }
  return null;
}

/* ============================================================
   Router
   ============================================================ */
const routes = { dashboard: viewDashboard, study: viewStudy, diagnostic: viewDiagnostic, mock: viewMock, flash: viewFlash };
let activeCleanup = null;

function currentRoute() {
  const h = (location.hash || "").replace(/^#\/?/, "");
  const name = h.split("/")[0] || "dashboard";
  return routes[name] ? name : "dashboard";
}
function setActiveNav(name) {
  document.querySelectorAll(".nav-item").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === name);
    if (a.dataset.route === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
}
function router() {
  if (typeof activeCleanup === "function") { try { activeCleanup(); } catch (_) { activeCleanup = null; } activeCleanup = null; }
  const name = currentRoute();
  setActiveNav(name);
  clear(VIEW());
  const main = document.getElementById("main");
  if (main) main.focus({ preventScroll: true });
  window.scrollTo(0, 0);
  activeCleanup = routes[name]() || null;
}

/* ============================================================
   Health (sidebar indicator)
   ============================================================ */
async function pollHealth() {
  const dot = document.getElementById("health-dot");
  const label = document.getElementById("health-label");
  try {
    const h = await api.get("/api/health");
    dot.className = "dot ok";
    const b = (h && h.bank) || {};
    label.textContent = (b.questions != null ? b.questions + " questions" : "online");
  } catch (e) {
    dot.className = "dot down";
    label.textContent = "backend offline";
  }
}

/* ============================================================
   DASHBOARD
   ============================================================ */
function viewDashboard() {
  const root = VIEW();
  root.append(pageHead("Dashboard", "Your readiness for NISM Series X-A at a glance."));
  const body = el("div", { class: "stack" });
  body.append(loadingState("Reading your model…"));
  root.append(body);

  Promise.allSettled([api.get("/api/health"), api.get("/api/status")]).then(([hr, sr]) => {
    clear(body);
    if (hr.status === "rejected" && hr.reason && hr.reason.kind === "network") {
      body.append(errorState(hr.reason, () => router()));
      return;
    }
    const health = hr.status === "fulfilled" ? hr.value : null;
    const status = sr.status === "fulfilled" ? sr.value : null;

    // Bank stats
    if (health && health.bank) {
      const b = health.bank;
      body.append(el("div", { class: "grid cols-3" }, [
        statTile("Concepts", b.concepts), statTile("Questions", b.questions), statTile("Caselets", b.caselets),
      ]));
    }

    if (sr.status === "rejected") {
      body.append(el("div", { class: "card" }, [
        emptyState("◇", "No model yet",
          "Run the diagnostic (36 questions, ~35 min) so Weeker can measure your mastery and predict your score.",
          el("a", { class: "btn primary", href: "#/diagnostic" }, "Start diagnostic")),
      ]));
      return;
    }

    const s = status || {};
    const cov = s.prediction ? s.prediction.coverage : null;
    if (!s.prediction || (cov != null && cov === 0 && (!s.heatmap || !s.heatmap.length))) {
      body.append(el("div", { class: "card" }, [
        emptyState("◇", "Not diagnosed yet",
          "Take the diagnostic so Weeker can seed your mastery model and start predicting.",
          el("a", { class: "btn primary", href: "#/diagnostic" }, "Start diagnostic")),
      ]));
      return;
    }

    // Prediction band
    if (s.prediction) body.append(predictionCard(s.prediction));

    // Due counts
    if (s.due_counts) {
      const d = s.due_counts;
      body.append(el("div", { class: "grid cols-2" }, [
        dueCard("Concepts due", d.concepts, "Review weak concepts in Study.", "#/study", "Study now"),
        dueCard("Flashcards due", d.flashcards, "Spaced-repetition cards ready.", "#/flash", "Review cards"),
      ]));
    }

    // Heatmap
    if (s.heatmap && s.heatmap.length) body.append(heatmapCard(s.heatmap));

    // Refill plan + calibration
    const twoCol = el("div", { class: "grid cols-2" });
    twoCol.append(refillCard(s.refill_plan));
    if (s.calibration && s.calibration.length) twoCol.append(calibrationCard(s.calibration));
    body.append(twoCol);
  });
}

function statTile(label, val) {
  return el("div", { class: "stat" }, [
    el("div", { class: "label", text: label }),
    el("div", { class: "num mono", text: val == null ? "—" : String(val) }),
  ]);
}
function dueCard(title, count, sub, href, cta) {
  return el("div", { class: "card" }, [
    el("div", { class: "row", style: "justify-content:space-between;align-items:flex-start" }, [
      el("div", {}, [el("h2", { text: title }), el("p", { class: "card-hint", text: sub })]),
      el("div", { class: "mono", style: "font-size:var(--fs-2xl);font-weight:650", text: count == null ? "—" : String(count) }),
    ]),
    el("a", { class: "btn" + (count ? " primary" : ""), href, style: "margin-top:var(--sp-3)" }, cta),
  ]);
}

function predictionCard(p) {
  const band = Array.isArray(p.band) ? p.band : [null, null];
  const lo = band[0], hi = band[1], exp = p.expected;
  const card = el("div", { class: "card" }, [el("h2", { text: "Predicted score" })]);

  card.append(el("div", { class: "grid cols-4", style: "margin-bottom:var(--sp-4)" }, [
    el("div", { class: "stat" }, [el("div", { class: "label", text: "Expected" }), el("div", { class: "num mono", text: exp == null ? "—" : num(exp, 1) })]),
    el("div", { class: "stat" }, [el("div", { class: "label", text: "Range" }), el("div", { class: "num mono sm", text: (lo == null ? "—" : num(lo, 0)) + "–" + (hi == null ? "—" : num(hi, 0)) })]),
    el("div", { class: "stat" }, [el("div", { class: "label", text: "Pass probability" }), el("div", { class: "num mono", text: pct(p.p_pass) })]),
    el("div", { class: "stat" }, [el("div", { class: "label", text: "Coverage" }), el("div", { class: "num mono", text: pct(p.coverage) })]),
  ]));

  // Visual band with expected marker
  if (lo != null && hi != null && exp != null && hi > lo) {
    const pad = (hi - lo) * 0.35 + 0.001;
    const dLo = lo - pad, dHi = hi + pad, span = dHi - dLo;
    const left = ((lo - dLo) / span) * 100, width = ((hi - lo) / span) * 100, mk = ((exp - dLo) / span) * 100;
    card.append(el("div", { class: "band-track", role: "img", "aria-label": "Expected " + num(exp, 1) + ", likely range " + num(lo, 0) + " to " + num(hi, 0) }, [
      el("div", { class: "rail" }),
      el("div", { class: "span", style: `left:${left}%;width:${width}%` }),
      el("div", { class: "mark", "data-v": num(exp, 1), style: `left:${mk}%` }),
    ]));
    card.append(el("div", { class: "row", style: "justify-content:space-between;color:var(--text-dim)" }, [
      el("span", { class: "xs mono", text: num(lo, 0) }), el("span", { class: "xs mono", text: num(hi, 0) }),
    ]));
  }

  // Pass-probability bar
  if (p.p_pass != null) {
    const good = p.p_pass >= 0.6;
    card.append(el("div", { style: "margin-top:var(--sp-4)" }, [
      el("div", { class: "row", style: "justify-content:space-between" }, [
        el("span", { class: "small muted", text: "Likelihood of passing" }),
        el("span", { class: "small mono", text: pct(p.p_pass) }),
      ]),
      el("div", { class: "bar " + (good ? "good" : "warn"), style: "margin-top:.35rem" }, el("i", { style: `width:${clamp(p.p_pass * 100, 0, 100)}%` })),
    ]));
  }
  return card;
}

function heatmapCard(heat) {
  // group by chapter, preserve order
  const groups = new Map();
  for (const it of heat) {
    const ch = it.chapter != null ? String(it.chapter) : "—";
    if (!groups.has(ch)) groups.set(ch, []);
    groups.get(ch).push(it);
  }
  const card = el("div", { class: "card" }, [
    el("div", { class: "row", style: "justify-content:space-between" }, [
      el("h2", { text: "Mastery heatmap" }),
      el("div", { class: "heat-legend" }, [el("span", { text: "low" }), el("span", { class: "ramp" }), el("span", { text: "high" })]),
    ]),
    el("p", { class: "card-hint", style: "margin-bottom:var(--sp-3)", text: "Each cell is a concept — brighter green means stronger mastery. Amber ring = due for review." }),
  ]);
  for (const [chap, items] of groups) {
    const cells = el("div", { class: "heat-cells" });
    items.forEach((it) => {
      const rgb = masteryRGB(it.mastery), fg = masteryFg(rgb);
      const flags = Array.isArray(it.flags) ? it.flags : [];
      const label = it.concept != null ? it.concept : "overall";
      const cell = el("div", {
        class: "heat-cell" + (it.due ? " due" : ""),
        style: `background:rgb(${rgb.join(",")});color:${fg}`,
        title: `${label} · mastery ${pct(it.mastery)}${it.due ? " · due" : ""}${flags.length ? " · " + flags.join(", ") : ""}`,
        "aria-label": `${chap} ${label}: mastery ${pct(it.mastery)}${it.due ? ", due" : ""}${flags.length ? ", flags " + flags.join(" ") : ""}`,
      }, [el("span", { text: Math.round(clamp(Number(it.mastery) || 0, 0, 1) * 100) })]);
      if (flags.length) cell.append(el("span", { class: "flag", "aria-hidden": "true" }));
      cells.append(cell);
    });
    card.append(el("div", { class: "heat-row" }, [el("div", { class: "chap", text: titleCase(chap) }), cells]));
  }
  return card;
}

function refillCard(plan) {
  const card = el("div", { class: "card" }, [el("h2", { text: "Tonight's refill plan" })]);
  if (!plan || !plan.length) {
    card.append(el("p", { class: "card-hint", text: "Nothing queued — you're all caught up." }));
    return card;
  }
  const list = el("ul", { class: "plan-list" });
  plan.forEach((item) => {
    if (item == null) return;
    if (typeof item === "object") {
      const primary = item.concept || item.chapter || item.title || item.name || item.label || item.kind || "Item";
      const rest = Object.entries(item).filter(([k]) => !["concept", "chapter", "title", "name", "label"].includes(k))
        .map(([k, v]) => `${titleCase(k)}: ${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ");
      list.append(el("li", {}, [el("span", { text: titleCase(String(primary)) }), rest ? el("span", { class: "muted small mono", text: rest }) : null]));
    } else {
      list.append(el("li", {}, [el("span", { text: String(item) })]));
    }
  });
  card.append(list);
  return card;
}

function calibrationCard(cal) {
  const card = el("div", { class: "card" }, [el("h2", { text: "Confidence calibration" })]);
  const cols = Object.keys(cal[0] || {});
  const tbl = el("table", { class: "table" });
  tbl.append(el("thead", {}, el("tr", {}, cols.map((c) => el("th", { text: titleCase(c) })))));
  const tb = el("tbody");
  cal.forEach((row) => {
    tb.append(el("tr", {}, cols.map((c) => {
      const v = row[c];
      const isNum = typeof v === "number";
      return el("td", { class: isNum ? "mono" : "", text: isNum ? (Number.isInteger(v) ? String(v) : num(v, 2)) : String(v == null ? "—" : v) });
    })));
  });
  tbl.append(tb);
  card.append(tbl);
  return card;
}

/* ============================================================
   STUDY  (core loop)
   ============================================================ */
function viewStudy() {
  const root = VIEW();
  root.append(pageHead("Study", "Adaptive practice — one question at a time."));
  const body = el("div", {});
  root.append(body);

  const st = { q: null, chosen: null, confidence: null, answered: false, result: null, sourceOpen: false, sourceData: null, sourceLoading: false, busy: false };

  function loadNext() {
    st.q = null; st.chosen = null; st.confidence = null; st.answered = false; st.result = null;
    st.sourceOpen = false; st.sourceData = null; st.sourceLoading = false;
    clear(body); body.append(loadingState("Finding your next question…"));
    api.get("/api/study/next").then((data) => {
      const q = data && data.question;
      if (!q) { renderEmpty(); return; }
      st.q = q; renderQ();
    }).catch((e) => { clear(body); body.append(errorState(e, loadNext)); });
  }

  function renderEmpty() {
    clear(body);
    body.append(el("div", { class: "card" }, emptyState("✓", "Nothing due right now",
      "You've cleared the queue. Come back later, or run a mock to pressure-test.",
      el("div", { class: "row", style: "justify-content:center" }, [
        el("a", { class: "btn", href: "#/dashboard" }, "Dashboard"),
        el("a", { class: "btn primary", href: "#/mock" }, "Start a mock"),
      ]))));
  }

  function submit() {
    if (st.busy || st.answered || !st.chosen) return;
    const conf = st.confidence || "unsure";
    st.busy = true; renderQ();
    api.post("/api/study/answer", { question_id: st.q.id, chosen_key: st.chosen, confidence: conf })
      .then((r) => { st.busy = false; st.answered = true; st.result = r; st.sourceData = r && r.source_peek ? { chunks: [r.source_peek] } : null; renderQ(); })
      .catch((e) => { st.busy = false; toast(e.message, "err"); renderQ(); });
  }

  function toggleSource() {
    st.sourceOpen = !st.sourceOpen;
    if (st.sourceOpen && !st.sourceData && !st.sourceLoading && st.q.concept != null) {
      st.sourceLoading = true; renderQ();
      api.get("/api/concept/" + encodeURIComponent(st.q.concept) + "/source")
        .then((d) => { st.sourceLoading = false; st.sourceData = d; renderQ(); })
        .catch(() => { st.sourceLoading = false; st.sourceData = { chunks: [] }; renderQ(); });
    } else { renderQ(); }
  }

  function renderQ() {
    const q = st.q; if (!q) return;
    clear(body);
    const card = el("div", { class: "card quiz" });
    card.append(el("div", { class: "q-meta" }, [
      q.chapter != null ? el("span", { class: "tag", text: titleCase(q.chapter) }) : null,
      q.concept != null ? el("span", { class: "tag violet", text: titleCase(q.concept) }) : null,
    ]));
    card.append(el("div", { class: "q-stem", text: q.stem }));
    card.append(optionList(q.options || [], st.chosen, st.answered && st.result ? st.result.correct_key : null, (k) => { if (!st.answered) { st.chosen = k; renderQ(); } }));

    if (!st.answered) {
      card.append(confidenceRow(st.confidence, (v) => { st.confidence = v; renderQ(); }));
      card.append(el("div", { class: "q-actions" }, [
        el("div", { class: "hint-line" }, [
          "Pick ", el("span", { class: "kbd", text: "a–d" }), " · confidence ", el("span", { class: "kbd", text: "1–3" }), " · submit ", el("span", { class: "kbd", text: "↵" }),
        ]),
        el("button", { class: "btn primary", disabled: !st.chosen || st.busy, onclick: submit }, st.busy ? "Submitting…" : "Submit"),
      ]));
    } else {
      card.append(revealPanel(st.result, st.q, st, toggleSource));
      card.append(el("div", { class: "q-actions" }, [
        el("div", { class: "hint-line" }, ["Next ", el("span", { class: "kbd", text: "n" }), " · peek source ", el("span", { class: "kbd", text: "p" })]),
        el("button", { class: "btn primary", onclick: loadNext }, "Next question →"),
      ]));
    }
    body.append(card);
  }

  function onKey(e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!st.q) return;
    if (!st.answered) {
      if (/^[1-3]$/.test(e.key) && st.chosen) {
        const c = CONF.find((x) => x.key === e.key); if (c) { st.confidence = c.val; renderQ(); e.preventDefault(); return; }
      }
      const opt = keyToOption(e, st.q.options || []);
      if (opt) { st.chosen = opt; renderQ(); e.preventDefault(); return; }
      if (e.key === "Enter") { submit(); e.preventDefault(); }
    } else {
      if (e.key === "n" || e.key === "Enter") { loadNext(); e.preventDefault(); }
      else if (e.key === "p") { toggleSource(); e.preventDefault(); }
    }
  }
  window.addEventListener("keydown", onKey);
  loadNext();
  return () => window.removeEventListener("keydown", onKey);
}

function revealPanel(r, q, st, toggleSource) {
  const ok = !!r.correct;
  const panel = el("div", { class: "reveal" });
  panel.append(el("div", { class: "reveal-head " + (ok ? "ok" : "no") }, [
    el("span", { "aria-hidden": "true", text: ok ? "✓" : "✕" }),
    el("span", { text: ok ? "Correct" : "Incorrect — answer is " + String(r.correct_key).toUpperCase() }),
  ]));
  const bodyEl = el("div", { class: "reveal-body" });
  if (r.explanation) bodyEl.append(el("p", { class: "explain", text: r.explanation }));

  if (r.delta) {
    const d = r.delta;
    bodyEl.append(el("div", { class: "delta" }, [
      deltaItem("θ (ability)", d.theta_before, d.theta_after, (x) => num(x, 2)),
      deltaItem("Mastery", d.mastery_before, d.mastery_after, (x) => pct(x)),
    ]));
  }
  if (r.misconception_fired) bodyEl.append(el("div", { class: "misc", text: "⚑ Misconception flagged: " + String(r.misconception_fired) }));

  bodyEl.append(el("button", { class: "btn ghost", style: "align-self:flex-start", "aria-expanded": st.sourceOpen ? "true" : "false", onclick: toggleSource },
    (st.sourceOpen ? "Hide source" : "Peek source") + " (p)"));
  if (st.sourceOpen) {
    if (st.sourceLoading) bodyEl.append(el("div", { class: "source" }, el("div", { class: "body muted", text: "Loading source…" })));
    else {
      const chunks = (st.sourceData && st.sourceData.chunks) || [];
      if (!chunks.length) bodyEl.append(el("div", { class: "source" }, el("div", { class: "body muted", text: "No source excerpt available." })));
      else chunks.forEach((c) => bodyEl.append(el("div", { class: "source" }, [
        c.pages != null ? el("div", { class: "pg", text: "pp. " + (Array.isArray(c.pages) ? c.pages.join("–") : c.pages) }) : null,
        el("div", { class: "body", text: c.text || "" }),
      ])));
    }
  }
  panel.append(bodyEl);
  return panel;
}
function deltaItem(label, before, after, fmt) {
  const dir = after != null && before != null ? (after > before ? "up" : after < before ? "down" : "") : "";
  const arrow = dir === "up" ? " ↑" : dir === "down" ? " ↓" : "";
  return el("div", { class: "d" }, [
    el("div", { class: "lbl", text: label }),
    el("div", { class: "val" }, [
      el("span", { text: before == null ? "—" : fmt(before) }),
      el("span", { class: "dim", text: " → " }),
      el("span", { class: dir, text: (after == null ? "—" : fmt(after)) + arrow }),
    ]),
  ]);
}

/* ============================================================
   DIAGNOSTIC  (measurement, no feedback)
   ============================================================ */
function viewDiagnostic() {
  const root = VIEW();
  root.append(pageHead("Diagnostic", "A one-time measurement to seed your mastery model."));
  const body = el("div", {});
  root.append(body);

  const st = { started: false, session: null, questions: [], answers: {}, idx: 0, submitting: false, result: null };

  function renderIntro() {
    clear(body);
    body.append(el("div", { class: "card" }, [
      emptyState("◇", "Diagnostic — 36 questions",
        "This measures where you stand across all chapters. It takes about 35 minutes and gives NO feedback during — that's what keeps the measurement honest. You can move back and forth before submitting.",
        el("button", { class: "btn primary lg", onclick: start }, "Begin diagnostic")),
      el("p", { class: "card-hint", style: "text-align:center;margin-top:var(--sp-3)", text: "Tip: answer every question, and set your confidence honestly." }),
    ]));
  }
  function start() {
    clear(body); body.append(loadingState("Assembling your diagnostic…"));
    api.post("/api/diagnostic/start").then((d) => {
      st.started = true; st.session = d.session_id; st.questions = d.questions || []; st.idx = 0;
      if (!st.questions.length) { clear(body); body.append(errorState({ message: "Diagnostic returned no questions." }, renderIntro)); return; }
      renderQ();
    }).catch((e) => { clear(body); body.append(errorState(e, start)); });
  }

  function renderQ() {
    const q = st.questions[st.idx]; const total = st.questions.length;
    const a = st.answers[q.id] || {};
    clear(body);

    const head = el("div", { class: "run-head" }, [
      el("div", { class: "run-progress" }, [
        el("div", { class: "row", style: "justify-content:space-between" }, [
          el("span", { class: "small muted", text: "Question " + (st.idx + 1) + " of " + total }),
          el("span", { class: "small mono", text: Object.keys(st.answers).length + "/" + total + " answered" }),
        ]),
        el("div", { class: "bar", style: "margin-top:.35rem" }, el("i", { style: `width:${((st.idx + 1) / total) * 100}%` })),
      ]),
      el("span", { class: "tag warn", text: "measurement — no feedback" }),
    ]);

    const card = el("div", { class: "card quiz" });
    card.append(el("div", { class: "q-meta" }, [q.chapter != null ? el("span", { class: "tag", text: titleCase(q.chapter) }) : null]));
    card.append(el("div", { class: "q-stem", text: q.stem }));
    card.append(optionList(q.options || [], a.chosen_key, null, (k) => { st.answers[q.id] = Object.assign({}, a, { question_id: q.id, chosen_key: k }); renderQ(); }));
    card.append(confidenceRow(a.confidence, (v) => { st.answers[q.id] = Object.assign({ question_id: q.id }, a, { confidence: v }); renderQ(); }));

    const last = st.idx === total - 1;
    card.append(el("div", { class: "q-actions" }, [
      el("button", { class: "btn", disabled: st.idx === 0, onclick: () => { st.idx--; renderQ(); } }, "← Prev"),
      el("div", { class: "hint-line" }, ["Pick ", el("span", { class: "kbd", text: "a–d" }), " · confidence ", el("span", { class: "kbd", text: "1–3" }), " · ", el("span", { class: "kbd", text: "← →" }), " move"]),
      last
        ? el("button", { class: "btn primary", onclick: doSubmit }, "Submit diagnostic")
        : el("button", { class: "btn primary", onclick: () => { st.idx++; renderQ(); } }, "Next →"),
    ]));
    body.append(el("div", {}, [head, card]));
  }

  function doSubmit() {
    const answered = Object.values(st.answers).filter((a) => a.chosen_key);
    const missing = st.questions.length - answered.length;
    if (missing > 0 && !confirm(missing + " question(s) are unanswered. Submit anyway? They'll be treated as skipped.")) return;
    const payload = answered.map((a) => ({ question_id: a.question_id, chosen_key: a.chosen_key, confidence: a.confidence || "guessing" }));
    st.submitting = true; clear(body); body.append(loadingState("Scoring & seeding your model…"));
    api.post("/api/diagnostic/submit", { answers: payload }).then((r) => { st.result = r; renderResult(r); })
      .catch((e) => { st.submitting = false; clear(body); body.append(errorState(e, () => renderQ())); });
  }

  function renderResult(r) {
    clear(body);
    const s = r.status || {};
    const wrap = el("div", { class: "stack" });
    wrap.append(el("div", { class: "card" }, [
      el("div", { class: "row", style: "justify-content:center;flex-direction:column;text-align:center" }, [
        el("div", { class: "ico", "aria-hidden": "true", style: "font-size:2rem", text: "✓" }),
        el("h2", { text: "Diagnostic complete" }),
        el("p", { class: "muted", text: r.seeded != null ? ("Seeded " + r.seeded + " concept estimates. Your model is live.") : "Your mastery model has been seeded." }),
        el("a", { class: "btn primary", href: "#/study", style: "margin-top:var(--sp-2)" }, "Start studying →"),
      ]),
    ]));
    if (s.prediction) wrap.append(predictionCard(s.prediction));
    if (s.heatmap && s.heatmap.length) wrap.append(heatmapCard(s.heatmap));
    body.append(wrap);
  }

  function onKey(e) {
    if (!st.started || st.result || st.submitting) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const q = st.questions[st.idx]; if (!q) return;
    const a = st.answers[q.id] || {};
    if (/^[1-3]$/.test(e.key) && a.chosen_key) { const c = CONF.find((x) => x.key === e.key); if (c) { st.answers[q.id] = Object.assign({ question_id: q.id }, a, { confidence: c.val }); renderQ(); e.preventDefault(); return; } }
    const opt = keyToOption(e, q.options || []);
    if (opt) { st.answers[q.id] = Object.assign({}, a, { question_id: q.id, chosen_key: opt }); renderQ(); e.preventDefault(); return; }
    if (e.key === "ArrowLeft" && st.idx > 0) { st.idx--; renderQ(); e.preventDefault(); }
    else if (e.key === "ArrowRight" && st.idx < st.questions.length - 1) { st.idx++; renderQ(); e.preventDefault(); }
  }
  window.addEventListener("keydown", onKey);
  renderIntro();
  return () => window.removeEventListener("keydown", onKey);
}

/* ============================================================
   MOCK  (timed, navigator, no feedback until submit)
   ============================================================ */
function viewMock() {
  const root = VIEW();
  root.append(pageHead("Mock exam", "Timed, exam-realistic. No feedback until you submit."));
  const body = el("div", {});
  root.append(body);

  const st = { phase: "choose", mock: null, questions: [], answers: {}, idx: 0, deadline: 0, timer: null, result: null };

  function renderChoose() {
    clear(body);
    body.append(el("div", { class: "grid cols-2" }, [
      mockChoice("Standalone", "A focused section of standalone questions.", () => start("standalone")),
      mockChoice("Full mock", "The complete exam pattern, including caselets — timed end-to-end.", () => start("full")),
    ]));
  }
  function mockChoice(title, desc, onStart) {
    return el("div", { class: "card" }, [
      el("h2", { text: title }), el("p", { class: "card-hint", style: "margin-bottom:var(--sp-4)", text: desc }),
      el("button", { class: "btn primary", onclick: onStart }, "Start " + title.toLowerCase()),
    ]);
  }

  function start(kind) {
    clear(body); body.append(loadingState("Setting up your " + kind + " mock…"));
    api.post("/api/mock/start", { kind }).then((d) => {
      st.mock = d; st.questions = d.questions || []; st.answers = {}; st.idx = 0; st.phase = "run";
      if (!st.questions.length) { clear(body); body.append(errorState({ message: "Mock returned no questions." }, renderChoose)); return; }
      if (d.duration_minutes) { st.deadline = Date.now() + d.duration_minutes * 60000; startTimer(); }
      renderRun();
    }).catch((e) => { clear(body); body.append(errorState(e, renderChoose)); });
  }

  function startTimer() {
    stopTimer();
    st.timer = setInterval(() => {
      const t = document.getElementById("mock-timer");
      const left = Math.max(0, st.deadline - Date.now());
      if (t) { t.textContent = fmtClock(left); t.className = "timer" + (left < 60000 ? " crit" : left < 300000 ? " low" : ""); }
      if (left <= 0) { stopTimer(); toast("Time's up — submitting your mock.", "err"); doSubmit(true); }
    }, 1000);
  }
  function stopTimer() { if (st.timer) { clearInterval(st.timer); st.timer = null; } }
  function fmtClock(ms) { const s = Math.round(ms / 1000); return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0"); }

  function renderRun() {
    const q = st.questions[st.idx]; const total = st.questions.length; const a = st.answers[q.id] || {};
    clear(body);

    const head = el("div", { class: "run-head" }, [
      el("div", { class: "run-progress" }, [
        el("div", { class: "row", style: "justify-content:space-between" }, [
          el("span", { class: "small muted", text: "Question " + (st.idx + 1) + " of " + total + (st.mock.total_marks ? " · " + st.mock.total_marks + " marks" : "") }),
          el("span", { class: "small mono", text: Object.keys(st.answers).filter((k) => st.answers[k].chosen_key).length + "/" + total + " answered" }),
        ]),
        el("div", { class: "bar", style: "margin-top:.35rem" }, el("i", { style: `width:${((st.idx + 1) / total) * 100}%` })),
      ]),
      st.mock.duration_minutes ? el("div", { id: "mock-timer", class: "timer", text: fmtClock(Math.max(0, st.deadline - Date.now())) }) : null,
    ]);

    const card = el("div", { class: "card quiz" });
    card.append(el("div", { class: "q-meta" }, [
      q.chapter != null ? el("span", { class: "tag", text: titleCase(q.chapter) }) : null,
      q.marks != null ? el("span", { class: "tag info", text: q.marks + (q.marks === 1 ? " mark" : " marks") }) : null,
      q.case_group_id != null ? el("span", { class: "tag violet", text: "caselet" }) : null,
    ]));
    if (q.case_scenario) card.append(el("div", { class: "q-scenario", text: q.case_scenario }));
    card.append(el("div", { class: "q-stem", text: q.stem }));
    card.append(optionList(q.options || [], a.chosen_key, null, (k) => { st.answers[q.id] = Object.assign({}, a, { question_id: q.id, chosen_key: k }); renderRun(); }));
    card.append(confidenceRow(a.confidence, (v) => { st.answers[q.id] = Object.assign({ question_id: q.id }, a, { confidence: v }); renderRun(); }));

    const last = st.idx === total - 1;
    card.append(el("div", { class: "q-actions" }, [
      el("button", { class: "btn", disabled: st.idx === 0, onclick: () => { st.idx--; renderRun(); } }, "← Prev"),
      el("span", { class: "hint-line" }, [el("span", { class: "kbd", text: "← →" }), " move · ", el("span", { class: "kbd", text: "a–d" }), " answer"]),
      last ? el("button", { class: "btn primary", onclick: () => doSubmit(false) }, "Submit mock")
        : el("button", { class: "btn primary", onclick: () => { st.idx++; renderRun(); } }, "Next →"),
    ]));

    // navigator
    const nav = el("div", { class: "card", style: "margin-top:var(--sp-4)" }, [
      el("div", { class: "row", style: "justify-content:space-between;margin-bottom:var(--sp-3)" }, [
        el("h2", { style: "margin:0", text: "Question navigator" }),
        el("button", { class: "btn primary", onclick: () => doSubmit(false) }, "Submit mock"),
      ]),
    ]);
    const grid = el("div", { class: "navgrid" });
    st.questions.forEach((qq, i) => {
      grid.append(el("button", {
        class: (st.answers[qq.id] && st.answers[qq.id].chosen_key ? "answered " : "") + (i === st.idx ? "current" : ""),
        "aria-label": "Go to question " + (i + 1) + (st.answers[qq.id] && st.answers[qq.id].chosen_key ? " (answered)" : ""),
        "aria-current": i === st.idx ? "true" : null,
        onclick: () => { st.idx = i; renderRun(); },
      }, String(i + 1)));
    });
    nav.append(grid);

    body.append(el("div", {}, [head, card, nav]));
  }

  function doSubmit(auto) {
    if (st.phase === "result") return;
    if (!auto) {
      const answered = Object.values(st.answers).filter((a) => a.chosen_key).length;
      const missing = st.questions.length - answered;
      if (missing > 0 && !confirm(missing + " question(s) unanswered. Submit the mock now?")) return;
    }
    stopTimer(); st.phase = "result"; clear(body); body.append(loadingState("Grading your mock…"));
    const payload = st.questions.map((q) => {
      const a = st.answers[q.id] || {};
      return { question_id: q.id, chosen_key: a.chosen_key || null, confidence: a.confidence || "guessing" };
    }).filter((a) => a.chosen_key);
    api.post("/api/mock/submit", { mock_id: st.mock.mock_id, answers: payload }).then((r) => { st.result = r; renderResult(r); })
      .catch((e) => { clear(body); body.append(errorState(e, () => { st.phase = "run"; renderRun(); })); });
  }

  function renderResult(r) {
    clear(body);
    const wrap = el("div", { class: "stack" });

    const marks = st.mock.total_marks;
    wrap.append(el("div", { class: "card" }, [
      el("h2", { text: "Result" }),
      el("div", { class: "grid cols-3", style: "margin-top:var(--sp-2)" }, [
        el("div", { class: "stat" }, [el("div", { class: "label", text: "Raw score" }), el("div", { class: "num mono", text: (r.raw_score == null ? "—" : num(r.raw_score, r.raw_score % 1 ? 1 : 0)) + (marks ? " / " + marks : "") })]),
        el("div", { class: "stat" }, [el("div", { class: "label", text: "Pass probability" }), el("div", { class: "num mono", text: r.prediction ? pct(r.prediction.p_pass) : "—" })]),
        el("div", { class: "stat" }, [el("div", { class: "label", text: "Expected (model)" }), el("div", { class: "num mono", text: r.prediction ? num(r.prediction.expected, 1) : "—" })]),
      ]),
    ]));

    if (r.per_section && r.per_section.length) {
      const sec = el("div", { class: "card" }, [el("h2", { text: "By section" })]);
      r.per_section.forEach((s) => {
        const ratio = s.max ? clamp(s.score / s.max, 0, 1) : 0;
        sec.append(el("div", { style: "margin-bottom:var(--sp-3)" }, [
          el("div", { class: "row", style: "justify-content:space-between" }, [
            el("span", { class: "small", text: titleCase(s.kind) }),
            el("span", { class: "small mono", text: num(s.score, s.score % 1 ? 1 : 0) + " / " + s.max }),
          ]),
          el("div", { class: "bar" + (ratio >= 0.6 ? " good" : ratio >= 0.4 ? " warn" : ""), style: "margin-top:.3rem" }, el("i", { style: `width:${ratio * 100}%` })),
        ]));
      });
      wrap.append(sec);
    }

    if (r.prediction) wrap.append(predictionCard(r.prediction));

    if (r.review && r.review.length) {
      const rank = (it) => {
        const wrong = String(it.chosen_key || "").toLowerCase() !== String(it.correct_key || "").toLowerCase();
        if (wrong && !it.chosen_key) return 1; // skipped
        if (wrong) return 0; return 2;
      };
      const sorted = r.review.slice().sort((a, b) => rank(a) - rank(b));
      const rev = el("div", { class: "card" }, [el("h2", { text: "Review" }), el("p", { class: "card-hint", style: "margin-bottom:var(--sp-4)", text: "Wrong answers first — study these before your next attempt." })]);
      sorted.forEach((it) => rev.append(reviewItem(it)));
      wrap.append(rev);
    }

    wrap.append(el("div", { class: "row", style: "justify-content:center" }, [
      el("a", { class: "btn", href: "#/dashboard" }, "Dashboard"),
      el("button", { class: "btn primary", onclick: () => { st.phase = "choose"; st.result = null; renderChoose(); } }, "New mock"),
    ]));
    body.append(wrap);
  }

  function reviewItem(it) {
    const wrong = String(it.chosen_key || "").toLowerCase() !== String(it.correct_key || "").toLowerCase();
    const item = el("div", { class: "review-item" + (wrong && it.chosen_key ? " confident-wrong" : ""), style: "margin-bottom:var(--sp-3)" });
    item.append(el("div", { class: "head" }, [
      el("span", { class: "small", text: it.stem }),
      it.marks != null ? el("span", { class: "tag info", text: it.marks + (it.marks === 1 ? " mark" : " marks") }) : null,
    ]));
    item.append(optionList(it.options || [], it.chosen_key, it.correct_key, noop));
    if (!it.chosen_key) item.append(el("p", { class: "misc", style: "margin-top:.5rem", text: "You skipped this question." }));
    if (it.explanation) item.append(el("p", { class: "explain small", style: "margin-top:.6rem", text: it.explanation }));
    return item;
  }

  function onKey(e) {
    if (st.phase !== "run") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const q = st.questions[st.idx]; if (!q) return;
    const a = st.answers[q.id] || {};
    if (/^[1-3]$/.test(e.key) && a.chosen_key) { const c = CONF.find((x) => x.key === e.key); if (c) { st.answers[q.id] = Object.assign({ question_id: q.id }, a, { confidence: c.val }); renderRun(); e.preventDefault(); return; } }
    const opt = keyToOption(e, q.options || []);
    if (opt) { st.answers[q.id] = Object.assign({}, a, { question_id: q.id, chosen_key: opt }); renderRun(); e.preventDefault(); return; }
    if (e.key === "ArrowLeft" && st.idx > 0) { st.idx--; renderRun(); e.preventDefault(); }
    else if (e.key === "ArrowRight" && st.idx < st.questions.length - 1) { st.idx++; renderRun(); e.preventDefault(); }
  }
  window.addEventListener("keydown", onKey);
  renderChoose();
  return () => { stopTimer(); window.removeEventListener("keydown", onKey); };
}

/* ============================================================
   FLASHCARDS
   ============================================================ */
function viewFlash() {
  const root = VIEW();
  root.append(pageHead("Flashcards", "Spaced repetition — flip, then grade honestly."));
  const body = el("div", { class: "flash-wrap" });
  root.append(body);

  const st = { card: null, flipped: false, busy: false };
  const GRADES = [
    { key: "1", grade: 1, label: "Forgot" },
    { key: "2", grade: 2, label: "Hard" },
    { key: "3", grade: 3, label: "Easy" },
  ];
  const front = (c) => c.front || c.prompt || c.question || c.term || c.q || "(front)";
  const back = (c) => c.back || c.answer || c.definition || c.response || c.a || "(back)";
  const cardId = (c) => (c.card_id != null ? c.card_id : c.id);

  function loadNext() {
    st.card = null; st.flipped = false;
    clear(body); body.append(loadingState("Fetching next card…"));
    api.get("/api/flash/next").then((d) => {
      const c = d && (d.card || (d.id || d.card_id ? d : null));
      if (!c) { renderEmpty(); return; }
      st.card = c; render();
    }).catch((e) => { clear(body); body.append(errorState(e, loadNext)); });
  }
  function renderEmpty() {
    clear(body);
    body.append(el("div", { class: "card" }, emptyState("✓", "No cards due",
      "Your flashcard queue is clear. New cards unlock as you study.",
      el("a", { class: "btn primary", href: "#/study" }, "Go study"))));
  }
  function grade(g) {
    if (st.busy || !st.card) return;
    st.busy = true;
    api.post("/api/flash/grade", { card_id: cardId(st.card), grade: g }).then(() => { st.busy = false; loadNext(); })
      .catch((e) => { st.busy = false; toast(e.message, "err"); });
  }
  function render() {
    const c = st.card; clear(body);
    const cardEl = el("div", {
      class: "flashcard", role: "button", tabindex: "0",
      "aria-label": st.flipped ? "Answer. Press to flip back." : "Prompt. Press to reveal answer.",
      onclick: () => { st.flipped = !st.flipped; render(); },
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); st.flipped = !st.flipped; render(); } },
    }, [
      el("span", { class: "side-label", text: st.flipped ? "Answer" : "Prompt" }),
      el("div", { class: "content", text: st.flipped ? back(c) : front(c) }),
      el("span", { class: "flip-hint", text: st.flipped ? "Grade below · click to flip back" : "Click or press Space / F to flip" }),
    ]);
    body.append(cardEl);

    if (st.flipped) {
      const g = el("div", { class: "conf", style: "justify-content:center;margin-top:var(--sp-4)" });
      GRADES.forEach((x) => g.append(el("button", { class: "conf-btn", onclick: () => grade(x.grade) }, [el("span", { class: "kbd", text: x.key }), el("span", { text: x.label })])));
      body.append(g);
    } else {
      body.append(el("p", { class: "hint-line", style: "text-align:center;margin-top:var(--sp-4)" }, [
        "Flip ", el("span", { class: "kbd", text: "space" }), " / ", el("span", { class: "kbd", text: "f" }),
      ]));
    }
  }
  function onKey(e) {
    if (!st.card || e.metaKey || e.ctrlKey || e.altKey) return;
    if (!st.flipped) { if (e.key === "f" || e.key === " ") { st.flipped = true; render(); e.preventDefault(); } return; }
    if (e.key === "f") { st.flipped = false; render(); e.preventDefault(); return; }
    const g = GRADES.find((x) => x.key === e.key); if (g) { grade(g.grade); e.preventDefault(); }
  }
  window.addEventListener("keydown", onKey);
  loadNext();
  return () => window.removeEventListener("keydown", onKey);
}

/* ============================================================
   Auth screens (login) + app shell toggle
   ============================================================ */
let healthTimer = null;

function buildLoginScreen(expired) {
  const errBox = el("div", { class: "login-error", id: "login-error", role: "alert", hidden: true });
  const userInput = el("input", {
    class: "input", id: "login-username", type: "text", name: "username",
    autocomplete: "username", required: true, placeholder: "you@example.com",
  });
  const passInput = el("input", {
    class: "input", id: "login-password", type: "password", name: "password",
    autocomplete: "current-password", required: true, placeholder: "Your password",
  });
  const submitBtn = el("button", { class: "btn primary lg", type: "submit", style: "width:100%" }, "Sign in");

  const form = el("form", {
    class: "login-card", novalidate: true,
    onsubmit: (e) => { e.preventDefault(); doLogin(userInput.value, passInput.value, errBox, submitBtn); },
  }, [
    el("div", { class: "brand", style: "justify-content:center" }, [
      el("span", { class: "brand-mark", html:
        "<svg viewBox='0 0 32 32' aria-hidden='true' focusable='false'>" +
        "<rect width='32' height='32' rx='8' fill='#151D30'></rect>" +
        "<path d='M6 9l3.5 14L13 12l3 11 3-11 3.5 11L26 9' fill='none' stroke='#A78BFA' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'></path>" +
        "</svg>" }),
      el("span", { class: "brand-name", text: "Weeker" }),
    ]),
    el("h1", { text: "Sign in" }),
    el("p", { class: "login-sub", text: "Adaptive NISM Series X-A prep." }),
    expired ? el("div", { class: "login-note", role: "status", text: "Your session expired. Please sign in again." }) : null,
    el("div", { class: "field" }, [el("label", { for: "login-username", text: "Username" }), userInput]),
    el("div", { class: "field" }, [el("label", { for: "login-password", text: "Password" }), passInput]),
    errBox,
    submitBtn,
  ]);

  return el("div", { class: "login-screen", id: "login-screen" }, [form]);
}

function showLoginError(errBox, msg) { errBox.textContent = msg; errBox.hidden = false; }

function doLogin(username, password, errBox, btn) {
  username = String(username || "").trim();
  if (!username || !password) { showLoginError(errBox, "Enter your username and password."); return; }
  errBox.hidden = true;
  btn.disabled = true; btn.textContent = "Signing in…";
  loginRequest(username, password).then((data) => {
    if (!data || !data.token) throw new ApiError("http", "Login failed — no token returned.");
    setAuth(data.token, username, data.expires_in);
    showApp();
  }).catch((e) => {
    btn.disabled = false; btn.textContent = "Sign in";
    showLoginError(errBox, (e && e.message) || "Login failed. Please try again.");
  });
}

function showLogin(expired) {
  if (healthTimer) { clearInterval(healthTimer); healthTimer = null; }
  const app = document.querySelector(".app");
  if (app) app.hidden = true;
  const userBox = document.getElementById("sidebar-user");
  if (userBox) userBox.hidden = true;
  const existing = document.getElementById("login-screen");
  if (existing) existing.remove();
  const scr = buildLoginScreen(!!expired);
  document.body.append(scr);
  const u = scr.querySelector("#login-username");
  if (u) u.focus();
}

function renderUserBox() {
  const box = document.getElementById("sidebar-user");
  if (!box) return;
  const name = document.getElementById("user-name");
  if (name) name.textContent = getUser() || "Signed in";
  box.hidden = false;
  const btn = document.getElementById("logout-btn");
  if (btn && !btn.dataset.wired) { btn.dataset.wired = "1"; btn.addEventListener("click", logout); }
}

function logout() { clearAuth(); showLogin(false); }

function showApp() {
  const existing = document.getElementById("login-screen");
  if (existing) existing.remove();
  const app = document.querySelector(".app");
  if (app) app.hidden = false;
  renderUserBox();
  if (!location.hash) location.replace("#/dashboard");
  router();
  pollHealth();
  if (!healthTimer) healthTimer = setInterval(pollHealth, 30000);
}

/* ============================================================
   Boot
   ============================================================ */
window.addEventListener("hashchange", () => { if (getToken() && !tokenExpired()) router(); });
window.addEventListener("DOMContentLoaded", () => {
  const expired = tokenExpired();
  if (!getToken() || expired) { clearAuth(); showLogin(expired); return; }
  showApp();
});
