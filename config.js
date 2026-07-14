/* ============================================================
   Weeker — runtime API base configuration.
   Loaded BEFORE app.js so app.js can read window.WEEKER_API_BASE.

   Local dev (`weeker serve` hosts the UI + API together):
     leave "" → same-origin, no CORS, nothing to change.

   Hosted static (GitHub Pages, Netlify, S3, …) talking to a
   remote API (e.g. Cloud Run): set this to the API origin, e.g.
     window.WEEKER_API_BASE = "https://weeker-etewfkqwma-el.a.run.app";
   No trailing slash. The remote API must allow this page's origin
   via CORS. This is the ONLY external origin the app ever calls.
   ============================================================ */
window.WEEKER_API_BASE = "https://weeker-etewfkqwma-el.a.run.app";
