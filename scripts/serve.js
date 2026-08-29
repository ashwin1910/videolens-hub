/**
 * A tiny local preview server, so you can see the site without installing the
 * Vercel CLI:
 *
 *     npm run serve        → http://localhost:3000
 *
 * It serves public/ as the site root and routes /api/* to the same handlers
 * Vercel will run, so the archive assistant behaves locally exactly as it does
 * in production (it reads OPENAI_API_KEY from .env).
 */

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PUBLIC = path.join(ROOT, "public");
const PORT = Number(process.env.PORT || 3000);

// Load .env so the API handlers find the key, without adding a dependency.
try {
  const env = await readFile(path.join(ROOT, ".env"), "utf8");
  for (const line of env.split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#") || !t.includes("=")) continue;
    const i = t.indexOf("=");
    const k = t.slice(0, i).trim();
    const v = t.slice(i + 1).trim().replace(/^["']|["']$/g, "");
    if (k && !(k in process.env)) process.env[k] = v;
  }
} catch {
  /* no .env — the static site still works, the assistant will report it */
}

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".mp4": "video/mp4",
  ".woff2": "font/woff2",
};

/** Minimal stand-in for the response object Vercel hands its functions. */
function shim(res) {
  res.status = (code) => {
    res.statusCode = code;
    return res;
  };
  res.json = (obj) => {
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify(obj));
    return res;
  };
  return res;
}

function readBody(req) {
  return new Promise((resolve) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch {
        resolve(raw);
      }
    });
  });
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  let pathname = decodeURIComponent(url.pathname);

  // --- API ----------------------------------------------------------------
  if (pathname.startsWith("/api/")) {
    const name = pathname.slice("/api/".length).replace(/\/$/, "");
    try {
      const mod = await import(`../api/${name}.js`);
      req.body = req.method === "POST" ? await readBody(req) : undefined;
      await mod.default(req, shim(res));
    } catch (err) {
      console.error(`api/${name} failed:`, err.message);
      shim(res).status(404).json({ error: `No such endpoint: /api/${name}` });
    }
    return;
  }

  // --- static -------------------------------------------------------------
  if (pathname === "/") pathname = "/index.html";
  // cleanUrls: /about → /about.html, matching vercel.json
  let file = path.join(PUBLIC, pathname);
  if (!file.startsWith(PUBLIC)) {
    shim(res).status(403).json({ error: "Forbidden" });
    return;
  }

  try {
    let s = await stat(file).catch(() => null);
    if (!s && !path.extname(file)) {
      const alt = `${file}.html`;
      if (await stat(alt).catch(() => null)) file = alt;
      else s = null;
    }
    if (s?.isDirectory()) file = path.join(file, "index.html");

    const buf = await readFile(file);
    res.setHeader("Content-Type", TYPES[path.extname(file)] || "application/octet-stream");
    res.setHeader("Cache-Control", "no-store");
    res.end(buf);
  } catch {
    res.statusCode = 404;
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.end("<h1>404</h1><p>Not found. Is public/ built? Try <code>npm run export</code>.</p>");
  }
});

server.listen(PORT, () => {
  console.log(`\n  Lens is running at http://localhost:${PORT}`);
  console.log(
    `  Archive assistant: ${
      process.env.OPENAI_API_KEY ? "configured" : "no OPENAI_API_KEY — chat will report it"
    }\n`
  );
});
