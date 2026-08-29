/**
 * GET /api/health — a quick read on whether a deployment is wired up correctly.
 * Useful straight after a Vercel deploy: it says whether the key is present
 * (never what it is) and how much of the archive actually shipped.
 */

import { readFile } from "node:fs/promises";
import path from "node:path";

export default async function handler(req, res) {
  const out = {
    ok: true,
    time: new Date().toISOString(),
    chatConfigured: Boolean(process.env.OPENAI_API_KEY),
    chatModel: process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini",
    archive: null,
  };

  try {
    const file = path.join(process.cwd(), "public", "data", "hub.json");
    const hub = JSON.parse(await readFile(file, "utf8"));
    out.archive = {
      generatedAt: hub.generatedAt,
      creators: (hub.creators || []).length,
      reels: (hub.reels || []).length,
      shortlisted: (hub.creators || []).reduce(
        (a, c) => a + (c?.counts?.kept || 0),
        0
      ),
    };
  } catch {
    out.ok = false;
    out.archive = { error: "public/data/hub.json is missing or unreadable" };
  }

  res.setHeader("Cache-Control", "no-store");
  return res.status(out.ok ? 200 : 500).json(out);
}
