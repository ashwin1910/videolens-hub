/**
 * POST /api/chat — the "Ask the archive" assistant.
 *
 * The page builds a grounded context out of the reels currently in scope and
 * sends it here. This function is the only place the OpenAI key is ever used;
 * it never reaches the browser.
 *
 * Request:  { system?: string, messages: [{ role, content }] }
 * Response: { reply: string }
 */

const MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini";

// The archive is machine-generated description, so the assistant is told to
// stay inside it and to hedge where the notes hedge. Pinned server-side rather
// than taken from the client, so the deployed key can only ever do this job.
const SYSTEM = [
  "You answer questions about a small archive of analysed Instagram fitness and",
  "nutrition reels. Use ONLY the notes provided in the user message.",
  "Be concise — 2-4 sentences, or a short list.",
  "Name the creator and the reel when it is relevant.",
  "If the notes do not cover something, say so plainly rather than guessing.",
  "These notes are machine-generated readings of video content, so hedge",
  "wherever the notes themselves hedge, and never present an inference as",
  "something that was definitely shown or said.",
].join(" ");

// Generous enough for the whole archive, small enough to bound cost per call.
const MAX_CHARS = 120000;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Use POST." });
  }

  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    return res.status(503).json({
      error:
        "OPENAI_API_KEY is not set on the server. Add it under Vercel → " +
        "Project → Settings → Environment Variables, then redeploy.",
    });
  }

  let body = req.body;
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch {
      return res.status(400).json({ error: "Body must be JSON." });
    }
  }

  const incoming = Array.isArray(body?.messages) ? body.messages : null;
  if (!incoming || incoming.length === 0) {
    return res.status(400).json({ error: "Expected a non-empty messages array." });
  }

  // Normalise, drop anything malformed, and bound the total size.
  const messages = [];
  let budget = MAX_CHARS;
  let truncated = false;
  const totalIncoming = incoming.reduce(
    (n, m) => n + (typeof m?.content === "string" ? m.content.length : 0),
    0,
  );
  for (const m of incoming) {
    const role = m?.role === "assistant" ? "assistant" : "user";
    const content = typeof m?.content === "string" ? m.content : String(m?.content ?? "");
    if (!content.trim()) continue;
    const clipped = content.slice(0, budget);
    if (clipped.length < content.length) truncated = true;
    budget -= clipped.length;
    messages.push({ role, content: clipped });
    if (budget <= 0) break;
  }
  if (messages.length === 0) {
    return res.status(400).json({ error: "No usable message content." });
  }

  const systemNote = truncated
    ? "Note: the archive notes were truncated to fit the context limit. Say so if your answer may be incomplete."
    : "";

  try {
    const upstream = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify({
        model: MODEL,
        temperature: 0.2,
        max_tokens: 600,
        messages: [{ role: "system", content: [SYSTEM, systemNote].filter(Boolean).join(" ") }, ...messages],
      }),
      // The function itself is capped at 30s (vercel.json), so give up first and
      // return a clear message rather than being killed mid-flight.
      signal: AbortSignal.timeout(25000),
    });

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => "");
      console.error("openai error", upstream.status, detail.slice(0, 500));
      // Don't leak upstream error bodies (they can echo the key's org details).
      const msg =
        upstream.status === 401
          ? "OpenAI rejected the API key."
          : upstream.status === 429
          ? "Rate limited or out of quota on the OpenAI account."
          : `OpenAI returned ${upstream.status}.`;
      return res.status(502).json({ error: msg });
    }

    const data = await upstream.json();
    const reply = data?.choices?.[0]?.message?.content?.trim();
    if (!reply) {
      return res.status(502).json({ error: "OpenAI returned an empty reply." });
    }

    res.setHeader("Cache-Control", "no-store");
    return res.status(200).json({ reply, model: MODEL });
  } catch (err) {
    console.error("chat handler failed", err);
    // A timeout or a failed connection is worth naming: it tells you the
    // deployment is fine and the network or OpenAI is not.
    if (err?.name === "TimeoutError" || err?.name === "AbortError") {
      return res.status(504).json({ error: "OpenAI did not respond in time." });
    }
    if (err?.name === "TypeError") {
      return res.status(502).json({ error: "Could not reach OpenAI from the server." });
    }
    return res.status(500).json({ error: "The assistant failed unexpectedly." });
  }
}
