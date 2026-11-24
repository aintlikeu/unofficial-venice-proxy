import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { VeniceWebClient, rateLimitWaitSeconds, readRateLimitState, upstreamHttpError } from "../src/client.mjs";
import { createVeniceOpenAiProxy, createVeniceOpenAiProxyServer } from "../src/proxy.mjs";

// Offline check of upstream failures: a request Venice never answered is
// retried once on a new connection, and a second failure reaches the client as
// a 502; a Venice 429 reaches it as a 429 with what Venice said about the
// reset. Runs the proxy server with a stubbed Venice client.

let passed = 0;
let failed = 0;
// Kept apart from console.log, which the checks silence around requests.
const log = console.log.bind(console);

function ok(label, condition, detail = "") {
  if (condition) {
    log(`  PASS  ${label}`);
    passed += 1;
  } else {
    console.error(`  FAIL  ${label}${detail ? ` — ${detail}` : ""}`);
    failed += 1;
  }
}

function section(title) {
  console.log(`\n── ${title} ──`);
}

// The error text seen live (2026-11-22 and 2026-11-23).
const CONNECTION_ERROR =
  "POST https://outerface.venice.ai/api/inference/chat: error sending request for uri (https://outerface.venice.ai/api/inference/chat): client error (SendRequest): client error (SendRequest): http2 error: stream error received: unspecific protocol error detected";

const fakeResponse = (status, headers = {}) => ({
  status,
  headers: new Headers(headers),
});

const okStream = () => ({ events: (async function* () { yield { kind: "content", content: "ok" }; })() });

function testClientRotation() {
  section("VeniceWebClient.rotateTransport");
  const client = new VeniceWebClient({ transport: "wreq" });
  let closed = 0;
  const old = {
    close: async () => {
      closed += 1;
    },
    getAllCookies: () => [{ name: "__session", value: "fresh", domain: ".venice.ai", path: "/" }],
  };
  client.wreqSession = old;
  client.wreqSeeded = true;
  ok("no rotation without a session to replace", new VeniceWebClient({ transport: "wreq" }).rotateTransport() === false);
  ok("no rotation on the fetch transport", Object.assign(new VeniceWebClient({ transport: "fetch" }), { wreqSession: old }).rotateTransport() === false);
  ok("generation starts at 0", client.transportGeneration === 0);
  ok("rotates for the current generation", client.rotateTransport(0) === true);
  ok("…drops the session and bumps the generation",
    client.wreqSession === null && client.wreqSeeded === false && client.transportGeneration === 1);
  ok("…keeps the cookies for the next session",
    client.cookies.export().some((cookie) => cookie.name === "__session" && cookie.value === "fresh"));
  ok("…leaves the old session open for requests still on it", closed === 0);
  client.wreqSession = { ...old };
  ok("a request from the old generation does not rotate again", client.rotateTransport(0) === false && client.transportGeneration === 1);
  ok("the current generation rotates", client.rotateTransport(1) === true && client.transportGeneration === 2);
}

function testRateLimitErrors() {
  section("upstreamHttpError / rateLimitWaitSeconds");
  const now = Date.parse("2026-11-24T12:00:00Z");
  const bare = upstreamHttpError(fakeResponse(429), "{}", now);
  ok("429 without headers: message unchanged, no wait",
    bare.message === "HTTP 429: {}" && bare.status === 429 && bare.rateLimit.retryAfterSeconds === undefined &&
      Object.keys(bare.rateLimit.headers).length === 0, JSON.stringify(bare.rateLimit));
  const full = upstreamHttpError(fakeResponse(429, {
    "x-ratelimit-type": "daily",
    "x-ratelimit-remaining": "0",
    "x-ratelimit-resets": "1795003200",
    "retry-after": "3600",
    "x-other": "no",
  }), "{}", now);
  ok("429 with headers: listed in the message in a fixed order",
    full.message === "HTTP 429: {} [x-ratelimit-type=daily x-ratelimit-remaining=0 x-ratelimit-resets=1795003200 retry-after=3600]",
    full.message);
  ok("…retry-after wins and gives the reset time",
    full.rateLimit.retryAfterSeconds === 3600 && full.rateLimit.resetAt === "2026-11-24T13:00:00.000Z", JSON.stringify(full.rateLimit));
  const other = upstreamHttpError(fakeResponse(500, { "retry-after": "5" }), "boom", now);
  ok("other statuses: no rateLimit, plain message", other.message === "HTTP 500: boom" && other.status === 500 && !other.rateLimit);
  const hostile = upstreamHttpError(fakeResponse(429, { "x-ratelimit-type": `a]b ${"x".repeat(300)}` }), "{}", now);
  ok("header values: no brackets, bounded length",
    !hostile.rateLimit.headers["x-ratelimit-type"].includes("]") && hostile.rateLimit.headers["x-ratelimit-type"].length <= 201,
    hostile.message.slice(0, 80));
  const broken = upstreamHttpError({ status: 429, headers: { get() { throw new Error("no"); } } }, "{}", now);
  ok("headers that throw are ignored", broken.message === "HTTP 429: {}" && broken.rateLimit.retryAfterSeconds === undefined);
  const noHeaders = upstreamHttpError({ status: 429 }, "{}", now);
  ok("a response without headers is handled", noHeaders.message === "HTTP 429: {}");

  const wait = (headers) => rateLimitWaitSeconds(headers, now);
  const cases = [
    ["retry-after seconds", { "retry-after": "120" }, 120],
    ["retry-after fractional", { "retry-after": "1.2" }, 2],
    ["retry-after HTTP date", { "retry-after": "Tue, 24 Nov 2026 12:30:00 GMT" }, 1800],
    ["retry-after 0", { "retry-after": "0" }, undefined],
    ["retry-after in the past", { "retry-after": "Tue, 24 Nov 2026 11:00:00 GMT" }, undefined],
    ["retry-after garbage, resets epoch s", { "retry-after": "soon", "x-ratelimit-resets": String(now / 1000 + 600) }, 600],
    ["resets epoch ms", { "x-ratelimit-resets": String(now + 90_000) }, 90],
    ["resets ISO date", { "x-ratelimit-resets": "2026-11-25T00:00:00Z" }, 43200],
    ["resets small number: not guessed", { "x-ratelimit-resets": "3600" }, undefined],
    ["resets in the past", { "x-ratelimit-resets": String(now / 1000 - 10) }, undefined],
    ["resets beyond a week: taken as misread", { "x-ratelimit-resets": String(now / 1000 + 8 * 86400) }, undefined],
    ["resets text without a date", { "x-ratelimit-resets": "tomorrow" }, undefined],
    ["nothing", {}, undefined],
  ];
  for (const [label, headers, expected] of cases) {
    const got = wait(headers);
    ok(`wait: ${label}`, got === expected, `${got} !== ${expected}`);
  }
}

const fakeJwt = (expSeconds) =>
  `e30.${Buffer.from(JSON.stringify({ exp: expSeconds })).toString("base64url")}.sig`;

async function testChatLimitMonitoring() {
  section("chat limit: recorded from every reply, warned at thresholds");
  const now = Date.parse("2026-11-24T18:27:24Z");
  const live = readRateLimitState(new Headers({
    "x-ratelimit-type": "chat", "x-ratelimit-remaining": "9997", "x-ratelimit-resets": "1795631244767",
  }), now);
  ok("live headers: type, remaining and the exact reset time",
    live?.type === "chat" && live.remaining === 9997 && live.resets_at === "2026-11-25T18:27:24.767Z" &&
      live.seen_at === "2026-11-24T18:27:24.000Z", JSON.stringify(live));
  ok("the reset time does not move with the clock",
    readRateLimitState(new Headers({ "x-ratelimit-type": "chat", "x-ratelimit-remaining": "1", "x-ratelimit-resets": "1795631244767" }), now + 1234)
      ?.resets_at === live.resets_at);
  ok("no type: ignored", readRateLimitState(new Headers({ "x-ratelimit-remaining": "5" })) === null);
  ok("remaining not a whole number: ignored",
    readRateLimitState(new Headers({ "x-ratelimit-type": "chat", "x-ratelimit-remaining": "-1" })) === null &&
      readRateLimitState(new Headers({ "x-ratelimit-type": "chat", "x-ratelimit-remaining": "lots" })) === null);
  ok("unreadable reset: null", readRateLimitState(new Headers({ "x-ratelimit-type": "chat", "x-ratelimit-remaining": "5", "x-ratelimit-resets": "3600" }))?.resets_at === null);

  const client = new VeniceWebClient({ transport: "fetch" });
  const seen = [];
  client.onRateLimit = (type, state, previous) => seen.push({ type, remaining: state.remaining, previous: previous?.remaining ?? null });
  client.recordRateLimit(fakeResponse(200, { "x-ratelimit-type": "chat", "x-ratelimit-remaining": "10", "x-ratelimit-resets": "1795631244767" }));
  client.recordRateLimit(fakeResponse(200, { "x-ratelimit-type": "scrape", "x-ratelimit-remaining": "99" }));
  client.recordRateLimit(fakeResponse(200, {}));
  client.recordRateLimit({ status: 200 });
  client.recordRateLimit(fakeResponse(429, { "x-ratelimit-type": "chat", "x-ratelimit-remaining": "0", "x-ratelimit-resets": "1795631244767" }));
  ok("kept per type, the latest wins", client.rateLimits.chat?.remaining === 0 && client.rateLimits.scrape?.remaining === 99);
  ok("callback gets each reply with the previous state",
    JSON.stringify(seen) === JSON.stringify([
      { type: "chat", remaining: 10, previous: null },
      { type: "scrape", remaining: 99, previous: null },
      { type: "chat", remaining: 0, previous: 10 },
    ]), JSON.stringify(seen));
  client.onRateLimit = () => {
    throw new Error("boom");
  };
  client.recordRateLimit(fakeResponse(200, { "x-ratelimit-type": "chat", "x-ratelimit-remaining": "7" }));
  ok("a throwing callback does not break the request", client.rateLimits.chat.remaining === 7);

  const proxy = createVeniceOpenAiProxy();
  const lines = [];
  const originalLog = console.log;
  const originalWarn = console.warn;
  console.log = (...args) => lines.push(`LOG ${args.join(" ")}`);
  console.warn = (...args) => lines.push(`WARN ${args.join(" ")}`);
  const resetsA = String(Date.now() + 3 * 3600_000);
  const resetsB = String(Date.now() + 27 * 3600_000);
  // Each reply's reset time jitters by a few ms, as live (24.765-24.798 s).
  let jitter = 0;
  const reply = (remaining, resets = resetsA) =>
    proxy.client.recordRateLimit(fakeResponse(200, { "x-ratelimit-type": "chat", "x-ratelimit-remaining": String(remaining), "x-ratelimit-resets": String(Number(resets) + (jitter++ % 3) * 17 - 17) }));
  try {
    for (const remaining of [9998, 5001, 5000, 4999, 3000, 1999, 3000, 1500]) reply(remaining);
    reply(9999, resetsB);
    reply(4000, resetsB);
    proxy.client.recordRateLimit(fakeResponse(200, { "x-ratelimit-type": "scrape", "x-ratelimit-remaining": "1" }));
  } finally {
    console.log = originalLog;
    console.warn = originalWarn;
  }
  const windowLines = lines.filter((line) => line.includes("venice chat limit: "));
  const warnLines = lines.filter((line) => line.includes("venice chat limit low"));
  ok("window start logged on the first reply and on a new reset time",
    windowLines.length === 2 && /^LOG .*remaining=9998 resets_at=\S+Z$/.test(windowLines[0]) &&
      windowLines[1].includes("remaining=9999") && windowLines[1].includes("(new window; the previous one ended with 1500 left)"),
    windowLines.join(" | "));
  ok("one warning per threshold, out-of-order replies ignored, reset after a new window",
    warnLines.length === 3 && warnLines[0].includes("remaining=5000 (at or below 5000)") &&
      warnLines[1].includes("remaining=1999 (at or below 2000)") && /in 3\.0 h|in 180 min|in 2\.9 h/.test(warnLines[1]) &&
      warnLines[2].includes("remaining=4000 (at or below 5000)") && warnLines.every((line) => line.startsWith("WARN ")),
    warnLines.join(" | "));
  ok("other limit types are not warned about", !lines.some((line) => line.includes("remaining=1 ")));
  const lowStart = createVeniceOpenAiProxy();
  const lowLines = [];
  console.warn = (...args) => lowLines.push(args.join(" "));
  console.log = () => {};
  try {
    lowStart.client.recordRateLimit(fakeResponse(429, { "x-ratelimit-type": "chat", "x-ratelimit-remaining": "0", "x-ratelimit-resets": resetsA }));
  } finally {
    console.log = originalLog;
    console.warn = originalWarn;
  }
  ok("first seen at 0 (a 429): one warning at the lowest threshold",
    lowLines.length === 1 && lowLines[0].includes("remaining=0 (at or below 10)"), lowLines.join(" | "));
}

async function testSessionFile() {
  section("session file: atomic write, best-effort save");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "venice-session-"));
  try {
    const file = path.join(dir, "session.json");
    const client = new VeniceWebClient({ transport: "fetch", sessionFile: file });
    client.sessionId = "sess_1";
    client.clerkJwt = fakeJwt(2_000_000_000);
    client.writeSavedSession({ lastValidatedAt: 1 });
    const first = JSON.parse(fs.readFileSync(file, "utf8"));
    ok("written and readable", first.sessionId === "sess_1" && first.lastValidatedAt === 1);
    ok("owner-only permissions", (fs.statSync(file).mode & 0o777) === 0o600, (fs.statSync(file).mode & 0o777).toString(8));
    ok("no temporary file left", fs.readdirSync(dir).join() === "session.json", fs.readdirSync(dir).join());

    // A write that fails (here: the temporary path is a directory) must leave
    // the old file whole; writing in place left it empty when the disk filled.
    fs.mkdirSync(`${file}.${process.pid}.tmp`);
    client.sessionId = "sess_2";
    let threw = null;
    try {
      client.writeSavedSession({ lastValidatedAt: 2 });
    } catch (error) {
      threw = error;
    }
    ok("a failed write throws", Boolean(threw));
    ok("…and the previous session file is intact", JSON.parse(fs.readFileSync(file, "utf8")).sessionId === "sess_1");
    fs.rmdirSync(`${file}.${process.pid}.tmp`);

    const proxy = createVeniceOpenAiProxy();
    proxy.client.sessionFile = file;
    proxy.client.sessionId = "sess_1";
    proxy.client.refreshClerkJwt = async () => fakeJwt(Math.floor(Date.now() / 1000) + 600);
    proxy.client.getUserSession = async () => ({ ok: true });
    proxy.client.writeSavedSession = () => {
      throw Object.assign(new Error("ENOSPC: no space left on device, write"), { code: "ENOSPC" });
    };
    const warnings = [];
    const originalWarn = console.warn;
    console.warn = (...args) => warnings.push(args.join(" "));
    let auth = null;
    try {
      auth = await proxy.ensureAuth();
    } finally {
      console.warn = originalWarn;
    }
    ok("refresh succeeds although the file cannot be written", auth?.source === "refresh" && Boolean(auth.login.clerkJwt));
    ok("…with a warning naming the file and the error",
      warnings.some((line) => line.includes("could not save the session file") && line.includes(file) && line.includes("ENOSPC")), warnings.join(" | "));

    const broken = createVeniceOpenAiProxy();
    fs.writeFileSync(file, "");
    broken.client.sessionFile = file;
    broken.client.sessionId = "sess_1";
    broken.client.refreshClerkJwt = async () => {
      throw new Error('HTTP 401: {"errors":[{"code":"session_expired"}]}');
    };
    warnings.length = 0;
    console.warn = (...args) => warnings.push(args.join(" "));
    let failure = null;
    try {
      await broken.ensureAuth();
    } catch (error) {
      failure = error;
    } finally {
      console.warn = originalWarn;
    }
    ok("refresh failure with an empty file: missing-session error",
      /No valid Venice web session found/.test(failure?.message ?? ""), failure?.message);
    ok("…and the refresh error is logged, not swallowed",
      warnings.some((line) => line.includes("refreshing the session in memory failed") && line.includes("session_expired")), warnings.join(" | "));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

async function testSharedSessionCreation() {
  section("VeniceWebClient.ensureWreqSession");
  const client = new VeniceWebClient({ transport: "wreq" });
  try {
    const [a, b, c] = await Promise.all([client.ensureWreqSession(), client.ensureWreqSession(), client.ensureWreqSession()]);
    ok("concurrent callers share one new session", a === b && b === c && client.wreqSession === a);
    client.rotateTransport();
    const [d, e] = await Promise.all([client.ensureWreqSession(), client.ensureWreqSession()]);
    ok("after a rotation they share one other session", d === e && d !== a);
    await a.close();
  } finally {
    await client.close();
  }
}

async function testProxyRetry() {
  const originalLog = console.log;
  const originalWarn = console.warn;
  const proxy = createVeniceOpenAiProxy();
  proxy.ensureAuth = async () => ({ login: { clerkJwt: "jwt" } });
  proxy.invalidateAuth = () => {};
  const rotations = [];
  let generation = 0;
  proxy.client.rotateTransport = (seen) => {
    rotations.push(seen);
    if (seen !== generation) {
      return false;
    }
    generation += 1;
    return true;
  };
  Object.defineProperty(proxy.client, "transportGeneration", { get: () => generation, configurable: true });

  // Each upstream call takes the next scripted outcome: "conn" throws the
  // connection error, "401" an auth error, "500" another error, else success.
  let script = [];
  let calls = 0;
  const next = () => {
    calls += 1;
    const step = script.shift() ?? "ok";
    if (step === "conn") throw new Error(CONNECTION_ERROR);
    if (step === "401") throw new Error('HTTP 401: {"error":"Expired Clerk session"}');
    if (step === "500") throw new Error('HTTP 500: {"error":"boom"}');
    if (typeof step === "object") throw upstreamHttpError(fakeResponse(step.status, step.headers), step.body ?? "{}");
  };
  proxy.client.openChatStream = async () => {
    next();
    return okStream();
  };
  proxy.client.chat = async () => {
    next();
    return { events: [{ kind: "content", content: "ok" }], text: "ok" };
  };
  proxy.client.openAgentChatStream = async () => {
    next();
    return { events: (async function* () { yield { event: "response.completed", data: { response: { status: "completed" } } }; })() };
  };
  proxy.client.openWorkflowChatStream = proxy.client.openAgentChatStream;

  const server = createVeniceOpenAiProxyServer(proxy);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}/v1`;
  let warnings = [];
  const post = async (body, steps) => {
    script = [...steps];
    calls = 0;
    rotations.length = 0;
    warnings = [];
    console.log = () => {};
    console.warn = (...args) => warnings.push(args.join(" "));
    try {
      const response = await fetch(`${base}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "zai-org-glm-5-2", messages: [{ role: "user", content: "hi" }], ...body }),
      });
      const text = await response.text();
      return { status: response.status, headers: response.headers, text, json: (() => { try { return JSON.parse(text); } catch { return null; } })() };
    } finally {
      console.log = originalLog;
      console.warn = originalWarn;
    }
  };

  try {
    section("one connection failure: retried on a new connection");
    const streamed = await post({ stream: true }, ["conn"]);
    ok("streamed request succeeds", streamed.status === 200 && streamed.text.includes('"content":"ok"'), `${streamed.status} ${streamed.text.slice(0, 200)}`);
    ok("upstream called twice", calls === 2, String(calls));
    ok("the connection was replaced once, for the generation the request used", rotations.join() === "0", rotations.join());
    const retryLine = warnings.find((line) => line.includes("upstream connection failed"));
    ok("the retry is logged with timing, in_flight and the error",
      /upstream connection failed after \d+ ms in_flight=1; retrying chat completion stream once on a new connection model=zai-org-glm-5-2: POST https:\/\/outerface/.test(retryLine ?? ""), retryLine);

    const plain = await post({ stream: false }, ["conn"]);
    ok("non-streamed request is retried too", plain.status === 200 && calls === 2, `${plain.status} ${calls}`);

    const withTools = await post({
      stream: false,
      tools: [{ type: "function", function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } } } } }],
    }, ["conn"]);
    ok("tool-emulated request is retried too", withTools.status === 200 && calls === 2, `${withTools.status} ${calls}`);

    const agent = await post({ model: "deepseek-v4-1-flash-free" }, ["conn"]);
    ok("agent model request is retried too", agent.status === 200 && calls === 2, `${agent.status} ${calls} ${agent.text.slice(0, 200)}`);

    section("two connection failures: 502");
    const twice = await post({ stream: true }, ["conn", "conn"]);
    ok("status 502", twice.status === 502, String(twice.status));
    ok("OpenAI-style error with code upstream_connection_error",
      twice.json?.error?.code === "upstream_connection_error" && twice.json.error.type === "server_error" &&
        twice.json.error.message.includes("error sending request for uri"), twice.text);
    ok("only one retry", calls === 2, String(calls));
    ok("request failed line has in_flight", warnings.some((line) => /request failed: POST \/v1\/chat\/completions in_flight=1 /.test(line)),
      warnings.join(" | "));

    section("other errors are not retried");
    const other = await post({ stream: true }, ["500"]);
    ok("upstream HTTP 500 stays a 500, one call", other.status === 500 && calls === 1 && !other.json?.error?.code, `${other.status} ${calls}`);
    ok("no connection retry logged", !warnings.some((line) => line.includes("upstream connection failed")));

    section("Venice 429: passed on as a 429");
    const limited = await post({ stream: true }, [{ status: 429, headers: { "retry-after": "1800", "x-ratelimit-type": "daily" } }]);
    ok("status 429 with Retry-After", limited.status === 429 && limited.headers.get("retry-after") === "1800",
      `${limited.status} ${limited.headers.get("retry-after")}`);
    ok("OpenAI-style rate limit error with the reset and the upstream reply",
      limited.json?.error?.type === "rate_limit_error" && limited.json.error.code === "rate_limit_exceeded" &&
        limited.json.error.message.includes("rate limiting this account; it resets in about 30 min (") &&
        limited.json.error.message.includes("HTTP 429: {} [x-ratelimit-type=daily retry-after=1800]"), limited.text);
    ok("no quota words that make opencode stop retrying",
      !/insufficient[-_\s]?(?:quota|balance)|quota[-_\s]?exceeded|budget exceeded|usage limit|limit exhausted/i.test(limited.json?.error?.message ?? ""));
    ok("one upstream call, no connection retry", calls === 1 && !warnings.some((line) => line.includes("upstream connection failed")));
    ok("the headers are in the request failed line",
      warnings.some((line) => line.includes("request failed:") && line.includes("[x-ratelimit-type=daily retry-after=1800]")), warnings.join(" | "));
    const unknown = await post({ stream: false }, [{ status: 429 }]);
    ok("no reset known: 429 without Retry-After, says so",
      unknown.status === 429 && unknown.headers.get("retry-after") === null &&
        unknown.json?.error?.message.startsWith("Venice is rate limiting this account; Venice did not say when it resets. Upstream reply: HTTP 429: {}"),
      `${unknown.status} ${unknown.text}`);
    const longWait = await post({ stream: true }, [{ status: 429, headers: { "retry-after": "36000" } }]);
    ok("a long wait reads in hours", longWait.status === 429 && longWait.json?.error?.message.includes("about 10.0 h"), longWait.text);
    const agentLimited = await post({ model: "deepseek-v4-1-flash-free" }, [{ status: 429, headers: { "retry-after": "60" } }]);
    ok("agent model: 429 too", agentLimited.status === 429 && agentLimited.headers.get("retry-after") === "60", `${agentLimited.status} ${agentLimited.text.slice(0, 200)}`);

    section("Venice 429 on the proxy's own tool retry, mid-stream");
    {
      let attempt = 0;
      const savedOpen = proxy.client.openChatStream;
      proxy.client.openChatStream = async () => {
        attempt += 1;
        if (attempt === 1) {
          return { events: (async function* () {
            yield { kind: "content", reasoning_content: "I will call get_weather." };
            yield { kind: "content", content: "\n" };
          })() };
        }
        throw failure();
      };
      let failure = () => upstreamHttpError(fakeResponse(429, { "retry-after": "120" }), "{}");
      try {
        const midStream = await post({
          stream: true,
          tools: [{ type: "function", function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } } } } }],
        }, []);
        const chunks = midStream.text.split("\n\n").map((block) => block.replace(/^data: /, "").trim()).filter(Boolean)
          .map((data) => (data === "[DONE]" ? data : JSON.parse(data)));
        ok("reasoning streamed, then a rate_limit_error chunk and [DONE]",
          midStream.status === 200 && attempt === 2 &&
            chunks.some((chunk) => chunk.choices?.[0]?.delta?.reasoning_content) &&
            chunks.at(-2)?.error?.type === "rate_limit_error" && chunks.at(-2)?.error?.message.includes("about 2 min") &&
            chunks.at(-1) === "[DONE]" && !chunks.some((chunk) => chunk.choices?.[0]?.finish_reason),
          JSON.stringify(chunks.slice(-3)));
        ok("the failure is logged", warnings.some((line) => line.includes("tool emulation stream failed") && line.includes("HTTP 429")),
          warnings.join(" | "));

        attempt = 0;
        failure = () => new Error(CONNECTION_ERROR);
        const otherFailure = await post({
          stream: true,
          tools: [{ type: "function", function: { name: "get_weather", parameters: { type: "object", properties: {} } } }],
        }, []);
        const otherChunks = otherFailure.text.split("\n\n").map((block) => block.replace(/^data: /, "").trim()).filter(Boolean)
          .map((data) => (data === "[DONE]" ? data : JSON.parse(data)));
        ok("another error there: a server_error chunk with its message and [DONE]",
          otherChunks.at(-2)?.error?.type === "server_error" && otherChunks.at(-2)?.error?.message.includes("error sending request for uri") &&
            otherChunks.at(-1) === "[DONE]", JSON.stringify(otherChunks.slice(-2)));
      } finally {
        proxy.client.openChatStream = savedOpen;
      }
    }

    section("chat limit in the outgoing line and /support/info");
    {
      const savedOpen = proxy.client.openChatStream;
      proxy.client.openChatStream = async () => {
        proxy.client.recordRateLimit(fakeResponse(200, { "x-ratelimit-type": "chat", "x-ratelimit-remaining": "4321", "x-ratelimit-resets": "1795631244767" }));
        return okStream();
      };
      const logged = [];
      try {
        script = [];
        console.log = (...args) => logged.push(args.join(" "));
        console.warn = () => {};
        const response = await fetch(`${base}/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model: "zai-org-glm-5-2", stream: true, messages: [{ role: "user", content: "hi" }] }),
        });
        await response.text();
        const info = await (await fetch(`${base.replace(/\/v1$/, "")}/support/info`)).json();
        console.log = originalLog;
        console.warn = originalWarn;
        ok("outgoing line has venice_chat_remaining",
          logged.some((line) => line.includes("outgoing POST /v1/chat/completions status=200") && line.endsWith(" venice_chat_remaining=4321")),
          logged.filter((line) => line.includes("outgoing")).join(" | "));
        ok("GET lines do not", !logged.some((line) => line.includes("outgoing GET") && line.includes("venice_chat_remaining")));
        ok("/support/info shows rate_limits.chat",
          info.rate_limits?.chat?.remaining === 4321 && info.rate_limits.chat.resets_at === "2026-11-25T18:27:24.767Z", JSON.stringify(info.rate_limits));
      } finally {
        console.log = originalLog;
        console.warn = originalWarn;
        proxy.client.openChatStream = savedOpen;
        delete proxy.client.rateLimits.chat;
      }
    }

    section("with an auth refresh");
    const authThenConn = await post({ stream: true }, ["conn", "401", "ok"]);
    ok("connection failure, then 401 on the retry, then success", authThenConn.status === 200 && calls === 3, `${authThenConn.status} ${calls}`);
    const connAfterAuth = await post({ stream: true }, ["401", "conn", "ok"]);
    ok("401, then connection failure, then success", connAfterAuth.status === 200 && calls === 3, `${connAfterAuth.status} ${calls}`);

    section("requests that failed together");
    script = [];
    calls = 0;
    rotations.length = 0;
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    let failing = 3;
    proxy.client.openChatStream = async () => {
      calls += 1;
      if (failing > 0) {
        failing -= 1;
        await gate;
        throw new Error(CONNECTION_ERROR);
      }
      return okStream();
    };
    console.log = () => {};
    console.warn = (...args) => warnings.push(args.join(" "));
    warnings = [];
    try {
      const requests = [1, 2, 3].map(() =>
        fetch(`${base}/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model: "zai-org-glm-5-2", stream: true, messages: [{ role: "user", content: "hi" }] }),
        }).then(async (response) => ({ status: response.status, text: await response.text() })),
      );
      while (calls < 3) {
        await new Promise((resolve) => setTimeout(resolve, 5));
      }
      ok("three requests in flight", proxy.inFlight === 3, String(proxy.inFlight));
      release();
      const results = await Promise.all(requests);
      ok("all three succeed on the retry", results.every((result) => result.status === 200), results.map((result) => result.status).join());
      ok("the connection was replaced once", rotations.length === 3 && generation === rotations[0] + 1, `${rotations.join()} gen=${generation}`);
      const lines = warnings.filter((line) => line.includes("upstream connection failed"));
      ok("one retry says new connection, the others renewed",
        lines.filter((line) => line.includes("on a new connection")).length === 1 &&
          lines.filter((line) => line.includes("on a renewed connection")).length === 2, lines.join(" | "));
      ok("in_flight shows all three", lines.every((line) => line.includes("in_flight=3")), lines.join(" | "));
    } finally {
      console.log = originalLog;
      console.warn = originalWarn;
    }
    await new Promise((resolve) => setTimeout(resolve, 20));
    ok("in_flight back to 0 once the responses closed", proxy.inFlight === 0, String(proxy.inFlight));
  } finally {
    server.close();
  }
}

async function main() {
  testClientRotation();
  testRateLimitErrors();
  await testChatLimitMonitoring();
  await testSessionFile();
  await testSharedSessionCreation();
  await testProxyRetry();
  console.log(`\n${"─".repeat(40)}`);
  console.log(`Results: ${passed} passed, ${failed} failed`);
  if (failed > 0) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
