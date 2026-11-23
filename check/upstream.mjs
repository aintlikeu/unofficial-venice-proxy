import { pathToFileURL } from "node:url";
import { VeniceWebClient } from "../src/client.mjs";
import { createVeniceOpenAiProxy, createVeniceOpenAiProxyServer } from "../src/proxy.mjs";

// Offline check of upstream connection failures: a request Venice never
// answered is retried once on a new connection, and a second failure reaches
// the client as a 502. Runs the proxy server with a stubbed Venice client.

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
      return { status: response.status, text, json: (() => { try { return JSON.parse(text); } catch { return null; } })() };
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
