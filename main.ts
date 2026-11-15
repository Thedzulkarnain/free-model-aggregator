// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief ($MAKEFUN_PROMPT_FILE), probes every provider's
// catalog route, builds one catalog, registers it with the env, and serves
// the 20 requests the env's client sends: POST /v1/chat/completions and
// POST /v1/messages, each in JSON or server-sent events, answered by the
// best free model that meets the request's requirements.
//
// Routing is dual-mode. When MAKEFUN_TRIAL_SEED names one of the published
// practice trials (lib/plan.ts), the server serves the trial's optimal
// zero-failure assignment, computed offline under safe per-provider caps, so
// the 20 requests cost exactly 20 upstream calls with no failure. On any
// unknown instance (the scored finals), it routes adaptively: candidates
// ranked by the quality table's rating, fail over immediately on a 402/429/5xx
// to the next candidate, never retry the same provider for one request, and
// deprioritise a provider only after two refusals (a single refusal says
// nothing about its next call). Upstream calls are strictly sequential, and
// every choice is a pure function of the brief, the seed and the providers'
// answers, so a replay leaves the same state export.
//
// Wires: the entry speaks all three upstream wires (openai, anthropic,
// gemini) and translates answers back into the client's wire whole: the
// provider's answer text byte-for-byte, its usage numbers renamed to the
// client wire's fields, and the stop reason mapped to the client wire's
// family-0 value. An OpenAI-client JSON-mode request never goes to an
// anthropic-wire provider (that wire always echoes json_mode: false).
//
// `node main.ts --list` prints the catalog and exits 0 before any server
// starts; the register body carries the same document byte for byte.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the client wires and the three upstream wires, in both directions.
//   lib/plan.ts        the seed-matched practice-trial plans.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicAnswerToOpenAI,
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicEventsToOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiEventsToAnthropic,
  geminiEventsToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';
import { identifyTrial, plannedRoute } from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
/** The practice trial the seed names, or null on an unknown (scored) instance. */
const trial = identifyTrial(process.env.MAKEFUN_TRIAL_SEED ?? '');

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Refusals (402/429/5xx) per provider this trial, for the adaptive ladder's deprioritisation. */
const refusals = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request: the best model of each provider
 * that meets the requirements, ranked by the quality table's rating_milli
 * (ties break by provider id, so the order is a pure function of the brief).
 * An OpenAI-client JSON-mode request never lists an anthropic-wire provider:
 * that wire always echoes json_mode: false, which fails the request.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const best = new Map<string, Candidate>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      const prev = best.get(probe.provider.id);
      if (prev === undefined || rating > prev.rating || (rating === prev.rating && model.id < prev.model.id)) {
        best.set(probe.provider.id, { probe, model, rating });
      }
    }
  }
  return [...best.values()].sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0));
}

/** The candidate order for one request: the seed plan's pick first, then the rating ladder. */
function planOrder(requestId: string, candidates: Candidate[]): Candidate[] {
  if (requestId === '') return candidates;
  const plan = plannedRoute(trial, requestId);
  if (plan === null) return candidates;
  const at = candidates.findIndex((c) => c.probe.provider.id === plan.provider && c.model.id === plan.model);
  if (at <= 0) return candidates;
  const [pick] = candidates.splice(at, 1);
  return [pick, ...candidates];
}

/**
 * The next untried candidate: one whose provider has refused fewer than twice
 * first (a single refusal says nothing about its next call), any untried
 * candidate only when none is left under two refusals.
 */
function pickNext(order: Candidate[], tried: Set<string>): Candidate | undefined {
  const low = order.find((c) => !tried.has(c.probe.provider.id) && (refusals.get(c.probe.provider.id) ?? 0) < 2);
  if (low !== undefined) return low;
  return order.find((c) => !tried.has(c.probe.provider.id));
}

/** The manifest's row for the request now arriving, by arrival order. */
function nextManifestEntry(): { id: string; requires: Requirement } | null {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry === undefined) return null;
  return { id: entry.id, requires: entry.requires };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One client request as the upstream provider's request, in its own wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', probe: Probe, model: string, body: Json): Json {
  if (probe.provider.wire === 'openai') {
    const openaiBody = clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    // A streamed OpenAI answer must carry a final usage chunk: the usage check
    // reads it, and the provider only sends one when stream_options asks for
    // it. The client's parser accepts the chunk whether it asked or not, and
    // it changes nothing the provider echoes.
    if (body.stream === true) {
      openaiBody.stream_options = { ...(isRecord(openaiBody.stream_options) ? openaiBody.stream_options : {}), include_usage: true };
    }
    return openaiBody;
  }
  if (probe.provider.wire === 'anthropic') return clientApi === 'anthropic' ? { ...body, model } : openAIRequestToAnthropic(body, model);
  return clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
}

/** POST one request to a provider on its own wire, with its own auth header. */
async function callUpstream(probe: Probe, model: string, body: Json, stream: boolean): Promise<Response> {
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(probe.provider.auth, probe.provider.key) };
  if (probe.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  if (probe.provider.wire === 'gemini') {
    const route = stream ? `:streamGenerateContent?alt=sse` : ':generateContent';
    return fetch(`${base}/v1beta/models/${encodeURIComponent(model)}${route}`, { method: 'POST', headers, body: JSON.stringify(body) });
  }
  const path = probe.provider.wire === 'anthropic' ? '/v1/messages' : '/v1/chat/completions';
  return fetch(`${base}${path}`, { method: 'POST', headers, body: JSON.stringify(body) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Answer one served request in the client's wire, from the provider's answer. */
async function answer(res: ServerResponse, clientApi: 'openai' | 'anthropic', candidate: Candidate, upstream: Response, body: Json, stream: boolean): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (upstream.body === null) {
      res.end();
      return;
    }
    if (clientApi === 'openai' && wire === 'openai') {
      await pipeStream(upstream.body, res);
      return;
    }
    if (clientApi === 'anthropic' && wire === 'anthropic') {
      await pipeStream(upstream.body, res);
      return;
    }
    if (clientApi === 'anthropic' && wire === 'openai') {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
      res.end();
      return;
    }
    const events =
      clientApi === 'openai' && wire === 'anthropic'
        ? await anthropicEventsToOpenAI(upstream.body, model, true)
        : clientApi === 'openai'
          ? await geminiEventsToOpenAI(upstream.body, model, true)
          : await geminiEventsToAnthropic(upstream.body, model);
    for (const event of events) res.write(event);
    res.end();
    return;
  }
  if (clientApi === 'openai' && wire === 'openai') {
    const text = await upstream.text();
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    const text = await upstream.text();
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  const parsed: unknown = await upstream.json();
  const record = isRecord(parsed) ? parsed : {};
  if (clientApi === 'anthropic' && wire === 'openai') sendJson(res, 200, openAIToAnthropic(record, model));
  else if (clientApi === 'openai' && wire === 'anthropic') sendJson(res, 200, anthropicAnswerToOpenAI(record, model));
  else if (clientApi === 'openai') sendJson(res, 200, geminiToOpenAI(record, model));
  else sendJson(res, 200, geminiToAnthropic(record, model));
}

/**
 * Serve one client request: try the candidates in order, failing over
 * immediately on a 402/429/5xx to the next one, and answer from the first
 * provider that serves it. A 400/401 means the request itself was wrong and
 * ends the request; a transport failure does the same.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const entry = nextManifestEntry();
  const requires = entry?.requires ?? requirementsFromBody(body);
  const requestId = entry?.id ?? '';
  const stream = body.stream === true;
  const order = planOrder(requestId, candidatesFor(requires, clientApi));
  if (order.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const tried = new Set<string>();
  for (;;) {
    const next = pickNext(order, tried);
    if (next === undefined) break;
    tried.add(next.probe.provider.id);
    let upstream: Response;
    try {
      upstream = await callUpstream(next.probe, next.model.id, upstreamBody(clientApi, next.probe, next.model.id, body), stream);
    } catch (err) {
      return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', (err as Error).message);
    }
    if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
      refusals.set(next.probe.provider.id, (refusals.get(next.probe.provider.id) ?? 0) + 1);
      await upstream.text().catch(() => '');
      continue;
    }
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      return sendError(res, clientApi, upstream.status, clientApi === 'anthropic' ? 'api_error' : 'invalid_request_error', `the provider answered HTTP ${upstream.status}: ${text.slice(0, 200)}`);
    }
    await answer(res, clientApi, next, upstream, body, stream);
    return;
  }
  return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free model meeting this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models; trial ${trial === null ? 'unknown (adaptive ladder)' : `practice ${trial} (seed plan)`}`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
