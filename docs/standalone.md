# Standalone build

One source tree (`grafana-plugins/invoicex-labeling-app/src`), two artifacts.

| Artifact | Command | Output | Host |
|---|---|---|---|
| Grafana app plugin | `npm run build` | `dist/` (tracked, shipped) | Grafana; `@grafana/{data,ui,runtime}`, react, react-dom, lodash are externals |
| Standalone SPA | `npm run build:standalone` or `make frontend-standalone` | `dist-standalone/` (gitignored) | none; everything bundled |

Run npm commands in `grafana-plugins/invoicex-labeling-app/`.

## Build-time env (standalone)

| Var | Default | Use |
|---|---|---|
| `INVOICEX_BASE_URL` | `''` (same origin) | prefix for `/pdf/{sha}`, `/whoami` |
| `INVOICEX_BASENAME` | `/` | router basename, script/chunk public path, pdf worker path |

## Host seam

`src/host/` = only code that differs between builds.

| File | Role |
|---|---|
| `types.ts` | `Host`: `currentUser()`, `pdfBytes(sha256)`, `pdfWorkerSrc` |
| `grafana.ts` | Grafana impl; only file allowed to import `@grafana/runtime` (plus `module.ts`) |
| `standalone.ts` | plain `fetch` impl |
| `index.ts` | `export { host } from './grafana'`; standalone webpack config swaps to `./standalone` |

Guard rail: `ast-grep scan` (in the app dir) errors on any `@grafana/runtime` specifier outside `src/host/grafana.ts` and `src/module.ts`.

## Contract: SPA -> server

| Route | Used by | Notes |
|---|---|---|
| `GET {INVOICEX_BASE_URL}/pdf/{sha256}` | `host.pdfBytes` | body = PDF bytes; non-2xx -> UI shows load failure |
| `GET {INVOICEX_BASE_URL}/whoami` | `host.currentUser` | JSON `{"user": "<string>"}`; any failure -> `submitted_by = 'unknown'` |
| `/api/*` | `src/utils/api.ts` | SPA hard-codes `/api`; PostgREST must see it with `/api` stripped (server proxy via `INVOICEX_API_UPSTREAM`, or your own proxy) |
| `{basename}pdf.worker.min.js`, `{basename}standalone.js`, chunks | browser | static files from `dist-standalone/` |

Client-side routes: `/queue`, `/label/:id`, `/schema`, `/model`, `/results`. With `INVOICEX_STATIC_DIR` set the Go server returns `index.html` for unknown non-file paths (except `/pdf/*`).

## Go standalone server (config; set on the server, not in the SPA)

Source: `pkg/cmd/standalone/main.go`, `pkg/internal/{identity,pdfsource}`.

| Item | Values |
|---|---|
| Routes | `GET /pdf/{sha}` (64 lowercase hex), `GET /whoami`, `GET /healthz`, `/api/*` (only if upstream set), `/` SPA (only if static dir set) |
| `INVOICEX_LISTEN` | default `:8081` |
| `INVOICEX_STATIC_DIR` | optional; serve `dist-standalone/` with `index.html` fallback; unset = no static route |
| `INVOICEX_API_UPSTREAM` | optional, e.g. `http://postgrest:3000`; set -> proxies `/api/*` to upstream with `/api` prefix stripped (PostgREST sees its native root); unset -> no `/api` route, front the API with your own proxy. Must be `http(s)://host[:port][/path]` or startup fails; upstream down -> 502 |
| `INVOICEX_PDF_SOURCE` | `local` (default) \| `http` \| `sharepoint` |
| `INVOICEX_PDF_LOCAL_DIR` | local source; default `/data/pdfs`; file `{sha}.pdf` |
| `INVOICEX_PDF_FETCH_URL_TEMPLATE` | http source (required); placeholders `{sha}`, `{source_id}`, `{drive_id}` |
| `INVOICEX_PDF_FETCH_AUTH_HEADER` | http source; optional value sent as `Authorization` |
| `POSTGREST_URL` | best-effort `rpc/get_document_source` lookup for sharepoint/http; default `http://postgrest:3000`; unused by local |
| `INVOICEX_IDENTITY` | `none` (default) \| `header` |
| `INVOICEX_IDENTITY_HEADER` | header name when identity=`header`; default `X-Forwarded-User` |

## Identity

Optional, pluggable. The SPA only calls `/whoami`; it sends no credentials scheme of its own and imposes none. `none` -> `unknown`. `header` -> value of the configured request header, else `unknown`. Auth, if any, is whatever sits in front (reverse proxy); the server sets no identity/Authorization headers on proxied `/api` calls.

## Isolation check

`@grafana/runtime` must not appear in `dist-standalone/`:

```
ast-grep scan --inline-rules 'id: iso
language: js
rule: {kind: string_fragment, regex: "@grafana/runtime"}' dist-standalone
```
