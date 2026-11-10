# MindMate LangGraph and MCP setup

The integration lives under `chatbot_feature/`. It extends the existing local
Llama/Sarvam application; it does not replace the model runtime, UI, or direct
`/api/chat` and `/api/chat/stream` endpoints. Set `MCP_ENABLED=true` to enable
`/api/agent/chat`. With it disabled, existing behavior is unchanged.

## Install and configure

Python 3.10+, Node.js 24+, `npx`, and Docker are recommended. Context7
uses Streamable HTTP; the other configured servers use stdio. Install the Python
client and LangGraph dependencies:

```sh
python3 -m pip install -r backend/requirements.txt
cp .env.example .env
```

Export values from `.env` in the process environment before starting the app.
The application does not parse `.env` automatically and never sends all process
environment variables to a server—only each server's allowlisted variables.

```sh
set -a
. ./.env
set +a
python3 -m backend.server --port 8080
```

Filesystem and Git roots must be absolute approved paths. Put the memory JSONL
and SQLite checkpoint database on durable private storage. Do not place secrets
in `servers.json`.

## Server matrix

| Server | Maintained launch/endpoint | Authentication | Transport | Agents |
| --- | --- | --- | --- | --- |
| Brave Search | `npx -y @brave/brave-search-mcp-server --transport stdio` | `BRAVE_API_KEY` | stdio | research |
| Filesystem | `npx -y @modelcontextprotocol/server-filesystem <root>` | approved root | stdio | coding, document |
| Memory | `npx -y @modelcontextprotocol/server-memory` | private `MEMORY_FILE_PATH` | stdio | memory |
| Playwright | `npx -y @playwright/mcp@latest --headless --isolated` | none | stdio | browser |
| GitHub | official `ghcr.io/github/github-mcp-server` image in read-only mode | PAT | stdio | coding |
| Context7 | `https://mcp.context7.com/mcp` | bearer API key | Streamable HTTP | coding |
| Firecrawl | `npx -y firecrawl-mcp` | `FIRECRAWL_API_KEY` | stdio | research |
| Sequential Thinking | `npx -y @modelcontextprotocol/server-sequential-thinking` | none | stdio | planning |
| Fetch | `mcp-server-fetch` (official PyPI package) | none | stdio | research, browser |
| Git | `mcp-server-git --repository <root>` (official PyPI package) | approved root | stdio | coding |
| Time | `mcp-server-time --local-timezone=<IANA zone>` (official PyPI package) | none | stdio | planning |

The source definitions and permissions are in
`chatbot_feature/mcp_connector/servers.json`. Disable a server by setting its
`enabled` field to `false`. Missing credentials produce
`authentication_required` health without breaking unrelated chat or MCP servers.

Official sources: [Brave](https://github.com/brave/brave-search-mcp-server),
[MCP reference servers](https://github.com/modelcontextprotocol/servers),
[Playwright](https://github.com/microsoft/playwright-mcp),
[GitHub](https://github.com/github/github-mcp-server),
[Context7](https://github.com/upstash/context7), and
[Firecrawl](https://github.com/firecrawl/firecrawl-mcp-server).

## Security model

- Tools are discovered at connection time; no tool names or call schemas are
  hard-coded into execution.
- Each agent receives only tools from its assigned servers. The manager repeats
  authorization immediately before execution.
- GitHub and Git are read-only. Browser mutations require an explicit
  `user_approved=true` execution argument. Filesystem access is sandboxed by the
  official server root.
- Literal loopback/private IP URLs are blocked before tool execution. Fetch also
  has its own private-address protections. Deployments should add an outbound
  proxy/domain allowlist for DNS-rebinding-resistant network policy.
- External output is labelled untrusted in both specialist and response prompts,
  truncated before returning to model context, and cannot change permissions.
- Memory writes are opt-in through explicit memory intent and every Memory MCP
  call must include the current `user:<id>:` namespace. Conversation checkpoints
  are separate from the Memory MCP knowledge graph.
- Tool concurrency, timeouts, graph recursion, delegation count, and tool calls
  per agent are bounded. Logs record agent/server/tool names, never arguments or
  credentials.

## Verification

Unit tests use fake MCP adapters and do not require secrets:

```sh
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s chatbot_feature/tests -v
cd frontend && npm run build
```

For real discovery, export credentials and roots, set `MCP_ENABLED=true`, start
the backend, then inspect `GET /api/status`, or run:

```sh
python3 -m chatbot_feature.mcp_connector.health
```

`connected` means discovery
succeeded; `authentication_required` means a required variable is absent;
`configuration_required` means an approved root/path is absent;
`unavailable` includes missing executables, incompatible runtime versions,
network failures, and server startup errors. A real tool is called only when the
supervisor routes a matching request and the model emits arguments conforming to
the dynamically discovered schema.

Consequential Playwright actions and all future mutation-capable GitHub tools
must be paired with a real user-confirmation UI before their permissions are
relaxed. The supplied GitHub configuration intentionally stays read-only.
