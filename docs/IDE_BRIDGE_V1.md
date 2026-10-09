# Token Workshed IDE Bridge v1

`/api/ide/v1` is the intentionally small, desktop-local bridge for native
Token Workshed companion surfaces. The JetBrains `0.2.0` plugin uses this
bridge when the App is available and falls back to vllm-mlx's OpenAI-compatible
`/v1` API when it is not. This bridge is not a manager API and does not expose
manager credentials, terminal, model download, or arbitrary file operations.
Agent and App-session features are additive in `/api/ide/v2` and require their
own explicit scopes.

## Transport and compatibility

- The bridge is served by the desktop UI endpoint, normally
  `http://127.0.0.1:7862`.
- Pairing and authorization-management routes are loopback-only. A remote UI
  host does not make IDE pairing remotely available.
- The v1 wire protocol is `protocol_version: "1"`. v2 clients use
  `protocol_version: "2"` and add a `requested_scopes` list. v1 clients remain
  compatible and do not receive Agent scopes.
- The desktop app owns the local runtime lifecycle. The plugin can request a
  local model switch after pairing, but cannot start, stop, download, or
  configure the runtime.

## Pairing and credentials

1. The plugin launches `tokenworkshed://ide/start` if the desktop app is not
   already running. The link contains no commands, model names, or credentials.
2. It creates `POST /pairing/requests` with its client ID, IDE name, plugin
   version, and protocol version.
3. The user approves or rejects the request in Desktop Settings. For a v2
   request, **Approve safe** omits `read_all`, `delete`, and `runs.approve`;
   **Approve all requested** is an explicit opt-in for those sensitive scopes.
   Approval returns a generated Bearer credential exactly once through
   `GET /pairing/status/{pairing_id}`.
4. Only a SHA-256 digest of the credential is retained in desktop state. The
   JetBrains plugin stores the raw credential in PasswordSafe.
5. Desktop Settings can revoke a client at any time; the next authenticated
   plugin request then returns HTTP `401`.

The credential is limited to these scopes:

- `ide.status.read`
- `ide.models.read`
- `ide.models.switch`
- `ide.assist.stream`
- `ide.companion.open`

There are no manager, terminal, or arbitrary-shell scopes. Agent/session
scopes are documented below and are never implied by a v1 pairing.

## Routes

| Route | Auth | Purpose |
| --- | --- | --- |
| `GET /status` | loopback | Version and restricted-capability handshake. |
| `POST /pairing/requests` | loopback | Create a pending approval request. |
| `GET /pairing/status/{id}` | loopback | Poll pending, rejected, or one-time approved credential. |
| `GET /pairing/pending` | loopback | Desktop Settings pending requests. |
| `POST /pairing/requests/{id}/decision` | loopback | Desktop Settings approval or rejection. |
| `GET /authorizations` | loopback | Desktop Settings authorized clients. |
| `POST /authorizations/{client_id}/revoke` | loopback | Revoke a client credential. |
| `GET /models` | Bearer | List local models and active model. |
| `POST /models/active` | Bearer | Request a local desktop model switch. |
| `POST /companion/open` | Bearer | Open the separate Rust/libcosmic IDE Companion window. |
| `POST /assist/stream` | Bearer | Stream constrained code-assistance proposals as NDJSON. |

All request models reject unknown fields. In particular, a client cannot add an
`agent` field or tunnel a manager request through this API.

## Code-assistance request and stream

`POST /assist/stream` accepts a message, one of `ask`, `fix`, `refactor`, or
`test`, optional previous chat turns, and up to eight explicitly supplied
project-relative context items. Each context item has a content hash. A
selection sends only the selected text plus offsets; without a selection the
plugin sends the current file.

Potentially sensitive paths (`.env`, private keys, credential/secret names,
and similar) require `sensitive_confirmed: true`; the plugin asks the user for
this confirmation per attachment. The bridge refuses absolute paths, path
traversal, and `.git` metadata.

The response uses `application/x-ndjson` and emits:

- `start` — chosen model and `agent: false`;
- `text_delta` — display-only prose;
- `patch` — a structured replacement bound to supplied context and hash;
- `command` — an argv-only command suggestion;
- `warning`, `error`, or `done`.

The desktop only generates patches. It never writes IDE files. The plugin
opens a native Diff, rechecks the file/selection hash, project path, and
read-only status, and applies only after the user confirms through the IDE
undo stack. New files are restricted to test paths and test mode.

Commands are never run automatically. Shell executables and shell metacharacters
are rejected; suggestions that would need a shell are copy-only.

## Rust/libcosmic IDE Companion

The standalone Rust/libcosmic Companion can still be opened by the desktop
application through `POST /companion/open`, using the desktop's existing local
API and model lifecycle. It is visually the compact left Chat surface: the
navigation rail, Chat title/status strip, model picker, message feed, and
composer remain; the desktop dashboard and other management pages are omitted.
The JetBrains plugin can invoke this route when the user has approved the
`ide.companion.open` scope.

The companion is not a new privilege boundary. It is launched only through
the paired `ide.companion.open` scope and inherits the desktop's local API
boundary. IDE project context handoff remains explicit and is never encoded
in the `tokenworkshed://` launch URL.

## Bridge v2: App sessions and Agent runs

`/api/ide/v2` is the optional App-ecosystem surface. It is loopback-hosted and
uses the same one-time pairing credential, but every endpoint checks an
independent scope:

- `ide.sessions.read` and `ide.sessions.read_all`
- `ide.sessions.write` and `ide.sessions.delete`
- `ide.sessions.chat`
- `ide.runs.read`, `ide.runs.stop`, and `ide.runs.approve`
- `ide.skills.read` and `ide.toolsets.read`

The plugin never receives the Hermes API key or manager token. The App invokes
its existing local OpenClaw/Hermes runtime and returns a normalized NDJSON
stream. Events include `run_started`, `message_started`, `assistant_delta`,
`tool_progress`, `approval_request`, `assistant_completed`, `run_completed`,
`error`, and terminal `done`. The plugin treats EOF before `done` as failure.

Sessions created from an IDE are tagged with the pairing client ID in App
state. A normal list only returns that client's sessions; importing all App
sessions requires `ide.sessions.read_all`. Session deletion requires
`ide.sessions.delete` and an explicit confirmation in the IDE.

Agent runs pause for an approval choice (`once`, `session`, `always`, or
`deny`). Stop and connection teardown signal cancellation to the App runner.
Tool output is structured and bounded; no raw terminal or arbitrary shell is
returned to the plugin.
