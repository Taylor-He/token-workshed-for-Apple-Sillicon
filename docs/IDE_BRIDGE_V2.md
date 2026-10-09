# IDE Bridge v2

IDE Bridge v2 makes the Token Workshed desktop App the control plane for
JetBrains App sessions and local Agent runs. It is additive to
[`IDE_BRIDGE_V1.md`](IDE_BRIDGE_V1.md). The current JetBrains connector uses
the App-owned v2/ACP path; it does not create a plugin window or silently fall
back to an unrelated server.

## Security boundary

The JetBrains client receives one scoped, one-time pairing credential. The App
never returns its manager token, Hermes API key, subprocess environment, or a
terminal endpoint. There is no `terminal`, `shell`, or arbitrary-command scope.
Tool arguments and progress are normalized by the App before they cross the
bridge.

The v2 pairing request can ask for these independent scopes:

| Scope | Purpose |
| --- | --- |
| `ide.sessions.read` | Read sessions created by this IDE client |
| `ide.sessions.read_all` | Import sessions created by other App clients |
| `ide.sessions.write` | Create, rename, and fork sessions |
| `ide.sessions.delete` | Delete an App session |
| `ide.sessions.chat` | Start an Agent run in a session |
| `ide.runs.read` | Read run metadata where exposed |
| `ide.runs.stop` | Stop a run owned by this client |
| `ide.runs.approve` | Answer an Agent approval request |
| `ide.skills.read` | Read App skill capability names |
| `ide.toolsets.read` | Read App toolset capability names |

`read_all`, `delete`, and `runs.approve` are deliberately explicit in the
pairing dialog. **Approve safe** never grants them; **Approve all requested**
is the explicit opt-in. A session created by the IDE stores its
`bridge_client_id` and is only returned to that client unless `read_all` was
approved.

## Endpoints

All endpoints are under `/api/ide/v2` and require the paired Bearer
credential, except status and the loopback pairing endpoints.

```text
GET    /status
POST   /pairing/requests
GET    /pairing/status/{pairing_id}
GET    /sessions
POST   /sessions
GET    /sessions/{session_id}
PATCH  /sessions/{session_id}
DELETE /sessions/{session_id}
GET    /sessions/{session_id}/messages
POST   /sessions/{session_id}/fork
POST   /sessions/{session_id}/chat/stream
POST   /runs/{run_id}/stop
POST   /runs/{run_id}/approval
GET    /skills
GET    /toolsets
```

Session ids created by the bridge are prefixed with `ide_`. Message content,
context files, and stored history have bounded sizes. Context paths are
relative project paths and sensitive files require an explicit confirmation
from the IDE before they are sent.

## Agent stream

`chat/stream` returns newline-delimited JSON. Every event has `session_id`,
`run_id`, `message_id`, a monotonically increasing `seq`, and a timestamp.
The event types are:

```text
run_started
message_started
assistant_delta
tool_started
tool_progress
tool_completed
tool_failed
approval_request
approval_policy
patch
artifact_warning
assistant_completed
run_completed
error
done
```

The ACP adapter treats `done: {"ok": true}` as the only successful
completion. Unexpected EOF, invalid JSON, an idle watchdog, or a stop request
is a failed/interrupted run. A patch event is always marked proposal-only and
is never written to the workspace. For the App-owned ACP client, approval is
represented as an App policy update; side-effecting MCP calls wait for the App
MCP decision gate rather than an IDE approval prompt.

## Compatibility

The App keeps all v1 routes and existing v1 credentials. A v1-only App or an
unpaired App is reported as unavailable by the current connector and does not
fabricate Agent functionality. Native Provider setup remains an explicit user
action in JetBrains AI Assistant settings.
