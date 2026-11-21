# RoboServer and remote mode

NextCore supports a manually started local server and an independent RoboServer
service. RoboServer's first release targets Apple Silicon Macs running macOS 14
or later.

## Prepare the server Mac

- Enable SSH for the server user and confirm that the account can install
  user-level applications in `~/Applications`.
- Provide Python 3.12, or install `uv` in the user's `PATH`, `~/.local/bin`, or
  `~/.cargo/bin`. RoboServer uses `uv` to install a user-level Python 3.12 when
  needed; it does not require root access.
- Keep an internet connection available for the first Python and SGLang runtime
  setup and for model downloads.

Copy `dist/token-workshed-roboserver.app` to the server Mac, or let NextCore
install that bundle over SSH when connecting. Opening the Dashboard only checks
the current service state. It does not install the runtime or start the service.
Choose **Start server** in the Dashboard to install the pinned Python 3.12 and
SGLang Runtime `v0.5.22`, then load a model when you need one.

The Dashboard keeps the RoboServer model library visible from its private
Hugging Face cache, including while the service is stopped. Its Performance
panel samples host CPU and unified-memory utilization while the service is
running. It shows SGLang model-memory allocation only when the backend reports
it; SGLang's Apple Silicon backend does not currently provide GPU compute
utilization, so the Dashboard does not estimate that value.

## Connect from NextCore

In **Server Settings → Remote RoboServer**, enter the server's SSH host,
username, port, and optional identity file. Choose **Inspect Host Key** and
compare the SHA-256 fingerprint with a trusted value from the server operator.
Choose **Trust Key & Connect** only after that comparison. NextCore installs or
starts the RoboServer user service over SSH and creates a private tunnel.

The RoboServer manager API listens on `127.0.0.1:17872` and SGLang listens on
`127.0.0.1:18010` on the server. Neither port is exposed to the network.
NextCore reaches them through SSH; its local inference tunnel uses port `8010`.
Model search, downloads, model selection, inference, and remote Agent tool
execution then use the server's model cache and data directory.

If the SSH tunnel drops temporarily, NextCore reopens the tunnel without
restarting a service that was intentionally stopped in the RoboServer
Dashboard. Use **Trust Key & Connect** to explicitly start the service again.

## Move Agent data

The **Move Agent Data** section previews imports in both directions. Select any
combination of persistent memories, Markdown Skills, and searchable chat
history. The destination reports new records, duplicates, conflicts, invalid
records, and credential-like records that were excluded. Import leaves existing
conflicts untouched and runs only after **Import Reviewed Data** is selected.

MCP definitions, credentials, scheduled jobs, run logs, and browser login state
are never included. Credential-like memories and Skills are omitted; matching
credential strings in searchable chat text are redacted. Imported chat history
is searchable by the destination Agent and does not merge the visible chat list.

## Local Mac control

In remote mode, computer, browser, terminal, and file tools run on RoboServer.
This release has no local-device relay from RoboServer to the NextCore Mac. A
request that explicitly targets the local Mac is blocked before tool execution
and receives a visible explanation. No local action is performed or represented
as completed.

## Current verification boundary

The bundles and their local service/API paths can be checked on the development
Mac. A live remote connection, model download, and remote inference still need a
separate Apple Silicon server with SSH enabled; those checks cannot be claimed
until such a server is configured.
