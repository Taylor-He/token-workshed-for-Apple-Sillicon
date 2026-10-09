# Token Workshed × JetBrains AI Assistant

This integration keeps JetBrains UI native and makes the Token Workshed App
the local control plane. Install the thin plugin from the sibling
`token-workshed-jetbrains` project, then open Token Workshed App Settings →
JetBrains Connections.

## Native Provider

Create an OpenAI-compatible Provider in JetBrains AI Assistant using the
values shown by Token Workshed:

- Base URL: the App's `http://127.0.0.1:<port>/v1` URL
- Model: the currently active local model
- API key: the restricted Provider key shown by the App
- Tool Calling: enabled when JetBrains exposes the option

The Provider exposes only model listing, Chat Completions, and text
Completions. It does not expose the desktop manager, Hermes credentials, or
the App's internal tool credentials.

Completion is marked experimental. When JetBrains sends `prompt` plus
`suffix`, Token Workshed uses the model tokenizer's FIM triplet when available;
otherwise it sends a bounded generic insertion prompt and records the fallback
diagnostic in App state.

## App-owned MCP

In JetBrains' native MCP settings, add a local stdio server using the **MCP
command** copied from the App. The command launches `token-workshed-mcp`,
which forwards `tools/list` and `tools/call` to the running App. Do not put an
API key or MCP server credentials in `acp.json` or the JetBrains MCP entry.

The App validates that a tool is configured, automatically executes only
read-only tools, and pauses side-effecting calls for an App decision. Every
call is recorded with its session ID and an argument hash. IntelliJ MCP
return is disabled by default and can only be enabled per project in App
policy.

## App-owned ACP Agent

On project startup the plugin wakes `token-workshed-desktop` on macOS, reads a
key-free loopback discovery record, and requests a user-approved pairing. On
approval it atomically merges exactly this public entry while preserving all
other entries:

```json
{
  "agent_servers": {
    "Token Workshed": {
      "command": "/absolute/path/to/python",
      "args": ["-m", "vllm_mlx.acp_adapter"]
    }
  }
}
```

The ACP adapter maps each ACP session to an App-persisted IDE session, so
`session/load` can restore it after an IDE restart. It advertises no IDE
filesystem or terminal capability. Agent output may contain structured patch
proposals, but the App and IDE do not apply them automatically.

The plugin has no custom window or configuration page. If setup is needed,
its actions open JetBrains' native AI Assistant settings or show a native
notification; all policy and approval choices remain in Token Workshed.
