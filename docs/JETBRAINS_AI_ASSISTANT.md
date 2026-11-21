# Token Workshed NextCore × JetBrains AI Assistant

The JetBrains connection provides model inference and completion through the
local NextCore service. The built-in SGLang Agent and its computer, browser,
terminal, file, and web tools are available only in the main NextCore chat UI.

## Native Provider

Create an OpenAI-compatible Provider in JetBrains AI Assistant using the
values shown by NextCore:

- Base URL: the App's `http://127.0.0.1:<port>/v1` URL
- Model: the currently active local model
- API key: the restricted Provider key shown by the App

The Provider exposes model listing, Chat Completions, and text Completions.
It forwards inference requests to SGLang but does not execute NextCore Agent
tools. Tool definitions supplied by an IDE client do not grant access to
NextCore's local computer, browser, terminal, or file tools. FIM completion
uses the cached model tokenizer when its full FIM token set is available;
otherwise NextCore reports the bounded generic insertion fallback as
experimental.

## MCP and ACP

The NextCore MCP bridge intentionally advertises an empty tool list and
rejects `tools/call`. ACP/IDE Agent session routes return HTTP 403. Remove old
Token Workshed ACP or MCP Agent entries from JetBrains configuration; they do
not provide access to the NextCore chat Agent. Use the main NextCore chat UI
for tasks that need web, desktop, terminal, or file tools.

## IDE Companion

The desktop can open the Rust/libcosmic IDE Companion for model chat and
completion. It remains an inference-only surface; Agent tools are not
forwarded to it or to the external IDE client.
