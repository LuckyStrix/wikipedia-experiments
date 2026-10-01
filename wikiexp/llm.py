"""Pluggable language-model backends for the RAG assistant: a local ollama server or Claude.

A backend turns ``(system prompt, chat messages)`` into a stream of text:

    backend = make_backend("ollama", ollama_url="http://localhost:11434", model="llama3.2:3b")
    for piece in backend.stream(system, [{"role": "user", "content": "..."}]):
        print(piece, end="")
    backend.info          # GenInfo: stop reason, token counts, seconds, notes (after the stream ends)

Anything with ``name``, ``model``, ``info`` and ``stream(system, messages, max_tokens)`` works, which
is how the tests and the screen's tests inject a fake. Problems a person can fix (ollama not running,
model not pulled, no API key, rate limited) raise ``LLMError`` with the message to show them.

- ``OllamaBackend``: ollama's HTTP API through the standard library only (``/api/chat``, streamed
  newline-delimited JSON). Works against another machine's ollama via ``url``.
- ``ClaudeBackend``: the official ``anthropic`` SDK, streaming through ``client.beta.messages.stream``
  with adaptive thinking, ``output_config.effort`` and server-side refusal fallbacks. The key comes
  from the environment (``ANTHROPIC_API_KEY``) and is never stored.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Iterator, Protocol, Sequence

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
BACKENDS = ("ollama", "claude")


class LLMError(Exception):
    """Something the user can act on; ``str(e)`` is written for them."""


@dataclass
class GenInfo:
    """What happened in the last generation (filled in as the stream finishes)."""
    stop_reason: str = ""          # "stop"/"end_turn", "length"/"max_tokens", "refusal", ...
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    first_token_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        return self.stop_reason in ("length", "max_tokens")

    @property
    def tokens_per_second(self) -> float:
        gen = self.seconds - self.first_token_seconds
        return self.output_tokens / gen if gen > 0 and self.output_tokens else 0.0


class Backend(Protocol):
    name: str
    model: str
    info: GenInfo

    def stream(self, system: str, messages: Sequence[dict], max_tokens: int = 1024) -> Iterator[str]:
        """Yield the answer as text pieces; raise LLMError for problems the user can fix."""


def complete(backend: Backend, system: str, messages: Sequence[dict], max_tokens: int = 256) -> str:
    """The whole answer as one string (for short helper prompts such as query rewriting)."""
    return "".join(backend.stream(system, messages, max_tokens)).strip()


# ── ollama ───────────────────────────────────────────────────────────────────

class OllamaBackend:
    name = "ollama"

    def __init__(self, url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_OLLAMA_MODEL,
                 num_ctx: int = 8192, temperature: float = 0.2, timeout: float = 600.0):
        self.url = (url or DEFAULT_OLLAMA_URL).strip()
        self.model = (model or DEFAULT_OLLAMA_MODEL).strip()
        self.num_ctx, self.temperature, self.timeout = num_ctx, temperature, timeout
        self.info = GenInfo()
        parsed = urllib.parse.urlparse(self.url if "://" in self.url else "http://" + self.url)
        self.url = self.url.rstrip("/")
        if not parsed.hostname:
            raise LLMError(f"'{url}' is not a valid ollama URL (try http://localhost:11434)")
        self._https = parsed.scheme == "https"
        self._host = parsed.hostname
        self._port = parsed.port or (443 if self._https else 80)
        self._base = parsed.path.rstrip("/")

    def _connect(self, timeout: float | None = None) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=timeout or self.timeout)

    def _unreachable(self, e: Exception) -> LLMError:
        return LLMError(f"Can't reach ollama at {self.url} ({e.__class__.__name__}). Start it with `ollama serve`, "
                        f"or set the Ollama server URL (settings, tab Ask) if it runs on another machine.")

    def _not_pulled(self) -> LLMError:
        return LLMError(f"ollama doesn't have the model '{self.model}'. Download it with: ollama pull {self.model}")

    def models(self) -> list[str]:
        """Names of the models the server has pulled."""
        try:
            conn = self._connect(10)
            conn.request("GET", self._base + "/api/tags")
            resp = conn.getresponse()
            data = json.loads(resp.read() or b"{}")
            conn.close()
        except (OSError, http.client.HTTPException) as e:
            raise self._unreachable(e) from e
        except ValueError as e:
            raise LLMError(f"{self.url} answered, but it doesn't look like ollama") from e
        return [m.get("name", "") for m in data.get("models", [])]

    def stream(self, system: str, messages: Sequence[dict], max_tokens: int = 1024) -> Iterator[str]:
        body = {"model": self.model, "stream": True,
                "messages": ([{"role": "system", "content": system}] if system else []) + list(messages),
                "options": {"num_ctx": self.num_ctx, "temperature": self.temperature, "num_predict": max_tokens}}
        self.info = info = GenInfo()
        t0 = time.time()
        try:
            conn = self._connect()
            conn.request("POST", self._base + "/api/chat", json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:      # refused, DNS, timeout, reset
            raise self._unreachable(e) from e
        try:
            if resp.status != 200:
                raw = resp.read().decode("utf-8", "replace")
                try:
                    err = json.loads(raw).get("error", raw)
                except ValueError:
                    err = raw
                if resp.status == 404 and ("not found" in err.lower() or "model" in err.lower()):
                    raise self._not_pulled()
                raise LLMError(f"ollama returned HTTP {resp.status}: {err.strip()[:300]}")
            for line in iter(resp.readline, b""):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if "error" in chunk:
                    err = str(chunk["error"])
                    raise self._not_pulled() if "not found" in err.lower() else LLMError(f"ollama error: {err}")
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    if not info.first_token_seconds:
                        info.first_token_seconds = time.time() - t0
                    yield piece
                if chunk.get("done"):
                    info.stop_reason = chunk.get("done_reason", "stop")
                    info.prompt_tokens = int(chunk.get("prompt_eval_count", 0))
                    info.output_tokens = int(chunk.get("eval_count", 0))
                    break
            else:
                if not info.stop_reason:
                    raise LLMError("ollama closed the connection before finishing the answer")
        except (socket.timeout, TimeoutError) as e:
            raise LLMError(f"ollama didn't answer within {self.timeout:.0f} s; a bigger model may need longer "
                           f"(or the machine is busy)") from e
        except (OSError, http.client.HTTPException) as e:
            raise LLMError(f"Lost the connection to ollama at {self.url} ({e.__class__.__name__})") from e
        finally:
            info.seconds = time.time() - t0
            conn.close()


# ── Claude ───────────────────────────────────────────────────────────────────

class ClaudeBackend:
    name = "claude"

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, effort: str = DEFAULT_EFFORT, client=None,
                 max_tokens: int = 16000):
        self.model = (model or DEFAULT_CLAUDE_MODEL).strip()
        self.effort = effort or DEFAULT_EFFORT
        self.max_tokens_cap = max_tokens
        self._client = client
        self.info = GenInfo()

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise LLMError("The claude backend needs the anthropic package: pip install anthropic") from e
            self._client = anthropic.Anthropic()     # credentials from the environment, never from config
        return self._client

    def stream(self, system: str, messages: Sequence[dict], max_tokens: int = 1024) -> Iterator[str]:
        import anthropic
        self.info = info = GenInfo()
        t0 = time.time()
        # Adaptive thinking spends output tokens before the answer, so leave room for it
        limit = max(max_tokens, 4096) if self.max_tokens_cap >= 4096 else max_tokens
        limit = min(limit, self.max_tokens_cap)
        try:
            with self.client.beta.messages.stream(
                    model=self.model, max_tokens=limit, system=system, messages=list(messages),
                    thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                    betas=[FALLBACK_BETA], fallbacks="default") as stream:
                for text in stream.text_stream:
                    if not info.first_token_seconds:
                        info.first_token_seconds = time.time() - t0
                    yield text
                final = stream.get_final_message()
            self._finish(final, info)
        except anthropic.AuthenticationError as e:
            raise LLMError("Claude rejected the API key. Set a valid ANTHROPIC_API_KEY in the environment "
                           "before starting the app (it is never stored in config.toml).") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"This API key isn't allowed to use {self.model}: {getattr(e, 'message', e)}") from e
        except anthropic.RateLimitError as e:
            wait = ""
            try:
                wait = f" Try again in {int(float(e.response.headers.get('retry-after', '')))} s."
            except (AttributeError, ValueError, TypeError):
                pass
            raise LLMError("Claude is rate limiting this key." + wait) from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {getattr(e, 'status_code', '?')}: {getattr(e, 'message', e)}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Can't reach the Claude API. Check your internet connection.") from e
        except TypeError as e:                  # the SDK says so when no credentials resolve at all
            if "authentication" in str(e).lower():
                raise LLMError("No Claude credentials found. Set ANTHROPIC_API_KEY in the environment before "
                               "starting the app (it is never stored in config.toml).") from e
            raise
        finally:
            info.seconds = time.time() - t0

    def _finish(self, final, info: GenInfo) -> None:
        info.stop_reason = getattr(final, "stop_reason", "") or ""
        usage = getattr(final, "usage", None)
        info.prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        info.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        iterations = getattr(usage, "iterations", None) or []
        if any(getattr(i, "type", "") == "fallback_message" for i in iterations):
            info.notes.append(f"a fallback model answered (served by {getattr(final, 'model', 'another model')})")
        if info.stop_reason == "refusal":
            details = getattr(final, "stop_details", None)
            why = getattr(details, "explanation", None) or getattr(details, "category", None)
            raise LLMError("Claude declined to answer this question" + (f" ({why})." if why else "."))
        if info.truncated:
            info.notes.append("the answer was cut off at the token limit")


def make_backend(name: str = "ollama", model: str = "", url: str = "", effort: str = "", **kw) -> Backend:
    """Backend by name with its model (blank = the default for that backend)."""
    if name == "ollama":
        return OllamaBackend(url or DEFAULT_OLLAMA_URL, model or DEFAULT_OLLAMA_MODEL, **kw)
    if name == "claude":
        return ClaudeBackend(model or DEFAULT_CLAUDE_MODEL, effort or DEFAULT_EFFORT, **kw)
    raise LLMError(f"unknown backend '{name}' (choose from {', '.join(BACKENDS)})")


def have_claude_credentials() -> bool:
    """Whether the environment looks set up for the claude backend (the SDK decides for real)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
