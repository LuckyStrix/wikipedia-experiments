"""Wiring for the Ask Wikipedia experiment: settings -> backend -> RAG. UI-free, used by the screen and CLI."""
from __future__ import annotations

from pathlib import Path

from pipeline.settings import Settings
from wikiexp import paths
from wikiexp.llm import (BACKENDS, DEFAULT_CLAUDE_MODEL, DEFAULT_EFFORT, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL,
                         Backend, make_backend)
from wikiexp.rag import RAG, Answer

# prompt budget (tokens of sources): a small local model on a CPU reads ~50-100 tokens/s, so keep it
# short there; Claude can take much more
BUDGET = {"ollama": 2400, "claude": 8000}
ANSWER_TOKENS = {"ollama": 700, "claude": 2000}


def load_settings() -> Settings:
    try:
        return Settings.load()
    except Exception:
        return Settings()


def backend_from(settings, name: str | None = None, model: str | None = None, url: str | None = None,
                 effort: str | None = None) -> Backend:
    """The backend named `name` (default: the setting) with this model, from settings and overrides."""
    name = name or str(settings["ask_backend"]).strip() or "ollama"
    if name == "claude":
        return make_backend("claude", model=model or str(settings["claude_model"]).strip() or DEFAULT_CLAUDE_MODEL,
                            effort=effort or str(settings["claude_effort"]).strip() or DEFAULT_EFFORT)
    return make_backend("ollama", model=model or str(settings["ollama_model"]).strip() or DEFAULT_OLLAMA_MODEL,
                        url=url or str(settings["ollama_url"]).strip() or DEFAULT_OLLAMA_URL)


def build_rag(data_dir: Path | None, backend: Backend, **kw) -> RAG:
    name = getattr(backend, "name", "ollama")
    kw.setdefault("budget_tokens", BUDGET.get(name, 2400))
    kw.setdefault("max_answer_tokens", ANSWER_TOKENS.get(name, 700))
    return RAG(Path(data_dir or paths.DATA), backend, **kw)


def describe(a: Answer) -> str:
    """One line of timing: 'retrieval 850 ms, generation 120 tokens at 6.1 tok/s (first token 4.2 s)'."""
    g = a.generation
    t = a.timings
    detail = (f" (search {t['search']:.0f}, read {t['read']:.0f}, rerank {t['rerank']:.0f} ms "
              f"over {int(t['passages'])} passages)") if "rerank" in t else ""
    bits = [f"retrieval {a.retrieval_ms:,.0f} ms{detail}"]
    if g.output_tokens:
        rate = f" at {g.tokens_per_second:.1f} tok/s" if g.tokens_per_second else ""
        first = f", first token {g.first_token_seconds:.1f} s" if g.first_token_seconds else ""
        bits.append(f"generation {g.output_tokens} tokens{rate}{first}")
    elif g.seconds:
        bits.append(f"generation {g.seconds:.1f} s")
    return ", ".join(bits)


__all__ = ["BACKENDS", "BUDGET", "load_settings", "backend_from", "build_rag", "describe"]
