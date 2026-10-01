"""Ask Wikipedia from the command line.

  python -m experiments.ask "Who designed the Eiffel Tower?"
  python -m experiments.ask "Why did the Tacoma Narrows Bridge collapse?" --show-passages
  python -m experiments.ask -i                       # a conversation: follow-up questions work
  python -m experiments.ask "..." --backend claude [--model claude-sonnet-5-5] [--effort high]
  python -m experiments.ask "..." --ollama-url http://gpu-pc:11434 --model llama3.1:8b

Needs the core, titles, text and keyword stages (semantic embeddings are used when present), and
either ollama running (default; `ollama pull llama3.2:3b`) or ANTHROPIC_API_KEY for --backend claude.
Settings come from config.toml ([ask] section / the app's Ask tab); the flags override them.
In interactive mode: /new starts a fresh conversation, an empty line quits.
"""
import argparse
import sys
import textwrap
from pathlib import Path

from wikiexp import paths
from wikiexp.llm import BACKENDS, LLMError
from wikiexp.rag import Answer, Conversation

from . import session


def show_sources(answer: Answer, passages: bool) -> None:
    if not answer.sources:
        return
    print("\nSources (* = cited):")
    for s in answer.sources:
        mark = "*" if s.n in answer.cited else " "
        print(f" {mark}[{s.n}] {s.label}   (score {s.passage.score:.2f})")
        if passages:
            print(textwrap.indent(textwrap.fill(s.passage.text.replace("\n", " "), 100), "      "))
    others = [r.title for r in answer.candidates[4:10]]
    if others and passages:
        print("  also considered: " + "; ".join(others))


def ask_once(rag, question: str, conv: Conversation, passages: bool) -> Answer | None:
    answer = None
    for ev in rag.ask(question, conv):
        if ev.kind == "status":
            print(f"({ev.data})", file=sys.stderr, flush=True)
        elif ev.kind == "token":
            print(ev.data, end="", flush=True)
        elif ev.kind == "done":
            answer = ev.data
    print()
    if answer is None:
        return None
    if answer.text != answer.raw_text:      # citations were fixed: show the corrected text
        print("\n(corrected) " + answer.text)
    if answer.error:
        print(f"\n! {answer.error}", file=sys.stderr)
    for n in answer.notes:
        print(f"! {n}", file=sys.stderr)
    show_sources(answer, passages)
    q = f" (searched: {answer.query})" if answer.query != answer.question else ""
    print(f"\n[{answer.backend} {answer.model}; {session.describe(answer)}]{q}")
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*", help="the question (several words need no quotes)")
    ap.add_argument("-i", "--interactive", action="store_true", help="a conversation with follow-up questions")
    ap.add_argument("--backend", choices=BACKENDS, help="ollama (local, default) or claude")
    ap.add_argument("--model", help="model name for the backend (default: from settings)")
    ap.add_argument("--ollama-url", help="ollama server, e.g. http://gpu-pc:11434")
    ap.add_argument("--effort", help="claude effort: low, medium, high, xhigh, max")
    ap.add_argument("--show-passages", action="store_true", help="print the text of each source passage")
    ap.add_argument("--articles", type=int, default=4, help="how many articles to read in full (default 4)")
    ap.add_argument("--budget", type=int, help="tokens of sources to put in the prompt")
    ap.add_argument("--data", help="data folder (default: the configured one)")
    args = ap.parse_args()
    if not args.question and not args.interactive:
        ap.error("give a question, or use -i")

    settings = session.load_settings()
    try:
        backend = session.backend_from(settings, args.backend, args.model, args.ollama_url, args.effort)
        kw = {"n_articles": args.articles}
        if args.budget:
            kw["budget_tokens"] = args.budget
        rag = session.build_rag(Path(args.data) if args.data else paths.DATA, backend, **kw)
        if not rag.hybrid.modes():
            sys.exit(f"No keyword index or embeddings in {rag.data_dir}: run the 'text' and 'keyword' stages "
                     f"(python -m pipeline.build_text; python -m pipeline.build_keyword).")
    except LLMError as e:
        sys.exit(str(e))
    conv = Conversation()
    try:
        if args.question:
            ask_once(rag, " ".join(args.question), conv, args.show_passages)
        if args.interactive:
            while True:
                try:
                    q = input("\nask> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q:
                    break
                if q == "/new":
                    conv.reset()
                    print("(new conversation)")
                    continue
                ask_once(rag, q, conv, args.show_passages)
    except LLMError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
