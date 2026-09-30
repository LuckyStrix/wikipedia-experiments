"""Wikipedia Experiments: terminal app for the data pipeline and the experiments.

    python app.py

Works in any terminal: Linux, macOS, Windows Terminal (no WSL needed), or over SSH.
The pipeline itself lives in pipeline/ and runs without the app too; see README.md.
"""
import sys

if sys.version_info < (3, 11):
    sys.exit("Python 3.11 or newer is required.")

from tui.app import WikiApp  # noqa: E402

if __name__ == "__main__":
    WikiApp().run()
