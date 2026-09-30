"""Move generated data between machines over SSH, and run commands on them.

Code travels through git; data (which is too big for git) travels with this tool. A typical flow:
run a GPU-heavy step on the Windows machine, then pull its output to wherever you're analysing.

  python -m tools.sync push gpu wiki.sqlite graph        # local data/... -> gpu's data/...
  python -m tools.sync pull gpu embeddings               # gpu's data/embeddings -> local data/
  python -m tools.sync run gpu "python -m pipeline.embed"  # git pull + run a command on gpu

Paths are relative to each machine's data directory. Machines are defined in config.toml
(see config.example.toml). Works with Windows' built-in OpenSSH server (no WSL) and Linux/macOS.
"""
import argparse
import base64
import subprocess
import sys

from wikiexp import paths


def machine(name):
    machines = paths.CONFIG.get("machines", {})
    if name not in machines:
        sys.exit(f"no [machines.{name}] in config.toml (known: {', '.join(machines) or 'none'})")
    m = dict(machines[name])
    m.setdefault("os", "posix")
    m.setdefault("data", m["repo"].rstrip("/") + "/data")
    return m


def remote_shell(m, command):
    """Wrap a command for the remote machine's shell.

    Windows OpenSSH hands commands to cmd.exe (or PowerShell, depending on setup), so PowerShell code
    is sent base64-encoded to survive either shell's quoting rules.
    """
    if m["os"] == "windows":
        encoded = base64.b64encode(command.encode("utf-16-le")).decode()
        return [f"powershell -NoProfile -EncodedCommand {encoded}"]
    return [command]


def remote_mkdir(m, path):
    if m["os"] == "windows":
        return remote_shell(m, f"New-Item -ItemType Directory -Force -Path '{path}' | Out-Null")
    return [f"mkdir -p '{path}'"]


def run(cmd, dry, show=None):
    print("+", show or " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def parent(rel):
    return rel.rstrip("/").rpartition("/")[0]


def push(m, items, dry):
    for rel in items:
        local = paths.DATA / rel
        if not local.exists():
            sys.exit(f"{local} does not exist")
        dest = "/".join(p for p in [m["data"], parent(rel)] if p)
        run(["ssh", m["host"], *remote_mkdir(m, dest)], dry, f"ssh {m['host']} mkdir {dest}")
        run(["scp", "-r", str(local), f"{m['host']}:{dest}/"], dry)


def pull(m, items, dry):
    for rel in items:
        dest = paths.DATA / parent(rel)
        if not dry:
            dest.mkdir(parents=True, exist_ok=True)
        run(["scp", "-r", f"{m['host']}:{m['data']}/{rel.rstrip('/')}", str(dest)], dry)


def remote_run(m, command, dry):
    if "repo" not in m:
        sys.exit("set repo = ... for this machine in config.toml")
    if m["os"] == "windows":
        full = f"Set-Location '{m['repo']}'; git pull --ff-only; if ($?) {{ {command} }}"
    else:
        full = f"cd '{m['repo']}' && git pull --ff-only && {command}"
    run(["ssh", "-t", m["host"], *remote_shell(m, full)], dry, f"ssh {m['host']} {full}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--dry-run", action="store_true", help="print commands without running them")
    ap.add_argument("action", choices=["push", "pull", "run"])
    ap.add_argument("machine")
    ap.add_argument("items", nargs=argparse.REMAINDER,
                    help="paths under data/ (push/pull), or a command (run)")
    args = ap.parse_args()
    if not args.items:
        ap.error("nothing to push, pull or run")
    m = machine(args.machine)
    if args.action == "push":
        push(m, args.items, args.dry_run)
    elif args.action == "pull":
        pull(m, args.items, args.dry_run)
    else:
        remote_run(m, " ".join(args.items), args.dry_run)


if __name__ == "__main__":
    main()
