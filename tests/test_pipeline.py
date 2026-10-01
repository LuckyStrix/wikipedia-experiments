import sys
import textwrap

import pytest

from pipeline import settings as S
from pipeline import stages as ST
from pipeline.parsers import ProgressParser, WgetParser
from pipeline.runner import PipelineRunner
from wikiexp.sqldump import int_rows, rows


# ── settings ─────────────────────────────────────────────────────────────────

def test_settings_roundtrip_keeps_machines(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[machines.gpu]\nhost = "me@pc"\nrepo = "C:/x"\nos = "windows"\n')
    s = S.Settings.load(cfg)
    assert s["dump_date"] == "20260901" and s["verify"] is True
    s["dump_date"] = "20261001"
    s["dl_text"] = False
    s["data_dir"] = "D:/wiki data"
    s.save(cfg)
    again = S.Settings.load(cfg)
    assert (again["dump_date"], again["dl_text"], again["data_dir"]) == ("20261001", False, "D:/wiki data")
    assert S.read_config(cfg)["machines"]["gpu"]["host"] == "me@pc"


def test_blank_paths_are_not_written(tmp_path):
    cfg = tmp_path / "config.toml"
    S.Settings().save(cfg)
    assert "paths" not in S.read_config(cfg) or not S.read_config(cfg)["paths"]


def test_invalid_dump_date():
    s = S.Settings({"dump_date": "2026-09"})
    assert s.invalid() == ["Dump date"]


# ── parsers ──────────────────────────────────────────────────────────────────

def test_wget_parser_weights_by_size():
    p = WgetParser({"small": 100, "big": 900}, ["small", "big"])
    assert p.feed("(1/2) downloading small")[0] == 0
    assert p.feed("(2/2) downloading big")[0] == 10
    assert p.feed(" 1K ........ 50% 5M 1s")[0] == 55


def test_wget_parser():
    p = WgetParser()
    assert p.feed("[2026-09-30 18:00:00] (2/4) downloading page.sql.gz") == (25, "page.sql.gz (2/4)")
    assert p.feed(" 65536K ........ ........ ........ ........ 50% 4.99M 35s") == (37, "page.sql.gz 50% (2/4)")
    assert p.hide(" 32768K ........ ........ ........ ........ 23% 4.98M 42s")
    assert not p.hide(" 32768K ........ ........ ........ ........ 30% 4.98M 42s")


def test_progress_parser():
    p = ProgressParser()
    assert p.feed("@progress 42 reading pagelinks") == (42, "reading pagelinks")
    assert p.hide("@progress 42 x") and not p.hide("[ 3s] reading")
    assert p.feed("[ 3s] reading") is None


# ── stages ───────────────────────────────────────────────────────────────────

def test_download_done_uses_latest_checksum(tmp_path):
    s = S.Settings({"dumps_dir": str(tmp_path), "dl_extras": False, "dl_text": False})
    names = [ST.dump_file(s, f).name for f in ST.CORE_DUMPS]
    log = tmp_path / "download.log"
    log.write_text("".join(f"{n}: FAILED\n" for n in names))
    assert not ST.download_done(s)
    log.write_text(log.read_text() + "".join(f"{n}: OK\n" for n in names))
    assert ST.download_done(s)
    s["dl_extras"] = True
    assert not ST.download_done(s)


def test_stage_commands_follow_settings():
    s = S.Settings({"verify": False, "titles_redirects": False, "dl_text": False})
    assert ST.BY_KEY["core"].command(s)[-1] == "--skip-verify"
    assert ST.BY_KEY["titles"].command(s)[-1] == "--no-redirects"
    dl = ST.BY_KEY["download"].command(s)
    assert "pagelinks.sql.gz" in dl and "pages-articles-multistream.xml.bz2" not in dl


# ── runner ───────────────────────────────────────────────────────────────────

def fake_stage(key, script, done=lambda s: False):
    return ST.Stage(key=key, name=f"Fake {key}", description="", command=lambda s: [sys.executable, "-c", script],
                    inputs=lambda s: [], outputs=lambda s: [], summary=lambda s: [], done_check=done)


def run(stages, settings, action):
    events = []
    r = PipelineRunner(settings, stages=stages,
                       on_log=lambda t, k: events.append(("log", t, k)),
                       on_state=lambda k, st, msg: events.append(("state", k, st)),
                       on_progress=lambda k, p, t: events.append(("progress", k, p, t)))
    action(r)
    r.wait(30)
    return events


def test_runner_reports_progress_and_hides_progress_lines(tmp_path):
    s = S.Settings({"data_dir": str(tmp_path)})
    script = "print('hello'); print('@progress 50 halfway'); print('bye')"
    ev = run([fake_stage("a", script)], s, lambda r: r.run_stage("a"))
    assert ("progress", "a", 50, "halfway") in ev
    outputs = [e[1] for e in ev if e[0] == "log" and e[2] == "output"]
    assert outputs == ["hello", "bye"]
    assert [e[2] for e in ev if e[0] == "state"] == ["running", "done"]
    assert (tmp_path / "runs" / "runs.json").exists() and (tmp_path / "runs" / "pipeline.log").exists()


def test_run_all_skips_done_and_stops_at_failure(tmp_path):
    s = S.Settings({"data_dir": str(tmp_path)})
    stages = [fake_stage("a", "", done=lambda s: True), fake_stage("b", "raise SystemExit(3)"),
              fake_stage("c", "print('never')")]
    ev = run(stages, s, lambda r: r.run_all())
    states = [(e[1], e[2]) for e in ev if e[0] == "state"]
    assert states == [("b", "running"), ("b", "failed")]


def test_stop(tmp_path):
    s = S.Settings({"data_dir": str(tmp_path)})
    stage = fake_stage("a", "import time\nprint('started', flush=True)\ntime.sleep(60)")

    def start_then_stop(r):
        import time
        r.run_stage("a")
        time.sleep(1)
        r.stop()
    ev = run([stage], s, start_then_stop)
    assert [e[2] for e in ev if e[0] == "state"] == ["running", "failed"]


# ── dump readers ─────────────────────────────────────────────────────────────

def test_sql_dump_readers(tmp_path):
    import gzip
    dump = tmp_path / "t.sql.gz"
    text = textwrap.dedent("""\
        CREATE TABLE `t` (
          `a` int(8) NOT NULL,
          `b` varbinary(255) NOT NULL,
          `c` int(11) DEFAULT NULL,
          PRIMARY KEY (`a`)
        ) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=binary;
        INSERT INTO `t` VALUES
        (1,'It\\'s_(a),test',NULL),
        (2,'back\\\\slash',-5);
        """)
    with gzip.open(dump, "wt") as f:
        f.write(text)
    assert list(rows(dump, ["a", "b", "c"])) == [(b"1", b"It's_(a),test", None), (b"2", b"back\\slash", b"-5")]

    nums = tmp_path / "n.sql.gz"
    with gzip.open(nums, "wt") as f:
        f.write("CREATE TABLE `n` (\n  `x` int NOT NULL,\n  `y` int NOT NULL\n);\n"
                "INSERT INTO `n` VALUES\n(1,2),\n(3,-4);\nINSERT INTO `n` VALUES\n(5,6);\n")
    import numpy as np
    arr = np.concatenate(list(int_rows(nums, 2)))
    assert arr.tolist() == [[1, 2], [3, -4], [5, 6]]


def test_parallel_parsing_matches_serial(tmp_path, monkeypatch):
    import gzip

    import numpy as np

    from wikiexp import sqldump
    dump = tmp_path / "p.sql.gz"
    lines = [f"({i},{i % 3},'Title_{i}_\\'q\\'',{i % 2})" for i in range(20000)]
    with gzip.open(dump, "wt") as f:
        f.write("CREATE TABLE `p` (\n  `id` int NOT NULL,\n  `ns` int NOT NULL,\n  `t` varbinary(9) NOT NULL,\n"
                "  `r` int NOT NULL\n);\nINSERT INTO `p` VALUES\n" + ",\n".join(lines) + ";\n")
    monkeypatch.setattr(sqldump, "BLOCK", 4096)   # many small blocks
    serial = list(sqldump.rows(dump, ["id", "t"], where=("ns", b"0"), workers=1))
    parallel = list(sqldump.rows(dump, ["id", "t"], where=("ns", b"0"), workers=4))
    assert parallel == serial and len(serial) == 6667
    assert serial[1] == (b"3", b"Title_3_'q'")

    ints = tmp_path / "i.sql.gz"
    with gzip.open(ints, "wt") as f:
        f.write("CREATE TABLE `i` (\n  `a` int NOT NULL,\n  `b` int NOT NULL\n);\nINSERT INTO `i` VALUES\n"
                + ",\n".join(f"({i},{-i})" for i in range(30000)) + ";\n")
    got = np.concatenate(list(sqldump.map_blocks(ints, functools_partial_int(), workers=4)))
    assert got.tolist() == [[i, -i] for i in range(30000)]


def functools_partial_int():
    import functools

    from wikiexp import sqldump
    return functools.partial(sqldump.parse_int_block, ncols=2)
