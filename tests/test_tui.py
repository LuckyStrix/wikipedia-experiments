"""Drive the app headlessly with Textual's pilot."""
import pytest

from pipeline import settings as S
from tests.test_pipeline import fake_stage
from tui.app import WikiApp


def make_app(tmp_path, stages, data_dir=None):
    s = S.Settings({"data_dir": str(data_dir or tmp_path / "data"), "dumps_dir": str(tmp_path / "dumps")})
    return WikiApp(settings=s, config_path=tmp_path / "config.toml", ui_state_path=tmp_path / "ui.json",
                   stages=stages)


async def wait_for(pilot, cond, seconds=15):
    for _ in range(int(seconds * 20)):
        if cond():
            return True
        await pilot.pause(0.05)
    return cond()


@pytest.mark.asyncio
async def test_run_all_from_keyboard(tmp_path):
    marker = tmp_path / "done.txt"
    stage = fake_stage("a", f"print('@progress 50 half'); open(r'{marker}', 'w').write('x'); print('hi')",
                       done=lambda s: marker.exists())
    app = make_app(tmp_path, [stage])
    async with app.run_test(size=(140, 45)) as pilot:
        assert app.cards["a"].state == "waiting"
        await pilot.press("r")
        assert await wait_for(pilot, lambda: app.cards["a"].state == "done")
        assert "Done" in app.cards["a"].status


@pytest.mark.asyncio
async def test_r_is_ignored_while_typing_in_settings(tmp_path):
    marker = tmp_path / "ran.txt"
    stage = fake_stage("a", f"open(r'{marker}', 'w')")
    app = make_app(tmp_path, [stage])
    async with app.run_test(size=(140, 45)) as pilot:
        app.query_one("#set-data_dir").focus()
        await pilot.press("r")
        await pilot.pause(0.5)
        assert not marker.exists()


@pytest.mark.asyncio
async def test_ctrl_s_saves_settings(tmp_path):
    app = make_app(tmp_path, [])
    async with app.run_test(size=(140, 45)) as pilot:
        app.settings["dump_date"] = "20261001"
        await pilot.press("ctrl+s")
        await pilot.pause(0.1)
    assert S.Settings.load(tmp_path / "config.toml")["dump_date"] == "20261001"


def log_text(richlog) -> str:
    return "\n".join("".join(seg.text for seg in line) for line in richlog.lines)


@pytest.mark.asyncio
async def test_six_degrees_only_accepts_real_titles(tiny_data, tmp_path):
    from experiments.six_degrees.screen import SixDegreesScreen
    from textual.widgets import Button, RichLog
    app = make_app(tmp_path, [], data_dir=tiny_data)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SixDegreesScreen(tiny_data)
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.finder is not None)
        a, b = screen.pickers
        find = screen.query_one("#sd-find", Button)

        # typing a partial title doesn't count as picked; Enter takes the top suggestion
        a.input.focus()
        await pilot.press(*"kevin ba")
        assert await wait_for(pilot, lambda: a.query_one("OptionList").display)
        assert a.match is None and find.disabled
        await pilot.press("enter")
        assert a.match.article == "Kevin Bacon"

        # a redirect with a typo: suggestions still find it
        b.input.focus()
        await pilot.press(*"mitochondira")
        assert await wait_for(pilot, lambda: b.query_one("OptionList").display)
        await pilot.press("enter")
        assert b.match.article == "Mitochondrion"
        assert not find.disabled

        # editing a picked title un-picks it (focus selects all, so move to the end first)
        b.input.focus()
        await pilot.press("end", "backspace")
        assert b.match is None and find.disabled
        await pilot.press(*"n")
        assert await wait_for(pilot, lambda: b.query_one("OptionList").display)
        await pilot.press("enter")

        await pilot.press("ctrl+f")
        results = screen.query_one("#sd-results", RichLog)
        assert await wait_for(pilot, lambda: "4 clicks" in log_text(results))
        assert "Footloose" in log_text(results)


@pytest.mark.asyncio
async def test_pipeline_keys_only_work_on_main_screen(tiny_data, tmp_path):
    from experiments.six_degrees.screen import SixDegreesScreen
    marker = tmp_path / "ran.txt"
    app = make_app(tmp_path, [fake_stage("a", f"open(r'{marker}', 'w')")], data_dir=tiny_data)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SixDegreesScreen(tiny_data)
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.finder is not None)
        screen.query_one("#sd-random").focus()
        await pilot.press("r", "ctrl+r")
        await pilot.pause(0.5)
        assert not marker.exists()
