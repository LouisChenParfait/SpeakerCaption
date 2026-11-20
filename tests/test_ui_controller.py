#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for the GUI controller.

Bug (2026-11-09): UIController.__init__ assigned ``self.stop = threading.Event()``,
which shadowed the ``stop()`` method. Clicking "■ 停止" raised
``TypeError: 'Event' object is not callable`` and the capture never stopped.

Bug (2026-11-09): the GUI wrote the *merged* settings (persisted values + CLI
overrides) back to settings.json on startup, so a one-off command line flag
such as ``--simulate tests/data/x.wav`` silently became permanent and every
later launch replayed that file instead of capturing the real system audio.
Command line flags must apply to the current launch only.

Run:  python tests/test_ui_controller.py
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from speakercaption.live_core import parse_args  # noqa: E402
from speakercaption.live_ui import UIController, _sanitize  # noqa: E402
from speakercaption.settings import UI_KEYS, load_settings  # noqa: E402


def _defaults():
    return {k: v for k, v in vars(parse_args([])).items() if k in UI_KEYS}


def _controller(tmp: Path, cli_overrides=None, persisted=None) -> UIController:
    defaults = _defaults()
    effective = _sanitize({**defaults, **(persisted or {}), **(cli_overrides or {})},
                           defaults)
    return UIController(parse_args([]), effective, defaults, tmp / "settings.json",
                        cli_overrides=cli_overrides, persisted=persisted)


def _session_opts(ctl: UIController, tmp: Path, wav: Path) -> dict:
    opts = dict(ctl.defaults)
    opts.update(simulate=str(wav), loop_sim=True, duration=0.0,
                final="none", outdir=str(tmp / "out"), formats="txt",
                translate=False, language="auto", no_streaming=False)
    return opts


def _wait_stopped(ctl, timeout=30):
    deadline = time.monotonic() + timeout
    while ctl.is_running() and time.monotonic() < deadline:
        time.sleep(0.1)
    return not ctl.is_running()


class TestStop(unittest.TestCase):
    def test_stop_is_callable_and_sets_event(self):
        with tempfile.TemporaryDirectory() as d:
            ctl = _controller(Path(d))
            self.assertTrue(callable(ctl.stop), "stop() 被同名属性覆盖了")
            self.assertFalse(ctl.stop_event.is_set())
            ctl.stop()
            self.assertTrue(ctl.stop_event.is_set())
            ctl.on_close()          # 关窗口走同一条路径，不能抛

    def test_stop_ends_pipeline(self):
        wav = ROOT / "tests" / "data" / "1-two-speakers-en.wav"
        if not wav.is_file():
            self.skipTest("缺测试音频，跳过端到端启停")
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctl = _controller(tmp)
            ctl.start(_session_opts(ctl, tmp, wav))
            time.sleep(3.0)
            self.assertTrue(ctl.is_running(), "pipeline 没能启动")
            ctl.stop()
            self.assertTrue(_wait_stopped(ctl), "点停止后 pipeline 仍未结束")
            self.assertTrue(ctl.stop_event.is_set())
            self.assertFalse(ctl.display.snapshot()["error"],
                             ctl.display.snapshot()["error"])

    def test_restart_after_stop(self):
        """停止 → 再开始：不能因旧的停止信号 / 残留线程而失败。"""
        wav = ROOT / "tests" / "data" / "1-two-speakers-en.wav"
        if not wav.is_file():
            self.skipTest("缺测试音频，跳过二次启动")
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctl = _controller(tmp)
            opts = _session_opts(ctl, tmp, wav)
            events = []
            for round_no in (1, 2):
                ctl.start(opts)
                events.append(ctl.stop_event)
                time.sleep(1.5)
                self.assertTrue(ctl.is_running(), "第 %d 轮没跑起来" % round_no)
                ctl.stop()
                self.assertTrue(_wait_stopped(ctl), "第 %d 轮停不下来" % round_no)
                self.assertFalse(ctl.display.snapshot()["error"],
                                 ctl.display.snapshot()["error"])
            # 每轮必须拿到独立停止事件，clear 旧事件不能复活上一轮线程
            self.assertIsNot(events[0], events[1], "两轮复用了同一个停止事件")


class TestCliOverrideNotPersisted(unittest.TestCase):
    """命令行参数只对当次启动覆盖，不能被 GUI 写回 settings.json。"""

    def test_cli_override_not_written_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cfg = tmp / "settings.json"
            overrides = {"simulate": "tests/data/0-four-speakers-zh.wav",
                         "duration": 30.0, "num_speakers": 2}
            ctl = _controller(tmp, cli_overrides=overrides)
            # 生效值里能看到命令行覆盖（界面显示它、本次运行用它）
            self.assertEqual(ctl.settings["simulate"], overrides["simulate"])
            self.assertEqual(ctl.settings["duration"], 30.0)
            # 界面初始化时会立刻保存一次——这一步不能把命令行值写进磁盘
            ctl.save(dict(ctl.settings))
            saved = load_settings(cfg)
            self.assertIsNone(saved.get("simulate"),
                              "命令行的 --simulate 被写进了 settings.json")
            self.assertEqual(float(saved.get("duration") or 0.0), 0.0)

    def test_user_change_does_persist(self):
        """用户在界面上真的改了，还是应该存下来。"""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cfg = tmp / "settings.json"
            overrides = {"num_speakers": 2}
            ctl = _controller(tmp, cli_overrides=overrides)
            opts = dict(ctl.settings)
            opts["num_speakers"] = 5          # 用户在下拉里改成「5 人」
            ctl.save(opts)
            saved = load_settings(cfg)
            self.assertEqual(int(saved["num_speakers"]), 5)

    def test_unrelated_change_keeps_cli_value_out(self):
        """改别的项时，没动过的那条命令行参数仍不落盘。"""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cfg = tmp / "settings.json"
            overrides = {"simulate": "x.wav"}
            ctl = _controller(tmp, cli_overrides=overrides)
            opts = dict(ctl.settings)
            opts["script"] = "traditional"     # 只改字形
            ctl.save(opts)
            saved = load_settings(cfg)
            self.assertEqual(saved["script"], "traditional")
            self.assertIsNone(saved.get("simulate"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
