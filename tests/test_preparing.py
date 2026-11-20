# -*- coding: utf-8 -*-
"""准备阶段反馈的回归测试。

Bug (2026-11-18): 点「开始听录」后要加载 VAD / CAM++ / Whisper 权重，
这段时间 stats 还没上来，状态条却仍显示「设置好选项后点开始听录」，
波形静止，用户分不清是在准备还是死机。
修法：LiveDisplay 增加 preparing 状态，LiveSession 分 5 步上报，
Qt / Tk 视图层据此显示动画与阶段文字，异常与结束必须清除该状态。
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speakercaption.live_core import (LiveSession, normalize_args, parse_args)
from speakercaption.live_gui import LiveDisplay
from speakercaption.live_ui import UIController
from speakercaption.settings import UI_KEYS

WAV = Path(__file__).resolve().parent / "data" / "1-two-speakers-en.wav"


class RecordingDisplay(LiveDisplay):
    def __init__(self):
        super().__init__()
        self.stages = []
        self.clear_count = 0

    def set_preparing(self, stage, detail="", step=0, total=0):
        self.stages.append((stage, step, total))
        super().set_preparing(stage, detail, step, total)

    def clear_preparing(self):
        self.clear_count += 1
        super().clear_preparing()


class DisplayStateTests(unittest.TestCase):
    def test_preparing_lifecycle(self):
        d = LiveDisplay()
        self.assertEqual(d.snapshot()["preparing"], {})
        d.set_preparing("加载 VAD", "silero", 1, 5)
        first = d.snapshot()["preparing"]
        self.assertEqual((first["stage"], first["step"], first["total"]),
                         ("加载 VAD", 1, 5))
        time.sleep(0.01)
        d.set_preparing("加载声纹", "", 2, 5)
        second = d.snapshot()["preparing"]
        self.assertEqual(second["stage"], "加载声纹")
        self.assertEqual(second["at"], first["at"], "阶段切换不能重置等待计时")
        d.clear_preparing()
        self.assertEqual(d.snapshot()["preparing"], {})

    def test_reset_clears_preparing_and_session(self):
        d = LiveDisplay()
        d.set_preparing("x", "", 1, 2)
        d.set_session({"stem": "live_2026"})
        d.reset()
        snap = d.snapshot()
        self.assertEqual(snap["preparing"], {})
        self.assertEqual(snap["session"], {})

    def test_session_survives_stats_updates(self):
        """采集开始后 set_stats 会整体替换 stats，会话信息不能塞在里面。"""
        d = LiveDisplay()
        d.set_session({"stem": "live_20261118_233520", "outdir": "out"})
        d.set_stats({"peak": 0.5, "device": "loopback"})
        self.assertEqual(d.snapshot()["session"]["stem"],
                         "live_20261118_233520")


def _controller(tmp):
    base = normalize_args(parse_args(["--outdir", str(Path(tmp) / "out")]))
    defaults = {k: v for k, v in vars(parse_args([])).items() if k in UI_KEYS}
    defaults["outdir"] = str(Path(tmp) / "out")
    return UIController(base, dict(defaults), defaults,
                        Path(tmp) / "settings.json")


def _wait(ctl, timeout=8.0):
    t0 = time.time()
    while ctl.is_running() and time.time() - t0 < timeout:
        time.sleep(0.02)


class ControllerPreparingTests(unittest.TestCase):
    def test_setup_failure_clears_preparing(self):
        """会话构造失败：不能把界面永远卡在加载动画上。"""
        with tempfile.TemporaryDirectory() as tmp:
            ctl = _controller(tmp)
            with patch("speakercaption.live_ui.LiveSession",
                       side_effect=RuntimeError("boom")):
                ctl.start(dict(ctl.settings))
            snap = ctl.display.snapshot()
            self.assertEqual(snap["preparing"], {})
            self.assertIn("boom", snap["error"])
            self.assertFalse(ctl.is_running())

    def test_pipeline_failure_clears_preparing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctl = _controller(tmp)
            with patch.object(LiveSession, "run_pipeline",
                              side_effect=RuntimeError("boom")):
                ctl.start(dict(ctl.settings))
                _wait(ctl)
            snap = ctl.display.snapshot()
            self.assertEqual(snap["preparing"], {})
            self.assertIn("boom", snap["error"])

    def test_start_file_uses_unique_stem_and_clears_state(self):
        """文件模式也要新建带时间戳的稿子，且转写完必须撤掉准备态。"""
        import speakercaption.file_mode as fm

        seen = []

        class FakeTranscriber:
            def __init__(self, *a, **kw):
                seen.append(kw)

            def run(self):
                return {"segments": [], "names": {}, "speakers": [1],
                        "elapsed": 0.1, "paths": {}, "single": True}

        with tempfile.TemporaryDirectory() as tmp:
            ctl = _controller(tmp)
            with patch.object(fm, "FileTranscriber", FakeTranscriber):
                ctl.start_file(dict(ctl.settings), str(Path(tmp) / "节目.wav"))
                _wait(ctl)
            self.assertTrue(seen, "FileTranscriber 没被构造")
            stem = seen[0]["stem"]
            self.assertRegex(stem, r"^节目_\d{8}_\d{6}(_\d+)?$")
            snap = ctl.display.snapshot()
            self.assertEqual(snap["session"]["stem"], stem)
            self.assertEqual(snap["preparing"], {})


class PipelineStageTests(unittest.TestCase):
    def test_pipeline_reports_all_stages_then_clears(self):
        if not WAV.is_file():
            self.skipTest("缺测试音频 %s" % WAV)
        with tempfile.TemporaryDirectory() as tmp:
            args = normalize_args(parse_args([
                "--outdir", str(Path(tmp) / "out"),
                "--final", "none", "--num-speakers", "1",
                "--duration", "1.0", "--no-gui",
                "--simulate", str(WAV),
            ]))
            disp = RecordingDisplay()
            session = LiveSession(args, display=disp)
            session.run_pipeline()
            names = [x[0] for x in disp.stages]
            self.assertIn("加载语音检测模型", names)
            self.assertIn("单人模式，跳过声纹", names)
            self.assertIn("加载流式识别模型", names)
            self.assertIn("加载句末精修模型", names)
            self.assertIn("打开音频设备", names)
            self.assertGreaterEqual(len(disp.stages), 5)
            self.assertGreaterEqual(disp.clear_count, 1)
            self.assertEqual(disp.snapshot()["preparing"], {})
            self.assertTrue(session.session_stem.startswith("live_"))


if __name__ == "__main__":
    unittest.main()
