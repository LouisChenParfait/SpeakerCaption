#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单人模式（不分说话人）回归测试。

用户场景：全程只有一个人的直播 / 讲座 / 录屏旁白比多人会议更常见，
此时不需要 CAM++ 声纹分人，也不该在画面和文本里反复写「说话人1：」。

入口：GUI「人数 → 单人（不分人）」、--num-speakers 1、--single-speaker，
三者统一归一到 args.num_speakers == 1 / args.single_speaker is True。

Run:  python tests/test_single_speaker.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from speakercaption.diarize import Segment  # noqa: E402
from speakercaption.live_core import build_args, normalize_args, parse_args  # noqa: E402
from speakercaption.live_gui import COUNT, LiveDisplay  # noqa: E402
from speakercaption.live_speaker import SingleSpeakerTracker  # noqa: E402
from speakercaption.outputs import to_srt, to_txt, write_outputs  # noqa: E402


def _segments():
    return [Segment(0.0, 2.0, 1, "大家好，欢迎来到直播。"),
            Segment(2.4, 4.0, 1, "今天我们聊一个话题。")]


class TestNormalize(unittest.TestCase):
    def test_cli_alias(self):
        a = normalize_args(parse_args(["--single-speaker"]))
        self.assertEqual(a.num_speakers, 1)
        self.assertTrue(a.single_speaker)

    def test_cli_num_speakers_one(self):
        a = normalize_args(parse_args(["--num-speakers", "1"]))
        self.assertTrue(a.single_speaker)

    def test_default_is_diarize(self):
        a = normalize_args(parse_args([]))
        self.assertEqual(a.num_speakers, 0)
        self.assertFalse(a.single_speaker)

    def test_gui_setting_one(self):
        base = normalize_args(parse_args([]))
        a = build_args(base, {"num_speakers": 1})
        self.assertTrue(a.single_speaker)
        b = build_args(base, {"num_speakers": 2})
        self.assertFalse(b.single_speaker)

    def test_gui_dropdown_has_single_choice(self):
        self.assertEqual(COUNT.label("1"), "单人（不分人）")
        self.assertEqual(COUNT.value("单人（不分人）"), "1")


class TestSingleSpeakerTracker(unittest.TestCase):
    def test_always_label_one(self):
        t = SingleSpeakerTracker()
        for n in (100, 16000, 48000):
            label, score, is_new = t.assign(np.zeros(n, dtype=np.float32))
            self.assertEqual(label, 1)
            self.assertFalse(is_new)
        self.assertEqual(t.num_speakers(), 1)
        self.assertEqual(t.display_name(), "说话人1")
        self.assertIsNone(t.last_label, "草稿应继续用黄色 ▷，不要染成说话人色")

    def test_save_writes_nothing(self):
        t = SingleSpeakerTracker()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "voices.json"
            t.save(p)
            self.assertFalse(p.exists(), "单人模式不该生成声纹文件")

    def test_custom_name(self):
        self.assertEqual(SingleSpeakerTracker(name="主播").display_name(), "主播")


class TestOutputs(unittest.TestCase):
    def test_txt_no_speaker_prefix(self):
        plain = to_txt(_segments(), timestamps=False, show_speaker=False)
        self.assertNotIn("说话人", plain)
        self.assertEqual(plain.splitlines()[0], "大家好，欢迎来到直播。")
        with_label = to_txt(_segments(), timestamps=False, show_speaker=True)
        self.assertIn("说话人1：", with_label)

    def test_txt_timestamps_still_work(self):
        line = to_txt(_segments()[:1], timestamps=True, show_speaker=False).strip()
        self.assertTrue(line.startswith("[00:00.0] "))
        self.assertNotIn("说话人", line)

    def test_srt_no_speaker_tag(self):
        s = to_srt(_segments(), show_speaker=False)
        self.assertNotIn("说话人", s)
        self.assertNotIn("[说话人1]", s)
        self.assertIn("-->", s)

    def test_write_outputs_keeps_json_speaker_field(self):
        with tempfile.TemporaryDirectory() as d:
            paths = write_outputs(_segments(), d, "live", timestamps=False,
                                  show_speaker=False, formats=("txt", "json"))
            txt = Path(paths["txt"]).read_text(encoding="utf-8")
            js = Path(paths["json"]).read_text(encoding="utf-8")
        self.assertNotIn("说话人", txt)
        # 机器可读格式保留 speaker=1，方便下游工具/后续重跑
        self.assertIn('"speaker": 1', js)


class TestDisplay(unittest.TestCase):
    def test_show_speaker_flag(self):
        d = LiveDisplay()
        self.assertTrue(d.snapshot()["show_speaker"])
        d.set_show_speaker(False)
        self.assertFalse(d.snapshot()["show_speaker"])
        d.reset()
        self.assertFalse(d.snapshot()["show_speaker"],
                         "新一轮 LiveSession 会显式设置，reset 不该把它改回 True")


if __name__ == "__main__":
    unittest.main(verbosity=2)
