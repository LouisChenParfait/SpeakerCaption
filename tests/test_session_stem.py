# -*- coding: utf-8 -*-
"""每次听录独立输出文件名的回归测试。

Bug (2026-11-18): 输出 stem 固定为 live，且启动时会 unlink 同名文件，
听完一段再听新直播，上一场文字稿直接被覆盖，找不回来。
修法：默认给 stem 加时间戳后缀，同秒连开自动加序号，不再删历史文件。
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speakercaption.diarize import Segment
from speakercaption.live_core import (LiveSession, _session_stem,
                                     normalize_args, parse_args)
from speakercaption.live_gui import LiveDisplay
from speakercaption.session_stem import (make_session_stem, safe_base,
                                         split_stamp)

FIXED = time.mktime(time.strptime("2026-11-18 23:35:20", "%Y-%m-%d %H:%M:%S"))


class StemFormatTests(unittest.TestCase):
    def test_timestamp_suffix_format(self):
        s = make_session_stem("live", None, when=FIXED)
        self.assertEqual(s, "live_20261118_233520")
        self.assertEqual(split_stamp(s), ("live", "20261118_233520"))

    def test_split_stamp_on_plain_name(self):
        self.assertEqual(split_stamp("live"), ("", ""))

    def test_safe_base_strips_media_extension(self):
        self.assertEqual(safe_base("武晔01.wav"), "武晔01")

    def test_safe_base_removes_illegal_chars(self):
        self.assertNotIn("/", safe_base("a/b:c"))
        self.assertNotIn(":", safe_base("a/b:c"))
        self.assertEqual(safe_base(""), "live")

    def test_safe_base_caps_length(self):
        self.assertLessEqual(len(safe_base("x" * 200)), 60)

    def test_none_mode_keeps_fixed_stem(self):
        args = normalize_args(parse_args(["--stem-suffix", "none",
                                          "--stem", "live"]))
        self.assertEqual(_session_stem(args, Path(".")), "live")

    def test_default_mode_is_timestamp(self):
        self.assertEqual(parse_args([]).stem_suffix, "timestamp")


class NoOverwriteTests(unittest.TestCase):
    def test_same_second_gets_suffix_2(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            s1 = make_session_stem("live", out, ("txt", "srt"), when=FIXED)
            (out / (s1 + ".txt")).write_text("上一场文字稿", encoding="utf-8")
            s2 = make_session_stem("live", out, ("txt", "srt"), when=FIXED)
            self.assertEqual(s2, s1 + "_2")
            self.assertEqual((out / (s1 + ".txt")).read_text(encoding="utf-8"),
                             "上一场文字稿")

    def test_checks_every_output_format(self):
        """只有 .srt 存在也必须避让，否则会覆盖上一场的字幕。"""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            s1 = make_session_stem("live", out, ("txt", "srt"), when=FIXED)
            (out / (s1 + ".srt")).write_text("1\n", encoding="utf-8")
            s2 = make_session_stem("live", out, ("txt", "srt"), when=FIXED)
            self.assertEqual(s2, s1 + "_2")

    def test_two_sessions_keep_both_transcripts(self):
        """真实走 write_all()：两次会话各写一份，旧稿与新稿并存。"""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "old_live.txt").write_text("很久以前的一场", encoding="utf-8")
            stems = []
            for text in ("第一场直播", "第二场直播"):
                args = normalize_args(parse_args([
                    "--outdir", str(out), "--final", "none",
                    "--num-speakers", "1", "--formats", "txt,json",
                ]))
                session = LiveSession(args, display=LiveDisplay())
                stems.append(session.session_stem)
                session.segments.append(Segment(0.0, 1.0, 1, text))
                session.write_all()
            self.assertNotEqual(stems[0], stems[1])
            self.assertEqual((out / "old_live.txt").read_text(encoding="utf-8"),
                             "很久以前的一场")
            for stem, text in zip(stems, ("第一场直播", "第二场直播")):
                self.assertIn(text, (out / (stem + ".txt")).read_text(encoding="utf-8"))
                doc = json.loads((out / (stem + ".json")).read_text(encoding="utf-8"))
                self.assertEqual(doc["session_id"], stem)
                self.assertIn("session_started_at", doc)

    def test_none_mode_still_overwrites_on_request(self):
        """显式 --stem-suffix none 保留旧的固定文件名行为（脚本批处理用）。"""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            args = normalize_args(parse_args([
                "--outdir", str(out), "--stem-suffix", "none",
                "--stem", "live", "--final", "none", "--num-speakers", "1",
                "--formats", "txt",
            ]))
            s = LiveSession(args, display=LiveDisplay())
            self.assertEqual(s.session_stem, "live")
            s.segments.append(Segment(0.0, 1.0, 1, "本次内容"))
            s.write_all()
            self.assertIn("本次内容", (out / "live.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
