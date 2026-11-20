#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在线分人（OnlineSpeakerTracker）回归测试。

重点复现一个真实踩过的坑
------------------------
广播/直播音频里 VAD 切出来的段特别长（实测 7~19 秒，因为电台几乎不停顿）。
旧实现用**整段**算一个 192 维声纹，一段里换了人就等于把几个人平均成一个
向量，后面所有人都被判成「离这个中心最近的」→ 整场只剩【说话人1】。

这里用一个假声纹提取器把这件事复现出来：它对长段做「多人方向平均」，
和真 CAM++ 的行为一致；切窗后每窗只有一个方向，测试就能验证新实现
是否真的把长段里的换人区分开了。

Run:  python tests/test_speaker_tracker.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from speakercaption.live_speaker import OnlineSpeakerTracker  # noqa: E402

# 两个「音色」方向，故意不是正交：余弦约 0.28，接近真人跨人相似度
_A = np.array([1.0, 0.3, 0.0, 0.0], dtype=np.float32)
_B = np.array([0.3, 1.0, 0.0, 0.0], dtype=np.float32)


class _FakeStream:
    def __init__(self):
        self.samples = None

    def accept_waveform(self, sample_rate, samples):
        self.samples = samples

    def input_finished(self):
        pass


class FakeExtractor:
    """假声纹提取器：按 3 秒子窗判音色，长段返回「多人方向平均」。

    这正是真 CAM++ 遇到「一段里换了人」时的行为：整段一个向量，
    谁都不像，谁都不像到能分开。
    """

    def __init__(self, dim=4):
        self.dim = dim

    def create_stream(self):
        return _FakeStream()

    def is_ready(self, stream):
        return stream.samples is not None and len(stream.samples) > 0

    def compute(self, stream):
        x = np.asarray(stream.samples, dtype=np.float32)
        win = 48000  # 3 秒
        acc = np.zeros(self.dim, dtype=np.float32)
        n = 0
        if len(x) >= win:
            for off in range(0, len(x) - win + 1, win // 2):
                w = x[off:off + win]
                acc += _A if (w > 0).mean() >= 0.5 else _B
                n += 1
        if n == 0:
            v = _A if (x > 0).mean() >= 0.5 else _B
            return v
        v = acc / n
        return v / (np.linalg.norm(v) + 1e-9)


def voice(seconds, speaker):
    """造一段「某人说话」的假音频：正数=甲，负数=乙。"""
    n = int(seconds * 16000)
    return np.full(n, 0.5 if speaker == "A" else -0.5, dtype=np.float32)


def mixed(first_sec, first_who, second_sec, second_who):
    """造一段「前脚甲、后脚乙」的音频：长段里换了人。"""
    return np.concatenate([voice(first_sec, first_who),
                            voice(second_sec, second_who)])


class TestWindowing(unittest.TestCase):
    """长段里的换人必须能分开。"""

    def setUp(self):
        self.t = OnlineSpeakerTracker(FakeExtractor(), max_speakers=4,
                                      threshold=0.60)

    def test_long_segment_with_speaker_change(self):
        """甲 12 秒 → 段内换成乙 → 乙 9 秒：应判成 1 / 2 / 2。

        第二段特意做成「前 3 秒甲、后 9 秒乙」：旧实现用整段算一个声纹，
        会把两个人平均成一个谁都不像的向量，然后错误地归给甲。
        """
        segs = [voice(12, "A"), mixed(3, "A", 9, "B"), voice(9, "B")]
        labels = []
        for seg in segs:
            label, _score, _new = self.t.assign(seg)
            labels.append(label)
        self.assertEqual(labels, [1, 2, 2],
                         "长段里换了人必须拆成两个说话人，而不是全是说话人1")

    def test_mixed_segment_leans_to_new_speaker(self):
        """一段里多数是乙时，整段应判给乙（而不是先出现的那个人）。"""
        self.t.assign(voice(12, "A"))
        label, _score, _new = self.t.assign(mixed(2, "A", 10, "B"))
        self.assertEqual(label, 2)

    def test_same_speaker_stays_one(self):
        """同一个人连续说：不能乱拆。"""
        labels = []
        for _ in range(4):
            label, _score, _new = self.t.assign(voice(8, "A"))
            labels.append(label)
        self.assertEqual(set(labels), {1})
        self.assertEqual(self.t.num_speakers(), 1)

    def test_two_speakers_alternating(self):
        """两人交替发言：各自稳定归到自己的标签。"""
        labels = []
        for who in ("A", "B", "A", "B", "A", "B"):
            label, _score, _new = self.t.assign(voice(6, who))
            labels.append(label)
        self.assertEqual(labels, [1, 2, 1, 2, 1, 2])
        self.assertEqual(self.t.num_speakers(), 2)

    def test_short_segment_inherits_previous_label(self):
        """太短的句子（<1 秒）没有可靠声纹，沿用上一句的标签。"""
        self.t.assign(voice(6, "A"))
        label, score, is_new = self.t.assign(voice(0.5, "B"))
        self.assertEqual(label, 1)
        self.assertEqual(score, -1.0)
        self.assertFalse(is_new)

    def test_max_speakers_is_respected(self):
        """限定 2 人时不能冒出第 3 个人。"""
        t = OnlineSpeakerTracker(FakeExtractor(), max_speakers=2, threshold=0.60)
        for who in ("A", "B", "C"):
            t.assign(voice(6, who))
        self.assertLessEqual(t.num_speakers(), 2)


class TestPrototypeBank(unittest.TestCase):
    """原型库：同一个人不同音色也要能归到同一人。"""

    def test_multiple_prototypes_per_speaker(self):
        t = OnlineSpeakerTracker(FakeExtractor(), max_speakers=4, threshold=0.60)
        t.assign(voice(6, "A"))
        n_before = len(t.protos.get(1, []))
        t.assign(voice(6, "A"))
        self.assertGreaterEqual(len(t.protos.get(1, [])), n_before)

    def test_centroids_view(self):
        """centroids 属性仍然可用（save/load 与外部引用依赖它）。"""
        t = OnlineSpeakerTracker(FakeExtractor(), max_speakers=4, threshold=0.60)
        t.assign(voice(6, "A"))
        self.assertIn(1, t.centroids)
        self.assertEqual(t.centroids[1].shape, (4,))

    def test_save_load_roundtrip(self):
        import json
        import tempfile
        t = OnlineSpeakerTracker(FakeExtractor(), max_speakers=4, threshold=0.60)
        t.assign(voice(6, "A"))
        t.assign(voice(6, "B"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "voices.json"
            t.save(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(len(data["speakers"]), 2)
            t2 = OnlineSpeakerTracker(FakeExtractor(), max_speakers=4, threshold=0.60)
            self.assertEqual(t2.load(p), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
