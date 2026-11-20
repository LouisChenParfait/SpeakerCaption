#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""界面选项的回归测试：Whisper 档位下拉与自定义路径的互不干扰。

背景
----
Whisper 权重有 2.9 GB，进不了 GitHub 仓库，所以「下哪一档」必须由使用者
自己选（README「选哪个档位？」）。界面上给的是「档位下拉 + 下载权重…」，
高级设置里给的是「档位名或本地路径」——两处必须能共存：
选了档位就覆盖路径，选了「自定义路径…」就保留路径。

Run:  python tests/test_gui_options.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speakercaption.live_gui import (   # noqa: E402
    WHISPER, WHISPER_CHOICES, WHISPER_CUSTOM, WHISPER_TIERS)


class TestWhisperTierChoices(unittest.TestCase):
    def test_all_six_tiers_offered(self):
        values = {v for _, v in WHISPER_CHOICES}
        for tier, _size, _desc in WHISPER_TIERS:
            self.assertIn(tier, values, "档位 %s 没出现在下拉里" % tier)

    def test_tiers_carry_size_hint(self):
        """每个档位都要标体积，分享者才知道要下多少。"""
        # 必须按「值」精确匹配：用 startswith 时 large-v3 会误命中
        # large-v3-turbo，再加上 set 无序，这个断言会随机失败。
        by_value = {val: lab for lab, val in WHISPER_CHOICES}
        for tier, size, _desc in WHISPER_TIERS:
            self.assertIn(tier, by_value, "%s 没有标签" % tier)
            self.assertIn(size, by_value[tier], "%s 的标签里没写体积" % tier)

    def test_default_tier_is_large_v3(self):
        """默认档位是 large-v3，不是 tiny（tiny 只是最小的一档）。"""
        # v2l 是「值 -> 标签」，键就是档位名
        self.assertIn("large-v3", WHISPER.v2l.keys())
        self.assertNotEqual(WHISPER.label("large-v3"), WHISPER.label("tiny"))

    def test_label_value_roundtrip(self):
        for tier, _size, _desc in WHISPER_TIERS:
            label = WHISPER.label(tier)
            self.assertEqual(WHISPER.value(label), tier)

    def test_custom_path_entry(self):
        """自定义路径是最后一项，值为 __custom__。"""
        self.assertEqual(WHISPER.value(WHISPER_CUSTOM), "__custom__")

    def test_unknown_value_falls_back_to_first(self):
        """存了一个不认识的档位名时，界面要显示点东西而不是空白。"""
        self.assertTrue(WHISPER.label("distil-large-v3"))


class TestSingleSpeakerDropdown(unittest.TestCase):
    def test_single_choice_label(self):
        from speakercaption.live_gui import COUNT
        self.assertEqual(COUNT.label("1"), "单人（不分人）")
        self.assertEqual(COUNT.value("单人（不分人）"), "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
