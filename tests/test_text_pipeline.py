#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pure-text tests: no models, no audio device, no network.

    python tests/test_text_pipeline.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speakercaption.live_core import normalize_args, parse_args  # noqa: E402
from speakercaption.settings import LiveOptions  # noqa: E402
from speakercaption.text_pipeline import (  # noqa: E402
    TextPipeline, convert_script, needs_translation, simplify_chinese,
    traditional_chinese,
)

TRAD = "忘記了那些文化革命的記憶"
SIMP = "忘记了那些文化革命的记忆"


class FakeTranslator:
    def __init__(self):
        self.calls = 0

    def translate(self, text, target="简体中文"):
        self.calls += 1
        return "译文：" + text


class ScriptTests(unittest.TestCase):
    def test_t2s(self):
        self.assertEqual(simplify_chinese(TRAD), SIMP)

    def test_s2t(self):
        self.assertEqual(traditional_chinese(SIMP).replace("著", "着"), TRAD)

    def test_asis(self):
        self.assertEqual(convert_script(TRAD, "as-is"), TRAD)

    def test_empty(self):
        self.assertEqual(convert_script("", "simplified"), "")


class TranslationTests(unittest.TestCase):
    def test_heuristic(self):
        self.assertTrue(needs_translation("The new generation has arrived"))
        self.assertFalse(needs_translation("这是一个测试说话人日志的音频"))
        self.assertFalse(needs_translation(""))

    def test_off_by_default(self):
        t = FakeTranslator()
        p = TextPipeline(translate=False, translator=t)
        self.assertEqual(p.process("The new generation has arrived"),
                         "The new generation has arrived")
        self.assertEqual(t.calls, 0)

    def test_cached_and_only_for_foreign_text(self):
        t = FakeTranslator()
        p = TextPipeline(translate=True, script="simplified", translator=t)
        out = p.process("The new generation has arrived")
        self.assertTrue(out.startswith("译文："))
        p.process("The new generation has arrived")   # cache hit
        p.process("这是一个测试")                     # pure Chinese -> skip
        self.assertEqual(t.calls, 1)

    def test_script_change_is_reversible_without_retranslation(self):
        t = FakeTranslator()
        p = TextPipeline(translate=True, translator=t)
        semantic = p.prepare("这是繁體中文")
        self.assertEqual(p.render(semantic, "simplified"), "这是繁体中文")
        self.assertEqual(p.render(semantic, "traditional"), "這是繁體中文")
        self.assertEqual(t.calls, 0)                   # pure Chinese, never translated


class OptionsTests(unittest.TestCase):
    def test_live_options(self):
        o = LiveOptions(script="simplified", timestamps=False)
        self.assertEqual(o.snapshot(), ("simplified", False, False))
        o.update(script="traditional", timestamps=True)
        self.assertEqual(o.snapshot(), ("traditional", True, False))

    def test_language_default_is_auto(self):
        args = normalize_args(parse_args([]))
        self.assertIsNone(args.language)
        self.assertEqual(args.script, "simplified")
        self.assertFalse(args.translate)
        self.assertFalse(args.timestamps)

    def test_traditional_alias(self):
        args = normalize_args(parse_args(["--traditional"]))
        self.assertEqual(args.script, "traditional")

    def test_translate_forces_auto_language(self):
        args = normalize_args(parse_args(["--translate", "--language", "zh"]))
        self.assertTrue(args.translate)
        self.assertIsNone(args.language)


if __name__ == "__main__":
    unittest.main(verbosity=2)
