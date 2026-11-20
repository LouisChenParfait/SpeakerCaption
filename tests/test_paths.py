#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""输出路径解析的回归测试。

相对路径必须相对于**工程根目录**解析，而不是进程当前工作目录：
否则换个目录启动（IDE 里点运行、别的脚本调用）结果就写到别处去了。

Run:  python tests/test_paths.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from speakercaption.live_core import ROOT as PKG_ROOT  # noqa: E402
from speakercaption.live_core import _resolve_outdir  # noqa: E402


class TestResolveOutdir(unittest.TestCase):
    def test_relative_path_is_anchored_to_project_root(self):
        self.assertEqual(_resolve_outdir("live_out"), PKG_ROOT / "live_out")
        self.assertEqual(_resolve_outdir("out/sub"), PKG_ROOT / "out" / "sub")

    def test_absolute_path_kept_as_is(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_resolve_outdir(d), Path(d))

    def test_empty_falls_back_to_live_out(self):
        self.assertEqual(_resolve_outdir(""), PKG_ROOT / "live_out")
        self.assertEqual(_resolve_outdir(None), PKG_ROOT / "live_out")

    def test_independent_of_cwd(self):
        """换工作目录不能改变结果路径。"""
        cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            self.assertEqual(_resolve_outdir("live_out"), PKG_ROOT / "live_out")
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
