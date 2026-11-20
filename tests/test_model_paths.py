# -*- coding: utf-8 -*-
"""模型来源解析的回归测试。

这些用例锁住三件事：
  1. 解析优先级：显式路径 > 工程内 models/whisper > HF 缓存 > 在线下载
  2. 工程内模型存在时**不发网络请求**（离线可用）
  3. build_asr 的键名兼容：实时传 model=，离线传 whisper_model=，都要认
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speakercaption.model_paths import (  # noqa: E402
    WHISPER_HOME, find_in_hf_cache, find_local, is_model_dir,
    resolve_whisper, size_of,
)


def make_model_dir(path: Path) -> Path:
    """造一个最小的 faster-whisper 模型目录。"""
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.bin").write_bytes(b"x" * 16)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


class TestIsModelDir(unittest.TestCase):
    def test_complete_dir_is_model(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_model_dir(Path(d) / "m")
            self.assertTrue(is_model_dir(p))

    def test_missing_model_bin_is_not_model(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m"
            p.mkdir()
            (p / "config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(is_model_dir(p))

    def test_missing_config_is_not_model(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m"
            p.mkdir()
            (p / "model.bin").write_bytes(b"x")
            self.assertFalse(is_model_dir(p))

    def test_nonexistent_dir(self):
        self.assertFalse(is_model_dir(Path("/definitely/not/here")))


class TestResolve(unittest.TestCase):
    def test_explicit_path_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_model_dir(Path(d) / "explicit")
            src, origin = resolve_whisper(str(p))
            self.assertEqual(origin, "local")
            self.assertEqual(Path(src).resolve(), p.resolve())

    def test_explicit_missing_path_is_not_silently_replaced(self):
        """用户填了路径但目录不存在 -> 报错，绝不偷偷换成别的模型。"""
        src, origin = resolve_whisper("/no/such/model/dir")
        self.assertIsNone(src)
        self.assertEqual(origin, "missing")

    def test_download_root_directory_itself(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_model_dir(Path(d))
            src, origin = resolve_whisper("large-v3", download_root=str(p))
            self.assertEqual(origin, "local")
            self.assertEqual(Path(src).resolve(), p.resolve())

    def test_download_root_plus_tier(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_model_dir(root / "small")
            src, origin = resolve_whisper("small", download_root=str(root))
            self.assertEqual(origin, "local")
            self.assertTrue(src.endswith("small"))

    def test_project_local_dir(self):
        """工程内 models/whisper/<档位>/ 优先于一切。"""
        target = WHISPER_HOME / "_test_tier"
        try:
            make_model_dir(target)
            src, origin = resolve_whisper("_test_tier")
            self.assertEqual(origin, "local")
            self.assertEqual(Path(src).resolve(), target.resolve())
        finally:
            if target.exists():
                for f in target.iterdir():
                    f.unlink()
                target.rmdir()

    def test_project_local_beats_hf_cache(self):
        """同名模型同时存在于工程内和 HF 缓存时，工程内优先。"""
        target = WHISPER_HOME / "_test_tier2"
        try:
            make_model_dir(target)
            src, origin = resolve_whisper("_test_tier2")
            self.assertEqual(origin, "local")
            self.assertEqual(Path(src).resolve(), target.resolve())
        finally:
            if target.exists():
                for f in target.iterdir():
                    f.unlink()
                target.rmdir()

    def test_unknown_falls_back_to_download(self):
        src, origin = resolve_whisper("_definitely_not_a_real_tier_xyz")
        self.assertEqual(origin, "download")
        self.assertEqual(src, "_definitely_not_a_real_tier_xyz")

    def test_default_is_large_v3(self):
        """留空时默认 large-v3；本机有没有缓存都会落到 large-v3 这档。"""
        src, origin = resolve_whisper("")
        self.assertIn(origin, ("download", "hf-cache", "local"))
        # 若本机有缓存，src 是路径；否则是档位名。两者都必须指向 large-v3。
        self.assertIn("large-v3", src)

class TestHelpers(unittest.TestCase):
    def test_find_local_with_path_input(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_model_dir(Path(d) / "m")
            hit = find_local(str(p))
            self.assertIsNotNone(hit)
            self.assertEqual(hit.resolve(), p.resolve())

    def test_find_local_rejects_non_model_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(find_local(str(Path(d))))

    def test_find_in_hf_cache_ignores_paths(self):
        self.assertIsNone(find_in_hf_cache("/some/path"))

    def test_size_of_known_tier(self):
        self.assertIn("GB", size_of("large-v3"))
        self.assertIn("MB", size_of("tiny"))
        self.assertIn("MB", size_of("small"))

    def test_size_of_unknown(self):
        self.assertEqual(size_of("not-a-tier"), "?")


class TestBuildAsrKwargCompat(unittest.TestCase):
    """实时传 model=/device=，离线传 whisper_model=，两者都要被认。"""

    def setUp(self):
        from speakercaption import asr
        self.asr = asr
        self.captured = {}

        class FakeWhisper:
            def __init__(self, model, device, compute_type, **kw):
                FakeWhisper.last = dict(model=model, device=device,
                                         compute_type=compute_type, **kw)

        self._orig = asr.WhisperASR
        asr.WhisperASR = FakeWhisper

    def tearDown(self):
        self.asr.WhisperASR = self._orig

    def test_live_style_kwargs(self):
        self.asr.build_asr("whisper", model="tiny", device="cpu",
                           compute_type="int8", language=None)
        last = self.asr.WhisperASR.last
        self.assertEqual(last["model"], "tiny")
        self.assertEqual(last["device"], "cpu")
        self.assertEqual(last["compute_type"], "int8")

    def test_cli_style_kwargs(self):
        self.asr.build_asr("whisper", whisper_model="medium",
                           whisper_device="cuda", whisper_compute="float16")
        last = self.asr.WhisperASR.last
        self.assertEqual(last["model"], "medium")
        self.assertEqual(last["device"], "cuda")
        self.assertEqual(last["compute_type"], "float16")

    def test_long_names_win_when_both_present(self):
        self.asr.build_asr("whisper", model="short", whisper_model="long")
        self.assertEqual(self.asr.WhisperASR.last["model"], "long")

    def test_download_root_passthrough(self):
        self.asr.build_asr("whisper", model="tiny", download_root="/tmp/x")
        self.assertEqual(self.asr.WhisperASR.last["download_root"], "/tmp/x")


class TestEnvOverride(unittest.TestCase):
    def test_hf_home_respected(self):
        """HF_HOME 改了，缓存目录查询要跟着走。"""
        from speakercaption import model_paths
        old = os.environ.get("HF_HOME")
        try:
            with tempfile.TemporaryDirectory() as d:
                os.environ["HF_HOME"] = d
                self.assertEqual(model_paths.hf_cache_dir(),
                                 Path(d) / "hub")
        finally:
            if old is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = old


if __name__ == "__main__":
    unittest.main()
