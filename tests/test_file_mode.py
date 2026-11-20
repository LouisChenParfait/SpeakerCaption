# -*- coding: utf-8 -*-
"""媒体文件听录 + Confucius 本地后端 的回归测试。

这些测试不加载真的大模型，全部用假后端替身，只测「接线对不对」。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speakercaption import asr as asr_mod
from speakercaption.confucius_local import find_project_dir, find_interpreter
from speakercaption.file_mode import MEDIA_SUFFIXES, FileTranscriber


class FakeDiarizer:
    """替身 Diarizer：直接返回预设分段，不加载声纹模型。"""

    def __init__(self, *a, **kw):
        self.elapsed = 0.0
        self.seg_model = "fake-seg.onnx"
        self.emb_model = "fake-emb.onnx"

    def process(self, audio, sr, merge_gap=0.0, progress=False, max_len=None):
        from speakercaption.diarize import Segment
        return [
            Segment(start=0.0, end=1.0, speaker=1, text=""),
            Segment(start=1.5, end=2.5, speaker=2, text=""),
        ]


class ExplodingDiarizer:
    """一旦被实例化就报错：用来证明单人模式**没有**走声纹分离。"""

    def __init__(self, *a, **kw):
        raise AssertionError("单人模式不该加载声纹分离模型")


def fake_vad_segments(*a, **kw):
    """替身 vad_segments：返回两段，其中一段故意带 2 号说话人标签。"""
    from speakercaption.diarize import Segment
    return [
        Segment(start=0.0, end=1.0, speaker=1, text=""),
        Segment(start=1.5, end=2.5, speaker=2, text=""),
    ]


class FakeBackend:
    """替身 ASR：记录收到的参数，直接回固定文本。"""

    def __init__(self, **kw):
        self.kw = kw
        self.calls = []
        self.closed = False

    def transcribe(self, audio, sample_rate=16000):
        self.calls.append((len(audio), sample_rate))
        return "测试文本"

    def close(self):
        self.closed = True


class FakePipeline:
    def __init__(self, **kw):
        self.kw = kw

    def process(self, text):
        return text

    def close(self):
        pass


class FileTranscriberTest(unittest.TestCase):
    """文件听录管线的接线测试。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="sc_filemode_")
        self.outdir = Path(self.tmp.name) / "out"
        self.audio = Path(self.tmp.name) / "a.wav"
        # 写一段 1 秒的静音 wav
        import numpy as np
        import soundfile as sf
        sf.write(str(self.audio), np.zeros(16000, dtype="float32"), 16000)

    def tearDown(self):
        self.tmp.cleanup()

    def _make(self, **kw):
        params = dict(
            path=self.audio, outdir=self.outdir,
            num_speakers=0, single_speaker=False,
            asr="none", formats=("txt", "json"), progress=None,
        )
        params.update(kw)
        return FileTranscriber(**params)

    def test_single_speaker_flattens_labels(self):
        """单人模式：所有段都标成 1 位说话人。"""
        import speakercaption.file_mode as fm
        old = fm.vad_segments
        fm.vad_segments = fake_vad_segments
        try:
            res = self._make(single_speaker=True, asr="none").run()
        finally:
            fm.vad_segments = old
        self.assertEqual(res["speakers"], [1])
        for seg in res["segments"]:
            self.assertEqual(seg.speaker, 1)
        self.assertTrue(res["single"])

    def test_single_speaker_never_loads_voiceprint_model(self):
        """单人模式只断句，不做声纹：不许碰 Diarizer（pyannote + CAM++）。"""
        import speakercaption.file_mode as fm
        old_d, old_v = fm.Diarizer, fm.vad_segments
        fm.Diarizer = ExplodingDiarizer
        fm.vad_segments = fake_vad_segments
        try:
            res = self._make(single_speaker=True, asr="none").run()
        finally:
            fm.Diarizer = old_d
            fm.vad_segments = old_v
        self.assertTrue(res["single"])
        self.assertEqual(len(res["segments"]), 2)

    def test_single_speaker_txt_has_no_speaker_prefix(self):
        """单人模式的 txt / srt 里不能出现「说话人1」。"""
        import speakercaption.file_mode as fm
        old_d, old_v, old_p = fm.Diarizer, fm.vad_segments, fm.build_pipeline
        fm.Diarizer = ExplodingDiarizer
        fm.vad_segments = fake_vad_segments
        fm.build_pipeline = lambda **kw: FakePipeline(**kw)
        try:
            res = self._make(single_speaker=True, asr="none",
                             formats=("txt", "srt", "json")).run()
        finally:
            fm.Diarizer = old_d
            fm.vad_segments = old_v
            fm.build_pipeline = old_p
        txt = Path(res["paths"]["txt"]).read_text(encoding="utf-8")
        srt = Path(res["paths"]["srt"]).read_text(encoding="utf-8")
        self.assertNotIn("说话人", txt)
        self.assertNotIn("说话人", srt)

    def test_diarize_mode_keeps_multiple_speakers(self):
        import speakercaption.file_mode as fm
        old = fm.Diarizer
        fm.Diarizer = FakeDiarizer
        try:
            res = self._make(num_speakers=2, asr="none").run()
        finally:
            fm.Diarizer = old
        self.assertEqual(res["speakers"], [1, 2])
        self.assertFalse(res["single"])

    def test_asr_backend_receives_options(self):
        """ASR 后端要拿到语言/热词等参数，不能丢。"""
        import speakercaption.file_mode as fm
        old_d, old_a, old_p = fm.Diarizer, fm.build_asr, fm.build_pipeline
        fm.Diarizer = FakeDiarizer
        fm.build_pipeline = lambda **kw: FakePipeline(**kw)
        captured = {}

        def fake_build_asr(name, **kw):
            captured["name"] = name
            captured["kw"] = kw
            return FakeBackend(**kw)

        fm.build_asr = fake_build_asr
        try:
            res = self._make(asr="confucius-local", language="zh",
                             hot_word="雷军", confucius_dir="/tmp/x").run()
        finally:
            fm.Diarizer = old_d
            fm.build_asr = old_a
            fm.build_pipeline = old_p

        self.assertEqual(captured["name"], "confucius-local")
        self.assertEqual(captured["kw"]["hot_word"], "雷军")
        self.assertEqual(captured["kw"]["confucius_dir"], "/tmp/x")
        self.assertEqual(captured["kw"]["language"], "zh")
        self.assertEqual(len(res["segments"]), 2)

    def test_outputs_written(self):
        """产物要真的落盘。"""
        import speakercaption.file_mode as fm
        old_d, old_p = fm.Diarizer, fm.build_pipeline
        fm.Diarizer = FakeDiarizer
        fm.build_pipeline = lambda **kw: FakePipeline(**kw)
        try:
            res = self._make(asr="whisper", formats=("txt", "srt", "json")).run()
        finally:
            fm.Diarizer = old_d
            fm.build_pipeline = old_p
        for key in ("txt", "srt", "json"):
            self.assertIn(key, res["paths"])
            self.assertTrue(os.path.isfile(res["paths"][key]), key)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self._make(path=Path(self.tmp.name) / "nope.wav").run()


class ConfuciusBackendTest(unittest.TestCase):
    """Confucius 本地后端的接线。"""

    def test_build_asr_supports_local(self):
        """build_asr 要认 confucius-local。"""
        import speakercaption.confucius_local as cl
        old = cl.build_local_asr
        seen = {}

        def fake(project_dir=None, hot_word="", timeout=300.0):
            seen["dir"] = project_dir
            seen["hot_word"] = hot_word
            return "BACKEND"

        cl.build_local_asr = fake
        try:
            out = asr_mod.build_asr("confucius-local",
                                    confucius_dir="/opt/r2t2",
                                    hot_word="苹果")
        finally:
            cl.build_local_asr = old
        self.assertEqual(out, "BACKEND")
        self.assertEqual(seen["dir"], "/opt/r2t2")
        self.assertEqual(seen["hot_word"], "苹果")

    def test_http_backend_accepts_both_url_kwarg_names(self):
        """实时传 url=、离线传 confucius_url=，两种都要被认。"""
        import speakercaption.asr as am
        old = am.ConfuciusASR
        seen = {}

        class FakeConf:
            def __init__(self, url=None, hot_word=""):
                seen["url"] = url

        am.ConfuciusASR = FakeConf
        try:
            am.build_asr("confucius", url="http://127.0.0.1:9999")
            self.assertEqual(seen["url"], "http://127.0.0.1:9999")
            am.build_asr("confucius", confucius_url="http://127.0.0.1:8888")
            self.assertEqual(seen["url"], "http://127.0.0.1:8888")
        finally:
            am.ConfuciusASR = old

    def test_unknown_backend_raises(self):
        with self.assertRaises(asr_mod.ASRError):
            asr_mod.build_asr("nope-backend")


class ConfuciusLocatorTest(unittest.TestCase):
    """目录/解释器探测。"""

    def test_explicit_dir_must_contain_cy_app(self):
        """显式目录里没有 cy_app 时，不能把那个目录当成 Confucius 项目。"""
        with tempfile.TemporaryDirectory() as d:
            # 空目录 → 不能返回它（要么 None，要么退回到其它候选目录）
            found = find_project_dir(d)
            self.assertNotEqual(str(found or "").rstrip("\\/"), d.rstrip("\\/"))
            # 放个 cy_app.* 就能认出来
            Path(d, "cy_app.cp312-win_amd64.pyd").write_bytes(b"x")
            self.assertEqual(Path(str(find_project_dir(d))).resolve(),
                             Path(d).resolve())

    def test_interpreter_prefers_own_py312(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            own = p / "py312" / "python.exe"
            own.parent.mkdir()
            own.write_text("")
            self.assertEqual(find_interpreter(p), str(own))

    def test_env_var_used(self):
        old = os.environ.get("CONFUCIUS_DIR")
        with tempfile.TemporaryDirectory() as d:
            Path(d, "cy_app.pyd").write_bytes(b"x")
            os.environ["CONFUCIUS_DIR"] = d
            try:
                self.assertIsNotNone(find_project_dir(None))
            finally:
                if old is None:
                    os.environ.pop("CONFUCIUS_DIR", None)
                else:
                    os.environ["CONFUCIUS_DIR"] = old


class MediaSuffixTest(unittest.TestCase):
    def test_common_formats_covered(self):
        for s in (".wav", ".mp3", ".m4a", ".mp4", ".mkv", ".flac"):
            self.assertIn(s, MEDIA_SUFFIXES)


if __name__ == "__main__":
    unittest.main()
