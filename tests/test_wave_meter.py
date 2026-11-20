# -*- coding: utf-8 -*-
"""真实音频波形的回归测试。

Bug / 需求 (2026-11-19): 中间的波形动画只吃一个 `set_energy(0..1)` 标量，
柱子形状全由 `sin()` 合成 —— 音量变了只是整体变高变矮，和声音本身无关。
修法：新增 `wave_meter.WaveMeter`，采集回调把真实 PCM 的 min/max 包络推进
环形缓冲，Qt 控件每 40ms 自己拉帧；文件模式画整段波形总览 + 进度播放头。

这些测试不加载任何模型、不开音频设备，只验证「数据确实是真实 PCM」和
「三种模式下的接线」。
"""
import math
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speakercaption import file_mode as fm
from speakercaption.live_audio import AudioSource
from speakercaption.live_core import LiveSession, normalize_args, parse_args
from speakercaption.live_ui import UIController
from speakercaption.settings import UI_KEYS
from speakercaption.wave_meter import (BIN_SEC, CAPACITY, MODE_FILE, MODE_IDLE,
                                       MODE_LIVE, MODE_PREPARING, WaveMeter,
                                       wave_envelope)

SR = 16000


def tone(sec=0.1, amp=0.3, freq=220):
    t = np.arange(int(SR * sec)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class EnvelopeTests(unittest.TestCase):
    def test_sine_envelope_is_bipolar_and_symmetric(self):
        e = wave_envelope(tone(0.2), 10)
        self.assertEqual(e.shape, (10, 2))
        self.assertTrue((e[:, 0] <= 0).all(), "min 列不该有正值")
        self.assertTrue((e[:, 1] >= 0).all(), "max 列不该有负值")
        self.assertAlmostEqual(float(-e[:, 0].min()), float(e[:, 1].max()), places=4)
        self.assertGreater(float(e[:, 1].max()), 0.5)

    def test_silence_is_exactly_flat(self):
        e = wave_envelope(np.zeros(1600, dtype=np.float32), 8)
        self.assertEqual(float(np.abs(e).sum()), 0.0)
        self.assertEqual(e.shape, (8, 2))

    def test_louder_signal_gives_taller_bars_but_same_shape(self):
        quiet = wave_envelope(tone(amp=0.05), 8)
        loud = wave_envelope(tone(amp=0.3), 8)
        self.assertGreater(float(loud[:, 1].max()), float(quiet[:, 1].max()))
        # tanh 是软饱和，不会像硬 clip 那样把大信号削成等高实心块
        self.assertLess(float(loud[:, 1].max()), 0.999)

    def test_short_input_does_not_crash(self):
        e = wave_envelope(np.array([0.5, -0.5, 0.25], dtype=np.float32), 10)
        self.assertEqual(e.shape, (10, 2))
        self.assertGreater(float(e[:, 1].max()), 0.0)
        self.assertLess(float(e[:, 0].min()), 0.0)

    def test_envelope_is_not_synthetic(self):
        """同一段静音 + 一段声音：前半必须完全平，后半必须起伏。"""
        a = np.concatenate([np.zeros(1600, dtype=np.float32), tone(0.1)])
        e = wave_envelope(a, 10)
        self.assertEqual(float(np.abs(e[:5]).sum()), 0.0)
        self.assertGreater(float(np.abs(e[5:]).sum()), 0.0)


class LiveRingTests(unittest.TestCase):
    def test_initial_state_is_idle(self):
        m = WaveMeter()
        self.assertEqual(m.mode, MODE_IDLE)
        self.assertFalse(m.active)

    def test_frame_is_left_padded_and_chronological(self):
        m = WaveMeter()
        m.begin_live()
        m.push(tone(0.1), SR)                 # 0.1s → 5 柱有声音
        f = m.frame(20)
        self.assertEqual(f.bars.shape, (20, 2))
        self.assertEqual(f.mode, MODE_LIVE)
        self.assertTrue(f.active)
        self.assertEqual(float(np.abs(f.bars[:15]).sum()), 0.0, "历史不足要左补零")
        self.assertGreater(float(np.abs(f.bars[15:]).sum()), 0.0)

    def test_begin_live_clears_previous_session(self):
        m = WaveMeter()
        m.begin_live()
        m.push(tone(0.1), SR)
        self.assertGreater(float(np.abs(m.frame(20).bars).sum()), 0.0)
        m.begin_live()
        self.assertEqual(float(np.abs(m.frame(20).bars).sum()), 0.0)

    def test_preparing_has_no_real_bars(self):
        m = WaveMeter()
        m.begin_preparing()
        f = m.frame(10)
        self.assertEqual(f.mode, MODE_PREPARING)
        self.assertTrue(f.active)
        self.assertEqual(float(np.abs(f.bars).sum()), 0.0)

    def test_ring_wraps_without_error(self):
        m = WaveMeter()
        m.begin_live()
        block = tone(0.1)
        for _ in range(int(CAPACITY / 5) + 20):
            m.push(block, SR)
        f = m.frame(110)
        self.assertEqual(f.bars.shape, (110, 2))
        self.assertTrue(np.isfinite(f.bars).all())
        self.assertGreater(float(np.abs(f.bars).sum()), 0.0)

    def test_finish_freezes_but_keeps_mode(self):
        m = WaveMeter()
        m.begin_live()
        m.push(tone(0.1), SR)
        m.finish()
        f = m.frame(10)
        self.assertFalse(f.active)
        self.assertEqual(f.mode, MODE_LIVE)

    def test_level_db_finite_and_tracks_amplitude(self):
        m = WaveMeter()
        m.begin_live()
        m.push(np.zeros(1600, dtype=np.float32), SR)
        silent_db = m.level_db()
        m.push(tone(0.1, amp=0.3), SR)
        loud_db = m.level_db()
        self.assertTrue(math.isfinite(silent_db))
        self.assertGreater(loud_db, silent_db)

    def test_peak_is_pre_compression(self):
        """显示压缩过的值不能反推成峰值，电平必须来自原始采样。"""
        m = WaveMeter()
        m.begin_live()
        m.push(tone(0.1, amp=0.9), SR)
        self.assertGreaterEqual(m.frame(1).peak, 0.85)

    def test_concurrent_push_and_frame(self):
        m = WaveMeter()
        m.begin_live()
        errors = []

        def writer():
            try:
                for _ in range(200):
                    m.push(tone(0.1), SR)
            except Exception as e:            # noqa: BLE001
                errors.append(e)

        ts = [threading.Thread(target=writer) for _ in range(2)]
        for t in ts:
            t.start()
        for _ in range(200):
            f = m.frame(110)
            if f.bars.shape != (110, 2) or not np.isfinite(f.bars).all():
                errors.append(ValueError("frame 损坏"))
        for t in ts:
            t.join(timeout=10)
        self.assertEqual(errors, [])


class FileOverviewTests(unittest.TestCase):
    def test_begin_file_builds_whole_file_waveform(self):
        audio = np.concatenate([np.zeros(8000, dtype=np.float32), tone(0.5)])
        m = WaveMeter()
        m.begin_file(audio, SR)
        f = m.frame(110)
        self.assertEqual(f.mode, MODE_FILE)
        self.assertTrue(f.active)
        self.assertEqual(f.bars.shape, (110, 2))
        self.assertEqual(float(np.abs(f.bars[:50]).sum()), 0.0, "前半静音应平")
        self.assertGreater(float(np.abs(f.bars[60:]).sum()), 0.0)

    def test_progress_clamped_and_kept_after_finish(self):
        m = WaveMeter()
        m.begin_file(tone(1.0), SR)
        m.set_progress(0.42)
        self.assertAlmostEqual(m.frame(10).progress, 0.42)
        m.set_progress(9.9)
        self.assertAlmostEqual(m.frame(10).progress, 1.0)
        m.finish()
        f = m.frame(10)
        self.assertEqual(f.mode, MODE_FILE)
        self.assertFalse(f.active)


class WiringTests(unittest.TestCase):
    def test_audio_source_pushes_real_pcm(self):
        """_put 是三种采集后端共用的唯一入口，必须把原始块送进 meter。"""
        m = WaveMeter()
        m.begin_live()
        src = AudioSource(wave_meter=m)
        block = tone(0.1)
        src._put(block, SR)
        f = m.frame(50)
        self.assertGreater(float(np.abs(f.bars).sum()), 0.0)
        self.assertGreater(f.peak, 0.2)
        # 也要确认音频本身没被可视化旁路吞掉
        raw, sr = src.q.get_nowait()
        self.assertEqual(sr, SR)
        self.assertEqual(len(raw), len(block))

    def test_visualization_failure_never_breaks_capture(self):
        class BrokenMeter:
            def push(self, *a, **kw):
                raise RuntimeError("visualization exploded")

        src = AudioSource(wave_meter=BrokenMeter())
        src._put(tone(0.1), SR)
        self.assertIsNone(src.wave_meter)
        self.assertFalse(src.q.empty())

    def test_controller_owns_meter_and_passes_it_to_session(self):
        seen = []

        class FakeSession:
            def __init__(self, args, display=None, stop=None, wave_meter=None):
                seen.append(wave_meter)
                if wave_meter is not None:
                    wave_meter.begin_live()      # 真会话在开流前会切到 live
                self.session_stem = "live_test"
                self.outdir = Path(args.outdir)
                self.session_formats = ("txt",)

            def run_pipeline(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            defaults = {k: v for k, v in vars(parse_args([])).items() if k in UI_KEYS}
            ctl = UIController(parse_args([]), dict(defaults), defaults,
                               Path(tmp) / "settings.json")
            with patch("speakercaption.live_ui.LiveSession", FakeSession):
                ctl.start(dict(ctl.settings))
                t0 = time.time()
                while ctl.is_running() and time.time() - t0 < 8:
                    time.sleep(0.02)
            self.assertEqual(seen, [ctl.wave_meter])
            self.assertEqual(ctl.wave_meter.mode, MODE_LIVE)
            self.assertFalse(ctl.wave_meter.active, "线程结束必须 finish()")

    def test_controller_passes_meter_to_file_transcriber(self):
        seen = []

        class FakeTranscriber:
            def __init__(self, *a, **kw):
                seen.append(kw)

            def run(self):
                return {"segments": [], "names": {}, "speakers": [1],
                        "elapsed": 0.1, "paths": {}, "single": True}

        with tempfile.TemporaryDirectory() as tmp:
            defaults = {k: v for k, v in vars(parse_args([])).items() if k in UI_KEYS}
            ctl = UIController(parse_args([]), dict(defaults), defaults,
                               Path(tmp) / "settings.json")
            with patch.object(fm, "FileTranscriber", FakeTranscriber):
                ctl.start_file(dict(ctl.settings), str(Path(tmp) / "节目.wav"))
                t0 = time.time()
                while ctl.is_running() and time.time() - t0 < 8:
                    time.sleep(0.02)
            self.assertTrue(seen)
            self.assertIs(seen[0]["wave_meter"], ctl.wave_meter)
            self.assertFalse(ctl.wave_meter.active)

    def test_file_transcriber_feeds_waveform_and_progress(self):
        """文件模式：解码后出整段波形，跑完进度到 100%。"""
        import soundfile as sf

        def fake_segments(*a, **kw):
            from speakercaption.diarize import Segment
            return [Segment(start=0.1, end=0.6, speaker=1, text="")]

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            sf.write(str(wav), tone(1.0), SR)
            m = WaveMeter()
            task = fm.FileTranscriber(wav, Path(tmp) / "out", single_speaker=True,
                                      asr="none", formats=("txt",), wave_meter=m)
            old = fm.vad_segments
            fm.vad_segments = fake_segments
            try:
                task.run()
            finally:
                fm.vad_segments = old
            f = m.frame(110)
            self.assertEqual(f.mode, MODE_FILE)
            self.assertAlmostEqual(f.progress, 1.0)
            self.assertGreater(float(np.abs(f.bars).sum()), 0.0)
            # finish() 由 UIController 的线程 finally 调用，转写器本身只管推数据
            self.assertTrue(m.active)


class EndToEndSimulateTests(unittest.TestCase):
    def test_simulated_capture_fills_real_waveform(self):
        """走完整条 LiveSession → AudioSource 文件回放：波形必须真的有数据。"""
        wav = Path(__file__).resolve().parent / "data" / "1-two-speakers-en.wav"
        if not wav.is_file():
            self.skipTest("缺测试音频 %s" % wav)
        with tempfile.TemporaryDirectory() as tmp:
            args = normalize_args(parse_args([
                "--outdir", str(Path(tmp) / "out"),
                "--final", "none", "--num-speakers", "1",
                # 这个 wav 的第 1 秒几乎静音，第 2 秒才有语音；
                # 跑 2.5s 才能验证「平线 → 起伏」都来自真实采样。
                "--duration", "2.5", "--no-gui", "--simulate", str(wav),
            ]))
            m = WaveMeter()
            m.begin_preparing()
            LiveSession(args, wave_meter=m).run_pipeline()
            f = m.frame(110)
            self.assertEqual(f.mode, MODE_LIVE)
            self.assertFalse(f.active, "run_pipeline 收尾必须 finish()")
            self.assertGreater(float(np.abs(f.bars).sum()), 0.0)
            col = np.abs(f.bars).max(axis=1)
            self.assertGreater(int((col > 0.3).sum()), 5, "说话段要有明显起伏")
            self.assertGreater(int((col < 0.01).sum()), 5, "静音段要真的是平线")


class QtWidgetTests(unittest.TestCase):
    def test_widget_uses_meter_frame(self):
        try:
            from PySide6.QtWidgets import QApplication
            from speakercaption.live_gui_qt import WaveWidget
        except ImportError:
            self.skipTest("PySide6 未安装")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication.instance() or QApplication([])
        m = WaveMeter()
        m.begin_live()
        m.push(tone(0.1), SR)
        w = WaveWidget(meter=m)
        w._tick()
        self.assertIsNotNone(w._bars)
        self.assertEqual(w._bars.shape, (WaveWidget.N, 2))
        self.assertGreater(float(np.abs(w._bars).sum()), 0.0)

        idle = WaveWidget()          # 无 meter：回退旧合成波形，不能崩
        idle.set_energy(0.5)
        idle._tick()
        self.assertIsNone(idle._bars)
        self.assertGreater(idle._energy, 0.0)
        w.deleteLater()
        idle.deleteLater()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
