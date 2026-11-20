#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WASAPI loopback re-open regression tests.

Bug (2026-11-09): after 停止 → 再开始 the second Pa_OpenStream returned
``OSError: [Errno -9999] Unanticipated host error``. Real-device A/B/C:

    same thread  open→close→open→close→open   all OK
    new thread, same PyAudio instance         -9999
    new thread, fresh PyAudio instance        -9999

So the cause is thread affinity of WASAPI/COM, not device release timing and
not security-software prompts. All PortAudio calls now run on one fixed
``_AudioHost`` thread; these tests assert that invariant with a fake module,
so they need no audio device.

Run:  python tests/test_live_audio.py
"""
from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import speakercaption.live_audio as la  # noqa: E402

HOST_ERR = OSError("[Errno -9999] Unanticipated host error")


class _FakeStream:
    def __init__(self):
        self.started = False
        self.closed = False
        self.open_thread = threading.get_ident()
        self.close_thread = None

    def start_stream(self):
        self.started = True

    def stop_stream(self):
        pass

    def close(self):
        self.close_thread = threading.get_ident()
        self.closed = True


class _FakePA:
    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls = 0
        self.streams = []
        self.constructed_thread = None

    def get_default_wasapi_loopback(self):
        return {"index": 1, "defaultSampleRate": 48000,
                "maxInputChannels": 2, "name": "fake loopback"}

    def open(self, **kw):
        self.calls += 1
        err = self.errors.pop(0) if self.errors else None
        if err is not None:
            raise err
        s = _FakeStream()
        self.streams.append(s)
        return s


def _fake_module(errors=()):
    """Return (fake pyaudiowpatch module, fake PyAudio instance, created list)."""
    mod = types.ModuleType("pyaudiowpatch")
    mod.paInt16 = 8
    mod.paContinue = 0
    pa = _FakePA(errors)
    created = []

    def make():
        pa.constructed_thread = threading.get_ident()
        created.append(pa)
        return pa

    mod.PyAudio = make
    return mod, pa, created


class TestWasapiRetry(unittest.TestCase):
    def setUp(self):
        self._pa_backup = la._PA

    def tearDown(self):
        la._PA = self._pa_backup

    def test_host_error_matches(self):
        self.assertTrue(la._is_host_error(HOST_ERR))
        self.assertTrue(la._is_host_error(OSError("paUnanticipatedHostError")))
        self.assertFalse(la._is_host_error(OSError("device busy")))

    def test_shared_pyaudio_is_singleton(self):
        mod, _pa, created = _fake_module()
        la._PA = None
        with mock.patch.dict(sys.modules, {"pyaudiowpatch": mod}):
            a = la._shared_pyaudio()
            b = la._shared_pyaudio()
        self.assertIs(a, b)
        self.assertEqual(len(created), 1, "PortAudio 被重复初始化")

    def test_host_error_retries_then_succeeds(self):
        mod, pa, _ = _fake_module([HOST_ERR, HOST_ERR])
        with mock.patch.dict(sys.modules, {"pyaudiowpatch": mod}), \
             mock.patch.object(la.time, "sleep", lambda *_: None):
            src = la.AudioSource(backend="wasapi")
            src._pa = pa
            stream = src._open_loopback_stream(1, 48000, 2, lambda *a: None)
        self.assertTrue(stream.started)
        self.assertEqual(pa.calls, 3)

    def test_other_error_raises_immediately(self):
        mod, pa, _ = _fake_module([OSError("device busy")])
        with mock.patch.dict(sys.modules, {"pyaudiowpatch": mod}), \
             mock.patch.object(la.time, "sleep", lambda *_: None):
            src = la.AudioSource(backend="wasapi")
            src._pa = pa
            with self.assertRaises(OSError):
                src._open_loopback_stream(1, 48000, 2, lambda *a: None)
        self.assertEqual(pa.calls, 1)

    def test_two_rounds_reuse_singleton(self):
        """完整 start/stop 两轮：模拟用户「停止 → 再开始」，不开真实设备。"""
        mod, _pa, created = _fake_module()
        la._PA = None
        fake_time = types.SimpleNamespace(time=time.time, sleep=lambda *_: None)
        with mock.patch.dict(sys.modules, {"pyaudiowpatch": mod}), \
             mock.patch.object(la, "time", fake_time):
            for round_no in (1, 2):
                src = la.AudioSource(backend="wasapi")
                src.start()
                deadline = time.time() + 3
                while time.time() < deadline:
                    if src.error or src.stats.snapshot()["device"] != "正在打开音频设备…":
                        break
                    time.sleep(0.01)
                self.assertIsNone(src.error, "第 %d 轮开流失败" % round_no)
                self.assertEqual(src.stats.snapshot()["device"], "fake loopback")
                src.stop()
        self.assertEqual(len(created), 1, "两轮听录重复初始化了 PortAudio")

    def test_all_portaudio_calls_on_one_thread(self):
        """两轮听录：PyAudio 初始化 / open / close 全部落在同一条音频线程。

        这是 -9999 的根因守卫：只要哪天有人把 open/close 挪回每轮新建的
        capture 线程，这条测试就会立刻失败。
        """
        mod, pa, created = _fake_module()
        la._PA = None
        fake_time = types.SimpleNamespace(time=time.time, sleep=lambda *_: None)
        with mock.patch.dict(sys.modules, {"pyaudiowpatch": mod}), \
             mock.patch.object(la, "time", fake_time):
            for round_no in (1, 2):
                src = la.AudioSource(backend="wasapi")
                src.start()
                deadline = time.time() + 3
                while time.time() < deadline:
                    if src.error or src.stats.snapshot()["device"] != "正在打开音频设备…":
                        break
                    time.sleep(0.01)
                self.assertIsNone(src.error, "第 %d 轮开流失败" % round_no)
                src.stop()
        host_tid = la._audio_host().thread.ident
        self.assertEqual(len(pa.streams), 2, "两轮应各开一条流")
        self.assertEqual(len(created), 1, "PortAudio 被重复初始化")
        self.assertEqual(pa.constructed_thread, host_tid,
                         "PyAudio 必须在音频线程里创建")
        for i, s in enumerate(pa.streams, 1):
            self.assertTrue(s.closed, "第 %d 条流没关掉" % i)
            self.assertEqual(s.open_thread, host_tid, "第 %d 条流在别的线程打开" % i)
            self.assertEqual(s.close_thread, host_tid, "第 %d 条流在别的线程关闭" % i)

    def test_list_devices_runs_on_host_thread(self):
        """--list-devices 也是 PortAudio 调用，同样必须在音频线程上。"""
        mod, pa, _ = _fake_module()
        la._PA = None
        fake_sd = types.ModuleType("sounddevice")
        fake_sd.query_devices = lambda: []
        with mock.patch.dict(sys.modules, {"pyaudiowpatch": mod,
                                           "sounddevice": fake_sd}):
            out = la.AudioSource.list_devices()
        self.assertIn("fake loopback", out)
        self.assertEqual(pa.constructed_thread, la._audio_host().thread.ident)

    def test_exhausted_retries_raise(self):
        mod, pa, _ = _fake_module([HOST_ERR] * 4)
        with mock.patch.dict(sys.modules, {"pyaudiowpatch": mod}), \
             mock.patch.object(la.time, "sleep", lambda *_: None):
            src = la.AudioSource(backend="wasapi")
            src._pa = pa
            with self.assertRaises(OSError):
                src._open_loopback_stream(1, 48000, 2, lambda *a: None)
        self.assertEqual(pa.calls, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
