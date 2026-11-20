"""Persistent UI settings and live-changeable runtime options.

GUI choices live in ``settings.json`` next to live.py, so the next launch
starts with the same translation / script / timestamp / backend choices.
Command-line flags still work and take precedence for that launch.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

# Keys exposed in the GUI and persisted. Anything else stays a pure CLI knob.
UI_KEYS = (
    "language", "script", "translate", "timestamps", "final", "num_speakers",
    "whisper_model", "whisper_device", "whisper_compute", "beam_size",
    "download_root",
    "confucius_url", "confucius_dir", "hot_word", "speaker_threshold",
    "speaker_names",
    "outdir", "formats", "duration", "silence_warn", "vad_min_silence",
    "translate_model", "ollama_url", "translate_timeout",
    "no_streaming", "enroll", "save_voices", "simulate", "loop_sim",
    "device", "backend", "media_file", "mode",
)


def load_settings(path) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if k in UI_KEYS}
    except Exception as e:                                # noqa: BLE001
        print("[设置] 读取 %s 失败(%s)，使用默认值" % (p, e), file=sys.stderr)
        return {}


def save_settings(path, data: dict) -> None:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        clean = {k: v for k, v in (data or {}).items() if k in UI_KEYS}
        p.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
    except Exception as e:                                # noqa: BLE001
        print("[设置] 写入 %s 失败: %s" % (p, e), file=sys.stderr)


class LiveOptions:
    """Options that can be flipped while a session is running.

    Only reversible, cheap transforms belong here: Chinese script and
    timestamps. Translation / language / backend / speaker count need a
    restart, so the GUI locks those while capturing.
    """
    def __init__(self, script="simplified", timestamps=False, translate=False):
        self._lock = threading.Lock()
        self.script = script
        self.timestamps = bool(timestamps)
        self.translate = bool(translate)

    def update(self, script=None, timestamps=None, translate=None):
        with self._lock:
            if script is not None:
                self.script = script
            if timestamps is not None:
                self.timestamps = bool(timestamps)
            if translate is not None:
                self.translate = bool(translate)

    def snapshot(self):
        with self._lock:
            return self.script, self.timestamps, self.translate
