"""Audio loading / resampling helpers.

Only numpy + soundfile are required. Higher-quality resamplers are used when
available (soxr > scipy > linear interpolation), so the dependency footprint
stays small on a fresh machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    frames: int
    duration: float
    subtype: str


def probe(path) -> AudioInfo:
    info = sf.info(str(path))
    return AudioInfo(
        path=str(path),
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        duration=info.duration,
        subtype=info.subtype,
    )


def _resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return data.astype(np.float32, copy=False)
    # 1) soxr: highest quality, tiny package
    try:
        import soxr

        return soxr.resample(data, src_sr, dst_sr).astype(np.float32)
    except ImportError:
        pass
    # 2) scipy polyphase
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(src_sr, dst_sr)
        return resample_poly(data, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        pass
    # 3) pure numpy linear interpolation (no extra dependency)
    n_out = int(round(len(data) * dst_sr / float(src_sr)))
    if n_out < 1:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(len(data), dtype=np.float64) / src_sr
    x_new = np.arange(n_out, dtype=np.float64) / dst_sr
    return np.interp(x_new, x_old, data).astype(np.float32)


def load_mono(path, target_sr: int = 16000):
    """Return (float32 mono samples, sample_rate). Always at target_sr."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    data = np.ascontiguousarray(data, dtype=np.float32)
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
        sr = target_sr
    return data, sr


def slice_audio(audio: np.ndarray, sr: int, start: float, end: float,
                pad: float = 0.0) -> np.ndarray:
    """Cut [start-pad, end+pad] seconds, clipped to the buffer."""
    a = max(0, int((start - pad) * sr))
    b = min(len(audio), int((end + pad) * sr))
    if b <= a:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(audio[a:b], dtype=np.float32)


def write_wav(path, audio: np.ndarray, sr: int = 16000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def load_any(path, target_sr: int = 16000):
    """Load any audio/video container by falling back to ffmpeg.

    soundfile handles wav/flac/ogg directly; mp3/m4a/mp4 and video files are
    decoded through ffmpeg into a temporary 16k mono wav first.
    """
    import shutil
    import subprocess
    import tempfile

    try:
        return load_mono(path, target_sr)
    except Exception as first:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError(
                "cannot decode %s (%s) and ffmpeg is not on PATH" % (path, first)
            ) from first
        with tempfile.TemporaryDirectory(prefix="speaker_caption_") as td:
            wav = Path(td) / "decoded.wav"
            cmd = [exe, "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
                   "-vn", "-ac", "1", "-ar", str(target_sr),
                   "-c:a", "pcm_s16le", str(wav)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0 or not wav.is_file():
                raise RuntimeError("ffmpeg failed: %s" % (p.stderr or "").strip()[:300])
            return load_mono(wav, target_sr)
