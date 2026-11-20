"""Speaker diarization wrapper around sherpa-onnx.

Pipeline:  wav -> pyannote segmentation 3.0 -> CAM++ embeddings
                -> fast clustering -> (speaker, start, end) segments

The segmentation model's **fp32** variant is the default: see docs/DEVLOG.md,
where the int8 variant was measured to merge two of four speakers.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEG = ROOT / "models" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
SEG_INT8 = ROOT / "models" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx"
DEFAULT_EMB = ROOT / "models" / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
DEFAULT_VAD = ROOT / "models" / "silero_vad.onnx"


@dataclass
class Segment:
    start: float
    end: float
    speaker: int          # 1-based, renormalised (0/1/3/7 -> 1/2/3/4)
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d


def check_models(segmentation_model=None, embedding_model=None):
    seg = Path(segmentation_model) if segmentation_model else DEFAULT_SEG
    emb = Path(embedding_model) if embedding_model else DEFAULT_EMB
    missing = [str(p) for p in (seg, emb) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing model file(s):\n  " + "\n  ".join(missing) +
            "\nRun: python scripts/download_models.py"
        )
    return seg, emb


def build_vad(model_path=None, *, threshold=0.5, min_silence=0.45,
              min_speech=0.25, max_speech=12.0):
    """构造 silero VAD（只断句，不分人）。

    单人直播 / 单人录屏用这条路径：只需要知道「哪儿是一句话」，
    完全不需要 pyannote 分割模型和 CAM++ 声纹模型。
    """
    import sherpa_onnx

    path = Path(model_path) if model_path else DEFAULT_VAD
    if not path.is_file():
        raise FileNotFoundError(
            "missing VAD model file: %s\nRun: python scripts/download_models.py"
            % path)
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(path)
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = min_silence
    cfg.silero_vad.min_speech_duration = min_speech
    cfg.silero_vad.max_speech_duration = max_speech
    cfg.silero_vad.window_size = 512
    cfg.sample_rate = 16000
    cfg.num_threads = 1
    if not cfg.validate():
        raise RuntimeError("VAD 配置无效")
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)


def vad_segments(audio, sample_rate=16000, vad=None, merge_gap=0.0,
                 max_len: float = None):
    """用 VAD 把音频切成语音段（**不分人**，speaker 恒为 1）。

    单人模式的离线听录用它：只断句、不算声纹，比跑一遍完整的
    pyannote + CAM++ 聚类快一个数量级，也不会出现「同一个人被判成
    两个人」这种误伤。
    """
    import numpy as np

    if vad is None:
        vad = build_vad()
    a = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    step = 1600  # 100ms 一块，和实时链路一致
    for off in range(0, len(a), step):
        vad.accept_waveform(a[off:off + step])
    vad.flush()

    segs = []
    while not vad.empty():
        seg = vad.front
        s = np.ascontiguousarray(np.asarray(seg.samples, dtype=np.float32))
        start = float(getattr(seg, "start", 0)) / 16000.0
        end = start + len(s) / 16000.0
        segs.append(Segment(start, end, 1))
        vad.pop()
    return merge_adjacent(segs, merge_gap, max_len) if merge_gap else segs


def _normalise_labels(segments: List[Segment]) -> List[Segment]:
    """Cluster labels are arbitrary and can be sparse (0,1,3,7...).

    Renumber by first appearance so users always see 说话人1..N.
    """
    mapping = {}
    for s in segments:
        if s.speaker not in mapping:
            mapping[s.speaker] = len(mapping) + 1
        s.speaker = mapping[s.speaker]
    return segments


def merge_adjacent(segments: List[Segment], gap: float = 0.0,
                   max_len: float = None) -> List[Segment]:
    """Join neighbouring segments from the same speaker separated by <= gap.

    Long-term transcripts read better when a speaker's short back-to-back
    turns are transcribed as one ASR request instead of many.

    ``max_len`` 给合并结果封顶：ASR 模型（Whisper 30 秒窗、Paraformer 同量级）
    对超长单段并不友好，广播音频里 VAD 切出的段动辄十几秒，不封顶会合并成
    40 秒以上的巨段，转写质量反而下降。封顶后到长度就另起一段。
    """
    if gap <= 0 or not segments:
        return segments
    out = [Segment(s.start, s.end, s.speaker, s.text) for s in segments]
    merged = [out[0]]
    for s in out[1:]:
        prev = merged[-1]
        too_long = (max_len is not None
                    and (prev.end - prev.start) >= float(max_len))
        if s.speaker == prev.speaker and s.start - prev.end <= gap and not too_long:
            prev.end = s.end
            if s.text:
                prev.text = (prev.text + " " + s.text).strip()
        else:
            merged.append(s)
    return merged


class Diarizer:
    def __init__(
        self,
        segmentation_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        num_speakers: int = -1,
        threshold: float = 0.7,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
        window_shift_ratio: float = 0.1,
    ):
        import sherpa_onnx

        seg, emb = check_models(segmentation_model, embedding_model)
        self.seg_model, self.emb_model = str(seg), str(emb)
        self.num_speakers = num_speakers
        self.threshold = threshold
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=self.seg_model,
                    window_shift_ratio=window_shift_ratio,
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=self.emb_model),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers, threshold=threshold
            ),
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )
        if not cfg.validate():
            raise RuntimeError("invalid diarization config (check model files)")
        self.sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
        self.sample_rate = self.sd.sample_rate

    def process(
        self,
        audio,
        sample_rate: int = 16000,
        merge_gap: float = 0.0,
        progress: bool = False,
        max_len: float = None,
    ) -> List[Segment]:
        import numpy as np

        if sample_rate != self.sample_rate:
            raise ValueError("audio must be %d Hz, got %d" % (self.sample_rate, sample_rate))
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        t0 = time.time()
        state = {"last": -1.0}

        def cb(done, total):
            pct = int(100.0 * done / max(total, 1))
            now = time.time()
            if pct != state["last"] and (pct % 10 == 0 or now - state["last_ts"] > 2):
                sys.stderr.write("\r[diarize] %d%% " % pct)
                sys.stderr.flush()
                state["last"], state["last_ts"] = pct, now
            return 0

        state["last_ts"] = t0
        if progress:
            result = self.sd.process(audio, callback=cb).sort_by_start_time()
            sys.stderr.write("\r[diarize] 100%\n")
        else:
            result = self.sd.process(audio).sort_by_start_time()

        segments = [Segment(float(r.start), float(r.end), int(r.speaker)) for r in result]
        segments = _normalise_labels(segments)
        segments = merge_adjacent(segments, merge_gap, max_len)
        self.elapsed = time.time() - t0
        return segments
