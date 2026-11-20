"""媒体文件听录：把离线版的整段管线包成一个可被 GUI 调用的任务。

为什么单独一个模块
------------------
实时听录是「增量、断句、逐句出字」，媒体文件听录是「拿到整段音频再一次性
处理」。两者共用同一批零件（声纹分人 / ASR / 简繁 / 翻译 / 写盘），但流程
不同，所以离线管线抽到这里，GUI 的「听录文件」按钮直接调它。

流程
----
    选文件 → ffmpeg/soundfile 解码 → 16k 单声道
      → 说话人分段（pyannote segmentation + CAM++ 声纹聚类）
      → 每段切音频 → ASR（whisper / confucius-local）
      → 简繁转换 / 可选翻译 → 写 txt/srt/json/rttm
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .asr import ASRError, build_asr
from .audio import load_any, probe, slice_audio
from .diarize import Diarizer, build_vad, merge_adjacent, vad_segments
from .outputs import speaker_name, write_outputs
from .text_pipeline import build_pipeline

ROOT = Path(__file__).resolve().parents[2]

# 支持的输入类型（ffmpeg 能解的都行）
MEDIA_SUFFIXES = [
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".ts", ".m4v",
]


class FileProgress:
    """文件听录进度回调（GUI 用它更新状态条）。"""

    def __call__(self, stage, detail=""):
        pass


class NullProgress(FileProgress):
    def __call__(self, stage, detail=""):
        pass


class FileTranscriber:
    """媒体文件听录任务。

    参数与离线版 CLI 对齐，但用 GUI 的选项语义（单人是 num_speakers==1）。
    """

    def __init__(self, path, outdir, *, num_speakers=0, single_speaker=False,
                 asr="whisper", language=None, script="simplified",
                 translate=False, translate_model="auto", ollama_url=None,
                 hot_word="", confucius_dir=None, whisper_model="large-v3",
                 whisper_device="auto", whisper_compute="auto", beam_size=5,
                 threshold=0.7, merge_gap=0.3, timestamps=False,
                 formats=("txt", "srt", "json", "rttm"), speaker_names="",
                 progress=None, stem=None, models_dir=None, wave_meter=None,
                 vad_threshold=0.5, vad_min_silence=0.45, vad_min_speech=0.25,
                 vad_max_speech=12.0, max_merge_sec=20.0):
        self.path = Path(path)
        self.outdir = Path(outdir)
        self.num_speakers = int(num_speakers or 0)
        self.single = bool(single_speaker or self.num_speakers == 1)
        self.asr = asr
        self.language = language
        self.script = script
        self.translate = bool(translate)
        self.translate_model = translate_model
        self.ollama_url = ollama_url
        self.hot_word = hot_word or ""
        self.confucius_dir = confucius_dir
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self.whisper_compute = whisper_compute
        self.beam_size = beam_size
        self.threshold = threshold
        self.merge_gap = merge_gap
        self.timestamps = bool(timestamps)
        self.formats = [f.strip().lower() for f in formats if f.strip()]
        self.speaker_names = speaker_names or ""
        self.progress = progress or NullProgress()
        self.stem = stem or self.path.stem
        self.models_dir = Path(models_dir) if models_dir else (ROOT / "models")
        # GUI 的真实波形总线：文件没有实时音频流，改用整段波形总览 + 进度播放头。
        self.wave_meter = wave_meter
        self.vad_threshold = float(vad_threshold)
        self.vad_min_silence = float(vad_min_silence)
        self.vad_min_speech = float(vad_min_speech)
        self.vad_max_speech = float(vad_max_speech)
        # 合并后的单段上限：ASR 模型对超长单段不友好（Whisper 30 秒窗）
        self.max_merge_sec = float(max_merge_sec) if max_merge_sec else None

    def _names(self):
        out = {}
        for part in self.speaker_names.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                out[int(k.strip())] = v.strip()
            except ValueError:
                pass
        return out

    def run(self):
        """执行听录。返回结果字典。"""
        t_all = time.time()
        names = self._names()

        if not self.path.is_file():
            raise FileNotFoundError("文件不存在: %s" % self.path)

        self.progress("probe", "读取文件信息")
        info = probe(self.path)
        self.progress("probe", "%.1f 秒 / %d Hz / %d ch"
                      % (info.duration, info.sample_rate, info.channels))

        self.progress("decode", "解码音频")
        audio, sr = load_any(self.path, target_sr=16000)
        duration = len(audio) / sr
        if audio.size == 0:
            raise ValueError("解码后没有音频数据")
        if self.wave_meter is not None:
            # 解码完就有整段 PCM 了，先算总览：文件模式看到的是这个文件的
            # 真实波形，不是按进度生成的装饰动画。
            self.wave_meter.begin_file(audio, sr)

        # 断句 + 声纹聚类
        if self.single:
            # 单人模式：只断句，**完全不做声纹**。
            # 既省掉 pyannote + CAM++ 两个模型的加载与推理（快一个数量级），
            # 也不会出现「明明选了单人，窗口里却还挂着【说话人1】」这种自相矛盾。
            self.progress("diarize", "单人模式：只断句，不做声纹")
            vad = build_vad(
                self.models_dir / "silero_vad.onnx",
                threshold=self.vad_threshold,
                min_silence=self.vad_min_silence,
                min_speech=self.vad_min_speech,
                max_speech=self.vad_max_speech,
            )
            segments = vad_segments(audio, sr, vad=vad, merge_gap=self.merge_gap,
                                    max_len=self.max_merge_sec)
            for s in segments:
                s.speaker = 1
        else:
            self.progress("diarize", "分离说话人")
            dia = Diarizer(num_speakers=-1 if self.num_speakers <= 0 else self.num_speakers,
                           threshold=self.threshold)
            segments = dia.process(audio, sr, merge_gap=self.merge_gap,
                                   max_len=self.max_merge_sec)
        speakers = sorted({s.speaker for s in segments})
        self.progress("diarize", "%d 段 / %d 位说话人" % (len(segments), len(speakers)))

        # ASR
        pipeline = build_pipeline(translate=self.translate, script=self.script,
                                  ollama_url=self.ollama_url,
                                  translate_model=self.translate_model)
        backend = None
        texts = []
        try:
            if self.asr and self.asr != "none":
                self.progress("asr", "加载 ASR 后端")
                backend = build_asr(
                    self.asr,
                    whisper_model=self.whisper_model,
                    whisper_device=self.whisper_device,
                    whisper_compute=self.whisper_compute,
                    language=self.language,
                    task="transcribe",
                    beam_size=self.beam_size,
                    confucius_dir=self.confucius_dir,
                    hot_word=self.hot_word,
                )
                for i, s in enumerate(segments, 1):
                    if s.speaker is None:
                        continue
                    chunk = slice_audio(audio, sr, s.start, s.end, pad=0.10)
                    try:
                        s.text = pipeline.process(backend.transcribe(chunk, sr))
                    except Exception as e:                        # noqa: BLE001
                        s.text = ""
                        self.progress("asr", "第 %d 段失败: %s" % (i, e))
                    texts.append(s.text)
                    if self.wave_meter is not None:
                        # 播放头按「段结束时间 / 总时长」推进，而不是按段数，
                        # 这样长段、静音间隙在波形上的位置和真实音频对得上。
                        self.wave_meter.set_progress(s.end / max(duration, 1e-6))
                    if i % 5 == 0 or i == len(segments):
                        self.progress("asr", "已转写 %d/%d 段" % (i, len(segments)))
        finally:
            if backend is not None and hasattr(backend, "close"):
                try:
                    backend.close()
                except Exception:
                    pass
            pipeline.close()

        # 写盘
        self.progress("write", "写出文件")
        meta = {
            "source": str(self.path),
            "duration_seconds": round(duration, 3),
            "sample_rate": sr,
            "num_speakers_detected": len(speakers),
            "num_speakers_requested": self.num_speakers,
            "speaker_mode": "single" if self.single else "diarize",
            "clustering_threshold": self.threshold,
            "asr_backend": self.asr,
            "language": self.language or "auto",
            "script": self.script,
            "translate": self.translate,
            "mode": "file",
        }
        self.outdir.mkdir(parents=True, exist_ok=True)
        paths = write_outputs(segments, self.outdir, self.stem, names=names,
                              timestamps=self.timestamps, meta=meta,
                              formats=self.formats, show_speaker=not self.single)
        elapsed = time.time() - t_all
        if self.wave_meter is not None:
            self.wave_meter.set_progress(1.0)
        if self.single:
            self.progress("done", "完成：%d 段 / 单人模式 / %.1f 秒"
                          % (len(segments), elapsed))
        else:
            self.progress("done", "完成：%d 段 / %d 位说话人 / %.1f 秒"
                          % (len(segments), len(speakers), elapsed))
        return {
            "segments": segments,
            "speakers": speakers,
            "names": names,
            "paths": paths,
            "duration": duration,
            "elapsed": elapsed,
            "single": self.single,
        }
