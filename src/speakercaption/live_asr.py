"""实时链路的 ASR：流式草稿 + 可选句末精修。

分层原因：真正「边听边出字」只能靠原生流式模型（Paraformer），
但它不带标点、没有热词。所以每句说完后再把这句音频送一次离线 ASR 精修，
草稿秒出、精修随后覆盖 —— 这就是原项目 hybrid 的思路，现在叠加说话人标签。

  final=none       只用流式草稿，零额外依赖，延迟最低
  final=whisper    faster-whisper 精修（GPU 上 2~3s 音频 < 1s），默认
  final=confucius  调用户自部署的 7861，中文标点/热词最好，但约 3~5s/句
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import asr as offline_asr


class StreamingASR:
    """sherpa-onnx 流式 Paraformer：持续喂 100ms 块，随时可取当前文字。"""
    BLOCK = 1600  # 100ms @ 16k

    def __init__(self, model_dir, num_threads: int = 2, enabled: bool = True):
        self.enabled = enabled
        self.rec = None
        self.stream = None
        if not enabled:
            return
        import sherpa_onnx
        d = Path(model_dir)
        need = ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"]
        missing = [f for f in need if not (d / f).is_file()]
        if missing:
            raise RuntimeError("流式模型缺文件 %s（目录 %s）。"
                               "运行 python scripts/download_models.py --with-streaming 补齐"
                               % (missing, d))
        self.rec = sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=str(d / need[0]), encoder=str(d / need[1]), decoder=str(d / need[2]),
            num_threads=num_threads, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search",
        )
        self.stream = self.rec.create_stream()

    def feed(self, samples: np.ndarray):
        if not self.enabled or samples.size == 0:
            return
        for i in range(0, len(samples), self.BLOCK):
            self.stream.accept_waveform(16000, samples[i:i + self.BLOCK])
            while self.rec.is_ready(self.stream):
                self.rec.decode_stream(self.stream)

    def text(self) -> str:
        if not self.enabled:
            return ""
        r = self.rec.get_result(self.stream)
        return (r if isinstance(r, str) else (getattr(r, "text", "") or "")).strip()

    def reset(self):
        if self.enabled:
            try:
                self.rec.reset(self.stream)
            except Exception:
                pass


class Finalizer:
    """句末精修。只返回 ASR 原文，简繁/翻译统一交给 TextPipeline。

    transcribe() 在后台线程调用，不阻塞实时采集。
    """
    def __init__(self, mode="whisper", **kw):
        self.mode = mode
        kw.pop("simplify", None)        # legacy flag, kept so old callers don't break
        self.backend = offline_asr.build_asr(mode, **kw)
        self.last_error = None

    @property
    def detected_language(self):
        return getattr(self.backend, "last_language", None)

    def transcribe(self, samples: np.ndarray) -> str:
        try:
            txt = self.backend.transcribe(np.ascontiguousarray(samples, dtype=np.float32), 16000)
            return (txt or "").strip()
        except Exception as e:                 # noqa: BLE001
            self.last_error = e
            return ""

    def close(self):
        fn = getattr(self.backend, "close", None)
        if fn:
            try:
                fn()
            except Exception:
                pass


def build_finalizer(mode: str, args):
    if mode == "none":
        return None
    # task="transcribe" is explicit on purpose: Whisper's translate task would
    # turn everything into English, which is never what a Chinese caption wants.
    kw = dict(task="transcribe")
    if mode == "whisper":
        kw.update(model=args.whisper_model, device=args.whisper_device,
                  compute_type=args.whisper_compute, language=args.language,
                  beam_size=args.beam_size, vad_filter=False,
                  download_root=getattr(args, "download_root", None))
    elif mode == "confucius":
        kw.update(url=args.confucius_url, hot_word=args.hot_word)
    elif mode == "confucius-local":
        # 进程内直调 Confucius，不用先起 7861 服务
        kw.update(confucius_dir=getattr(args, "confucius_dir", None),
                  hot_word=args.hot_word)
    return Finalizer(mode, **kw)
