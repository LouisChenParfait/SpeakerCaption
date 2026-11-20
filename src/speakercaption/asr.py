"""Pluggable ASR backends.

Design rule for the open-source build: **nothing here depends on a private
binary, a compiled module, or a hard-coded local service.** The Confucius
backend is only an HTTP client for a service the user deploys themselves; it
ships no weights and no vendor code.

    --asr none        diarization only (rttm/json with empty text)
    --asr whisper     faster-whisper, MIT, works everywhere (default)
    --asr confucius   optional high-quality Chinese backend via Gradio HTTP
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Some Windows setups push even 127.0.0.1 through HTTP_PROXY, which makes
# requests to a local Gradio server return 404. Belt-and-braces safeguard.
for _k in ("NO_PROXY", "no_proxy"):
    _hosts = "127.0.0.1,localhost,::1"
    _cur = os.environ.get(_k, "")
    if _hosts not in _cur:
        os.environ[_k] = (_cur + "," + _hosts).strip(",")


class ASRError(RuntimeError):
    pass


from .text_pipeline import (  # noqa: F401
    convert_script, simplify_chinese, traditional_chinese,
)
from .model_paths import (  # noqa: F401
    WHISPER_HOME, resolve_whisper, size_of,
)


class NullASR:
    name = "none"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        return ""


class WhisperASR:
    name = "whisper"

    def __init__(self, model="large-v3", device="auto", compute_type="auto",
                 language=None, beam_size=5, vad_filter=False, download_root=None,
                 task="transcribe", initial_prompt=None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ASRError("faster-whisper not installed: pip install faster-whisper") from e

        # --- 模型来源解析 -------------------------------------------------
        # 三条路，优先级固定：
        #   1. 用户填的显式路径 / download_root 指定的目录
        #   2. 工程内 models/whisper/<档位>       ← clone 后下载一次即可离线用
        #   3. HuggingFace 自动下载               ← 兜底，可用 hf-mirror 镜像
        source, origin = resolve_whisper(model, download_root)
        if source is None:
            raise ASRError(
                "找不到 whisper 模型：%s\n"
                "  · 若你填的是路径，请确认目录里有 model.bin 和 config.json\n"
                "  · 或运行：python scripts/download_models.py --with-whisper" % model)
        if origin == "local":
            print("[asr] whisper 模型（本地目录）: %s" % source)
        elif origin == "hf-cache":
            print("[asr] whisper 模型（HF 缓存）: %s" % source)
        else:
            print("[asr] whisper 模型即将下载: %s（约 %s）"
                  % (source, size_of(source)))
            print("[asr] 国内网络可先设 HF_ENDPOINT=https://hf-mirror.com")
        # 下载目标：优先工程内 models/whisper/<档位>，这样权重跟着工程走，
        # 换机器时拷 models/ 目录即可，不用重新下 2.9 GB。
        if origin == "download" and not download_root:
            download_root = str(WHISPER_HOME / source)

        compute = {
            "auto": None,
            "cuda": "float16",
            "cpu": "int8",
        }.get(compute_type, compute_type)
        dev = device
        if dev == "auto":
            # ctranslate2 is a hard dependency of faster-whisper, so probing it
            # avoids pulling in torch (2+ GB) just to detect a GPU.
            try:
                import ctranslate2
                dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                try:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    dev = "cpu"
        if compute is None:
            compute = "float16" if dev == "cuda" else "int8"

        self.device, self.compute = dev, compute
        kw = {"device": dev, "compute_type": compute}
        if download_root:
            kw["download_root"] = str(download_root)
        try:
            self.model = WhisperModel(source, **kw)
        except Exception as e:
            # 下载失败是最常见的新用户问题：给一条能照着做的指引，
            # 而不是把 huggingface_hub 的原始异常直接甩到脸上。
            msg = str(e)
            if "LocalEntryNotFoundError" in msg or "not found in the local cache" in msg \
                    or "offline" in msg.lower() or "Connection" in msg:
                raise ASRError(
                    "Whisper 模型 %s 拿不到：%s\n"
                    "  解决办法（任选其一）：\n"
                    "   1) 手动下载到工程内（之后完全离线可用）：\n"
                    "      python scripts/download_models.py --with-whisper %s\n"
                    "   2) 国内网络走镜像：\n"
                    "      set HF_ENDPOINT=https://hf-mirror.com   (Windows)\n"
                    "      export HF_ENDPOINT=https://hf-mirror.com  (macOS/Linux)\n"
                    "   3) 换小档位：--whisper-model small"
                    % (source, type(e).__name__, source)) from e
            if dev == "cuda":
                print("[asr] CUDA init failed (%s), falling back to CPU int8" % e)
                self.device, self.compute = "cpu", "int8"
                kw = {"device": "cpu", "compute_type": "int8"}
                if download_root:
                    kw["download_root"] = str(download_root)
                self.model = WhisperModel(source, **kw)
            else:
                raise
        # "auto"/""/"none" all mean: let Whisper detect the spoken language.
        # Forcing zh on English audio makes Whisper decode a Chinese translation
        # instead of the English words (see text_pipeline.py).
        self.language = None if str(language).lower() in ("auto", "none", "", "null") else language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.task = task or "transcribe"
        self.initial_prompt = initial_prompt or None
        self.last_language = None
        self.model_name = source
        print("[asr] whisper model=%s device=%s compute=%s language=%s task=%s" %
              (source, self.device, self.compute, self.language or "auto", self.task))

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.size == 0:
            return ""
        segs, info = self.model.transcribe(
            audio,
            language=self.language,
            task=self.task,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        # Segments are a generator: it must be consumed before reading info.
        text = "".join(s.text for s in segs).strip()
        try:
            self.last_language = info.language
        except Exception:
            pass
        return text


class ConfuciusASR:
    """HTTP client for a user-deployed Confucius4-R2T2 Gradio service.

    Uses the Gradio HTTP protocol directly (upload -> call -> SSE) instead of
    gradio_client, which hides real HTTP errors and sometimes returns empty
    result lists. Nothing from the vendor package is bundled here.
    """

    name = "confucius"

    def __init__(self, url="http://127.0.0.1:7861", hot_word="", timeout=300.0):
        try:
            import httpx
        except ImportError as e:
            raise ASRError("httpx not installed: pip install httpx") from e
        self.base = str(url).rstrip("/")
        self.hot_word = hot_word or ""
        self._tmp = tempfile.TemporaryDirectory(prefix="speaker_caption_")
        self.wav = str(Path(self._tmp.name) / "segment.wav")
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=8.0), trust_env=False
        )
        self.prefix = self._detect_prefix()
        print("[asr] confucius endpoint=%s%s hot_word=%s" %
              (self.base, self.prefix, self.hot_word or "(none)"))

    def _detect_prefix(self):
        last = "unknown"
        for p in ("/gradio_api", ""):
            try:
                r = self.client.get(self.base + p + "/info", timeout=8)
                if r.status_code == 200:
                    return p
                last = "HTTP %d" % r.status_code
            except Exception as e:
                last = "%s: %s" % (type(e).__name__, e)
        raise ASRError("cannot reach Confucius service at %s (%s)" % (self.base, last))

    @staticmethod
    def _clean(txt):
        txt = "" if txt is None else str(txt)
        return txt.split("|")[0].replace("<asr_text>", "") \
                  .replace("\r", " ").replace("\n", " ").strip()

    def _once(self):
        with open(self.wav, "rb") as f:
            r = self.client.post(self.base + self.prefix + "/upload",
                                 files={"files": ("segment.wav", f, "audio/wav")})
        if r.status_code != 200:
            raise ASRError("upload HTTP %d: %s" % (r.status_code, r.text[:200]))
        up = r.json()
        if not isinstance(up, (list, tuple)) or not up:
            raise ASRError("upload returned an empty list: %s" % r.text[:200])

        payload = {"data": [
            {"path": up[0], "meta": {"_type": "gradio.FileData"},
             "orig_name": "segment.wav"},
            None, "音频", "文本", self.hot_word, "Confucius4-R2T2-model",
        ]}
        r2 = self.client.post(self.base + self.prefix + "/call/do_text", json=payload)
        if r2.status_code != 200:
            raise ASRError("call HTTP %d: %s" % (r2.status_code, r2.text[:200]))
        eid = (r2.json() or {}).get("event_id")
        if not eid:
            raise ASRError("no event_id: %s" % r2.text[:200])

        final, server_err, ev = None, None, None
        with self.client.stream(
            "GET", self.base + self.prefix + "/call/do_text/" + eid
        ) as s:
            for line in s.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    ev = line[6:].strip()
                elif line.startswith("data:"):
                    raw = line[5:].strip()
                    if ev == "error":
                        server_err = raw
                        break
                    try:
                        final = json.loads(raw)
                    except Exception:
                        pass
        if server_err:
            raise ASRError("server error: %s" % server_err[:300])
        if not isinstance(final, (list, tuple)) or not final:
            return ""            # silence -> empty result, not an error
        return final[0]

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.size == 0:
            return ""
        sf.write(self.wav, audio, sample_rate, subtype="PCM_16")
        try:
            return self._clean(self._once())
        except Exception:
            time.sleep(0.6)
            return self._clean(self._once())

    def close(self):
        try:
            self.client.close()
        finally:
            self._tmp.cleanup()


def build_asr(name, **kw):
    name = (name or "none").lower()
    if name == "none":
        return NullASR()
    if name == "whisper":
        # 实时的 build_finalizer 传的是 model=/device=/compute_type=，
        # 离线的 cli.py 传的是 whisper_model=/whisper_device=。
        # 两种键名都认，否则 --whisper-model / --whisper-device 会被静默忽略，
        # 永远退回默认的 large-v3 + auto 设备。
        def pick(short, long, default):
            v = kw.get(long, None)
            if v is None:
                v = kw.get(short, None)
            return default if v is None else v
        return WhisperASR(
            model=pick("model", "whisper_model", "large-v3"),
            device=pick("device", "whisper_device", "auto"),
            compute_type=pick("compute_type", "whisper_compute", "auto"),
            language=kw.get("language"),
            beam_size=kw.get("beam_size", 5),
            vad_filter=kw.get("vad_filter", False),
            download_root=kw.get("download_root"),
            task=kw.get("task", "transcribe"),
            initial_prompt=kw.get("initial_prompt"),
        )
    if name == "confucius":
        # 注意：实时的 build_finalizer 传的是 url=，离线的 cli.py 传的是
        # confucius_url=。两种都认，否则自定义 --confucius-url 会被静默忽略。
        url = kw.get("confucius_url") or kw.get("url") or "http://127.0.0.1:7861"
        return ConfuciusASR(url=url, hot_word=kw.get("hot_word", ""))
    if name in ("confucius-local", "confucius_local"):
        # 进程内直调，不需要用户先启动 7861 服务
        from .confucius_local import build_local_asr
        return build_local_asr(project_dir=kw.get("confucius_dir"),
                               hot_word=kw.get("hot_word", ""),
                               timeout=kw.get("confucius_timeout", 300.0))
    raise ASRError("unknown ASR backend: %s" % name)
