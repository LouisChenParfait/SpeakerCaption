"""Text post-processing: Chinese script conversion + optional EN -> ZH translation.

Why this module exists
----------------------
Two recurring complaints from real live-caption sessions:

* "为什么听读出来都是繁体字？"
  faster-whisper large-v3 (and Confucius) frequently emit **traditional**
  Chinese even for Mandarin audio. There is no ASR flag to force simplified.

* "听到的英文为什么直接变成汉字？"
  The live entry used to default to ``--language zh``. Whisper is then told
  "output Chinese" regardless of what is actually spoken; on English speech
  it decodes a Chinese translation for some sentences and leaves English for
  others. Measured: forced zh -> info.language='zh'; auto -> info.language='en'
  with the original English text. Whisper large-v3 is also inconsistent about
  simplified/traditional, which is why one run can contain both.

Fix: auto-detect the spoken language (``language=None``) and do everything
else as an **explicit, opt-in text step**:

    ASR raw text --[translate?]--> semantic text --[script?]--> displayed text

Translation is off by default; script conversion to simplified is on by
default. Both are exposed as UI options in the live window.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from typing import Optional

# Local Ollama / Confucius endpoints must never be routed through the proxy.
for _k in ("NO_PROXY", "no_proxy"):
    _hosts = "127.0.0.1,localhost,::1"
    if _hosts not in os.environ.get(_k, ""):
        os.environ[_k] = (os.environ.get(_k, "") + "," + _hosts).strip(",")

SCRIPT_SIMPLIFIED = "simplified"
SCRIPT_TRADITIONAL = "traditional"
SCRIPT_ASIS = "as-is"
SCRIPT_CHOICES = (SCRIPT_SIMPLIFIED, SCRIPT_TRADITIONAL, SCRIPT_ASIS)

_CONVERTERS = {}
_WARNED = False
_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def _converter(name: str):
    """Lazily build an OpenCC converter; warn once if the optional dep is absent."""
    global _WARNED
    if name in _CONVERTERS:
        return _CONVERTERS[name]
    try:
        from opencc import OpenCC
        conv = OpenCC(name)
    except Exception:                                     # noqa: BLE001
        conv = None
        if not _WARNED:
            _WARNED = True
            print("[text] 简繁转换需要 opencc：pip install opencc-python-reimplemented",
                  file=sys.stderr)
    _CONVERTERS[name] = conv
    return conv


def convert_script(text: str, script: str = SCRIPT_SIMPLIFIED) -> str:
    """t2s / s2t / pass-through. Never raises when opencc is unavailable."""
    if not text or script == SCRIPT_ASIS:
        return text
    name = "s2t" if script == SCRIPT_TRADITIONAL else "t2s"
    conv = _converter(name)
    return conv.convert(text) if conv else text


def simplify_chinese(text: str) -> str:
    """Backward-compatible helper used by the offline CLI."""
    return convert_script(text, SCRIPT_SIMPLIFIED)


def traditional_chinese(text: str) -> str:
    """s2t counterpart of simplify_chinese()."""
    return convert_script(text, SCRIPT_TRADITIONAL)


def has_latin(text: str) -> bool:
    """Heuristic: enough Latin letters to be worth sending to a translator."""
    return bool(text) and len(_LATIN.findall(text)) >= 3


def needs_translation(text: str) -> bool:
    """Pure Chinese text must not be sent to a translator."""
    if not has_latin(text):
        return False
    latin = len(_LATIN.findall(text))
    cjk = len(_CJK.findall(text))
    return latin >= 3 and latin >= cjk * 0.4


class OllamaTranslator:
    """Tiny local-LLM translator over Ollama's /api/chat.

    No API key, no cloud call, no bundled weights. ``model=\"auto\"`` asks
    Ollama which models are installed and prefers a translation-specific one.
    If Ollama is unreachable we warn once and return the original text, so a
    live session never dies because translation failed.
    """

    PREFER = ("hy-mt", "mt2", "translate", "translator", "qwen", "glm",
              "gemma", "mistral", "llama", "gpt-oss")

    def __init__(self, url="http://127.0.0.1:11434", model="auto", timeout=60.0):
        import httpx
        self.base = str(url).rstrip("/")
        self.requested_model = model or "auto"
        self._model = None if self.requested_model == "auto" else self.requested_model
        self._timeout = float(timeout)
        self._warned = False
        self.last_model = None
        self._lock = threading.Lock()
        self._build_client()

    def _build_client(self):
        import httpx
        # Windows 上复用 Ollama 的 keep-alive 连接偶发
        # ReadError [WinError 10038]（在非套接字上尝试操作），因此每请求新建连接。
        self.client = httpx.Client(
            timeout=httpx.Timeout(self._timeout, connect=5.0),
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=4),
            headers={"Connection": "close"},
        )

    def _rebuild_client(self):
        try:
            self.client.close()
        except Exception:
            pass
        self._build_client()

    def resolve_model(self):
        if self._model:
            return self._model
        try:
            r = self.client.get(self.base + "/api/tags")
            r.raise_for_status()
            names = [m.get("name") or m.get("model") or ""
                     for m in r.json().get("models", [])]
            names = [n for n in names if n]
            for key in self.PREFER:
                for n in names:
                    if key in n.lower():
                        self._model = n
                        break
                if self._model:
                    break
            if not self._model and names:
                self._model = names[0]
        except Exception as e:                            # noqa: BLE001
            self._warn("无法读取 Ollama 模型列表 (%s: %s)" % (type(e).__name__, e))
            return None
        if self._model:
            print("[翻译] 使用本地模型 %s @ %s" % (self._model, self.base), flush=True)
        return self._model

    def _warn(self, msg):
        if not self._warned:
            self._warned = True
            print("[翻译] %s；本次保留原文" % msg, file=sys.stderr, flush=True)

    def translate(self, text: str, target: str = "简体中文") -> str:
        model = self.resolve_model()
        if not model or not text:
            return text
        prompt = (
            "你是翻译器。把下面内容中的外文译成%s，已有的中文原样保留，"
            "不要合并或改写中文。只输出译文，不要解释、不要加引号、不要加前缀。\n\n%s"
            % (target, text)
        )
        payload = {
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.1},
            "keep_alive": "10m",
        }
        with self._lock:
            for attempt in range(2):
                try:
                    r = self.client.post(self.base + "/api/chat", json=payload)
                    r.raise_for_status()
                    out = (r.json().get("message") or {}).get("content") or ""
                    out = _THINK.sub("", out).strip().strip('"').strip("“”").strip()
                    out = re.sub(r"^(译文|翻译)[:：]\s*", "", out).strip()
                    return out or text
                except Exception as e:                    # noqa: BLE001
                    if attempt == 0:
                        # keep-alive 失效时换一条全新连接再试一次
                        self._rebuild_client()
                        continue
                    self._warn("调用 Ollama 失败 (%s: %s)" % (type(e).__name__, e))
                    return text

    def warmup(self):
        """Load the model in the background so the first sentence isn't blocked."""
        try:
            self.translate("warmup")
        except Exception:                                 # noqa: BLE001
            pass

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


class TextPipeline:
    """Thread-safe post-processing shared by the capture and GUI threads.

    ``prepare()`` is the expensive, semantic step (translation) and is cached;
    ``render()`` is the cheap, reversible step (script conversion). Keeping
    them separate lets the UI switch 简体/繁体 mid-session without re-calling
    the ASR or the LLM.
    """

    def __init__(self, translate=False, script=SCRIPT_SIMPLIFIED,
                 translator: Optional[OllamaTranslator] = None):
        self._lock = threading.Lock()
        self.translate = bool(translate)
        self.script = script if script in SCRIPT_CHOICES else SCRIPT_SIMPLIFIED
        self.translator = translator
        self._cache = {}

    def configure(self, translate=None, script=None):
        with self._lock:
            if translate is not None:
                self.translate = bool(translate)
            if script in SCRIPT_CHOICES:
                self.script = script

    def snapshot(self):
        with self._lock:
            return self.translate, self.script

    def prepare(self, raw: str) -> str:
        """ASR text -> semantic text. Translation is opt-in and cached."""
        if not raw:
            return ""
        with self._lock:
            do_translate = self.translate
        if not do_translate or not needs_translation(raw):
            return raw
        with self._lock:
            hit = self._cache.get(raw)
        if hit is not None:
            return hit
        out = self.translator.translate(raw) if self.translator else raw
        with self._lock:
            self._cache[raw] = out
        return out

    def render(self, semantic: str, script: str = None) -> str:
        """Semantic text -> displayed text (script conversion only)."""
        return convert_script(semantic, script or self.script)

    def process(self, raw: str) -> str:
        return self.render(self.prepare(raw))

    def close(self):
        if self.translator:
            self.translator.close()


def build_pipeline(translate=False, script=SCRIPT_SIMPLIFIED,
                   ollama_url="http://127.0.0.1:11434",
                   translate_model="auto", translate_timeout=60.0):
    translator = OllamaTranslator(ollama_url, translate_model,
                                  translate_timeout) if translate else None
    return TextPipeline(translate=translate, script=script, translator=translator)
