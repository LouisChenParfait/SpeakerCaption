"""Confucius 本地后端：进程内直调，不依赖外部 Gradio 服务。

三种后端的取舍
--------------
| 后端 | 需要用户先准备什么 | 备注 |
|---|---|---|
| `confucius`（HTTP） | 自己起 7861 Gradio 服务 | 旧行为，保留兼容 |
| `confucius-local`（本模块） | 装好 Confucius4-R2T2 即可 | **自动拉起子进程**，无需手动开服务 |
| `whisper` | 无 | 默认 |

`confucius-local` 做的事
------------------------
1. 找到 Confucius4-R2T2 项目目录（配置项 / 环境变量 / 常见安装位置）；
2. 用**它自带的 py312 解释器**起一个 worker 子进程（那个环境里已有
   torch 2.8 + CUDA，SpeakerCaption 自己的 venv 不需要装 torch）；
3. worker 进程内直接调 `cy_app.do_text()`，把结果经 stdout 的 JSON 行
   传回来。

好处：省 7861 端口、不用手动启服务、模型只加载一次（跨多句复用），
并且 SpeakerCaption 侧零新增重量级依赖。
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .asr import ASRError

# Confucius4-R2T2 的常见安装位置（按优先级）
CANDIDATE_DIRS = [
    r"D:\AI_TTS\Confucius4-R2T2",
    r"D:\Confucius4-R2T2",
    r"C:\AI_TTS\Confucius4-R2T2",
    r"C:\Confucius4-R2T2",
]

# Confucius 自带解释器的相对路径
INTERPRETER_REL = os.path.join("py312", "python.exe")


def find_project_dir(explicit=None):
    """定位 Confucius4-R2T2 目录。

    顺序：显式配置 → 环境变量 → 常见安装位置。
    只认「里面有 cy_app 的目录」才算数，光有目录名不够。
    """
    cands = []
    if explicit:
        cands.append(str(explicit))
    env = os.environ.get("CONFUCIUS_DIR") or os.environ.get("CONFUCIUS4_R2T2_DIR")
    if env:
        cands.append(env)
    cands.extend(CANDIDATE_DIRS)

    for c in cands:
        if not c:
            continue
        d = Path(c).expanduser()
        if not d.is_dir():
            continue
        # cy_app 可能是 cy_app.cp312-win_amd64.pyd / cy_app.pyd / cy_app.py，
        # 所以按前缀通配匹配，不能只认死文件名。
        if any(d.glob("cy_app.*")):
            return d.resolve()
    return None


def find_interpreter(project_dir: Path):
    """找 Confucius 自带的 Python 解释器。

    优先用它自己的 py312（torch 已经装好）；退而求其次用当前解释器
    （只有在当前环境也装了 torch 时才可行）。
    """
    own = project_dir / "py312" / "python.exe"
    if own.is_file():
        return str(own)
    exe = Path(sys.executable)
    # 当前解释器能否 import torch
    try:
        import subprocess as _sp
        r = _sp.run([str(exe), "-c", "import torch"], capture_output=True, timeout=60)
        if r.returncode == 0:
            return str(exe)
    except Exception:
        pass
    return None


class ConfuciusLocalASR:
    """进程内调用 Confucius4-R2T2 的 ASR 后端。

    与 `ConfuciusASR`（HTTP）接口一致：`transcribe(audio, sr) -> str`。
    """

    name = "confucius-local"

    def __init__(self, project_dir=None, hot_word="", timeout=300.0,
                 warmup=True):
        self.hot_word = hot_word or ""
        self.timeout = float(timeout)

        self.project_dir = find_project_dir(project_dir)
        if not self.project_dir:
            raise ASRError(
                "找不到 Confucius4-R2T2 项目目录。请在高级设置里填 "
                "\"Confucius 目录\"，或设置环境变量 CONFUCIUS_DIR。")

        self.python = find_interpreter(self.project_dir)
        if not self.python:
            raise ASRError(
                "找不到可用的 Python 解释器来运行 Confucius "
                "(期望 %s)。" % (self.project_dir / "py312" / "python.exe"))

        self._tmp = tempfile.TemporaryDirectory(prefix="speaker_caption_r2t2_")
        self.wav = str(Path(self._tmp.name) / "segment.wav")
        # 必须是 RLock：_request() 持有它时还会调 _ensure_worker()，
        # 而 _ensure_worker() 内部也要拿同一把锁。普通 Lock 不可重入 → 自死锁。
        self._lock = threading.RLock()
        self._proc = None
        self._req_id = 0
        self._reader = None
        self._rx = None
        self._rxq = None
        self._stderr_tail = []

        print("[asr] confucius-local project=%s" % self.project_dir)
        print("[asr] confucius-local python=%s" % self.python)
        if warmup:
            self._ensure_worker()

    # ---------- 子进程生命周期 ----------

    def _worker_script(self):
        # worker 脚本跟本文件同目录
        return Path(__file__).resolve().parent / "confucius_worker.py"

    def _ensure_worker(self):
        """确保 worker 子进程已就绪（幂等）。"""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            script = self._worker_script()
            if not script.is_file():
                raise ASRError("worker 脚本缺失: %s" % script)

            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            # Confucius 的 hf 下载镜像设置，避免子进程尝试联网
            env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            env.setdefault("HF_HUB_DISABLE_XET", "1")
            # 子进程不需要代理
            env.pop("HTTP_PROXY", None)
            env.pop("HTTPS_PROXY", None)

            self._proc = subprocess.Popen(
                [self.python, "-u", str(script), str(self.project_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_dir),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            # 队列必须在读线程启动**之前**建好，否则读线程可能往 None 队列里 put
            self._rxq = queue.Queue()
            self._reader = threading.Thread(target=self._drain_stderr,
                                            args=(self._proc,), daemon=True)
            self._reader.start()
            self._rx = threading.Thread(target=self._read_loop, daemon=True)
            self._rx.start()

            # 等 ready
            ready = self._read_message(deadline=time.time() + 180.0)
            if not ready or not ready.get("ok"):
                err = (ready or {}).get("error", "worker 启动失败")
                self._kill()
                raise ASRError("Confucius worker 启动失败: %s" % err)
            print("[asr] confucius-local worker 已就绪")

    def _drain_stderr(self, proc):
        """把 worker 的 stderr 丢掉（否则管道写满会卡死子进程）。

        但保留最后若干行，出错时打印出来便于定位。
        """
        tail = []
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                del tail[:-20]
        except Exception:
            pass
        self._stderr_tail = tail

    def _read_loop(self):
        """后台线程：把 worker stdout 的 JSON 行塞进队列。

        必须单独读 stdout，否则 `readline()` 在管道上会阻塞，
        无法实现"等 N 秒就超时"。
        """
        proc = self._proc
        q = self._rxq
        if proc is None or proc.stdout is None or q is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                q.put(msg)
        except Exception:
            pass

    def _read_message(self, deadline):
        """从 worker stdout 读一条消息；超时返回 None。"""
        remaining = max(0.2, deadline - time.time())
        try:
            return self._rxq.get(timeout=remaining)
        except Exception:
            return None

    def _request(self, audio_path):
        """发一条请求并等回应。"""
        with self._lock:
            self._ensure_worker()
            self._req_id += 1
            rid = self._req_id
            payload = {"id": rid, "audio": audio_path, "hot_word": self.hot_word}
            try:
                self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except Exception as e:
                self._kill()
                raise ASRError("Confucius worker 通信中断: %s" % e)

            deadline = time.time() + self.timeout
            while True:
                msg = self._read_message(deadline)
                if msg is None:
                    raise ASRError("Confucius worker 超时（%.0fs）%s"
                                   % (self.timeout, self._stderr_hint()))
                if msg.get("id") != rid:
                    continue
                if not msg.get("ok"):
                    raise ASRError("Confucius 转写失败: %s%s"
                                   % (msg.get("error"), self._stderr_hint()))
                return msg.get("text", "") or ""

    def _stderr_hint(self):
        """把 worker stderr 的末尾几行附在错误后面，便于定位。"""
        tail = getattr(self, "_stderr_tail", None)
        if not tail:
            return ""
        return "\n  worker stderr: " + " | ".join(tail[-5:])

    # ---------- ASR 接口 ----------

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio is None or audio.size == 0:
            return ""
        a = np.asarray(audio, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        # 写临时 wav 交给 worker（它自己解码处理采样率）
        sf.write(self.wav, a, sample_rate, subtype="PCM_16")
        try:
            return self._request(self.wav)
        except ASRError:
            # 偶发通信/推理失败，重建 worker 重试一次
            self._kill()
            time.sleep(0.8)
            return self._request(self.wav)

    def _kill(self):
        proc = self._proc
        self._proc = None
        self._rx = None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def close(self):
        try:
            self._kill()
        finally:
            try:
                self._tmp.cleanup()
            except Exception:
                pass


def build_local_asr(project_dir=None, hot_word="", timeout=300.0):
    return ConfuciusLocalASR(project_dir=project_dir, hot_word=hot_word,
                              timeout=timeout)
