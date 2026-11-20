"""实时音频采集：抓「电脑正在播放的声音」，输出到 16k 单声道队列。

三个可插拔后端，跨平台仍能成立，只是各平台的开箱即用程度不同：

  wasapi   Windows 默认。pyaudiowpatch 拿默认输出设备的回环，
           不需要启用「立体声混音」，USB/HDMI/蓝牙声卡通吃。
  sd       其它平台（或 Windows 上想指定设备）：sounddevice 选 monitor 设备，
           macOS 用 BlackHole / Soundflower，Linux 用 PulseAudio monitor。
  file     离线 wav 按实时节奏回放，用于无系统声音时验收整条链路。

历史踩坑（见原项目 DEVLOG 与用户级记忆）：
  - WASAPI 回环给的是**交织**多声道 (L,R,L,R...)，必须 reshape 后取均值，
    否则相当于半速 + 梳状滤波，ASR 会彻底听错。
  - 安全软件（卡巴斯基麦克风保护）拦下时，流仍能「打开成功」但数据全 0，
    且开流耗时会明显变长。故这里同时记录电平、有声占比与开流耗时。
  - 异常退出后回环句柄可能被锁，表现为开流几十秒；优雅 stop/close，
    不调用 terminate（该版本会在 terminate 时挂起）。
  - 同一进程「停止 → 再开始」二次开流必现 -9999。实测分组对照：同一线程
    关流→再开→再开全部成功；换线程（无论复用还是新建 PyAudio）一律 -9999。
    根因是 WASAPI 的 COM / 设备接口绑定在 Pa_Initialize 所在线程，与设备释放
    耗时、安全软件授权都无关。故全进程固定一条音频线程（_AudioHost），
    所有 PyAudio 初始化、open、close 都在它上面执行；每轮只换数据队列。
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

TARGET_SR = 16000

# ---------------------------------------------------------------------------
# 全进程唯一的音频线程。
#
# 真实设备分组对照（2026-11-09，WASAPI 回环 idx=47 48000Hz/2ch）：
#   same-thread     同一线程 open→close→open→close→open     全部 OK
#   cross-same-pa   换线程、复用同一个 PyAudio                -9999
#   cross-new-pa    换线程、新建 PyAudio                      -9999
# 因此 -9999 既不是「设备释放滞后」也不是安全软件拦截，而是 PortAudio 在
# Pa_Initialize 的线程上初始化 WASAPI/COM，后续 Pa_OpenStream 必须回到该线程。
# 原实现每轮 AudioSource.start() 都 new 一个 capture 线程，第二轮必挂。
_PA_LOCK = threading.Lock()
_PA = None
_STREAM_LOCK = threading.Lock()
_HOST_LOCK = threading.Lock()
_AUDIO_HOST = None


class _AudioHost:
    """串行执行所有 PortAudio 调用的那一条线程。"""

    def __init__(self):
        self.tasks: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="audio-host",
                                       daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            fn = self.tasks.get()
            if fn is None:
                return
            try:
                fn()
            except BaseException as e:      # noqa: BLE001
                sys.stderr.write("[采集] 音频线程任务异常: %s: %s\n"
                                 % (type(e).__name__, e))

    def submit(self, fn):
        """异步跑长任务（一次听录），返回完成事件。"""
        done = threading.Event()

        def wrapper():
            try:
                fn()
            finally:
                done.set()

        self.tasks.put(wrapper)
        return done

    def call(self, fn, timeout: float = 15.0):
        """同步跑短任务并回传结果/异常；不可在 host 线程内递归调用。"""
        out = {}
        done = threading.Event()

        def wrapper():
            try:
                out["value"] = fn()
            except BaseException as e:      # noqa: BLE001
                out["error"] = e
            finally:
                done.set()

        self.tasks.put(wrapper)
        if not done.wait(timeout):
            raise TimeoutError("音频设备线程无响应（%.0fs）" % timeout)
        if "error" in out:
            raise out["error"]
        return out.get("value")


def _audio_host() -> _AudioHost:
    global _AUDIO_HOST
    with _HOST_LOCK:
        if _AUDIO_HOST is None:
            _AUDIO_HOST = _AudioHost()
        return _AUDIO_HOST


def _shared_pyaudio():
    """PyAudio 实例必须在音频线程里创建，否则 COM 归属线程就错了。"""
    global _PA
    host = _audio_host()
    if threading.current_thread() is not host.thread:
        return host.call(_shared_pyaudio, timeout=15)
    with _PA_LOCK:
        if _PA is None:
            import pyaudiowpatch as pyaudio
            _PA = pyaudio.PyAudio()
        return _PA


def _close_stream_obj(s):
    """兼容 PyAudio Stream(stop_stream) 与 sounddevice InputStream(stop)。"""
    try:
        if hasattr(s, "stop_stream"):
            s.stop_stream()
        elif hasattr(s, "stop"):
            s.stop()
    except Exception:
        pass
    try:
        s.close()
    except Exception:
        pass


def _is_host_error(exc: BaseException) -> bool:
    # 去空格后同时覆盖 "Unanticipated host error" 与 PortAudio 的
    # "paUnanticipatedHostError" 两种写法。
    msg = str(exc).lower().replace(" ", "")
    return "-9999" in msg or "unanticipatedhosterror" in msg or "hosterror" in msg


def resample(data: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    """零依赖重采样。整数倍用 boxcar 均值（自带抗混叠），非整数用线性插值。"""
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr or data.size == 0:
        return np.ascontiguousarray(data)
    if src_sr > dst_sr and src_sr % dst_sr == 0:
        r = src_sr // dst_sr
        n = data.size // r
        return np.ascontiguousarray(data[:n * r].reshape(-1, r).mean(axis=1))
    n_out = int(round(data.size * dst_sr / float(src_sr)))
    if n_out <= 1:
        return np.ascontiguousarray(data[:1])
    x_old = np.arange(data.size, dtype=np.float64) * (dst_sr / float(src_sr))
    x_new = np.arange(n_out, dtype=np.float64)
    return np.ascontiguousarray(np.interp(x_new, x_old, data).astype(np.float32))


class CaptureStats:
    """线程安全的电平 / 静音看门狗状态，GUI 与控制台共用。"""
    def __init__(self, silence_warn_sec: float = 20.0):
        self.lock = threading.Lock()
        self.t_open = 0.0
        self.open_sec = 0.0
        self.device = "正在打开音频设备…"
        self.peak = 0.0
        self.total = 0
        self.sound = 0
        self.last_sound_ts = 0.0
        self.started_ts = 0.0
        self.silence_warn_sec = float(silence_warn_sec)
        self.blocked_hint = False

    def on_open(self, device: str, open_sec: float):
        with self.lock:
            self.device = device
            self.open_sec = open_sec
            self.started_ts = time.time()
            self.last_sound_ts = time.time()
            self.blocked_hint = open_sec > 2.0

    def on_block(self, peak: float, n: int, rms: float = 0.0):
        now = time.time()
        with self.lock:
            self.peak = max(peak, self.peak * 0.85)
            self.total += n
            # 用 RMS 判定「真的有声音」。WASAPI 回环在完全静音时也会给微弱底噪，
            # 只看 peak 会把底噪算成有声，导致静音看门狗永远不报警（实测踩过）。
            # rms > 0.003 约 -50 dBFS；被安全软件拦下时数据是严格全 0，必被识别。
            if rms > 0.003 and peak > 0.005:
                self.sound += n
                self.last_sound_ts = now

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            silent = now - self.last_sound_ts if self.last_sound_ts else 0.0
            return {
                "device": self.device,
                "open_sec": self.open_sec,
                "peak": self.peak,
                "silent_sec": silent,
                "sound_ratio": (self.sound / self.total) if self.total else 0.0,
                "blocked_hint": self.blocked_hint,
                "warn": self.silence_warn_sec > 0 and silent > self.silence_warn_sec,
            }


class AudioSource:
    """把采集后端统一成：start() 后台跑、read() 阻塞取 16k 单声道块、stop() 收尾。"""

    def __init__(self, backend: str = "auto", device=None, simulate: str = None,
                 loop: bool = True, stats: Optional[CaptureStats] = None,
                 max_q: int = 300, wave_meter=None):
        self.backend = backend
        self.device = device
        self.simulate = simulate
        self.loop = loop
        self.stats = stats or CaptureStats()
        # 真实波形总线（GUI 才有；--no-gui 的 CLI 路径为 None，零开销）。
        # 只在这里接一次，WASAPI / sounddevice / 文件回放三种后端就都覆盖了。
        self.wave_meter = wave_meter
        self.q: queue.Queue = queue.Queue(maxsize=max_q)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._host_done: Optional[threading.Event] = None
        self._pa = None
        self._stream = None
        self.error: Optional[BaseException] = None
        self.eof = threading.Event()
        self._eof_seen = False

    # ---------- 公共 ----------
    @property
    def device_kind(self) -> str:
        if self.simulate:
            return "file"
        if self.backend == "auto":
            return "wasapi" if sys.platform == "win32" else "sd"
        return self.backend

    def start(self):
        kind = self.device_kind
        target = {"wasapi": self._run_wasapi, "sd": self._run_sd,
                  "file": self._run_file}[kind]
        wrapped = self._wrap(target)
        if kind in ("wasapi", "sd"):
            # 关键：所有 PortAudio 调用固定在同一条线程上。
            # 每轮换的是 AudioSource（队列/状态），绝不能换线程。
            self._host_done = _audio_host().submit(wrapped)
        else:
            self._thread = threading.Thread(target=wrapped, name="capture",
                                            daemon=True)
            self._thread.start()

    def _wrap(self, fn):
        def run():
            try:
                fn()
            except BaseException as e:      # noqa: BLE001
                self.error = e
                sys.stderr.write("[采集] 失败: %s: %s\n" % (type(e).__name__, e))
            finally:
                self.eof.set()
                self.q.put(None)
        return run

    def read(self, timeout: float = 0.5):
        """返回 float32 16k 单声道，或 None（超时/结束）。

        结束哨兵是最后入队的，FIFO 保证它之前的所有音频块都已被取走，
        因此遇到哨兵时不会丢尾部数据。
        """
        if self._eof_seen:
            return None
        try:
            item = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is None:
            self._eof_seen = True
            return None
        raw, sr = item
        return resample(raw, sr)

    def stop(self):
        self._stop.set()
        if self._host_done is not None:
            # _run_wasapi / _run_sd 会在自己的 finally 里关流，这里只等它跑完，
            # 不能从本线程去 close：跨线程 close 同样会污染 PortAudio 状态。
            if not self._host_done.wait(timeout=8):
                # 极端情况（回调卡死）：宁可跨线程关一次，也别把设备锁死。
                with _STREAM_LOCK:
                    s, self._stream = self._stream, None
                if s is not None:
                    _close_stream_obj(s)
        elif self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self.device_kind in ("wasapi", "sd"):
            # 让 WASAPI 会话在音频线程内释放干净，再接受下一轮任务。
            time.sleep(0.25)

    def _close_stream(self):
        # 只由音频线程执行。外部线程（兜底）调用时转交给 host，绝不直接关。
        with _STREAM_LOCK:
            s = self._stream
            self._stream = None
        if s is None:
            return
        host = _audio_host()
        if threading.current_thread() is host.thread:
            _close_stream_obj(s)
        else:
            try:
                host.call(lambda: _close_stream_obj(s), timeout=3)
            except Exception:
                _close_stream_obj(s)
        # 故意不调用 PyAudio.terminate()：该版本在回环流上会挂起；
        # PyAudio 实例由 _shared_pyaudio() 全进程持有，随进程退出回收。

    def _put(self, raw, sr):
        if self.q.full():
            try:
                self.q.get_nowait()         # 实时优先：宁可丢旧数据，不累积延迟
            except queue.Empty:
                pass
        pk = float(np.abs(raw).max()) if raw.size else 0.0
        rms = float(np.sqrt(np.mean(raw * raw))) if raw.size else 0.0
        self.stats.on_block(pk, raw.size, rms)
        if self.wave_meter is not None:
            # 回调线程上只做一次降采样（4096 样本量级，几十微秒），
            # 不碰 Qt、不写文件；波形柱来自这块真实 PCM 的 min/max。
            try:
                self.wave_meter.push(raw, sr)
            except Exception:                       # noqa: BLE001
                # 可视化是旁路：任何异常都不能把音频采集搞挂。
                self.wave_meter = None
        try:
            self.q.put_nowait((raw, sr))
        except queue.Full:
            pass

    # ---------- Windows: WASAPI 回环 ----------
    def _open_loopback_stream(self, idx, sr, ch, callback):
        """开流并退避重试：刚停止就重开时，WASAPI 会话释放滞后会报 -9999。"""
        import pyaudiowpatch as pyaudio
        delays = (0.0, 0.4, 0.9, 1.6)
        for i, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            if self._stop.is_set():
                raise RuntimeError("采集已停止，取消开流")
            stream = None
            try:
                stream = self._pa.open(
                    format=pyaudio.paInt16, channels=ch, rate=sr, input=True,
                    input_device_index=idx, frames_per_buffer=4096,
                    stream_callback=callback)
                stream.start_stream()
                return stream
            except Exception as e:                       # noqa: BLE001
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                if not _is_host_error(e) or i == len(delays) - 1:
                    if _is_host_error(e):
                        print("[提示] WASAPI 开流失败，依次检查："
                              "① 系统是否正在播放声音、默认输出设备是否被拔出或切换；"
                              "② 等 1~2 秒再点「开始听录」；"
                              "③ 若安全软件弹出「正在试图访问麦克风」则点「立即允许」；"
                              "④ 仍失败则在管理员命令行执行 "
                              "net stop Audiosrv && net start Audiosrv", flush=True)
                    raise
                print("[采集] 回环开流失败（%s），%.1fs 后重试 %d/%d"
                      % (e, delays[i + 1], i + 1, len(delays) - 1), flush=True)
                self._close_stream()

    def _run_wasapi(self):
        import pyaudiowpatch as pyaudio
        t0 = time.time()
        self._pa = _shared_pyaudio()
        try:
            lb = self._pa.get_default_wasapi_loopback()
        except Exception as e:
            raise RuntimeError("获取默认 WASAPI 回环失败: %s（确认有声音在播；"
                               "若之前异常退出过，重启 Windows 音频服务）" % e)
        idx = int(lb["index"])
        sr = int(lb["defaultSampleRate"]) or 48000
        ch = int(lb["maxInputChannels"]) or 2

        def callback(in_data, frame_count, time_info, status):
            if status:
                sys.stderr.write("[采集] status: %s\n" % status)
            arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                arr = arr.reshape(-1, ch).mean(axis=1)
            self._put(arr, sr)
            return (None, pyaudio.paContinue)

        self._stream = self._open_loopback_stream(idx, sr, ch, callback)
        self.stats.on_open(str(lb["name"]), time.time() - t0)
        print("[采集] WASAPI 回环 idx=%d %s %dHz/%dch，开流 %.2fs"
              % (idx, lb["name"], sr, ch, time.time() - t0), flush=True)
        if self.stats.snapshot()["blocked_hint"]:
            print("[提示] 开流偏慢，若状态条变红请检查系统是否正在播放声音", flush=True)
        try:
            while not self._stop.is_set():
                time.sleep(0.15)
        finally:
            self._close_stream()

    # ---------- 其它平台 / 指定设备 ----------
    def _run_sd(self):
        import sounddevice as sd
        t0 = time.time()
        dev = self.device
        if dev is None:
            for i, d in enumerate(sd.query_devices()):
                name = str(d.get("name", "")).lower()
                if any(k in name for k in ("monitor", "blackhole", "soundflower", "loopback")):
                    dev, sr, ch = i, int(d["default_samplerate"]) or 48000, max(1, int(d["max_input_channels"]))
                    break
            if dev is None:
                raise RuntimeError(
                    "未找到系统声音回环设备。macOS 请装 BlackHole，Linux 请用 PulseAudio monitor，"
                    "或用 --device <编号> 指定（--list-devices 查看）")
        else:
            info = sd.query_devices(int(dev))
            sr = int(info["default_samplerate"]) or 48000
            ch = max(1, int(info["max_input_channels"]))
        ch = min(ch, 2)

        def callback(indata, frames, time_info, status):
            if status:
                sys.stderr.write("[采集] status: %s\n" % status)
            arr = np.asarray(indata, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            self._put(np.ascontiguousarray(arr), sr)

        self._stream = sd.InputStream(device=dev, channels=ch, samplerate=sr,
                                      dtype="float32", callback=callback,
                                      blocksize=int(sr * 0.08))
        self._stream.start()
        self.stats.on_open("sounddevice #%s" % dev, time.time() - t0)
        print("[采集] sounddevice dev=%s %dHz/%dch，开流 %.2fs"
              % (dev, sr, ch, time.time() - t0), flush=True)
        try:
            while not self._stop.is_set():
                time.sleep(0.15)
        finally:
            self._close_stream()

    def _run_file(self):
        import soundfile as sf
        path = Path(self.simulate)
        if not path.is_file():
            raise FileNotFoundError("模拟音频不存在: %s" % path)
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        self.stats.on_open("文件回放 %s" % path.name, 0.0)
        print("[采集] 文件回放 %s (%.1fs, 循环=%s)" %
              (path.name, len(data) / sr, self.loop), flush=True)
        block = int(sr * 0.1)
        while not self._stop.is_set():
            i = 0
            while i < len(data) and not self._stop.is_set():
                self._put(np.ascontiguousarray(data[i:i + block]), sr)
                i += block
                time.sleep(0.1)            # 按实时节奏喂，模拟真实直播
            if not self.loop:
                break
        # 让主循环有机会把缓冲里的尾巴处理完
        time.sleep(0.3)

    # ---------- 设备列表 ----------
    @staticmethod
    def list_devices():
        return _audio_host().call(_list_devices_impl, timeout=20)


def _list_devices_impl():
    """设备枚举也是 PortAudio 调用，必须在音频线程上跑。"""
    lines = []
    if sys.platform == "win32":
        try:
            p = _shared_pyaudio()
            lb = p.get_default_wasapi_loopback()
            lines.append("WASAPI 默认回环: idx=%s  %s  %sHz"
                         % (lb.get("index"), lb.get("name"), lb.get("defaultSampleRate")))
        except Exception as e:
            lines.append("WASAPI 回环不可用: %s" % e)
    try:
        import sounddevice as sd
        lines.append("--- sounddevice ---")
        for i, d in enumerate(sd.query_devices()):
            lines.append("%3d  %-58s in=%s out=%s sr=%s" %
                         (i, d.get("name", ""), d.get("max_input_channels"),
                          d.get("max_output_channels"), d.get("default_samplerate")))
    except Exception as e:
        lines.append("sounddevice 不可用: %s" % e)
    return "\n".join(lines)


def security_soft_running() -> bool:
    """粗判常见安全软件进程，用于给「被拦截」提示做旁证。"""
    try:
        import subprocess
        # 中文 Windows 的 tasklist 输出是 GBK；text=True 强制 utf-8 会在
        # subprocess 的 reader 线程里抛 UnicodeDecodeError（虽不致命但刷屏）。
        raw = subprocess.run(["tasklist", "/fo", "csv", "/nh"],
                             capture_output=True, timeout=5).stdout or b""
        out = raw.decode("utf-8", "ignore").lower()
        return any(k in out for k in ('"avp.exe"', '"kavfs.exe"', '"avpui.exe"'))
    except Exception:
        return False
