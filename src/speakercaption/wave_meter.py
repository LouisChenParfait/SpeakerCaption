"""真实音频波形总线：把采集到的 PCM 送到界面上，而不是画一条假正弦。

为什么需要单独一个模块
----------------------
旧版 `WaveWidget` 只有 `set_energy(0..1)` 一个标量，波形形状完全由
`sin(i * 0.35 + phase)` 之类的函数合成 —— 音量变了只是整体变高变矮，
柱子怎么起伏和真实声音毫无关系。用户看到的是装饰动画，不是音频。

这里把「音频 → 波形」拆成一条独立、线程安全、低开销的通路：

    采集回调 (WASAPI / sounddevice / 文件回放)
        └─ AudioSource._put(raw, sr)
             └─ WaveMeter.push()  → 每根柱 = 20ms 真实采样的 min/max 包络
                  └─ 环形缓冲 (512 柱 ≈ 10s)
                       └─ WaveWidget 每 40ms 取最近 110 柱 → 画双极波形

文件听录没有实时流，走另一条：`begin_file(audio, sr)` 一次性算出整段
文件的波形总览，`set_progress()` 随 ASR 段推进点亮已转写部分 + 播放头。

设计约束（改这里前先读）
------------------------
* **push() 在 PortAudio 回调线程里跑**，只做一次 reshape + min/max，
  不分配大对象、不弹日志、不碰 Qt；GUI 只读 `frame()`，锁内只做拷贝。
* **形状必须保真**：每柱存原始采样的 min/max 两个值，画的时候上下各画一半，
  所以看到的就是波形本身。只有 `DISPLAY_GAIN` 这个固定显示增益和 ±1 限幅，
  没有自动增益、没有随机数、没有平滑成「漂亮曲线」。
* **静音必须是平线**：全 0 输入 → 全 0 包络，一眼能区分「没声音」和「没在采」。
* **不依赖 Qt / 不依赖 GUI**：命令行 `--no-gui` 时压根不创建 meter，
  `AudioSource(wave_meter=None)` 走零开销分支。
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

# 每根波形柱覆盖的真实音频时长（秒）。110 柱 ≈ 2.2 秒历史。
BIN_SEC = 0.020
# 显示增益：语音常见 RMS 在 -20 ~ -30 dBFS，×3 才能在 64px 高度里看清。
# 这是固定增益而非 AGC，幅度比例仍然真实，不会把悄悄话和喊叫拉平。
DISPLAY_GAIN = 3.0
# 环形缓冲容量（柱）：约 10 秒，远大于窗口宽度，GUI 掉几帧也不丢形。
CAPACITY = 512
# 单个采集块最多切多少柱，防止异常大的 block 一次性刷掉整屏。
MAX_PUSH_BINS = 16
# 文件总览的分辨率：比窗口宽，downsample 时用 max-pooling 保住尖峰。
FILE_BINS = 512

MODE_IDLE = "idle"
MODE_PREPARING = "preparing"
MODE_LIVE = "live"
MODE_FILE = "file"


def wave_envelope(audio, bins: int, gain: float = DISPLAY_GAIN) -> np.ndarray:
    """把一段 PCM 降采样成 ``(bins, 2)`` 的 ``[min, max]`` 包络。

    每一柱就是原始采样点在那一小段时间里的真实最小值 / 最大值，
    不是取绝对值、不是 RMS 包络、更不是正弦。唯一的处理是固定的
    `tanh` 显示压缩：把小音量抬到看得见，同时避免硬限幅把大音量
    削成一条实心方块。压缩是单调的，不改变波形的起伏与相对强弱。
    """
    a = np.asarray(audio, dtype=np.float32).ravel()
    nb = max(1, int(bins))
    out = np.zeros((nb, 2), dtype=np.float32)
    if a.size == 0:
        return out
    per = a.size // nb
    if per <= 0:
        # 样本比柱数还少：逐个样本落到对应柱，没落到的柱保持 0。
        for i, x in enumerate(a):
            j = i * nb // a.size
            v = float(x)
            out[j, 0] = min(out[j, 0], v)
            out[j, 1] = max(out[j, 1], v)
    else:
        v = a[:per * nb].reshape(nb, per)
        out[:, 0] = v.min(axis=1)
        out[:, 1] = v.max(axis=1)
    np.tanh(out * gain, out=out)
    return out


def _pool_to(bars: np.ndarray, n: int) -> np.ndarray:
    """把高分辨率包络降到 n 柱：对 min 取最小、对 max 取最大，保住尖峰。"""
    src = np.asarray(bars, dtype=np.float32)
    if src.shape[0] == n:
        return src.copy()
    if src.shape[0] < n:
        idx = np.linspace(0, src.shape[0] - 1, n).astype(int)
        return src[idx].copy()
    edges = np.linspace(0, src.shape[0], n + 1).astype(int)
    out = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        seg = src[lo:hi]
        out[i, 0] = seg[:, 0].min()
        out[i, 1] = seg[:, 1].max()
    return out


@dataclass(frozen=True)
class WaveFrame:
    """一帧给视图层的数据。`bars` 是 ``(n, 2)`` 的 [min, max]，列 0/1 分别负/正。"""

    bars: np.ndarray
    level: float          # 0..1，最近一块的显示电平（RMS × 2 与 peak 的折中）
    peak: float           # 0..1，最近一块的真实峰值
    progress: float       # 文件模式 0..1；其它模式恒 0
    mode: str
    active: bool


class WaveMeter:
    """线程安全的波形 / 电平总线。采集线程只写，GUI 线程只读。

    生命周期由 `UIController` 驱动::

        begin_preparing()   # 点开始，模型还没加载：界面显示呼吸待机
        begin_live()        # 音频设备打开前：清空历史，开始滚动真实波形
        finish()            # 停止 / 出错 / 结束：冻结，界面回到待机
        begin_file(audio)   # 文件解码完成：整段总览
        set_progress(x)     # 每转写完一段：推进播放头
    """

    def __init__(self, capacity: int = CAPACITY, gain: float = DISPLAY_GAIN):
        self.capacity = max(64, int(capacity))
        self.gain = float(gain)
        self._lock = threading.Lock()
        self._buf = np.zeros((self.capacity, 2), dtype=np.float32)
        self._w = 0                       # 下一个写入位置
        self._count = 0                   # 有效历史柱数（上限 capacity）
        self._level = 0.0
        self._peak = 0.0
        self._mode = MODE_IDLE
        self._active = False
        self._progress = 0.0
        self._file_bars = None

    # ---------- 生命周期 ----------
    def begin_preparing(self):
        """模型加载 / 文件探测阶段：还没有音频，界面走待机呼吸。"""
        with self._lock:
            self._mode = MODE_PREPARING
            self._active = True
            self._clear_locked()
            self._file_bars = None
            self._progress = 0.0

    def begin_live(self):
        """实时采集即将开始：清掉上一场的历史，避免两次听录波形连在一起。"""
        with self._lock:
            self._mode = MODE_LIVE
            self._active = True
            self._clear_locked()
            self._file_bars = None
            self._progress = 0.0

    def begin_file(self, audio, sr=16000):
        """文件解码完成：算出整段波形总览，播放头归零。"""
        bars = wave_envelope(audio, FILE_BINS, self.gain)
        with self._lock:
            self._mode = MODE_FILE
            self._active = True
            self._clear_locked()
            self._file_bars = bars
            self._progress = 0.0

    def set_progress(self, frac: float):
        with self._lock:
            self._progress = max(0.0, min(1.0, float(frac)))

    def finish(self):
        """不再有音频进来。文件模式保留总览，实时模式由视图回到待机。"""
        with self._lock:
            self._active = False

    def _clear_locked(self):
        self._buf.fill(0.0)
        self._w = 0
        self._count = 0
        self._level = 0.0
        self._peak = 0.0

    # ---------- 采集线程写入 ----------
    def push(self, audio, sr: int):
        """吃一块原始 PCM（float32，单声道），切柱后写进环形缓冲。

        在 PortAudio 回调里调用，必须快：一次 reshape + 两次 reduction，
        4096 样本量级约几十微秒，锁只在写环时持有。
        """
        a = np.asarray(audio, dtype=np.float32).ravel()
        if a.size == 0:
            return
        sr = max(1, int(sr or 16000))
        nb = int(round(a.size / float(sr) / BIN_SEC))
        nb = max(1, min(MAX_PUSH_BINS, nb))
        env = wave_envelope(a, nb, self.gain)
        # 峰值取压缩前的原始采样，压过的包络已经饱和，反推不回真实峰值。
        pk = float(np.max(np.abs(a)))
        rms = float(np.sqrt(np.mean(a * a)))
        with self._lock:
            for i in range(nb):
                self._buf[self._w] = env[i]
                self._w = (self._w + 1) % self.capacity
            self._count = min(self.capacity, self._count + nb)
            self._level = rms
            self._peak = min(1.0, pk)

    # ---------- GUI 线程读取 ----------
    def _recent_locked(self, n: int) -> np.ndarray:
        bars = np.zeros((n, 2), dtype=np.float32)
        valid = min(n, self._count)
        if valid:
            idx = (self._w - valid + np.arange(valid)) % self.capacity
            bars[n - valid:] = self._buf[idx]
        return bars

    def frame(self, n: int) -> WaveFrame:
        """取最近 n 柱（左旧右新）。文件模式返回整段总览的降采样。"""
        n = max(1, int(n))
        with self._lock:
            mode, active = self._mode, self._active
            level, peak = self._level, self._peak
            progress = self._progress
            if mode == MODE_FILE and self._file_bars is not None:
                bars = _pool_to(self._file_bars, n)
            else:
                bars = self._recent_locked(n)
        return WaveFrame(bars=bars,
                         level=max(level * 2.0, peak * 0.5),
                         peak=peak, progress=progress,
                         mode=mode, active=active)

    def level_db(self) -> float:
        """给状态条电平格用的 dBFS（下限 -60，避免 log10(0)）。"""
        f = self.frame(1)
        return 20.0 * math.log10(max(f.level, 1e-5))

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active
