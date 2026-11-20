"""SpeakerCaption Qt 视图层 —— 「曜石 × 火焰」设计实现（PySide6）。

与旧 Tkinter 视图层（live_gui.py）并行存在：live_ui.py 优先导入本模块，
PySide6 缺失时回退旧版。控制层、引擎层、LiveDisplay 状态容器完全复用，
本文件只负责「渲染 + 交互」。

设计令牌（照此改，不要临时加颜色）
----------------------------------
* **曜石灰阶**：暖调炭黑 BG < PANEL < ELEV < CONTROL < CONTROL_HI，明度差分层。
* **火焰能量色**：取自博百文化凤凰标志（#B30F00 → #D94B00 → #F2600C → #FFC24B），
  只给两处：主操作按钮、声波/电平可视化。运行中主按钮熄灭成红色「停止」。
* **字幕是主角**：15pt 占满剩余空间；定稿暖白、草稿琥珀、说话人六色轮转。
* **分组代替堆砌**：输出 / 输入 / 引擎三组，组名 10.5pt 加宽字距。
* **克制的光**：发光只给主按钮、波形与状态点。
"""
from __future__ import annotations

import html
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer, QRectF, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QIcon, QLinearGradient,
                           QPainter, QPainterPath, QPen)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QFileDialog, QFrame, QGraphicsOpacityEffect,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListView, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QStackedWidget, QTextEdit,
                               QVBoxLayout, QWidget)

from .live_gui import (ADVANCED_FIELDS, COUNT, DEVICE, FINAL, LANG, SCRIPT,
                       WHISPER, WHISPER_CUSTOM, LiveDisplay, _coerce)
from .outputs import stamp
from .wave_meter import (BIN_SEC, MODE_FILE as WM_FILE, MODE_LIVE as WM_LIVE)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets" / "brand"

# ---------- 设计令牌 ----------
BG = "#100C0D"          # 窗口底 / 字幕区
PANEL = "#171214"       # 面板
ELEV = "#211A1C"        # 抬升层 / 状态条
CONTROL = "#2C2326"     # 控件底
CONTROL_HI = "#3A2E32"  # 控件悬停
LINE = "#0A0708"        # 凹陷分隔线

FG = "#F4EEE8"          # 主文字
MUTED = "#AFA49B"       # 次要
FAINT = "#857972"       # 最弱（4.6:1，修复旧版 FAINT 不达标）

FLAME_DEEP = "#B30F00"  # 凤凰深红
BRAND = "#D94B00"       # 品牌橙
EMBER = "#F2600C"       # 主操作
EMBER_HI = "#FF7A1F"    # 主操作悬停
GOLD = "#FFC24B"        # 火焰尖端
DANGER = "#E5482F"      # 停止 / 危险
WARN = "#FFB35C"        # 警告 / 草稿

# 说话人六色：与火焰同温层，黑底高对比，label 从 1 开始轮转
SPEAKER_COLORS = ["#FFC24B", "#3FD8C2", "#74B9FF", "#C4A5FF", "#FF92B0", "#A9DE70"]
DRAFT_COLOR = "#FFB35C"

MODE_LIVE = "live"
MODE_FILE = "file"

APP_QSS = """
* { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }
QWidget { background: @BG@; color: @FG@; font-size: 10pt; }
QToolTip { background: @ELEV@; color: @FG@; border: 1px solid #000; padding: 4px 8px; }

QFrame#band { background: @PANEL@; border-bottom: 1px solid @LINE@; }
QFrame#statusbar { background: @ELEV@; border-top: 1px solid @LINE@; }
QLabel { background: transparent; }
QLabel#muted { color: @MUTED@; }
QLabel#faint { color: @FAINT@; }
QLabel#grpname { color: @FAINT@; font-size: 8pt; }
QLabel#pill { background: @ELEV@; color: @MUTED@; border-radius: 11px;
              padding: 4px 12px; font-family: Consolas, monospace; font-size: 8.5pt; }
QFrame#divider { background: #000000; max-width: 1px; }

QPushButton#primary {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 @EMBER_HI@, stop:1 @BRAND@);
  color: #2A1306; font-size: 11pt; font-weight: bold;
  border: none; border-radius: 9px; padding: 10px 24px;
}
QPushButton#primary:hover {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF8A33, stop:1 @EMBER@);
}
QPushButton#primary[stop="true"] {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F05A3C, stop:1 @DANGER@);
  color: #FFF3EF;
}
QPushButton#primary:disabled {
  background: @CONTROL@; color: @FAINT@;
}
QPushButton.ghost {
  background: @CONTROL@; color: @FG@; border: none; border-radius: 9px; padding: 7px 13px;
}
QPushButton.ghost:hover { background: @CONTROL_HI@; }
QPushButton.ghost:disabled { color: @FAINT@; background: @ELEV@; }

QPushButton.seg {
  background: transparent; color: @MUTED@; border: none; border-radius: 12px;
  padding: 5px 16px; font-size: 9.5pt;
}
QPushButton.seg:hover { color: @FG@; }
QPushButton.seg:checked {
  background: @CONTROL_HI@; color: @FG@;
  border-bottom: 2px solid @EMBER@;
}
QFrame#segbox { background: @ELEV@; border-radius: 15px; }

QComboBox {
  background: @CONTROL@; border: none; border-radius: 6px;
  padding: 6px 26px 6px 11px; color: @FG@; font-size: 9.5pt;
}
QComboBox:hover { background: @CONTROL_HI@; }
QComboBox:disabled { color: @FAINT@; background: @ELEV@; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow { image: url(@CHEVRON@); width: 10px; height: 6px; }
QComboBox QAbstractItemView {
  background: @ELEV@; color: @FG@; border: 1px solid #000;
  selection-background-color: @CONTROL_HI@; outline: none; padding: 4px;
}

QCheckBox { color: @FG@; font-size: 9.5pt; spacing: 7px; }
QCheckBox:disabled { color: @FAINT@; }
QCheckBox::indicator {
  width: 15px; height: 15px; border-radius: 4px;
  background: @ELEV@; border: 1px solid #000;
}
QCheckBox::indicator:checked {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 @EMBER_HI@, stop:1 @BRAND@);
  border: none; image: url(@CHECK@);
}
QCheckBox::indicator:disabled { background: @ELEV@; }

QLineEdit {
  background: @ELEV@; border: 1px solid #000; border-radius: 6px;
  color: @FG@; padding: 7px 12px; font-family: Consolas, monospace; font-size: 9.5pt;
  selection-background-color: @BRAND@;
}
QLineEdit:disabled { color: @FAINT@; }

QTextEdit#captions {
  background: @BG@; border: none; padding: 6px 10px;
  font-size: 15pt; line-height: 1.9;
}
QScrollBar:vertical {
  background: transparent; width: 10px; margin: 2px;
}
QScrollBar::handle:vertical { background: @CONTROL@; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: @CONTROL_HI@; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QDialog { background: @PANEL@; }
QFrame#groupbox { border-top: 1px solid #000; margin-top: 14px; }
QLabel#grouptitle { color: @BRAND@; font-size: 8.5pt; }
QFrame#note {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                              stop:0 rgba(217,75,0,30), stop:1 rgba(217,75,0,10));
  border: 1px solid rgba(217,75,0,76); border-radius: 12px;
}
QMessageBox QLabel { color: @FG@; }
QMessageBox QPushButton {
  background: @CONTROL@; border-radius: 8px; padding: 6px 18px; min-width: 64px;
}
QMessageBox QPushButton:hover { background: @CONTROL_HI@; }

/* 准备阶段的不定态进度条：火焰渐变，Qt 自带往复动画 */
QProgressBar#prep {
  background: @ELEV@; border: none; border-radius: 4px;
}
QProgressBar#prep::chunk {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                              stop:0 @FLAME_DEEP@, stop:0.5 @BRAND@, stop:1 @GOLD@);
  border-radius: 4px;
}
"""


def _qss() -> str:
    s = APP_QSS
    for name, val in [("BG", BG), ("PANEL", PANEL), ("ELEV", ELEV),
                      ("CONTROL", CONTROL), ("CONTROL_HI", CONTROL_HI),
                      ("LINE", LINE), ("FG", FG), ("MUTED", MUTED),
                      ("FAINT", FAINT), ("BRAND", BRAND), ("EMBER", EMBER),
                      ("EMBER_HI", EMBER_HI), ("GOLD", GOLD), ("DANGER", DANGER),
                      ("WARN", WARN), ("FLAME_DEEP", FLAME_DEEP)]:
        s = s.replace("@%s@" % name, val)
    s = s.replace("@CHECK@", (ASSETS / "check.svg").as_posix())
    s = s.replace("@CHEVRON@", (ASSETS / "chevron.svg").as_posix())
    return s


def svg_pixmap(name: str, height: int) -> "QPixmap":
    """把品牌 SVG 渲染成指定高度的 QPixmap。"""
    from PySide6.QtGui import QPixmap
    r = QSvgRenderer(str(ASSETS / name))
    size = r.defaultSize()
    w = max(1, round(height * size.width() / max(1, size.height())))
    pm = QPixmap(w, height)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    r.render(p)
    p.end()
    return pm


def flame_color(t: float) -> QColor:
    """0..1 → 深红 → 品牌橙 → 火焰金 的插值。"""
    def mix(a, b, k):
        return tuple(round(a[i] + (b[i] - a[i]) * k) for i in range(3))
    if t < 0.45:
        c = mix((179, 15, 0), (217, 75, 0), t / 0.45)
    elif t < 0.8:
        c = mix((217, 75, 0), (242, 96, 12), (t - 0.45) / 0.35)
    else:
        c = mix((242, 96, 12), (255, 194, 75), (t - 0.8) / 0.2)
    return QColor(*c)


def _tri_pixmap(color_hex: str, kind: str = "play", size: int = 28) -> "QPixmap":
    """自绘播放/停止图标：雅黑不含 ▶ 字形，画出来的才不依赖字体回退。"""
    from PySide6.QtGui import QPixmap
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color_hex))
    p.setPen(Qt.NoPen)
    s = size
    if kind == "play":
        path = QPainterPath()
        path.moveTo(s * 0.30, s * 0.20)
        path.lineTo(s * 0.78, s * 0.50)
        path.lineTo(s * 0.20 + 2, s * 0.80)
        path.closeSubpath()
        p.drawPath(path)
    else:
        p.drawRoundedRect(QRectF(s * 0.26, s * 0.26, s * 0.48, s * 0.48), s * 0.09, s * 0.09)
    p.end()
    return pm


# ---------- 火焰波形（真实音频包络） ----------
class WaveWidget(QWidget):
    """声波即火焰 —— 但每根柱子都来自真实 PCM，不是合成正弦。

    * **实时听录**：滚动显示最近 ``N`` 柱（约 ``N * BIN_SEC`` 秒）的
      min/max 包络，左旧右新。静音就是一条平线，说话时才起伏，
      柱子的高低、停顿、爆发都是采集到的原始采样。
    * **文件听录**：显示整段文件的波形总览；已转写部分点亮火焰色，
      未转写部分是暗色，播放头按 ASR 段落的结束时间前进。
    * **准备 / 待机**：这时确实还没有音频可画，退回低幅呼吸线，
      表示「程序活着但没在采」。

    没有 meter（Tk 回退、单元测试、``--no-gui``）时完全走旧版
    ``set_energy()`` 合成波形，保证接口向后兼容。
    """

    N = 110

    def __init__(self, parent=None, meter=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.meter = meter
        self._phase = 0.0
        self._energy = 0.0
        self._target = 0.0
        self._bars = None                       # (N,2) 真实包络；None = 待机
        self._smooth = np.zeros((self.N, 2), dtype=np.float32)
        self._frame = None
        # 无字体的离屏环境（CI / offscreen 截图）画文字只会得到豆腐块，索性不画。
        self._has_fonts = bool(QFontDatabase.families())
        self.setToolTip(
            "真实音频波形 · 每根柱为一个 %.0fms 片段的实际采样峰谷值，"
            "约 %.1f 秒滚动窗口（左旧右新）。静音时会变成一条平线。"
            % (BIN_SEC * 1000, self.N * BIN_SEC))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    @property
    def has_meter(self) -> bool:
        return self.meter is not None

    def set_energy(self, v: float):
        """无 meter 时的旧接口：归一化电平驱动合成呼吸波形。"""
        self._target = max(0.0, min(1.0, v))

    def _tick(self):
        self._phase += 0.09
        if self.meter is not None:
            f = self.meter.frame(self.N)
            self._frame = f
            if f.mode == WM_FILE:
                # 整段总览是静态数据，转写完也留着，供回看本次文件。
                self._bars = f.bars
            elif f.mode == WM_LIVE and f.active:
                # 只做轻度时间平滑，抑制 40ms 轮询带来的抖动；
                # 柱子取值仍是真实包络，不会凭空造出起伏。
                self._smooth = self._smooth * 0.25 + f.bars * 0.75
                self._bars = self._smooth
            else:
                self._bars = None
                self._smooth.fill(0.0)
        self._energy += (self._target - self._energy) * 0.10
        self.update()

    def _paint_standby(self, p, w, h, mid, gap, bw):
        """没有音频时的低幅呼吸线：只表达「程序活着」。"""
        grad = QLinearGradient(0, h, 0, 0)
        grad.setColorAt(0.0, QColor(FLAME_DEEP))
        grad.setColorAt(0.45, QColor(BRAND))
        grad.setColorAt(0.8, QColor(EMBER))
        grad.setColorAt(1.0, QColor(GOLD))
        p.setBrush(grad)
        p.setPen(Qt.NoPen)
        for i in range(self.N):
            x = i * gap + gap * 0.5
            env = math.sin(i / self.N * math.pi)
            if self._energy > 0.02:
                noise = (math.sin(i * 0.35 + self._phase * 2.1) * 0.5
                         + math.sin(i * 0.13 - self._phase * 1.3) * 0.35
                         + math.sin(i * 0.71 + self._phase * 3.7) * 0.15)
                amp = (0.18 + abs(noise) * 0.82) * self._energy * env
            else:
                amp = (0.015 + 0.01 * math.sin(self._phase * 0.9 + i * 0.22)) * env
            bh = max(1.6, amp * h * 0.92)
            p.setOpacity(0.35 + min(0.65, amp * 2.2))
            p.drawRoundedRect(QRectF(x - bw / 2, mid - bh / 2, bw, bh), bw / 2, bw / 2)
        p.setOpacity(1.0)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2
        n = self.N
        gap = w / n
        bw = max(2.0, gap * 0.42)

        if self._bars is None or self._frame is None:
            self._paint_standby(p, w, h, mid, gap, bw)
            p.end()
            return

        f = self._frame
        # 中线：完全静音时画面只剩这一条线，一眼能看出「没声音」。
        p.setPen(QPen(QColor(CONTROL_HI), 1))
        p.drawLine(0, int(mid), int(w), int(mid))
        p.setPen(Qt.NoPen)
        for i in range(n):
            lo, hi = float(self._bars[i, 0]), float(self._bars[i, 1])
            y1 = mid - hi * (h * 0.45)
            y2 = mid - lo * (h * 0.45)
            bh = max(1.6, y2 - y1)
            amp = min(1.0, max(abs(lo), abs(hi)))
            x = i * gap + gap * 0.5
            if f.mode == WM_FILE:
                played = (i + 0.5) / n <= f.progress
                c = (flame_color(min(1.0, 0.20 + amp * 1.5)) if played
                     else QColor(CONTROL_HI))
                alpha = (0.40 + min(0.60, amp * 2.4)) if played else 0.55
            else:
                c = flame_color(min(1.0, 0.15 + amp * 1.4))
                alpha = 0.30 + min(0.70, amp * 2.4)
            p.setOpacity(alpha)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(x - bw / 2, y1, bw, bh), bw / 2, bw / 2)
        p.setOpacity(1.0)

        if f.mode == WM_FILE:
            px = int(min(w - 1, max(0.0, f.progress * w)))
            p.setPen(QPen(QColor(GOLD), 1.5))
            p.drawLine(px, 1, px, int(h) - 1)
        if self._has_fonts:
            font = QFont("Consolas")
            font.setPointSize(8)
            p.setFont(font)
            p.setPen(QColor(FAINT))
            if f.mode == WM_FILE:
                label = "文件波形 %d%%" % int(round(f.progress * 100))
            else:
                label = ("%.1f dB" % (20 * math.log10(max(f.level, 1e-5)))
                         if f.active else "— dB")
            p.drawText(QRectF(w - 118, 2, 110, 14),
                       Qt.AlignRight | Qt.AlignVCenter, label)
        p.end()


# ---------- 状态条电平格 ----------
class LevelBars(QWidget):
    N = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(64, 16)
        self._n = 0

    def set_db(self, db: float):
        self._n = int(round(max(0.0, min(1.0, (db + 60) / 55.0)) * self.N))
        self.update()

    def clear(self):
        self._n = 0
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        w = 4.0
        gap = (self.width() - w * self.N) / (self.N - 1)
        for i in range(self.N):
            bh = 3 + (i / self.N) * 12
            x = i * (w + gap)
            if i < self._n:
                p.setBrush(flame_color(i / (self.N - 1)))
            else:
                p.setBrush(QColor(CONTROL))
            p.drawRoundedRect(QRectF(x, self.height() - bh, w, bh), 1, 1)
        p.end()


# ---------- 字幕区渲染 ----------
def _cap_html(it: dict, show_ts: bool, show_speaker: bool) -> str:
    lab = int(it.get("label", 1) or 1)
    color = SPEAKER_COLORS[(lab - 1) % len(SPEAKER_COLORS)]
    parts = []
    if show_ts:
        parts.append('<span style="color:%s; font-family:Consolas; font-size:8.5pt;">[%s]</span> '
                     % (FAINT, stamp(it.get("start", 0.0))))
    if show_speaker:
        parts.append('<b style="color:%s;">[%s]</b>&nbsp;'
                     % (color, html.escape(str(it.get("name", "说话人%d" % lab)))))
    parts.append('<span style="color:%s;">%s</span>'
                 % (color, html.escape((it.get("text") or "").strip())))
    return '<div style="margin:2px 0 6px 0;">' + "".join(parts) + "</div>"


def _draft_html(text: str, label, show_speaker: bool) -> str:
    if label and show_speaker:
        color = SPEAKER_COLORS[(int(label) - 1) % len(SPEAKER_COLORS)]
    else:
        color = DRAFT_COLOR
    return ('<div style="margin:2px 0 6px 0;"><span style="color:%s;">'
            '<img src="img://tri" width="13" height="13"> %s'
            '<span style="color:%s;"> ▍</span></span></div>'
            % (color, html.escape(text), GOLD))


# ---------- 空状态引导 ----------
class EmptyState(QWidget):
    STEPS = [
        ("设置好上方选项后点「开始听录」", "主按钮是唯一的高饱和色块，一眼定位"),
        ("全程只有一个人讲话时", "把「人数」选成「单人（不分人）」，可跳过声纹分人、画面更干净"),
        ("首次采集若被安全软件拦截", "请在弹窗点「允许」并勾选记住"),
        ("窗口会保留全部定稿文字", "多人时用不同颜色区分说话人；每次听录自动新建带时间戳的文字稿，上一场不会被覆盖"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        ghost = QLabel(self)
        pm = svg_pixmap("logo-mark.svg", 460)
        ghost.setPixmap(pm)
        ghost.setFixedSize(pm.size())
        eff = QGraphicsOpacityEffect(ghost)
        eff.setOpacity(0.07)
        ghost.setGraphicsEffect(eff)
        ghost.move(self.width() - pm.width(), 0)
        self._ghost = ghost

        box = QWidget(self)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(18)
        for i, (title, sub) in enumerate(self.STEPS, 1):
            row = QHBoxLayout()
            row.setSpacing(14)
            num = QLabel("%02d" % i)
            num.setStyleSheet("color:%s; font-family:Consolas; font-size:15pt; font-weight:bold;" % GOLD)
            num.setFixedWidth(34)
            num.setAlignment(Qt.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(2)
            b = QLabel(title)
            b.setStyleSheet("color:%s; font-size:11pt; font-weight:bold;" % FG)
            s = QLabel(sub)
            s.setStyleSheet("color:%s; font-size:9pt;" % FAINT)
            col.addWidget(b)
            col.addWidget(s)
            row.addWidget(num)
            row.addLayout(col)
            row.addStretch(1)
            lay.addLayout(row)
        lay.addStretch(1)
        self._box = box

    def resizeEvent(self, e):
        super().resizeEvent(e)
        bw = 470
        self._box.setGeometry(max(20, (self.width() - bw) // 2 - 40),
                              max(16, (self.height() - 300) // 2),
                              bw, min(320, self.height() - 32))
        pm = self._ghost.pixmap()
        self._ghost.move(self.width() - int(pm.width() * 0.72),
                         (self.height() - pm.height()) // 2)


# ---------- 准备阶段：脉冲火点 + 阶段进度 ----------
class EmberPulse(QWidget):
    """三颗依次亮起的火点，表示「程序活着、正在忙」。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(58, 18)
        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(70)

    def _tick(self):
        self._t += 0.14
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        for i in range(3):
            phase = (self._t - i * 0.6) % 3.0
            k = max(0.0, 1.0 - abs(phase - 0.6) / 1.0)
            c = flame_color(0.5 + 0.5 * k)
            c.setAlphaF(0.30 + 0.70 * k)
            r = 3.0 + 1.8 * k
            p.setBrush(c)
            p.drawEllipse(QRectF(11 + i * 18 - r, 9 - r, 2 * r, 2 * r))
        p.end()


class PreparingState(QWidget):
    """模型加载 / 文件转写期间的中央反馈层。

    准备阶段字幕区是空的，只靠状态条一行字很容易被当成死机；
    这里给脉冲动画 + 不定态进度条 + 当前阶段 + 已等待秒数 + 本次文件名。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignCenter)
        card = QWidget()
        card.setFixedWidth(540)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(30, 26, 30, 26)
        lay.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(EmberPulse())
        self.title = QLabel("正在准备，请稍等…")
        self.title.setStyleSheet("color:%s; font-size:16pt; font-weight:bold;" % FG)
        head.addWidget(self.title)
        head.addStretch(1)
        lay.addLayout(head)

        self.stage = QLabel("")
        self.stage.setStyleSheet("color:%s; font-size:11pt;" % GOLD)
        self.stage.setWordWrap(True)
        lay.addWidget(self.stage)

        self.bar = QProgressBar()
        self.bar.setObjectName("prep")
        self.bar.setRange(0, 0)          # 不定态：Qt 自带往复动画
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        lay.addWidget(self.bar)

        self.steps = QLabel("")
        self.steps.setStyleSheet("font-family:Consolas; font-size:10pt;")
        lay.addWidget(self.steps)

        self.hint = QLabel("")
        self.hint.setObjectName("faint")
        self.hint.setStyleSheet("font-size:9pt;")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        root.addWidget(card, 0, Qt.AlignCenter)

    def set_state(self, prep: dict, session: dict, mode: str):
        self.title.setText("正在转写文件，请稍等…" if mode == MODE_FILE
                           else "正在准备，请稍等…")
        step = int(prep.get("step", 0) or 0)
        total = int(prep.get("total", 0) or 0)
        prog = "(%d/%d) " % (step, total) if total else ""
        detail = (prep.get("detail") or "").strip()
        self.stage.setText("%s%s%s" % (prog, prep.get("stage", ""),
                                       ("  " + detail) if detail else ""))
        if total:
            marks = []
            for i in range(1, total + 1):
                if i < step:
                    marks.append('<span style="color:%s;">●</span>' % GOLD)
                elif i == step:
                    marks.append('<span style="color:%s; font-size:14pt;">●</span>' % EMBER_HI)
                else:
                    marks.append('<span style="color:%s;">○</span>' % CONTROL_HI)
            self.steps.setText(" ".join(marks) +
                               '　<span style="color:%s;">%d / %d 步</span>'
                               % (FAINT, step, total))
        else:
            self.steps.setText("")
        waited = max(0.0, time.monotonic() - float(prep.get("at") or time.monotonic()))
        tail = ("本次文字稿 %s.txt" % session["stem"]) if session.get("stem") else ""
        self.hint.setText("已等待 %.0f 秒 · 首次加载模型通常需要十几秒，界面仍可操作%s"
                          % (waited, ("　·　" + tail) if tail else ""))


# ---------- 高级设置 ----------
# 分组：把 21 个字段按职责聚类，不再是一整张平铺表单
ADV_GROUPS = [
    ("Whisper", ["whisper_model", "whisper_device", "whisper_compute",
                 "download_root", "beam_size"]),
    ("Confucius 精修", ["confucius_url", "confucius_dir"]),
    ("文本 / 翻译", ["hot_word", "ollama_url", "translate_model", "translate_timeout"]),
    ("分人 / 声纹", ["speaker_threshold", "speaker_names", "enroll", "save_voices"]),
    ("VAD / 采集", ["vad_min_silence", "silence_warn", "duration"]),
    ("输出 / 调试", ["outdir", "formats", "simulate"]),
]


class AdvancedDialog(QDialog):
    """模态高级选项：分组表单 + 模型来源说明 + 两个布尔开关。"""

    def __init__(self, parent, opts: dict):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("高级设置")
        self.resize(720, 780)
        self.setMinimumSize(640, 480)
        self._opts = opts
        self._fvars = {}
        self._field_meta = {k: (cap, kind, ch) for k, cap, kind, ch in ADVANCED_FIELDS}

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 16)
        root.setSpacing(10)

        title = QLabel("高级选项")
        title.setStyleSheet("font-size:13pt; font-weight:bold;")
        root.addWidget(title)
        sub = QLabel("停止后可修改，改动会自动保存")
        sub.setObjectName("faint")
        root.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setStyleSheet("background:%s;" % PANEL)
        blay = QVBoxLayout(body)
        blay.setContentsMargins(0, 6, 8, 0)
        blay.setSpacing(4)

        for gtitle, keys in ADV_GROUPS:
            gt = QLabel(gtitle.upper() if gtitle == "Whisper" else gtitle)
            gt.setObjectName("grouptitle")
            f = QFont(gt.font())
            f.setLetterSpacing(QFont.PercentageSpacing, 150)
            gt.setFont(f)
            blay.addWidget(gt)
            line = QFrame()
            line.setObjectName("groupbox")
            line.setFixedHeight(1)
            blay.addWidget(line)
            grid = QGridLayout()
            grid.setHorizontalSpacing(18)
            grid.setVerticalSpacing(8)
            for i, key in enumerate(keys):
                caption, kind, choices = self._field_meta[key]
                lab = QLabel(caption)
                lab.setObjectName("muted")
                lab.setStyleSheet("font-size:9pt;")
                if kind == "choice":
                    w = QComboBox()
                    w.setView(QListView())
                    w.addItems(choices.labels)
                    w.setCurrentText(choices.label(opts.get(key, "auto")))
                    self._fvars[key] = (w, kind, choices)
                else:
                    w = QLineEdit()
                    w.setText(str(opts.get(key, "") if opts.get(key) is not None else ""))
                    self._fvars[key] = (w, kind, None)
                cell = QVBoxLayout()
                cell.setSpacing(3)
                cell.addWidget(lab)
                cell.addWidget(w)
                grid.addLayout(cell, i // 2, i % 2)
            blay.addLayout(grid)

        note = QFrame()
        note.setObjectName("note")
        nlay = QVBoxLayout(note)
        nlay.setContentsMargins(14, 12, 14, 12)
        nt = QLabel("Whisper 模型从哪来？")
        nt.setStyleSheet("color:%s; font-weight:bold; font-size:9.5pt;" % GOLD)
        np_ = QLabel("首次使用点主界面的「下载权重…」按档位自动下载；"
                     "也可把同事的 models/whisper/<档位>/ 整个目录拷过来离线使用；"
                     "高级玩家可在上方直接填自定义模型路径。国内网络可设 "
                     "HF_ENDPOINT=https://hf-mirror.com")
        np_.setObjectName("muted")
        np_.setStyleSheet("font-size:9pt;")
        np_.setWordWrap(True)
        nlay.addWidget(nt)
        nlay.addWidget(np_)
        blay.addWidget(note)
        blay.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        bools = QHBoxLayout()
        self.v_nostream = QCheckBox("关闭流式草稿（整句说完才出字）")
        self.v_nostream.setChecked(bool(opts.get("no_streaming", False)))
        self.v_loop = QCheckBox("模拟回放循环")
        self.v_loop.setChecked(bool(opts.get("loop_sim", False)))
        bools.addWidget(self.v_nostream)
        bools.addWidget(self.v_loop)
        bools.addStretch(1)
        root.addLayout(bools)

        foot = QHBoxLayout()
        foot.addStretch(1)
        ok = QPushButton("确定")
        ok.setObjectName("primary")
        ok.setStyleSheet("font-size:10pt; padding: 8px 28px;")
        ok.clicked.connect(self.accept)
        foot.addWidget(ok)
        root.addLayout(foot)

    def values(self) -> dict:
        out = {}
        for key, (w, kind, choices) in self._fvars.items():
            if choices:
                out[key] = choices.value(w.currentText())
            else:
                out[key] = _coerce(kind, w.text())
        out["no_streaming"] = self.v_nostream.isChecked()
        out["loop_sim"] = self.v_loop.isChecked()
        return out


# ---------- 主窗口 ----------
class MainWindow(QWidget):
    """曜石 × 火焰主窗口。controller 协议与旧版一致：
    start(opts) / start_file(opts, path) / stop() / is_running() /
    apply_live(opts) / save(opts) / on_close() / settings / defaults。
    """

    def __init__(self, display: LiveDisplay, controller, title=None):
        super().__init__()
        self.display = display
        self.controller = controller
        self.opts = dict(controller.settings)
        self._ready = False
        self._running = False
        self._last_running = False
        self._stopping = False
        self._stop_at = 0.0
        self._mode = MODE_LIVE
        self._dl_state = {"result": None}
        self._widget_map = {}

        self.setWindowTitle(title or display.title)
        self.resize(1200, 620)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setWindowOpacity(0.98)
        try:
            self.setWindowIcon(QIcon(svg_pixmap("logo-mark.svg", 64)))
        except Exception:
            pass

        self._build()
        self._bind()
        self.apply_mode(self.opts.get("mode") if self.opts.get("mode") in (MODE_LIVE, MODE_FILE) else MODE_LIVE)
        self._ready = True

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start(150)

    # ---------- 布局 ----------
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ② 模式栏
        header = QFrame()
        header.setObjectName("band")
        hh = QHBoxLayout(header)
        hh.setContentsMargins(14, 9, 14, 9)
        hh.setSpacing(14)

        logo = QLabel()
        logo.setPixmap(svg_pixmap("logo-mark.svg", 24))
        hh.addWidget(logo)
        name = QLabel("SpeakerCaption")
        name.setStyleSheet("font-weight:bold; letter-spacing:2px; font-size:10.5pt;")
        hh.addWidget(name)
        words = QLabel()
        words.setPixmap(svg_pixmap("brand-words.svg", 16))
        hh.addWidget(words)

        segbox = QFrame()
        segbox.setObjectName("segbox")
        sl = QHBoxLayout(segbox)
        sl.setContentsMargins(3, 3, 3, 3)
        sl.setSpacing(2)
        self.seg_live = QPushButton("实时听录")
        self.seg_file = QPushButton("听录文件")
        for b in (self.seg_live, self.seg_file):
            b.setProperty("class", "seg")
            b.setCheckable(True)
            sl.addWidget(b)
        self.seg_live.setChecked(True)
        hh.addWidget(segbox)

        tagline = QLabel("抓电脑正在播的声音　·　或直接转写音视频文件")
        tagline.setObjectName("faint")
        tagline.setStyleSheet("font-size:9pt;")
        hh.addWidget(tagline)
        hh.addStretch(1)
        self.engine_pill = QLabel("● Whisper large-v3 · 就绪")
        self.engine_pill.setObjectName("pill")
        hh.addWidget(self.engine_pill)
        root.addWidget(header)

        # ③ 文件行
        self.file_row = QFrame()
        self.file_row.setObjectName("band")
        fl = QVBoxLayout(self.file_row)
        fl.setContentsMargins(14, 8, 14, 8)
        fl.setSpacing(5)
        fr = QHBoxLayout()
        fr.setSpacing(10)
        flab = QLabel("媒体文件")
        flab.setObjectName("muted")
        self.file_edit = QLineEdit(str(self.opts.get("media_file", "") or ""))
        self.file_edit.setPlaceholderText("未选择文件")
        self.browse_btn = QPushButton("浏览…")
        self.browse_btn.setProperty("class", "ghost")
        fr.addWidget(flab)
        fr.addWidget(self.file_edit, 1)
        fr.addWidget(self.browse_btn)
        fl.addLayout(fr)
        fhint = QLabel("支持 mp3 / m4a / wav / flac / mp4 / mkv（视频自动抽音轨）· 按原文件名存到输出目录")
        fhint.setObjectName("faint")
        fhint.setStyleSheet("font-size:8.5pt;")
        fl.addWidget(fhint)
        root.addWidget(self.file_row)
        self.file_row.hide()

        # ④ 控制面板
        deck = QFrame()
        deck.setObjectName("band")
        dl = QVBoxLayout(deck)
        dl.setContentsMargins(14, 10, 14, 9)
        dl.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setSpacing(14)
        self.start_btn = QPushButton(" 开始听录")
        self.start_btn.setObjectName("primary")
        self.start_btn.setIcon(QIcon(_tri_pixmap("#2A1306", "play")))
        row1.addWidget(self.start_btn)

        self.v_translate = QCheckBox("译成中文")
        self.v_translate.setChecked(bool(self.opts.get("translate", False)))
        self.v_script = self._combo(SCRIPT, SCRIPT.label(self.opts.get("script", "simplified")), "script")
        self.v_ts = QCheckBox("时间戳")
        self.v_ts.setChecked(bool(self.opts.get("timestamps", False)))
        g_out = self._group("输出", [self.v_translate, self._field("字形", self.v_script), self.v_ts])
        row1.addLayout(g_out)

        self.v_lang = self._combo(LANG, LANG.label(self.opts.get("language") or "auto"), "language")
        self.v_count = self._combo(COUNT, COUNT.label(str(self.opts.get("num_speakers", 0))), "num_speakers")
        row1.addLayout(self._group("输入", [self._field("音频语言", self.v_lang),
                                            self._field("人数", self.v_count)]))
        row1.addStretch(1)
        dl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.v_final = self._combo(FINAL, FINAL.label(self.opts.get("final", "whisper")), "final")
        _wm = str(self.opts.get("whisper_model") or "large-v3")
        self.v_whisper = self._combo(
            WHISPER, WHISPER.label(_wm) if _wm in WHISPER.v2l else WHISPER_CUSTOM,
            "whisper_model")
        row2.addLayout(self._group("引擎", [self._field("精修后端", self.v_final),
                                            self._field("Whisper 档位", self.v_whisper)]))
        self.adv_btn = QPushButton("高级设置…")
        self.adv_btn.setProperty("class", "ghost")
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.setProperty("class", "ghost")
        self.dl_btn = QPushButton("下载权重…")
        self.dl_btn.setProperty("class", "ghost")
        row2.addWidget(self.adv_btn)
        row2.addWidget(self.reset_btn)
        row2.addWidget(self.dl_btn)
        dl.addLayout(row2)
        tip = QLabel("默认：英文原文、中文简体、无时间戳；人数选「单人」可跳过分人。"
                     "「译成中文」需本机 Ollama；运行中可切换字形和时间戳。")
        tip.setObjectName("faint")
        tip.setStyleSheet("font-size:8.5pt;")
        dl.addWidget(tip)
        root.addWidget(deck)

        # ⑤ 字幕舞台：波形 + 字幕 / 空状态
        stage = QWidget()
        st = QVBoxLayout(stage)
        st.setContentsMargins(0, 0, 0, 0)
        st.setSpacing(0)
        # 控制器持有真实波形总线；老版本 controller / 测试桩没有这个属性时
        # 自动退回旧的合成呼吸波形，不会抛 AttributeError。
        self.wave = WaveWidget(meter=getattr(self.controller, "wave_meter", None))
        st.addWidget(self.wave)
        self.stack = QStackedWidget()
        self.captions = _CaptionView()
        # 流式草稿的 ▷ 前缀：雅黑不含该字形，注册一张自绘的琥珀小三角。
        # 注意：QTextEdit.setHtml 会清空文档的资源缓存（实测），
        # 所以每次 setHtml 前都要重新注册，不能只在这里注册一次。
        self._tri_pm = _tri_pixmap(DRAFT_COLOR, "play", 26)
        self._register_draft_icon()
        self.empty = EmptyState()
        self.preparing = PreparingState()
        self.stack.addWidget(self.captions)
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.preparing)
        st.addWidget(self.stack, 1)
        root.addWidget(stage, 1)

        # ⑥ 状态条
        status = QFrame()
        status.setObjectName("statusbar")
        sh = QHBoxLayout(status)
        sh.setContentsMargins(14, 7, 14, 7)
        sh.setSpacing(12)
        self.level = LevelBars()
        self.prep_bar = QProgressBar()
        self.prep_bar.setObjectName("prep")
        self.prep_bar.setRange(0, 0)        # 不定态：加载模型没有可信百分比
        self.prep_bar.setTextVisible(False)
        self.prep_bar.setFixedWidth(110)
        self.prep_bar.setFixedHeight(6)
        self.prep_bar.hide()
        self.status_label = QLabel("设置好选项后点「开始听录」")
        self.status_label.setStyleSheet("font-family:Consolas; font-size:9.5pt; color:%s;" % MUTED)
        self.session_label = QLabel("")
        self.session_label.setObjectName("faint")
        self.session_label.setStyleSheet("font-family:Consolas; font-size:8.5pt;")
        sh.addWidget(self.level)
        sh.addWidget(self.prep_bar)
        sh.addWidget(self.status_label, 1)
        sh.addWidget(self.session_label)
        root.addWidget(status)

    def _combo(self, choices, current, key):
        cb = QComboBox()
        cb.setView(QListView())
        cb.addItems(choices.labels)
        cb.setCurrentText(current)
        self._widget_map[key] = cb
        return cb

    def _field(self, tag, widget):
        h = QHBoxLayout()
        h.setSpacing(6)
        lab = QLabel(tag)
        lab.setObjectName("muted")
        lab.setStyleSheet("font-size:9pt;")
        h.addWidget(lab)
        h.addWidget(widget)
        return h

    def _group(self, name, items):
        h = QHBoxLayout()
        h.setSpacing(10)
        lab = QLabel(name)
        lab.setObjectName("grpname")
        f = QFont(lab.font())
        f.setLetterSpacing(QFont.PercentageSpacing, 160)
        lab.setFont(f)
        h.addWidget(lab)
        for it in items:
            h.addLayout(it) if isinstance(it, QHBoxLayout) else h.addWidget(it)
        sep = QFrame()
        sep.setObjectName("divider")
        sep.setFixedHeight(22)
        h.addWidget(sep)
        return h

    # ---------- 绑定 ----------
    def _bind(self):
        self.start_btn.clicked.connect(self._start_click)
        self.seg_live.clicked.connect(lambda: self.apply_mode(MODE_LIVE))
        self.seg_file.clicked.connect(lambda: self.apply_mode(MODE_FILE))
        self.browse_btn.clicked.connect(self._browse)
        self.adv_btn.clicked.connect(self._open_advanced)
        self.reset_btn.clicked.connect(self._restore_defaults)
        self.dl_btn.clicked.connect(self._download_whisper)
        self._widget_map["translate"] = self.v_translate
        self._widget_map["timestamps"] = self.v_ts
        for w in (self.v_translate, self.v_ts):
            w.toggled.connect(self._on_change)
        for cb in (self.v_script, self.v_lang, self.v_count, self.v_final, self.v_whisper):
            cb.currentTextChanged.connect(self._on_change)
        self.file_edit.textChanged.connect(self._on_change)

    # ---------- 选项收集 / 模式 ----------
    def collect(self) -> dict:
        out = dict(self.opts)
        out.update({
            "translate": self.v_translate.isChecked(),
            "script": SCRIPT.value(self.v_script.currentText()),
            "timestamps": self.v_ts.isChecked(),
            "language": LANG.value(self.v_lang.currentText()),
            "final": FINAL.value(self.v_final.currentText()),
            "num_speakers": int(COUNT.value(self.v_count.currentText())),
            "media_file": self.file_edit.text(),
            "mode": self._mode,
        })
        wm = WHISPER.value(self.v_whisper.currentText())
        if wm and wm != "__custom__":
            out["whisper_model"] = wm
        return out

    def _on_change(self, *_):
        if not self._ready:
            return
        cur = self.collect()
        self.opts.update(cur)
        if self._running:
            self.controller.apply_live(cur)
        else:
            self.controller.save(cur)

    def apply_mode(self, mode):
        self._mode = mode
        self.seg_live.setChecked(mode == MODE_LIVE)
        self.seg_file.setChecked(mode == MODE_FILE)
        self.file_row.setVisible(mode == MODE_FILE)
        if not self._running:
            self.start_btn.setText(" 开始转写" if mode == MODE_FILE else " 开始听录")
        if self._ready:
            self._on_change()

    # ---------- 运行控制 ----------
    def _start_click(self):
        if self._running:
            if self._stopping:
                return
            self._stopping = True
            self._stop_at = time.monotonic()
            self.controller.stop()
            self.start_btn.setText("正在停止…")
            self.start_btn.setEnabled(False)
            self._set_status("正在停止…（等待最后几句定稿）", MUTED)
            return
        cur = self.collect()
        if self._mode == MODE_FILE:
            path = (cur.get("media_file") or "").strip()
            if not path:
                self._set_status("⚠ 请先选择要听录的音视频文件", WARN)
                return
            if not Path(path).is_file():
                self._set_status("文件不存在: %s" % path, "#FF7A6E")
                return
            self.controller.start_file(cur, path)
        else:
            self.controller.start(cur)
        # 会话构造失败时 controller 已把错误写进 display，不要假装在运行
        if not self.controller.is_running():
            return
        # 立刻给反馈：模型加载动辄十几秒，不能等到音频流开起来才更新状态条
        self._set_status("⏳ 正在准备，请稍等…（正在加载模型）", GOLD)
        self._set_running_ui(True)

    def _set_running_ui(self, running):
        self._running = running
        self._stopping = False
        self.start_btn.setProperty("stop", running)
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.start_btn.setText(" 停止" if running else
                               (" 开始转写" if self._mode == MODE_FILE else " 开始听录"))
        self.start_btn.setIcon(QIcon(_tri_pixmap(
            "#FFF3EF" if running else "#2A1306", "stop" if running else "play")))
        self.start_btn.setEnabled(True)
        live_keys = ("script", "timestamps")
        for key, w in self._widget_map.items():
            w.setEnabled((not running) or key in live_keys)
        for b in (self.adv_btn, self.reset_btn, self.dl_btn,
                  self.seg_live, self.seg_file, self.file_edit, self.browse_btn):
            b.setEnabled(not running)
        self._sync_engine_pill(running, bool(self.display.snapshot().get("preparing")))
        if not running:
            self.wave.set_energy(0)
            self.level.clear()

    def _sync_engine_pill(self, running, preparing=False):
        tier = WHISPER.value(self.v_whisper.currentText())
        tier = "" if tier == "__custom__" else (" " + (tier or ""))
        if preparing:
            txt, col = "● 正在加载模型…", GOLD
        elif running:
            txt, col = "● Whisper%s · 推理中" % tier, GOLD
        else:
            txt, col = "● Whisper%s · 就绪" % tier, MUTED
        if self.engine_pill.text() != txt:
            self.engine_pill.setText(txt)
            self.engine_pill.setStyleSheet("color:%s;" % col)

    # ---------- 高级设置 / 恢复默认 / 下载 ----------
    def _open_advanced(self):
        dlg = AdvancedDialog(self, dict(self.opts))
        if dlg.exec() == QDialog.Accepted:
            self.opts.update(dlg.values())
            _wm = str(self.opts.get("whisper_model") or "")
            self._ready = False
            self.v_whisper.setCurrentText(
                WHISPER.label(_wm) if _wm in WHISPER.v2l else WHISPER_CUSTOM)
            self._ready = True
            self.controller.save(dict(self.opts))
            self._set_status("高级选项已保存", MUTED)

    def _restore_defaults(self):
        d = dict(self.controller.defaults)
        self.opts.update(d)
        self._ready = False
        self.v_translate.setChecked(bool(d.get("translate", False)))
        self.v_script.setCurrentText(SCRIPT.label(d.get("script", "simplified")))
        self.v_ts.setChecked(bool(d.get("timestamps", False)))
        self.v_lang.setCurrentText(LANG.label(d.get("language") or "auto"))
        self.v_final.setCurrentText(FINAL.label(d.get("final", "whisper")))
        self.v_count.setCurrentText(COUNT.label(str(d.get("num_speakers", 0))))
        _wm = str(d.get("whisper_model") or "large-v3")
        self.v_whisper.setCurrentText(WHISPER.label(_wm) if _wm in WHISPER.v2l else WHISPER_CUSTOM)
        self._ready = True
        self.controller.save(dict(self.opts))
        self._set_status("已恢复默认设置", MUTED)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择要听录的音视频文件", "",
            "音视频文件 (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma "
            "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.wmv *.ts *.m4v);;所有文件 (*.*)")
        if p:
            self.file_edit.setText(p)
            self._on_change()

    def _download_whisper(self):
        tier = WHISPER.value(self.v_whisper.currentText())
        if not tier or tier == "__custom__":
            QMessageBox.information(
                self, "Whisper 权重",
                "先在上方下拉里选一个档位（tiny / base / small / medium /\n"
                "large-v3 / large-v3-turbo），再点「下载权重…」。\n\n"
                "如果已经有一份本地模型了，去「高级设置」把「Whisper 模型」\n"
                "填成它的完整路径即可，不用下载。")
            return
        if self.controller.is_running():
            return
        self.display.set_notice("正在下载 Whisper 权重 %s …（可继续使用界面）" % tier)
        self.dl_btn.setEnabled(False)
        dl_state = self._dl_state

        def work():
            import subprocess
            script = ROOT / "scripts" / "download_models.py"
            cmd = [sys.executable, str(script), "--with-whisper", tier]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=3600)
                out = (p.stdout or "") + (p.stderr or "")
                ok = p.returncode == 0
            except Exception as e:  # noqa: BLE001
                out = "%s: %s" % (type(e).__name__, e)
                ok = False
            tail = "\n".join([ln for ln in out.strip().splitlines() if ln.strip()][-10:])
            dl_state["result"] = (ok, tier, tail)

        threading.Thread(target=work, name="whisper-dl", daemon=True).start()

    # ---------- 状态条 ----------
    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            "font-family:Consolas; font-size:9.5pt; color:%s;" % color)

    def _register_draft_icon(self):
        """把自绘琥珀三角注册进字幕文档的资源缓存。

        QTextEdit.setHtml 会清空文档资源缓存（实测），所以 refresh 里
        每次 setHtml 前都要重注册一次。
        """
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QTextDocument
        self.captions.document().addResource(
            QTextDocument.ImageResource, QUrl("img://tri"), self._tri_pm)

    # ---------- 主循环轮询（等价旧版 refresh）----------
    def refresh(self):
        snap = self.display.snapshot()
        prep = snap.get("preparing") or {}
        sess = snap.get("session") or {}

        # 下载结果回到主线程弹窗
        if self._dl_state["result"] is not None:
            ok, tier, tail = self._dl_state["result"]
            self._dl_state["result"] = None
            self.dl_btn.setEnabled(not self._running)
            if ok:
                self.display.set_notice("Whisper %s 已下载到 models/whisper/%s/" % (tier, tier))
                QMessageBox.information(
                    self, "Whisper 权重",
                    "档位 %s 下载完成。\n\n位置：models/whisper/%s/\n\n"
                    "这个目录可以整个拷给同事，拷过去就完全离线可用。\n\n%s"
                    % (tier, tier, tail))
            else:
                self.display.set_notice("Whisper %s 下载失败（详见弹窗）" % tier)
                QMessageBox.critical(
                    self, "Whisper 权重下载失败",
                    "档位 %s 下载失败。\n\n%s\n\n"
                    "国内网络可改用镜像：\n"
                    "python scripts/download_models.py --with-whisper %s "
                    "--endpoint https://hf-mirror.com" % (tier, tail, tier))

        if snap["dirty"]:
            html_parts = []
            for it in snap["finals"]:
                html_parts.append(_cap_html(it, snap["show_timestamps"],
                                            snap.get("show_speaker", True)))
            if snap["partial"]:
                html_parts.append(_draft_html(snap["partial"], snap["partial_label"],
                                              snap.get("show_speaker", True)))
            self._register_draft_icon()
            self.captions.setHtml("".join(html_parts))
            sb = self.captions.verticalScrollBar()
            sb.setValue(sb.maximum())
            self.display.clear_dirty()

        # 舞台页面：准备态优先（此时字幕区是空的，只靠状态条容易被当成死机）
        is_empty = not snap["finals"] and not snap["partial"] and not snap["error"]
        target = (self.preparing if prep else
                  (self.empty if is_empty else self.captions))
        if self.stack.currentWidget() is not target:
            self.stack.setCurrentWidget(target)
        if prep:
            self.preparing.set_state(prep, sess, self._mode)
        self.prep_bar.setVisible(bool(prep))
        self.session_label.setText(
            ("本次: %s.txt" % sess["stem"]) if sess.get("stem") else "")
        self.session_label.setToolTip(
            ("输出目录：%s" % sess["outdir"]) if sess.get("outdir") else "")

        # 波形：有 meter 时控件自己每 40ms 拉真实 PCM 包络，这里绝不再喂
        # 由 sin() 合成的能量值，否则会把真实波形盖掉。
        # 无 meter（Tk 回退 / 旧 controller 测试桩）才走原来的标量合成波形。
        stats = snap["stats"]
        if self._running:
            if prep:
                if not self.wave.has_meter:
                    self.wave.set_energy(0.14 + 0.08 * math.sin(time.monotonic() * 2.2))
                self.level.clear()
            elif stats.get("file_stage"):
                if not self.wave.has_meter:
                    self.wave.set_energy(0.55 + 0.15 * math.sin(time.monotonic() * 2))
                self.level.clear()
            else:
                if self.wave.has_meter and self.wave.meter.active:
                    db = self.wave.meter.level_db()
                else:
                    peak = float(stats.get("peak", 0.0))
                    db = 20 * math.log10(max(peak, 1e-5))
                    if not self.wave.has_meter:
                        self.wave.set_energy(max(0.0, min(1.0, (db + 55) / 45)))
                self.level.set_db(db)

        # 状态条优先级（与旧版一致）
        text, color = self._status_of(snap)
        self._set_status(text, color)

        running = self.controller.is_running()
        if running != self._last_running:
            self._last_running = running
            self._set_running_ui(running)
        self._sync_engine_pill(self._running, bool(prep))

    def _status_of(self, snap):
        if snap.get("error"):
            return "⚠ " + str(snap["error"])[:150], "#FF7A6E"
        # 「停止中」要压过准备态：用户在加载模型时点停止，必须立刻看到回应
        if self._stopping:
            waited = time.monotonic() - self._stop_at
            return (("正在停止…（等待最后几句定稿，已 %.0fs）" % waited) if waited < 12
                    else ("仍在收尾，可关闭窗口强制退出（已等待 %.0fs）" % waited), MUTED)
        prep = snap.get("preparing") or {}
        if prep:
            waited = max(0.0, time.monotonic() -
                         float(prep.get("at") or time.monotonic()))
            step, total = int(prep.get("step", 0)), int(prep.get("total", 0))
            prog = "%d/%d" % (step, total) if total else ""
            parts = [p for p in (prog, prep.get("stage", ""),
                                 prep.get("detail", "")) if p]
            return ("⏳ 正在准备，请稍等…  %s · 已 %.0fs" %
                    ("  ".join(parts), waited), GOLD)
        stats = snap["stats"]
        if stats.get("file_stage"):
            stage = stats.get("file_stage")
            detail = stats.get("file_detail", "")
            return ("文件听录 · %s%s" % (stage, ("  " + detail) if detail else ""),
                    GOLD if stage == "done" else MUTED)
        if stats:
            peak = float(stats.get("peak", 0.0))
            db = 20 * math.log10(max(peak, 1e-5))
            if stats.get("warn"):
                return ("电平 %5.1f dB  ⚠ 连续 %.0fs 没收到声音 —— 多半是安全软件拦了麦克风，"
                        "请点『允许』并勾选记住" % (db, stats.get("silent_sec", 0)), "#FF7A6E")
            if stats.get("blocked_hint") and stats.get("sound_ratio", 1.0) < 0.01:
                return "电平 %5.1f dB  ⚠ 开流偏慢且几乎全程静音，检查授权弹窗" % db, WARN
            return "电平 %5.1f dB  ● 采集中  %s" % (db, str(stats.get("device", ""))[:44]), GOLD
        notice = snap.get("notice", "")
        if notice and not self._running:
            return notice, MUTED
        return (("选择媒体文件后点「开始转写」" if self._mode == MODE_FILE
                 else "设置好选项后点「开始听录」"), MUTED)

    def closeEvent(self, e):
        self.controller.on_close()
        super().closeEvent(e)


class _CaptionView(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setObjectName("captions")
        self.setReadOnly(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)


def gui_main(display: LiveDisplay, controller, title=None):
    """与旧版 gui_main 同签名。创建 QApplication（若不存在）并进入事件循环。"""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(_qss())
    win = MainWindow(display, controller, title)
    win.show()
    try:
        app.exec()
    finally:
        controller.stop()
