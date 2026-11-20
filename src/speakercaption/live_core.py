"""实时听录主逻辑：系统声音 → VAD 断句 → 流式出字 → 在线分人 → 句末精修 → 写盘。

一句话说清和离线版的区别：
  离线版 diarize.py 要拿到整段音频再全局聚类；
  这里每句 VAD 一结束就**立刻**给一个说话人标签，绝不回头改标签。
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import __version__
from .diarize import Segment
from .outputs import write_outputs
from .session_stem import make_session_stem
from .live_audio import AudioSource, CaptureStats, security_soft_running
from .live_asr import StreamingASR, build_finalizer
from .live_gui import LiveDisplay
from .live_speaker import (OnlineSpeakerTracker, SingleSpeakerTracker,
                          build_extractor, DEFAULT_WIN_SEC)
from .settings import LiveOptions, UI_KEYS
from .text_pipeline import (SCRIPT_CHOICES, SCRIPT_SIMPLIFIED, build_pipeline)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="live.py",
        description="实时抓取电脑正在播放的声音，流式转写并标注说话人（会议 / 直播播客）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version="SpeakerCaption live " + __version__)

    g = p.add_argument_group("采集")
    g.add_argument("--backend", choices=["auto", "wasapi", "sd", "file"], default="auto",
                   help="auto: Windows 用 WASAPI 回环，其它平台用 sounddevice")
    g.add_argument("--device", default=None, help="sounddevice 设备编号/名称（--list-devices 查看）")
    g.add_argument("--list-devices", action="store_true", help="列出采集设备后退出")
    g.add_argument("--simulate", default=None, metavar="WAV",
                   help="用 wav 按实时节奏回放，验证整条链路（自动切到 file 后端）")
    g.add_argument("--loop-sim", action="store_true", help="模拟回放循环")
    g.add_argument("--duration", type=float, default=0, help="采集多少秒后自动退出（0=一直跑）")
    g.add_argument("--silence-warn", type=float, default=20.0,
                   help="连续多少秒无声音就在状态条报警（0=关闭）")

    g = p.add_argument_group("说话人")
    g.add_argument("--num-speakers", type=int, default=0,
                   help="0=自动新增；1=单人直播（完全不做分人）；2~6=已知人数")
    g.add_argument("--single-speaker", action="store_true",
                   help="--num-speakers 1 的快捷开关：单人直播，跳过声纹模型，"
                        "窗口与 txt/srt 都不显示说话人")
    g.add_argument("--speaker-threshold", type=float, default=0.60,
                   help="声纹余弦阈值：高于判为同一人；实测同人 0.72~0.77、跨人 0.30~0.38")
    g.add_argument("--speaker-names", default="", help='例: "1=主持人,2=嘉宾"')
    g.add_argument("--enroll", default=None, help="载入上次保存的声纹，保持跨次说话人标签一致")
    g.add_argument("--save-voices", default=None,
                   help="退出时把声纹存到该路径（默认 <outdir>/voices.json）")

    g = p.add_argument_group("ASR / 文本")
    g.add_argument("--final", choices=["none", "whisper", "confucius",
                                       "confucius-local"], default="whisper",
                   help="句末精修后端；none=只用流式草稿（延迟最低、无标点）；"
                        "confucius-local=进程内直调 Confucius，不用先起 7861 服务")
    g.add_argument("--no-streaming", action="store_true",
                   help="关掉流式草稿，只在整句说完才出字")
    g.add_argument("--language", default="auto",
                   help="音频语言 auto/zh/en；强制 zh 会把英文语音解码成中文译文，默认自动检测")
    g.add_argument("--script", choices=list(SCRIPT_CHOICES), default=SCRIPT_SIMPLIFIED,
                   help="中文字形：简体 / 繁体 / 保持 ASR 原样（默认简体）")
    g.add_argument("--traditional", action="store_true", help="--script traditional 的快捷开关")
    g.add_argument("--simplify", action="store_true", help="--script simplified 的快捷开关（兼容旧参数）")
    g.add_argument("--translate", action="store_true",
                   help="把识别出的外文译成中文（需本地 Ollama，默认关闭）")
    g.add_argument("--translate-model", default="auto", help="Ollama 模型名；auto=自动挑已安装的翻译模型")
    g.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    g.add_argument("--translate-timeout", type=float, default=60.0)
    g.add_argument("--hot-word", default="", help="Confucius 热词，逗号分隔")
    g.add_argument("--confucius-url", default="http://127.0.0.1:7861",
                   help="Confucius HTTP 服务地址（仅 --final confucius 用）")
    g.add_argument("--confucius-dir", default=None,
                   help="Confucius4-R2T2 项目目录（仅 --final confucius-local 用）；"
                        "留空则自动在常见安装位置查找，也可用环境变量 CONFUCIUS_DIR")
    g.add_argument("--whisper-model", default="large-v3",
                   help="档位名（tiny/base/small/medium/large-v3）或本地模型目录路径")
    g.add_argument("--whisper-device", default="auto", choices=["auto", "cuda", "cpu"])
    g.add_argument("--whisper-compute", default="auto")
    g.add_argument("--download-root", default=None,
                   help="Whisper 权重目录；留空则用工程内 models/whisper/<档位>，"
                        "或回退到 HuggingFace 自动下载")
    g.add_argument("--beam-size", type=int, default=5)
    g.add_argument("--paraformer-dir", default=str(MODELS / "streaming-paraformer"))

    g = p.add_argument_group("VAD")
    g.add_argument("--vad-threshold", type=float, default=0.5)
    g.add_argument("--vad-min-silence", type=float, default=0.45, help="静音多久算一句结束")
    g.add_argument("--vad-min-speech", type=float, default=0.25)
    g.add_argument("--vad-max-speech", type=float, default=12.0, help="单句最长，超过强制断句")

    g = p.add_argument_group("输出 / 界面")
    g.add_argument("--outdir", default=str(ROOT / "live_out"))
    g.add_argument("--stem", default="live", help="输出文件名（live.txt / live.srt / ...）")
    g.add_argument("--stem-suffix", choices=["timestamp", "none"], default="timestamp",
                   help="每次听录自动加时间后缀并新建文件，绝不覆盖上一场文字稿；"
                        "none=沿用固定文件名并覆盖（旧行为，供脚本批处理用）")
    g.add_argument("--timestamps", action="store_true", help="txt 每行前加时间戳（默认不加）")
    g.add_argument("--formats", default="txt,srt,json,rttm")
    g.add_argument("--no-gui", action="store_true", help="不弹窗口，仅控制台 + 文件")
    g.add_argument("--config", default=str(ROOT / "settings.json"),
                   help="图形界面选项保存位置")
    g.add_argument("--models-dir", default=str(MODELS))
    return p.parse_args(argv)


def normalize_args(args):
    """Resolve legacy aliases and keep mutually exclusive options sane."""
    # 单人模式只有一个状态位：num_speakers == 1。GUI 的「单人（不分人）」、
    # 命令行 --num-speakers 1 与 --single-speaker 三条路都归一到这里。
    if getattr(args, "single_speaker", False):
        args.num_speakers = 1
    args.single_speaker = (int(getattr(args, "num_speakers", 0) or 0) == 1)
    if getattr(args, "traditional", False):
        args.script = "traditional"
    elif getattr(args, "simplify", False):
        args.script = SCRIPT_SIMPLIFIED
    if args.script not in SCRIPT_CHOICES:
        args.script = SCRIPT_SIMPLIFIED
    lang = str(getattr(args, "language", "auto") or "auto").lower()
    args.language = None if lang in ("auto", "none", "null", "") else lang
    # 译中文时先让 Whisper 忠实转写原语言，再由本地模型翻译文本。
    # 若同时强制 zh，Whisper 会把英文直接解码成中文，等于重复翻译。
    if getattr(args, "translate", False) and args.language in ("zh", "yue", "wuu"):
        print("[文本] 已开启译中文，语言提示自动改为 auto（避免 Whisper 抢先翻译）", flush=True)
        args.language = None
    return args


def explicit_args(argv):
    """Return only the CLI flags the user actually typed (GUI override)."""
    args = parse_args(argv)
    defaults = vars(parse_args([]))
    return {k: v for k, v in vars(args).items()
            if k in defaults and v != defaults[k] and k != "no_gui"}


def build_args(base, opts: dict):
    """CLI namespace overlaid with GUI settings."""
    d = vars(base).copy()
    for k, v in (opts or {}).items():
        if k in UI_KEYS:
            d[k] = v
    d["traditional"] = False
    d["simplify"] = False
    return normalize_args(argparse.Namespace(**d))


def parse_names(spec: str):
    out = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[int(k.strip())] = v.strip()
            except ValueError:
                pass
    return out


def build_vad(args, models_dir: Path):
    """实时断句用的 silero VAD（与文件听录共用同一套参数）。"""
    vad_file = models_dir / "silero_vad.onnx"
    if not vad_file.is_file():
        raise RuntimeError("缺 VAD 模型 %s，运行 python scripts/download_models.py --with-live" % vad_file)
    from .diarize import build_vad as _build_vad
    return _build_vad(
        vad_file,
        threshold=args.vad_threshold,
        min_silence=args.vad_min_silence,
        min_speech=args.vad_min_speech,
        max_speech=args.vad_max_speech,
    )


def _resolve_outdir(value) -> Path:
    """输出目录：相对路径一律相对于工程根目录解析。

    不能直接用 Path(value)：那会相对于**进程当前工作目录**解析，
    从别的目录启动（IDE 里点运行、别的脚本调用）就会把结果写到意想不到的
    地方，找都找不到。工程根目录才是稳定的锚点。
    """
    p = Path(value) if value else (ROOT / "live_out")
    if not p.is_absolute():
        p = ROOT / p
    return p


def _formats_of(args) -> tuple:
    return tuple(x.strip() for x in str(args.formats or "txt").split(",") if x.strip())


def _session_stem(args, outdir: Path) -> str:
    """给本次会话定文件名：默认带时间戳新建，--stem-suffix none 保留固定名。"""
    if getattr(args, "stem_suffix", "timestamp") != "timestamp":
        return str(args.stem)
    return make_session_stem(args.stem, outdir, _formats_of(args))


class LiveSession:
    def __init__(self, args, display=None, stop=None, live_options=None,
                 wave_meter=None):
        self.args = args
        self.models = Path(args.models_dir)
        self.outdir = _resolve_outdir(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        # 每次会话独立文件名：听完一段再听新直播时，上一场文字稿要留得住。
        # args 与 self.args 是同一个 Namespace，改这里等同于改后面所有写出点。
        args.stem = _session_stem(args, self.outdir)
        self.session_stem = str(args.stem)
        self.session_formats = _formats_of(args)
        self.session_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.stop = stop or threading.Event()
        # 定稿线程必须有自己的结束标志：GUI 二次启动会复用并 clear 外部 stop，
        # 若 final_worker 拿它当循环条件，上一轮的 daemon 线程会被「复活」并
        # 永久驻留，继续写同一个 display / 占着上一轮的 ASR 模型。
        self._final_stop = threading.Event()
        self.display = display or LiveDisplay()
        self.live_options = live_options or LiveOptions(
            script=args.script, timestamps=args.timestamps,
            translate=args.translate)
        self.pipeline = build_pipeline(
            translate=args.translate, script=args.script,
            ollama_url=args.ollama_url, translate_model=args.translate_model,
            translate_timeout=args.translate_timeout)
        self.stats = CaptureStats(args.silence_warn)
        # GUI 的真实波形总线；命令行 --no-gui 时为 None。
        self.wave_meter = wave_meter
        self.final_q: queue.Queue = queue.Queue()
        self.segments: list = []
        self.seg_lock = threading.RLock()   # apply_settings() may call write_all()
        self.io_lock = threading.RLock()    # GUI 切换设置时不能和定稿线程抢写文件
        self.uid = 0
        self.names = parse_names(args.speaker_names)
        self.single = bool(getattr(args, "single_speaker", False))
        self.display.set_show_speaker(not self.single)
        self.t0 = 0.0
        self.tracker = None
        self.last_lang = None
        self.partial_lags = []       # 语音起点 → 首个草稿字上屏
        self.final_lags = []         # 一句说完 → 精修文本上屏
        self._speech_onset_wall = None
        self._lag_recorded = False
        self._draft_hidden_notice = False

    # ---------- 输出 ----------
    def write_all(self):
        with self.io_lock:
            with self.seg_lock:
                segs = sorted(self.segments, key=lambda s: s.start)
            if not segs:
                return
            script, timestamps, translate = self.live_options.snapshot()
            meta = {
                "mode": "live", "session_id": self.session_stem,
                "session_started_at": self.session_started_at,
                "source": self.stats.snapshot().get("device", ""),
                "num_segments": len(segs),
                "num_speakers": len({s.speaker for s in segs}),
                "speaker_mode": "single" if self.single else "diarize",
                "final_backend": self.args.final,
                "language": self.args.language or "auto",
                "script": script, "translate": translate,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_outputs(segs, self.outdir, self.args.stem, names=self.names,
                          timestamps=timestamps, meta=meta,
                          show_speaker=not self.single,
                          formats=self.session_formats)

    # ---------- 准备阶段反馈 ----------
    def _stage(self, stage, detail="", step=0, total=5):
        """把「正在准备」报到窗口：模型加载动辄十几秒，界面不能一片死寂。"""
        self.display.set_preparing(stage, detail, step, total)
        print("[准备] (%d/%d) %s %s" % (step, total, stage, detail), flush=True)

    def apply_settings(self):
        """GUI flipped 简体/繁体 or 时间戳: re-render history and rewrite files."""
        script, timestamps, _ = self.live_options.snapshot()
        with self.seg_lock:
            for s in self.segments:
                semantic = getattr(s, "semantic", s.text)
                s.text = self.pipeline.render(semantic, script)
        self.display.rebuild(lambda t: self.pipeline.render(t, script), timestamps)
        self.write_all()

    def final_worker(self, finalizer):
        while not self._final_stop.is_set() or not self.final_q.empty():
            try:
                item = self.final_q.get(timeout=0.4)
            except queue.Empty:
                continue
            uid, seg, samples, queued_at = item
            if finalizer is not None and samples.size:
                refined = finalizer.transcribe(samples)
                if refined:
                    semantic = self.pipeline.prepare(refined)
                    seg.semantic = semantic
                    script = self.live_options.snapshot()[0]
                    seg.text = self.pipeline.render(semantic, script)
                self.last_lang = finalizer.detected_language
            else:
                seg.semantic = seg.text
            with self.seg_lock:
                self.segments.append(seg)
            self.final_lags.append(time.time() - queued_at)
            self.display.update_final(uid, seg.text, semantic=getattr(seg, "semantic", seg.text))
            self.write_all()
            if self.args.no_gui:
                # 单人模式不显示说话人前缀，和窗口 / txt / srt 的输出口径保持一致。
                tag = "" if self.single else "[%s] " % self.tracker.display_name(seg.speaker)
                print("[定稿] %s%s" % (tag, seg.text), flush=True)

    # ---------- 入口 ----------
    def run_pipeline(self):
        """Only the capture pipeline. The GUI (main thread) owns the lifecycle."""
        return self._run_pipeline()

    def run(self):
        if self.args.list_devices:
            print(AudioSource.list_devices())
            return 0
        return self._run_pipeline()

    # ---------- 主循环 ----------
    def _run_pipeline(self):
        args = self.args

        # 组装组件（先加载模型，出错要早于开音频设备）
        import sherpa_onnx  # noqa: F401
        self._stage("加载语音检测模型", "silero VAD", 1)
        vad = build_vad(args, self.models)
        self._stage("单人模式，跳过声纹" if self.single else "加载声纹模型",
                    "" if self.single else "CAM++ 切窗在线分人", 2)
        if self.single:
            # 单人直播：不加载 CAM++ 声纹模型，也不做每句 embedding。
            # 既省 CPU/内存，也不会把同一个人的短句误判成新说话人。
            self.tracker = SingleSpeakerTracker(name=self.names.get(1, ""))
            if args.enroll:
                print("[说话人] 单人模式忽略 --enroll（本模式不区分说话人）", flush=True)
        else:
            emb = self.models / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
            if not emb.is_file():
                raise RuntimeError("缺声纹模型 %s" % emb)
            extractor = build_extractor(emb)
            tracker_win_sec = DEFAULT_WIN_SEC
            self.tracker = OnlineSpeakerTracker(
                extractor,
                max_speakers=args.num_speakers or None,
                threshold=args.speaker_threshold,
                names=self.names,
                win_sec=tracker_win_sec,
            )
            if args.enroll:
                n = self.tracker.load(args.enroll)
                print("[说话人] 已载入 %d 个注册声纹: %s" % (n, args.enroll), flush=True)
                if args.num_speakers and args.num_speakers < n:
                    print("[警告] --num-speakers=%d 小于已载入声纹数 %d，以已载入为准"
                          % (args.num_speakers, n), flush=True)

        # 中文流式 Paraformer 在纯英文语音上只会吐出中文谐音，徒增误导。
        # 用户明确选了英文时干脆关掉黄色草稿，只留句末精修。
        streaming = not args.no_streaming and args.language != "en"
        self._stage("加载流式识别模型", "Paraformer" if streaming else "本次不启用草稿", 3)
        stream_asr = StreamingASR(args.paraformer_dir, enabled=streaming)
        if not args.no_streaming and not streaming:
            print("[链路] 音频语言=en，已关闭中文流式草稿（Paraformer 只认中文）", flush=True)
        self._stage("加载句末精修模型",
                    "%s（首次加载权重约十几秒）" % args.final if args.final != "none"
                    else "未启用精修", 4)
        finalizer = build_finalizer(args.final, args)
        speaker_stage = "单人模式（跳过声纹）" if self.single else "CAM++ 切窗在线分人"
        print("[链路] VAD → %s → %s → %s" %
              ("流式 Paraformer" if stream_asr.enabled else "整句转写",
               speaker_stage,
               "句末精修 " + args.final if finalizer else "无精修"), flush=True)
        print("[文本] 语言=%s 字形=%s 译中文=%s 时间戳=%s" %
              (args.language or "auto", args.script,
               "开" if args.translate else "关",
               "开" if args.timestamps else "关"), flush=True)
        if self.single:
            print("[说话人] 单人模式：不做分人，窗口与 txt/srt 不显示说话人", flush=True)
        else:
            print("[说话人] 人数上限=%s 阈值=%.2f（长段切 %g 秒窗投票）"
                  % (args.num_speakers or "自动", args.speaker_threshold,
                     tracker_win_sec), flush=True)
        if security_soft_running() and sys.platform == "win32":
            print("[提示] 检测到安全软件进程：若弹出麦克风授权框请点「允许」，"
                  "勾选「记住我的选择」可免后续重复授权", flush=True)

        # 本地大模型首次加载要十几秒，放到后台预热，别让第一句卡住。
        translator = getattr(self.pipeline, "translator", None)
        if args.translate and translator is not None:
            print("[文本] 后台预热本地翻译模型…", flush=True)
            threading.Thread(target=translator.warmup, name="translator-warmup",
                             daemon=True).start()

        # 只有固定文件名模式（--stem-suffix none）才重建旧文件。
        # 默认的带时间戳会话文件名本来就唯一，绝不删任何历史文字稿。
        if getattr(args, "stem_suffix", "timestamp") == "none":
            for fmt in self.session_formats:
                p = self.outdir / ("%s.%s" % (args.stem, fmt))
                if p.exists():
                    p.unlink()
        print("[输出] 本次结果 → %s" % (self.outdir / (args.stem + ".txt")), flush=True)

        src = AudioSource(backend=args.backend, device=args.device,
                          simulate=args.simulate, loop=args.loop_sim, stats=self.stats,
                          wave_meter=self.wave_meter)
        if args.simulate:
            src.backend = "file"

        # 句末精修线程（GUI 由主线程持有，见 run()）
        wt = threading.Thread(target=self.final_worker, args=(finalizer,),
                              name="finalizer", daemon=True)
        wt.start()

        self._stage("打开音频设备", "准备开始采集系统声音", 5)
        if self.wave_meter is not None:
            # 模型都加载完了，接下来进来的每一块 PCM 都要进真实波形。
            # 在这里切模式能顺手清掉上一场听录留下的历史柱。
            self.wave_meter.begin_live()
        src.start()
        self.display.clear_preparing()      # 准备完成，状态条交回电平显示
        self.t0 = time.time()
        last_stats = 0.0
        reason = "用户停止"
        try:
            while not self.stop.is_set():
                chunk = src.read(timeout=0.2)
                if chunk is None:
                    if src.error:
                        reason = "采集失败"
                        self.display.set_error(
                            "采集失败：%s（确认系统正在播放声音、默认输出设备正常；"
                            "安全软件若弹授权框请点「允许」）" % str(src.error)[:80])
                        break
                    if src.eof.is_set():
                        reason = "音频回放结束" if src.device_kind == "file" else "采集线程结束"
                        break
                    continue
                if chunk.size == 0:
                    continue

                vad.accept_waveform(chunk)
                stream_asr.feed(chunk)

                # 首字延迟基准 = VAD 判定「有人开始说话」的墙钟时刻。
                # 不能用 VAD 的样本索引 + 启动时刻反推：循环回放 / 长时间运行后
                # 音频时间轴与墙钟会错位，算出来的延迟是假的（实测踩过）。
                if vad.is_speech_detected():
                    now = time.time()
                    if stream_asr.enabled:
                        if self._speech_onset_wall is None:
                            self._speech_onset_wall = now
                        draft_now = stream_asr.text()
                        # 定稿一旦判定为非中文语音，中文流式模型的草稿就是谐音噪声，
                        # 继续上屏看起来又像"英文被翻成中文"，直接隐藏。
                        if self.last_lang and self.last_lang not in ("zh", "yue", "wuu"):
                            if draft_now and not self._draft_hidden_notice:
                                print("[文本] 检测到 %s 语音，隐藏中文流式草稿，以句末定稿为准"
                                      % self.last_lang, flush=True)
                                self._draft_hidden_notice = True
                            draft_now = ""
                        if draft_now and not self._lag_recorded:
                            self.partial_lags.append(max(0.0, now - self._speech_onset_wall))
                            self._lag_recorded = True
                        script = self.live_options.snapshot()[0]
                        self.display.set_partial(
                            self.pipeline.render(draft_now, script),
                            label=None if self.single else self.tracker.last_label)
                else:
                    self._speech_onset_wall = None
                    self._lag_recorded = False

                while not vad.empty():
                    seg = vad.front
                    samples = np.ascontiguousarray(
                        np.asarray(seg.samples, dtype=np.float32))
                    # VoiceActivityDetector 的 start 是样本索引，不是秒
                    start = float(getattr(seg, "start", 0)) / 16000.0
                    end = start + len(samples) / 16000.0
                    draft = stream_asr.text() if stream_asr.enabled else ""
                    if self.last_lang and self.last_lang not in ("zh", "yue", "wuu"):
                        draft = ""
                    script = self.live_options.snapshot()[0]
                    shown = self.pipeline.render(draft, script)
                    label, score, is_new = self.tracker.assign(samples)
                    name = self.tracker.display_name(label)
                    self.uid += 1
                    item = Segment(start=start, end=end, speaker=label, text=shown)
                    item.semantic = draft
                    self.display.add_final({
                        "uid": self.uid, "label": label, "name": name,
                        "text": shown or "（转写中…）", "semantic": draft,
                        "start": start, "end": end,
                    })
                    if is_new:
                        print("[说话人] 检测到新说话人 → %s（共 %d 人）"
                              % (name, self.tracker.num_speakers()), flush=True)
                    self.final_q.put((self.uid, item, samples.copy(), time.time()))
                    vad.pop()
                    stream_asr.reset()

                now = time.time()
                if now - last_stats > 0.3:
                    self.display.set_stats(self.stats.snapshot())
                    last_stats = now
                if args.duration and now - self.t0 >= args.duration:
                    reason = "到达 --duration"
                    self.stop.set()
        except KeyboardInterrupt:
            reason = "Ctrl+C"
        finally:
            self.stop.set()
            self._final_stop.set()
            self.display.clear_preparing()   # 出错/提前退出也不能把界面卡在「准备中」
            src.stop()
            if self.wave_meter is not None:
                self.wave_meter.finish()
            # 等定稿队列排空，避免最后几句丢失
            deadline = time.time() + 20
            while not self.final_q.empty() and time.time() < deadline:
                time.sleep(0.1)
            wt.join(timeout=3)
            if finalizer:
                finalizer.close()
            self.pipeline.close()

        # 收尾统计
        snap = self.stats.snapshot()
        voices = Path(args.save_voices) if args.save_voices else self.outdir / "voices.json"
        if not self.single:
            self.tracker.save(voices)
        print("-" * 72, flush=True)
        print("[结束] %s  运行 %.1fs  共 %d 句 / %d 位说话人"
              % (reason, time.time() - self.t0, len(self.segments),
                 len({s.speaker for s in self.segments})), flush=True)
        tail = ("声纹已存 %s" % voices) if not self.single else "单人模式（未生成声纹文件）"
        print("[结束] 有声占比 %.1f%%  开流耗时 %.2fs  %s"
              % (100 * snap["sound_ratio"], snap["open_sec"], tail), flush=True)
        if self.partial_lags:
            print("[延迟] 首字上屏 平均 %.2fs / 最大 %.2fs（%d 句）"
                  % (sum(self.partial_lags) / len(self.partial_lags),
                     max(self.partial_lags), len(self.partial_lags)), flush=True)
        if self.final_lags:
            print("[延迟] 句末定稿 平均 %.2fs / 最大 %.2fs（%d 句，后端 %s）"
                  % (sum(self.final_lags) / len(self.final_lags),
                     max(self.final_lags), len(self.final_lags), args.final), flush=True)
        if snap["sound_ratio"] < 0.01 and sys.platform == "win32":
            print("[警告] 本次几乎全程静音：确认系统确实在播放声音、默认输出设备"
                  "与回环设备一致；若安全软件有「麦克风保护」，检查是否拦截了本项目"
                  "的 python.exe（--list-devices 可查回环设备）。", flush=True)
        for fmt in self.session_formats:
            p = self.outdir / ("%s.%s" % (args.stem, fmt))
            if p.exists():
                print("[输出] %s (%d 字节)" % (p, p.stat().st_size), flush=True)
        return 0 if not src.error else 1


def main(argv=None):
    args = normalize_args(parse_args(argv))
    try:
        if args.no_gui:
            return LiveSession(args).run()
        # Tkinter must own the main thread. The UI controller starts/stops the
        # capture pipeline in a worker thread and owns the shared LiveDisplay.
        from .live_ui import run_ui
        return run_ui(args, explicit_args(argv))
    except Exception as e:                     # noqa: BLE001
        print("[致命错误] %s: %s" % (type(e).__name__, e), file=sys.stderr, flush=True)
        return 2
