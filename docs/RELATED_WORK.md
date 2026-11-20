# 同类项目调研（Related Work）

> 调研日期：2026-11-09
> 方法：GitHub Search API（`search/repositories`，按 star 降序）+ 各项目 README 原文核对
> 目的：确认 SpeakerCaption 是否与已有开源项目「撞车」，并给出可写进 README 的差异化定位。
> star 数与 `push` 日期为调研当日快照，会随时间变化。

## 一句话结论

**「语音转写 + 说话人分离」是红海（GitHub 上 `speaker diarization` 相关仓库 2875 个），
但我们占的生态位是空的**：
用纯 ONNX 的轻量分离模型、零 HF token、零 torch、CPU 可跑、35 MB 模型、
输出 `txt/srt/json/rttm` 四种格式、ASR 可插拔（含中文热词后端）的**跨平台命令行听录工具**。

关键词交叉检索结果（均为 GitHub 仓库搜索）：

| 检索式 | 命中数 | 说明 |
|---|---:|---|
| `speaker diarization` | 2875 | 大类，红海 |
| `diarization transcription` | 1629 | 大类 |
| `speaker diarization whisper` | 731 | 主流组合 |
| `Confucius R2T2 diarization` | **0** | 中文热词后端 + 分离，无人做 |
| `sherpa 3dspeaker diarization` | **0** | 同一套 ONNX 底座做成应用，无人做 |
| `speaker diarization rttm srt whisper` | 1 | 仅 1 个 2-star 项目 |
| `podcast transcription diarization cli` | 4 | 全部 0 star |
| `loopback audio diarization` | 4 | 全部 0 star，实时系统音 + 分离仍空白 |
| `SpeakerCaption` | **0** | 仓库名未被占用 |

名称占用：GitHub 仓库 `SpeakerCaption` 0 命中、GitHub 用户/组织 `SpeakerCaption` 404、
PyPI `speakercaption` 与 `speaker-caption` 均 404。**名字可用。**

## 主要同类项目

### A. 同一功能，重型技术栈（真竞品，但部署门槛完全不同）

| 项目 | Star | 许可 | 最近 push | 分离方案 | 关键门槛 |
|---|---:|---|---|---|---|
| [m-bain/whisperX](https://github.com/m-bain/whisperX) | 25.2k | BSD-2 | 2026-10-22 | pyannote `speaker-diarization-community-1` | **必须 HF token + 接受模型协议**；torch + wav2vec2 对齐模型；作者自述「Diarization is far from perfect」 |
| [MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization) | 5.7k | BSD-2 | 2026-08-15 | NeMo MarbleNet VAD + TitaNet + MSDD + Demucs 人声分离 | Cython/FFmpeg 前置、NeMo 全家桶、Windows 安装困难；README 直言「参数还硬编码在 diarize.py 里，CLI 参数以后再加」；不支持指定人数、重叠说话人未处理 |
| [aTrain](https://github.com/aTrainTranscription/aTrain) | 1.2k | AGPL-3.0 | 2026-11-08 | pyannote.audio | 桌面 GUI / Microsoft Store 安装包为主，无 CLI；面向质性研究（MAXQDA、ATLAS.ti、nVivo 导出）；AGPL 有传染性 |
| [JSchmie/ScrAIbe](https://github.com/JSchmie/ScrAIbe) | 66 | GPL-3.0 | 2025-01 | whisper + pyannote | 已近两年未更新，GPL |

whisperX 输出 srt/vtt/json 且支持 `--min_speakers/--max_speakers`，功能面上最接近，
但它的分离是**可选附加功能**（默认不开），核心卖点是词级强制对齐，不是听录成稿。

### B. 实时流式方向（主场景不同，仅边缘重叠）

| 项目 | Star | 许可 | 最近 push | 说明 |
|---|---:|---|---|---|
| [Zackriya-Solutions/meetily](https://github.com/Zackriya-Solutions/meetily) | 32.5k | MIT | 2026-11-07 | 会议纪要桌面应用（Rust），实时录音 + LLM 总结，Windows/macOS |
| [QuentinFuxa/WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | 11.2k | Apache-2.0 | 2026-11-08 | 低延迟实时流式 ASR；默认分离用 NeMo Sortformer，`diart` 分支才需 HF 登录；有 `wlk transcribe file.wav --format srt` 离线 CLI |

WhisperLiveKit 的重叠点在于它也提供单文件离线 CLI，但：
- 主场景是流式服务（WebSocket / OpenAI 兼容 API），离线只是附带；
- 只文档化了 `srt` 导出，**没有 txt/json/rttm，也没承诺导出文件带说话人标签**；
- 说话人数只能设 Sortformer 的 `1~4` **上限**，不是精确人数；
- 每个 diarization frame 强制解析到单一说话人，**明确不支持重叠说话人输出**；
- sortformer / canary 依赖 `nemo-toolkit[asr]`（torch + Lightning 重依赖），多组 extras 互相冲突需分环境。

### C. 端到端新路线（技术上最值得警惕，但硬件门槛相反）

| 项目 | Star | 许可 | 最近 push | 说明 |
|---|---:|---|---|---|
| [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) | 2.3k | Apache-2.0 | 2026-10-21 | 0.9B 端到端模型，一次推理同时出转写 + `[S01]` 说话人 + 时间戳 + 声学事件；50+ 语言，中文评测 SOTA，支持热词 prompt，自带字幕 Web UI 导出 JSON/SRT/ASS |

这是 2026-07 开源的新路线，中文播客 cpCER 5.97，效果强于拼接式方案。
但它要求 **transformers 5.x + torch + vLLM/SGLang + CUDA（README 基准基于 H100）**，
`flash-attn` 与长音频 `max_new_tokens=65536` 都是 GPU 场景；Windows 本地部署基本不现实。
**它的存在反而说明了本项目的价值**：给没有大显存、只想 `pip install` 后处理一个 wav 的人留一条路。

### D. 中文桌面字幕工具（用户群有交集，产出物不同）

| 项目 | Star | 许可 | 最近 push | 说明 |
|---|---:|---|---|---|
| [corvo007/MioSub](https://github.com/corvo007/MioSub) | 840 | AGPL-3.0 | 2026-10-20 | Electron 一站式字幕生成器：下载 → 转录 → 翻译 → 压制；有「说话人自动标注」；默认要 Gemini API Key；产出是字幕/成片，不是听录文稿 |
| [Homelab-00/TranscriptionSuite](https://github.com/homelab-00/TranscriptionSuite) | 810 | GPL-3.0 | 2026-10-04 | 本地优先桌面 App，多后端 + 分离 + 日历模式 |
| [OpenTranscribe](https://github.com/attevon-llc/OpenTranscribe) | 109 | AGPL-3.0 | 2026-11-08 | Docker + FastAPI + Svelte 自托管平台，带搜索与协作 |

这一组都是**带界面的应用/平台**，不是可嵌进脚本的 CLI 库；且清一色 AGPL/GPL。

### E. 底座（我们站在它们肩上，不是竞品）

- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 15.8k / Apache-2.0
- [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) 10.8k / MIT
- [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) 3.2k / Apache-2.0
- [modelscope/FunASR](https://github.com/modelscope/FunASR) 21k / MIT

sherpa-onnx 自带 `OfflineSpeakerDiarization` API 和官方 Python 示例，但那是**示例脚本**，
不是可直接用的成品工具（没有多格式导出、没有 ASR 后端抽象、没有说话人命名、没有 CLI 参数体系）。

## 差异化定位（建议直接写进 README / 项目简介）

不要把自己描述成「又一个 Whisper + 说话人分离工具」——那是 731 个仓库的红海。
准确的自我描述是：

> **零门槛的离线听录 CLI**：不用 HuggingFace token、不用接受模型协议、不用 torch / NeMo /
> pyannote、不用 GPU，35 MB 模型、`pip install` 后即可对一段播客/访谈录音出
> `txt / srt / json / rttm` 四份带说话人标记的文稿；ASR 可换成任意后端，
> 包括支持中文热词的自部署 Confucius4-R2T2。全链路 Apache-2.0 / MIT 宽松许可。

四个别人没有同时满足的点：

1. **无 gated model**：所有分离模型是可直链下载的 ONNX，不需要 HF 账号或协议勾选。
   WhisperX / whisper-diarization / aTrain 都卡在这一步，这也是 Windows 用户最常见的劝退点。
2. **可插拔 ASR + 中文热词**：`confucius` 后端把中文专名/热词能力接进分离结果，
   检索式 `Confucius R2T2 diarization` 命中 0，是独有组合。
3. **四格式一次出齐，含 rttm**：rttm 是分离领域的标准交换格式，
   便于把结果喂给别的可视化/评估工具；同类 CLI 基本只给 srt。
4. **许可干净**：Apache-2.0 主仓 + MIT/Apache 依赖；对照组里 aTrain / MioSub /
   OpenTranscribe / ScrAIbe 都是 AGPL 或 GPL，商用集成要顾虑传染性。

## 风险与规避

- **端到端模型正在吞掉拼接式路线**：MOSS-Transcribe-Diarize 这类模型在效果上更强。
  规避方式：README 明确说明「本项目卖的是可离线、可换 ASR、可审计、CPU 可跑」，
  不宣称 SOTA 准确率；有条件的用户可以直接去看端到端方案。
- **不要对标实时流式 ASR 服务器**：WhisperLiveKit（11k）与 meetily（32k）占满的是
  「低延迟流式服务 / WebSocket API / 会议纪要 App」这块。本项目后来新增的 `live.py`
  是「本机抓系统回环声音 → 本地字幕窗口」的单机 CLI，不做服务端、不做多用户并发、
  不做会议机器人；主打的仍是零门槛与离线可审计，形态与它们不冲突。
- **不要做 GUI / 平台**：aTrain、MioSub、TranscriptionSuite、OpenTranscribe 已经很挤，
  且都是 GPL 系。CLI + 库形态反而是空位，也更适合被别的工具集成。
- **命名**：`SpeakerCaption` 在 GitHub 与 PyPI 均未被占用，可以注册；
  但这个词偏「字幕」，容易和 MioSub 那类字幕工具混淆，
  简介里务必第一句就写清「离线听录 / 说话人分离 / CLI」。

## 引用方式

在 README 的 Related Work 小节里链接本文件即可；不需要把上面所有项目都搬进 README，
否则会显得像在给自己找参照物。README 里放一句定位 + 一个链接足够。
