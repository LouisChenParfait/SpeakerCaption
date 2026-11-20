# Third-party notices / 第三方许可清单

本项目自身代码采用 **Apache-2.0**（见 [LICENSE](LICENSE)）。
下面列出所有运行时依赖与模型的许可归属。**模型权重一律不随本仓库分发**，
首次运行时由 `scripts/download_models.py` 从上游官方 release 下载。

## 1. Python 包（pip 安装，不打包进仓库）

| 包 | 许可 | 用途 |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Apache-2.0 (Xiaomi Corporation) | diarization 运行时 / ONNX Runtime 封装 |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | ONNX 推理 |
| [soundfile](https://github.com/bastibe/python-soundfile) | BSD-3-Clause | 音频读写 |
| [numpy](https://numpy.org/) | BSD-3-Clause | 数值计算 |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | 可选 ASR 后端（默认） |
| [ctranslate2](https://github.com/OpenNMT/CTranslate2) | MIT | faster-whisper 推理引擎 |
| [OpenAI Whisper weights](https://huggingface.co/Systran/faster-whisper-large-v3) | MIT | 可选 ASR 权重，由 faster-whisper 自动下载 |
| [httpx](https://github.com/encode/httpx) | BSD-3-Clause | 可选 Confucius 后端的 HTTP 客户端 |
| [soxr](https://github.com/dofuuz/python-soxr) | LGPL-2.1 | 可选高质量重采样 |
| [opencc-python-reimplemented](https://github.com/yichen0831/OpenCC-Python) | MIT | 简繁转换（默认简体，可选繁体/原样） |
| [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) | Apache-2.0 | 实时模式 Windows WASAPI 回环采集（仅 win32 安装；仓库 `LICENSE.txt` 为 Apache-2.0，GitHub API 显示 NOASSERTION 只是未自动识别） |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) | MIT | 实时模式 macOS / Linux 采集 |

**可选的外部服务（不随本仓库分发、不打包任何代码或权重）**：

| 服务 | 许可 | 用途 |
|---|---|---|
| [Ollama](https://github.com/ollama/ollama) | MIT | 仅当勾选「译成中文」时，本机 `127.0.0.1:11434` 提供文本翻译；不开启则完全不访问 |

用户自行选择 Ollama 模型；模型各自的许可由用户自行确认，本仓库不内置、不分发任何 LLM 权重。

## 2. 模型权重（下载到 `models/`，被 `.gitignore` 排除）

| 模型 | 许可 | 版权 | 文件 | 大小 |
|---|---|---|---|---|
| pyannote segmentation 3.0 (ONNX 转换版) | **MIT** | CNRS | `models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` (及 `model.int8.onnx`) | 5.72 MB / 1.47 MB |
| 3D-Speaker CAM++ zh-en speaker embedding | **Apache-2.0** | ModelScope / Alibaba | `models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | 26.97 MB |
| Silero VAD (ONNX, v4/v5 旧版) | **MIT** | Silero Team | `models/silero_vad.onnx` | 0.61 MB |
| Streaming Paraformer bilingual zh-en (int8) | **Apache-2.0** | ModelScope / damo（模型卡 `license: apache-2.0`） | `models/streaming-paraformer/{encoder,decoder}.int8.onnx` + `tokens.txt` | 约 226 MB |
| faster-whisper 权重（可选 ASR） | **MIT** | OpenAI Whisper / SYSTRAN 转换 | `models/whisper/<档位>/`（或 HF 缓存） | tiny 72 MB / base 145 MB / small 461 MB / medium 1.42 GB / large-v3 2.88 GB |

### Whisper 权重的两种落地位置

Whisper 权重**不随仓库分发**（2.88 GB 太大，且 GitHub 单文件上限 100 MB），
有两条等效路径，许可都是 MIT：

1. **工程内目录**（推荐，可离线、可整目录分享）
   `python scripts/download_models.py --with-whisper large-v3`
   → 下载到 `models/whisper/large-v3/`（真实文件，不是缓存符号链接）。
   把这个目录连同 `models/` 一起打包给同事，对方**完全不需要联网**。

2. **HuggingFace 缓存**（首次运行自动下载）
   不做上述操作时，`WhisperModel("large-v3")` 会自动从 HuggingFace 拉取到
   `~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/`。
   国内网络可设 `HF_ENDPOINT=https://hf-mirror.com` 走镜像。

来源：

- 分段模型压缩包 https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  （由 https://huggingface.co/pyannote/segmentation-3.0 转换；包内自带 `LICENSE`，MIT 全文，
  要求再分发时保留版权与许可声明。本仓库不分发该包，故声明在此。）
- 声纹模型 https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
  （上游项目 https://github.com/modelscope/3D-Speaker 的 `LICENSE` 为 Apache-2.0）
- VAD 模型 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  （https://github.com/snakers4/silero-vad ，MIT）
- 流式 ASR 模型 https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en
  （模型卡首行 `license: apache-2.0`；ONNX 由 ModelScope
  `damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online` 转换而来。
  GitHub release 上的同名 tar.bz2 约 1 GB 含全精度档，本项目只用 HF 上的 int8 三件套）
- Whisper 权重 https://huggingface.co/Systran/faster-whisper-large-v3
  （模型卡 `license: mit`；上游 https://github.com/openai/whisper 同样是 MIT。
  可选镜像 https://hf-mirror.com/Systran/faster-whisper-large-v3）

## 3. Confucius4-R2T2（仅可选后端，本项目不分发任何组件）

`--asr confucius` 只是通过标准 Gradio HTTP 协议调用**用户自行部署**的服务，
代码中不含厂商源码、编译模块或权重。

| 资产 | 许可 |
|---|---|
| Confucius4-R2T2 仓库代码 | Apache-2.0 |
| Confucius4-R2T2 **模型权重** | **网易模型使用许可协议（NetEase Model Use License，非 OSI 开源许可）** |

关键约束（以官方 `MODEL_LICENSE_zh` 为准，中英文版本冲突时以中文为准）：

- **§3.4 a)** 下游分发必须受同等条款约束 → 不要把权重放进本仓库，也不要把项目
  整体标成 MIT/Apache 就以为覆盖了权重。
- **§2.2** 月活 > 1 亿或年收入 > 1 亿人民币需向有道单独申请商业许可。
- **§3.4 c)** 不得用该模型改进其他商业 AI 模型。
- **§4.2 禁用场景**：医疗诊断、自动驾驶、军事、关键基础设施控制、
  **大规模生物识别监控**、自动化决策（信贷/就业评估）。

因此 README 中的正确表述是：

```
Code: Apache-2.0 / Weights: NetEase Model Use License (not OSI)
```

另外：本机原部署包里的 `cy_app.cp312-win_amd64.pyd` 是编译产物，无源码、
绑定私有服务链路，**绝不上传 GitHub**。

## 4. 测试音频

`tests/data/*.wav` 由 `scripts/download_models.py --with-test-audio` 从
sherpa-onnx 官方 release 下载（`1-two-speakers-en.wav`、`0-four-speakers-zh.wav`），
仅用于本地验收，**不随仓库分发**（`.gitignore` 已排除）。

## 5. 使用限制提醒

说话人分离会处理**声纹**，属于生物识别信息。请在处理他人录音前取得同意、
遵守所在地法律（如个人信息保护法 / GDPR），并避免用于大规模监控或自动化决策。
本项目不提供身份识别能力：输出的是「说话人1/2」这类匿名标签，不关联真实姓名，
除非使用者自己通过 `--speaker-names` 显式映射。

## 6. 许可文本获取

- MIT (pyannote)：随模型压缩包一并提供，路径
  `models/sherpa-onnx-pyannote-segmentation-3-0/LICENSE`
- Apache-2.0：本仓库根目录 `LICENSE`
- 网易模型使用许可：见 Confucius4-R2T2 上游仓库的 `MODEL_LICENSE_zh`
