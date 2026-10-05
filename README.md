# CapsWriter-RS

[CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) 的 Rust 重实现——**完全离线**的语音输入与文件转录工具。

**双击一个 exe，按住鼠标左键说话，松开上屏。**

## 与原版的关系

| 项目 | 原版 CapsWriter-Offline | CapsWriter-RS |
|---|---|---|
| 语言 | Python 3.14 | Rust |
| ASR 引擎 | sherpa-onnx / 自研 ONNX+GGUF 管线 | 同一套 SenseVoice 双 ONNX 模型（ort 直接推理） |
| 模型文件 | `models/` | **完全通用，直接复制即用** |
| 启动 | 数秒（解包+解释器） | < 1 秒 |
| 常驻内存 | 300MB+ | 显著更低 |
| 触发键 | CapsLock / 鼠标 X2 | **鼠标左键**（默认）/ X2 / CapsLock |
| 界面 | 托盘 + Toast | 无（控制台日志） |
| LLM 角色 | ✅ | ✅（OpenAI 兼容接口） |
| GGUF 引擎（Qwen3-ASR 等） | ✅ | 路线图中 |

## 快速开始

```
1. capswriter-rs.exe          双击启动（自动拉起内置识别服务端）
2. 按住鼠标左键说话，松开上屏
```

- **短按左键（< 250ms）**＝普通点击，完全不受影响
- **按住期间移动鼠标超过阈值**＝拖拽/框选，自动取消识别
- 也可以按住**鼠标侧键 X2** 说话（短按自动补发原按键）
- 往 exe 上**拖拽音视频文件**＝转录出 `.srt` / `.txt` / `.json`

## 命令行

```
capswriter-rs                    听写模式（默认）
capswriter-rs server             仅启动识别服务端
capswriter-rs transcribe <文件>  文件转录
capswriter-rs <文件>             同上（支持拖拽）
```

## 架构

```
┌────────────────────── capswriter-rs.exe ──────────────────────┐
│  client 线程组                    server 线程                 │
│  ├ Win32 钩子(鼠标/键盘)          ├ WebSocket :6321            │
│  ├ cpal WASAPI 录音   ──ws──▶    ├ SenseVoice 引擎            │
│  ├ 后处理(热词/规则/标点)          │   ├ mel 特征(纯 Rust)      │
│  ├ 剪贴板+SendInput 上屏          │   ├ Encoder ONNX (ort)     │
│  └ 录音/日记归档                   │   ├ CTC Decoder ONNX      │
│                                  │   └ spm 解码(纯 Rust)      │
└───────────────────────────────────────────────────────────────┘
```

- 保持 C/S 架构：客户端与服务端通过 WebSocket（默认 127.0.0.1:6321）通信
- **单进程单 exe**：启动时端口空闲则自动在进程内拉起服务端；端口被占则复用已有服务（可把 `server_addr` 指向局域网另一台机器）
- 热词/规则/归档在客户端侧，识别在服务端侧——与原版职责划分一致

## 配置

全部在 `config.toml`（exe 同目录），模型路径、触发键、阈值、输出方式均可调。

托盘菜单提供“重新加载配置与热词”和“重新加载配置与模型”。前者适用于热词、规则、LLM、输出和触发参数；后者会在当前录音结束后重新创建 ASR 引擎，成功后才替换正在使用的模型。配置或模型加载失败时保留旧配置和旧模型。ONNX Runtime DLL、llama.cpp 动态库目录属于进程级资源，切换这些路径仍需使用“重启程序”。
热词 `hot.txt` 与规则 `hot-rule.txt` **格式与原版完全一致**，可直接复用。

### 热词说明

原版使用 CTC 概率空间 + 音素 RAG 匹配；本版实现为**拼音模糊匹配**（无声调拼音编辑距离），
阈值 `hot_thresh`（默认 0.80）。同音/近音错字可替换，准确率略低于原版方案，后续可升级。

### LLM 角色

`config.toml` 中开启 `[llm] enabled = true` 并配置角色。识别结果以角色名开头时交给 LLM：

> 「润色 帮我把这段话整理通顺」→ 剥离「润色」前缀 → 送 LLM → 上屏润色结果

## 模型

使用原版 CapsWriter 的 SenseVoice-Small 模型（放入 `models/SenseVoice-Small/Sensevoice-Small-ONNX/`）：

```
SenseVoice-Encoder.int8.onnx   (240MB)
SenseVoice-CTC.int8.onnx       (13MB)
tokenizer.bpe.model
```

int8 与 fp16 均支持（修改 config.toml 中文件名即可）。模型下载：[CapsWriter Models Release](https://github.com/HaujetZhao/CapsWriter-Offline/releases/tag/models)

## 构建

```
cargo build --release
```

依赖：Rust stable (MSVC)、VS C++ Build Tools（链接期）。
文件转录支持 wav/mp3/flac/ogg/m4a 内置解码，其他格式（mp4 等）需 ffmpeg 在 PATH。

### 运行库部署（onnxruntime.dll）

本项目使用 ort 的 `load-dynamic` 模式，**运行时动态加载 onnxruntime.dll**，不静态链接。
exe 同目录（或 `config.toml` 的 `onnxruntime_dll` 指定路径）需放置：

- `onnxruntime.dll` **≥ 1.22.0**（int8 模型用到 2-bit MatMulNBits 算子，1.20 及以下会报 unsupported operator）
- 获取方式：`pip download onnxruntime==1.22.0`（或从 [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases) 下载），从 wheel 中提取 `onnxruntime.dll`
- 也支持自定义动态库版本：设置环境变量 `ORT_DYLIB_PATH` 指向任意 onnxruntime.dll

## 当前限制与路线图

- [x] SenseVoice 引擎（CPU，多线程）
- [x] 按住说话 / 短按穿透 / 拖拽取消 / X2 与 CapsLock 触发
- [x] 热词（拼音模糊）+ 正则规则 + 末尾标点处理
- [x] 文件转录（srt/txt/json，VAD 由分段重叠拼接替代）
- [x] 录音归档 + 日记
- [x] LLM 角色（OpenAI 兼容）
- [ ] DirectML / CUDA 加速（ort 执行提供器切换，管线已就绪）
- [ ] Qwen3-ASR / Fun-ASR-Nano（GGUF，需 FFI llama.cpp）
- [ ] CTC 概率空间热词雷达（对齐原版音素 RAG）
- [ ] 托盘图标 / Toast 提示
- [ ] Linux 服务端（钩子与上屏保持 Windows-only）

## 注意事项

- 想在**管理员窗口**中输入，需以管理员身份运行本程序（Windows SendInput 权限规则）
- 左键按住说话与「按住拖拽」的区分靠 `move_cancel_px`，可按习惯调整
- 麦克风设备采样率非 16k 时自动高质量重采样（Kaiser 窗 FIR，对齐 scipy）

## License

MIT（沿用原版思路与模型格式，模型版权归原作者所有）
