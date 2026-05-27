# ⚒️ PromptForge — AI绘画提示词锻造工具

ComfyUI 智能提示词节点包，集成 LLM 对话、人物一致性、剧情分镜、规则系统、标签预设、翻译等功能。

融合 [pandai-plus](https://github.com/Liuxd-1230/pandai-plus) 和 [ComfyUI-Prompt-Assistant](https://github.com/yawiii/ComfyUI-Prompt-Assistant) 的精华。

---

## ✨ 功能概览

### 🤖 LLM 对话
- 支持 OpenAI 兼容 API（DeepSeek / 智谱 / OpenAI 等）
- 支持本地 Ollama 模型
- 多轮对话历史管理（滑动窗口）

### 👤 人物一致性
- 人物外貌锚点：定义角色外观，保持多图一致
- 角色描述自动注入 prompt
- 支持多人场景合并

### 📖 剧情分镜
- 将长剧情自动拆分成多个场景
- 每个场景自动生成图片 prompt
- 支持批量输出

### 🖼️ 图片分析
- 视觉 LLM 分析图片内容/风格/人物
- 基于参考图生成 img2img prompt
- 风格迁移 prompt 生成

### ⚙️ 规则系统（12套预置规则）
- **扩写规则**：通用扩写、人像大师、Tags 风格、Qwen-Edit、Kontext、Wan 视频
- **视觉反推**：像素级描述、Tag 风格反推、编辑重绘
- **翻译规则**：中↔英 AI 绘画术语精准翻译
- **视频规则**：视频复刻、视频分镜解构

### 🏷️ 标签预设
- CSV 标签库管理（动漫/画质/摄影/3D/人物/服装/场景）
- 支持多套标签切换
- 一键插入常用标签

### 🔄 提示词翻译
- 中↔英 AI 绘画术语精准翻译
- 保护权重符号、专有名词
- 支持词典翻译 + LLM 翻译

---

## 📦 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Liuxd-1230/PromptForge.git
pip install -r PromptForge/requirements.txt
```

重启 ComfyUI 即可使用。

---

## 🔧 全部节点

### LLM 对话类

| 节点 | 说明 |
|------|------|
| PromptForge LLM 对话 | 通用 LLM 对话（支持 OpenAI/Ollama） |
| PromptForge 人物锚点 | 定义角色外观，保持多图一致 |
| PromptForge 剧情分镜 | 长文本自动拆分为场景 |
| PromptForge 图片分析 | 视觉 LLM 分析图片 |
| PromptForge Prompt 增强 | LLM 扩写 + 负面词生成 |

### Prompt 工具类

| 节点 | 说明 |
|------|------|
| PromptForge Prompt 构建 | 组合正面/负面 prompt + 标签注入 |
| PromptForge 标签预设 | 加载/管理 CSV 标签库 |
| PromptForge 规则引擎 | 加载/切换 prompt 规则文件 |
| PromptForge 翻译 | 中↔英 AI 绘画术语翻译 |
| PromptForge Prompt 拆分 | 按分隔符拆分为 4 段 |

---

## 🔗 工作流示例

### 1. 基础对话生图
```
[API 配置] → [LLM 对话] → 图片 prompt
```

### 2. 人物一致性生图
```
[人物锚点: 小红] → [LLM 对话] → 一致的角色 prompt → 生图
```

### 3. 剧情分镜 → 批量生图
```
[酒馆剧情] → [剧情分镜] → [场景选择] → prompt → 生图
```

### 4. 规则增强 prompt
```
[用户输入] → [规则引擎: Tags风格] → 优化后的 prompt → 生图
```

### 5. 图片反推 → 风格迁移
```
[参考图] → [图片分析] → [Prompt 增强] → img2img prompt → 生图
```

---

## 📂 项目结构

```
PromptForge/
├── __init__.py              # ComfyUI 节点注册
├── nodes_prompt.py          # Prompt 工具节点（5个）
├── nodes_llm.py             # LLM 对话节点（5个）
├── requirements.txt         # Python 依赖
├── config/
│   ├── rules/
│   │   ├── expand/          # 扩写规则（6套）
│   │   ├── vision/          # 视觉反推规则（3套）
│   │   ├── translate/       # 翻译规则（1套）
│   │   └── video/           # 视频规则（2套）
│   ├── tags/
│   │   └── anime_tags.csv   # 动漫标签库（80+标签）
│   └── presets/
│       └── default_config.json
└── README.md
```

---

## 📋 预置规则列表

| 分类 | 规则名 | 用途 |
|------|--------|------|
| 扩写 | 通用扩写 | 全学科自动识别领域，深度扩写 |
| 扩写 | 人像大师 | 人像摄影专用，五维细节填充 |
| 扩写 | Tags 风格 | Danbooru 标签流，SD 专用权重 |
| 扩写 | Qwen-Edit 指令 | 图像编辑指令优化 |
| 扩写 | Kontext 指令 | Flux Kontext 编辑指令 |
| 扩写 | Wan 视频 | 通义万相视频提示词 |
| 视觉 | 像素级描述 | 全要素提取，像素级精度 |
| 视觉 | Tag 风格反推 | Danbooru 标签流反推 |
| 视觉 | 编辑重绘 | 编辑指令生成 |
| 翻译 | 中英翻译 | AI 绘画术语精准翻译 |
| 视频 | 视频复刻 | 精准复刻或创意重构 |
| 视频 | 视频分镜 | 视频→分镜→结构化指令 |

---

## 🗺️ 开发路线

- [x] 基础框架 + 节点注册
- [x] LLM 对话 + 人物一致性
- [x] 剧情分镜 + 图片分析
- [x] 规则系统 + 标签预设
- [x] 翻译 + Prompt 工具
- [ ] Anima 适配节点（LLLite、加速）
- [ ] 小助手 UI（节点悬浮按钮）

---

## 📜 致谢

- [pandai-plus](https://github.com/Liuxd-1230/pandai-plus) — 人物一致性、剧情分镜基础
- [ComfyUI-Prompt-Assistant](https://github.com/yawiii/ComfyUI-Prompt-Assistant) — 规则系统、标签库灵感

---

## 📄 License

MIT
