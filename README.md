# ⚒️ PromptForge — AI绘画提示词锻造工具

ComfyUI 智能提示词节点包，集成 LLM 对话、人物一致性、剧情分镜、规则系统、标签预设、翻译等功能。

融合 [pandai-plus](https://github.com/Liuxd-1230/pandai-plus) 和 [ComfyUI-Prompt-Assistant](https://github.com/yawiii/ComfyUI-Prompt-Assistant) 的精华。

---

## ✨ 功能概览

| 模块 | 节点数 | 说明 |
|------|--------|------|
| Config 配置 | 2 | API 配置、API 测试 |
| Character 人物 | 2 | 人物锚点、人物合并 |
| LLM 对话 | 6 | 对话、分镜、场景选择/查看/批量、图片分析 |
| Prompt 工具 | 5 | 构建、标签预设、规则引擎、翻译、图生图 |
| **总计** | **15** | |

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

### Config 配置类

| 节点 | 说明 |
|------|------|
| PromptForge API 配置 | 连接 DeepSeek / 智谱 / OpenAI / Ollama |
| PromptForge API 测试 | 测试连通性 + 获取模型列表 |

### Character 人物类

| 节点 | 说明 |
|------|------|
| PromptForge 人物锚点 | 定义角色外观，保持多图一致 |
| PromptForge 人物合并 | 多角色合并，支持链式连接 |

### LLM 对话类

| 节点 | 说明 |
|------|------|
| PromptForge LLM 对话 | 多轮对话 + 历史管理 + prompt文件注入 |
| PromptForge 剧情分镜 | 长文本拆分场景（scene/shot/beat 模式） |
| PromptForge 场景选择 | 从场景列表选择指定场景 |
| PromptForge 场景查看 | 查看所有场景详情 |
| PromptForge 批量输出 | 批量输出所有 prompt |
| PromptForge 图片分析 | 视觉 LLM 分析图片内容/风格 |

### Prompt 工具类

| 节点 | 说明 |
|------|------|
| PromptForge Prompt 构建 | 组合正面/负面 prompt + 标签注入 |
| PromptForge 标签预设 | CSV 标签库管理，一键插入 |
| PromptForge 规则引擎 | 加载/切换 prompt 规则文件（15套预置） |
| PromptForge 翻译 | 中↔英 AI 绘画术语精准翻译 |
| PromptForge 图生图Prompt | 基于参考图分析生成 img2img prompt |

---

## 📋 预置规则（15套）

### 扩写规则（6套）
| 规则 | 用途 |
|------|------|
| 扩写-通用 | 全学科自动识别领域，深度扩写 |
| 扩写-人像大师 | 人像摄影专用，五维细节填充 |
| 扩写-Tags风格 | Danbooru 标签流，SD 专用权重 |
| Qwen-Image-Edit 指令优化 | 图像编辑指令优化 |
| Kontext 指令优化并翻译 | Flux Kontext 编辑指令 |
| Wan 视频提示词 | 通义万相视频提示词 |

### 视觉反推规则（8套）
| 规则 | 用途 |
|------|------|
| 像素级描述 | 全要素提取，像素级精度 |
| 图像描述-Tag风格 | Danbooru 标签流反推 |
| 图像编辑重绘 | 编辑指令生成 |
| Qwen-Edit 指令优化-视觉版 | 结合视觉推理的编辑指令 |
| Kontext 指令优化-视觉版 | 英文编辑指令，角色一致性 |
| Detail Caption | 英文详细描述反推 |
| Caption-Tags | 英文标签流反推 |
| 图像到视频提示词 | 图生视频 prompt |

### 翻译规则（1套）
| 规则 | 用途 |
|------|------|
| 中英翻译 | AI 绘画术语精准翻译 |

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

### 5. 图片反推 → img2img
```
[参考图] → [图片分析] → [图生图Prompt] → img2img prompt → 生图
```

---

## 📂 项目结构

```
PromptForge/
├── __init__.py              # 节点注册
├── nodes_config.py          # API 配置/测试 (2)
├── nodes_character.py       # 人物锚点/合并 (2)
├── nodes_llm.py             # LLM 对话/分镜/分析 (6)
├── nodes_prompt.py          # Prompt 工具 (5)
├── config/
│   ├── rules/
│   │   ├── expand/          # 扩写规则 (6套)
│   │   ├── vision/          # 视觉反推规则 (8套)
│   │   └── translate/       # 翻译规则 (1套)
│   ├── tags/
│   │   └── anime_tags.csv   # 标签库 (80+)
│   └── presets/
│       └── default_config.json
├── requirements.txt
└── README.md
```

---

## 🗺️ 开发路线

- [x] 基础框架 + 15个节点
- [x] 规则系统（15套预置规则）
- [x] 标签预设（80+ 标签）
- [ ] Anima 适配节点
- [ ] 小助手 UI

---

## 📜 致谢

- [pandai-plus](https://github.com/Liuxd-1230/pandai-plus) — 人物一致性、剧情分镜基础
- [ComfyUI-Prompt-Assistant](https://github.com/yawiii/ComfyUI-Prompt-Assistant) — 规则系统、标签库灵感

---

## 📄 License

MIT
