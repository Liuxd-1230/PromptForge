"""
PromptForge - Image Nodes
Img2Img Prompt, Style Transfer Prompt
"""


# ============================================================
# 1. 图生图Prompt节点
# ============================================================
class Img2ImgPromptNode:
    """基于参考图分析和用户需求，生成img2img prompt"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_config": ("API_CONFIG",),
                "reference_analysis": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "参考图的分析结果（从图片分析节点连接）"
                }),
                "user_request": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "你的修改需求，如：把背景换成海边，保持人物不变..."
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": "deepseek-chat",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("img2img_prompt", "negative_prompt")
    FUNCTION = "generate_img2img_prompt"
    CATEGORY = "PromptForge/Image"

    def generate_img2img_prompt(self, api_config, reference_analysis, user_request,
                                model="deepseek-chat"):
        from openai import OpenAI

        api_url = api_config["api_url"]
        api_key = api_config["api_key"]

        if not model or model.strip() == "":
            model = api_config.get("default_model", "deepseek-chat")

        base_url = api_url if api_url.endswith("/v1") else f"{api_url}/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)

        system_prompt = """You are an expert image generation prompt engineer specializing in img2img workflows.

Given a reference image analysis and user modification request, generate:
1. A detailed img2img prompt that preserves the original image's composition while applying the requested changes
2. A negative prompt to avoid unwanted artifacts

Rules:
- Keep the original image's style, lighting, and composition unless the user explicitly asks to change them
- The prompt should be in English
- Use natural language descriptions, not structured tags
- Start with quality tags: masterpiece, best quality, score_7
- Output format: two paragraphs separated by "---NEGATIVE---"

Example output:
masterpiece, best quality, score_7, [detailed description of the modified image]

---NEGATIVE---
worst quality, low quality, blurry, deformed, ugly"""

        user_prompt = f"""Reference image analysis:
{reference_analysis}

User modification request:
{user_request}

Generate the img2img prompt and negative prompt."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=0.7,
                stream=False
            )

            result = response.choices[0].message.content or ""

            if "---NEGATIVE---" in result:
                parts = result.split("---NEGATIVE---")
                img2img_prompt = parts[0].strip()
                negative_prompt = parts[1].strip()
            else:
                img2img_prompt = result.strip()
                negative_prompt = "worst quality, low quality, blurry, deformed, ugly"

            return (img2img_prompt, negative_prompt)

        except Exception as e:
            return (f"[Error] {str(e)}", "worst quality, low quality")


# ============================================================
# 2. 风格迁移Prompt节点
# ============================================================
class StyleTransferPromptNode:
    """基于内容描述和风格描述，生成风格迁移prompt"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "内容描述：你想生成什么内容..."
                }),
                "style_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "风格描述：参考图的艺术风格..."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("transfer_prompt", "negative_prompt")
    FUNCTION = "generate_transfer_prompt"
    CATEGORY = "PromptForge/Image"

    def generate_transfer_prompt(self, content_description, style_description):
        transfer_prompt = f"masterpiece, best quality, score_7, {content_description}, {style_description}"
        negative_prompt = "worst quality, low quality, blurry, deformed, ugly, text, watermark"

        return (transfer_prompt, negative_prompt)


# ============================================================
# Register all nodes
# ============================================================
NODE_CLASS_MAPPINGS = {
    "Img2ImgPromptNode": Img2ImgPromptNode,
    "StyleTransferPromptNode": StyleTransferPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Img2ImgPromptNode": "PromptForge 图生图Prompt",
    "StyleTransferPromptNode": "PromptForge 风格迁移Prompt",
}
