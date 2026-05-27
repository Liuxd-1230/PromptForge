"""
PromptForge - Prompt Nodes
构建、标签、规则、翻译、图生图
"""
import json
import os
from pathlib import Path


class PromptBuilder:
    """
    Assembles a final prompt from positive/negative components, with optional
    weighting, prefix/suffix injection, and tag-set merging.
    """

    @classmethod
    def INPUT_TYPES(cls):
        tag_files = _get_tag_files()
        return {
            "required": {
                "positive": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Main positive prompt text",
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Main negative prompt text",
                }),
            },
            "optional": {
                "prefix": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prepended to positive prompt",
                }),
                "suffix": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text appended to positive prompt",
                }),
                "negative_prefix": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prepended to negative prompt",
                }),
                "tag_preset": (tag_files if tag_files else ["(no presets)"],),
                "separator": ("STRING", {
                    "default": ", ",
                    "tooltip": "Separator between prompt segments",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "build"
    CATEGORY = "PromptForge/Prompt"

    def build(self, positive, negative, prefix="", suffix="",
              negative_prefix="", tag_preset="(no presets)", separator=", "):
        parts_pos = []
        parts_neg = []

        # Tag preset injection
        if tag_preset != "(no presets)":
            tag_data = _load_json(_CONFIG_DIR / "tags" / f"{tag_preset}.json")
            if isinstance(tag_data, dict):
                parts_pos.extend(tag_data.get("positive", []))
                parts_neg.extend(tag_data.get("negative", []))
            elif isinstance(tag_data, list):
                parts_pos.extend(tag_data)

        # Prefix / suffix
        if prefix.strip():
            parts_pos.append(prefix.strip())
        if positive.strip():
            parts_pos.append(positive.strip())
        if suffix.strip():
            parts_pos.append(suffix.strip())

        if negative_prefix.strip():
            parts_neg.append(negative_prefix.strip())
        if negative.strip():
            parts_neg.append(negative.strip())

        final_pos = separator.join(parts_pos)
        final_neg = separator.join(parts_neg)

        return (final_pos, final_neg)


# ---------------------------------------------------------------------------
# Node 2: TagPresetManager
# ---------------------------------------------------------------------------


class TagPresetManager:
    """
    Load, preview, and save tag presets. Presets are JSON files stored under
    config/tags/.  Each preset maps to {"positive": [...], "negative": [...]}.
    """

    @classmethod
    def INPUT_TYPES(cls):
        tag_files = _get_tag_files()
        return {
            "required": {
                "operation": (["load", "save", "list"],),
                "preset_name": ("STRING", {
                    "default": "default",
                    "tooltip": "Name of the tag preset",
                }),
            },
            "optional": {
                "positive_tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Positive tags (one per line or comma-separated)",
                }),
                "negative_tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative tags (one per line or comma-separated)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_tags", "negative_tags", "info")
    FUNCTION = "manage"
    CATEGORY = "PromptForge/Preset"

    def manage(self, operation, preset_name, positive_tags="", negative_tags=""):
        path = _CONFIG_DIR / "tags" / f"{preset_name}.json"

        if operation == "list":
            files = _get_tag_files()
            info = "Available presets:\n" + "\n".join(f"  - {f}" for f in files) if files else "No presets found."
            return ("", "", info)

        elif operation == "load":
            data = _load_json(path)
            if isinstance(data, dict):
                pos = "\n".join(data.get("positive", []))
                neg = "\n".join(data.get("negative", []))
            else:
                pos = neg = ""
            return (pos, neg, f"Loaded preset: {preset_name}")

        elif operation == "save":
            pos_list = [t.strip() for t in re.split(r"[,\n]", positive_tags) if t.strip()]
            neg_list = [t.strip() for t in re.split(r"[,\n]", negative_tags) if t.strip()]
            _save_json(path, {"positive": pos_list, "negative": neg_list})
            return (positive_tags, negative_tags, f"Saved preset: {preset_name}")

        return ("", "", "Unknown operation")


# ---------------------------------------------------------------------------
# Node 3: PromptRuleEngine
# ---------------------------------------------------------------------------


class PromptRuleEngine:
    """
    Apply transformation rules to a prompt string. Rules are loaded from
    config/rules/<category>/ as JSON files.

    Rule format:
    {
      "rules": [
        {"match": "regex_pattern", "replace": "replacement", "description": "..."},
        ...
      ]
    }
    """

    @classmethod
    def INPUT_TYPES(cls):
        categories = _get_rule_dirs()
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": "Input prompt to apply rules to",
                }),
                "category": (categories if categories else ["(no rules)"],),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_rule_match": ("STRING", {
                    "default": "",
                    "tooltip": "Additional regex match pattern (applied first)",
                }),
                "custom_rule_replace": ("STRING", {
                    "default": "",
                    "tooltip": "Replacement string for custom rule",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "changes")
    FUNCTION = "apply_rules"
    CATEGORY = "PromptForge/Rules"

    def apply_rules(self, prompt, category, enabled,
                    custom_rule_match="", custom_rule_replace=""):
        if not enabled:
            return (prompt, "Rules disabled")

        changes = []

        # Apply custom rule first
        if custom_rule_match and custom_rule_match.strip():
            try:
                new_prompt, n = re.subn(custom_rule_match, custom_rule_replace, prompt)
                if n > 0:
                    changes.append(f"Custom rule: {n} match(es)")
                    prompt = new_prompt
            except re.error as e:
                changes.append(f"Custom rule error: {e}")

        # Load and apply category rules
        rules_dir = _CONFIG_DIR / "rules" / category
        if not rules_dir.exists():
            return (prompt, "No rules found" if not changes else "\n".join(changes))

        for rule_file in sorted(rules_dir.glob("*.json")):
            data = _load_json(rule_file)
            rules = data.get("rules", []) if isinstance(data, dict) else []
            for rule in rules:
                match_pat = rule.get("match", "")
                replace_str = rule.get("replace", "")
                desc = rule.get("description", rule_file.stem)
                try:
                    new_prompt, n = re.subn(match_pat, replace_str, prompt)
                    if n > 0:
                        changes.append(f"[{desc}] {n} replacement(s)")
                        prompt = new_prompt
                except re.error:
                    changes.append(f"Invalid regex in {rule_file.name}: {match_pat}")

        return (prompt, "\n".join(changes) if changes else "No changes applied")


# ---------------------------------------------------------------------------
# Node 4: PromptTranslator
# ---------------------------------------------------------------------------


class PromptTranslator:
    """
    Translate prompt text between languages using an external LLM call
    (via PromptForge LLM node) or simple dictionary-based translation.

    This node provides the interface; actual translation is delegated to
    the LLM backend or can be performed with a local glossary file
    stored at config/translations/<lang>.json.
    """

    LANGUAGES = [
        "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
        "ar", "th", "vi", "id", "tr",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "tooltip": "Text to translate",
                }),
                "source_lang": (cls.LANGUAGES,),
                "target_lang": (cls.LANGUAGES,),
            },
            "optional": {
                "glossary_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional custom glossary: each line is 'source|target'",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("translated", "info")
    FUNCTION = "translate"
    CATEGORY = "PromptForge/Translate"

    def translate(self, text, source_lang, target_lang, glossary_override=""):
        if source_lang == target_lang:
            return (text, "Source and target are the same language")

        # Try loading a glossary
        glossary = {}
        if glossary_override.strip():
            for line in glossary_override.strip().splitlines():
                if "|" in line:
                    src, tgt = line.split("|", 1)
                    glossary[src.strip()] = tgt.strip()
        else:
            glossary_path = _CONFIG_DIR / "translations" / f"{source_lang}_{target_lang}.json"
            glossary = _load_json(glossary_path)

        if glossary:
            translated = text
            for src, tgt in glossary.items():
                translated = re.sub(re.escape(src), tgt, translated, flags=re.IGNORECASE)
            return (translated, f"Translated via glossary ({len(glossary)} entries)")

        return (
            text,
            f"No glossary found for {source_lang}->{target_lang}. "
            "Connect an LLM node for full translation support."
        )


# ---------------------------------------------------------------------------
# Node 5: PromptSplitter
# ---------------------------------------------------------------------------


class Img2ImgPromptNode:
    """
    图生图Prompt生成节点
    基于参考图片分析和角色锚点，生成适合图生图的prompt
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_config": ("API_CONFIG",),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "deepseek-chat",
                }),
                "reference_analysis": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "参考图片的分析结果（从图片分析节点获取）"
                }),
                "user_request": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "你想要的效果，如：保持风格不变，但把人物换成小红"
                }),
            },
            "optional": {
                "character_list": ("CHARACTER_LIST",),
                "preserve_elements": (["style", "composition", "lighting", "all", "none"], {
                    "default": "style",
                    "tooltip": "从参考图保留哪些元素"
                }),
                "strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "tooltip": "变换强度建议（0.3=微调，0.5=中等，0.7=较大变化）"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("img2img_prompt", "negative_prompt", "suggested_strength")
    FUNCTION = "generate_prompt"
    CATEGORY = "PromptForge/Image"

    def generate_prompt(self, api_config, model, reference_analysis, user_request,
                        character_list=None, preserve_elements="style", strength=0.7):

        from openai import OpenAI

        api_url = api_config["api_url"]
        api_key = api_config["api_key"]

        if not model or model.strip() == "":
            model = api_config.get("default_model", "deepseek-chat")

        base_url = api_url if api_url.endswith("/v1") else f"{api_url}/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 构建角色信息
        character_info = ""
        if character_list:
            char_parts = []
            for char in character_list:
                desc = f"- {char['name']}: {char['age']} {char['gender']}"
                if char['appearance']:
                    desc += f", {char['appearance']}"
                if char['clothing']:
                    desc += f", wearing {char['clothing']}"
                if char['features']:
                    desc += f", {char['features']}"
                char_parts.append(desc)
            character_info = "Characters to include:\n" + "\n".join(char_parts)

        # 保留元素说明
        preserve_desc = {
            "style": "art style, color palette, overall aesthetic",
            "composition": "composition, framing, camera angle",
            "lighting": "lighting, shadows, atmosphere",
            "all": "style, composition, lighting, and overall mood",
            "none": "only the core concept, everything else can change"
        }

        system_prompt = """You are an expert Stable Diffusion prompt engineer specializing in img2img.

Your task is to generate a prompt for img2img generation based on:
1. A reference image analysis
2. User's request for changes
3. Character descriptions (if provided)

Output TWO things separated by [NEGATIVE]:
1. The positive prompt (what you want)
2. The negative prompt (what to avoid)

Example format:
beautiful woman in red dress standing in garden, masterpiece, best quality, highly detailed
[NEGATIVE]
blurry, low quality, deformed, ugly, bad anatomy, extra limbs"""

        user_prompt = f"""Reference image analysis:
{reference_analysis}

User request: {user_request}

{character_info if character_info else ""}

Preserve from reference: {preserve_desc.get(preserve_elements, "style")}

Generate img2img prompt that combines the reference image's {preserve_elements} with the user's request.
If character descriptions are provided, incorporate them into the prompt for character consistency.

Output format:
[positive prompt]
[NEGATIVE]
[negative prompt]"""

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

            result = response.choices[0].message.content

            # 解析输出
            if "[NEGATIVE]" in result:
                parts = result.split("[NEGATIVE]")
                img2img_prompt = parts[0].strip()
                negative_prompt = parts[1].strip()
            else:
                img2img_prompt = result.strip()
                negative_prompt = "blurry, low quality, deformed, ugly, bad anatomy, extra limbs, watermark, text"

        except Exception as e:
            img2img_prompt = f"[Error] {str(e)}"
            negative_prompt = ""

        return (img2img_prompt, negative_prompt, strength)


# ============================================================
# 3. 图片混合/风格迁移提示节点
# ============================================================
class


NODE_CLASS_MAPPINGS = {
    "PromptBuilder": PromptBuilder,
    "TagPresetManager": TagPresetManager,
    "PromptRuleEngine": PromptRuleEngine,
    "PromptTranslator": PromptTranslator,
    "Img2ImgPromptNode": Img2ImgPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptBuilder": "PromptForge Prompt 构建",
    "TagPresetManager": "PromptForge 标签预设",
    "PromptRuleEngine": "PromptForge 规则引擎",
    "PromptTranslator": "PromptForge 翻译",
    "Img2ImgPromptNode": "PromptForge 图生图Prompt",
}
