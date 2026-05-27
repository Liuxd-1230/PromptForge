"""
PromptForge - Prompt Building Nodes

Nodes for constructing, transforming, and managing image generation prompts.
Includes: tag presets, rule engine, translation, prompt assembly/composition.
"""

import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"


def _load_json(path: Path):
    """Load a JSON file, returning empty dict/list on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: Path, data):
    """Persist data as JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_tag_files():
    """Return a list of available tag preset filenames (without extension)."""
    tag_dir = _CONFIG_DIR / "tags"
    tag_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in tag_dir.glob("*.json"))


def _get_rule_dirs():
    """Return available rule categories (subdirs under config/rules)."""
    rules_dir = _CONFIG_DIR / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        d.name for d in rules_dir.iterdir() if d.is_dir()
    )


# ---------------------------------------------------------------------------
# Node 1: PromptBuilder
# ---------------------------------------------------------------------------

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

class PromptSplitter:
    """
    Split a long prompt into structured segments for weighted or
    section-based editing (e.g., subject, style, environment, lighting).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": "Full prompt to split",
                }),
            },
            "optional": {
                "delimiter": ("STRING", {
                    "default": "|",
                    "tooltip": "Delimiter used to separate prompt sections",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("section_1", "section_2", "section_3", "section_4")
    FUNCTION = "split_prompt"
    CATEGORY = "PromptForge/Prompt"

    def split_prompt(self, prompt, delimiter="|"):
        sections = [s.strip() for s in prompt.split(delimiter) if s.strip()]
        # Pad to 4 sections
        while len(sections) < 4:
            sections.append("")
        return tuple(sections[:4])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "PromptBuilder": PromptBuilder,
    "TagPresetManager": TagPresetManager,
    "PromptRuleEngine": PromptRuleEngine,
    "PromptTranslator": PromptTranslator,
    "PromptSplitter": PromptSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptBuilder": "Prompt Builder",
    "TagPresetManager": "Tag Preset Manager",
    "PromptRuleEngine": "Prompt Rule Engine",
    "PromptTranslator": "Prompt Translator",
    "PromptSplitter": "Prompt Splitter",
}
