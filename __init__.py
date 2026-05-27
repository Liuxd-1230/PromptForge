"""
PromptForge - ComfyUI Custom Node Pack for AI Image Generation Prompts

Combines features from:
- pandai-plus: LLM chat, character consistency, story splitting, image analysis
- ComfyUI-Prompt-Assistant: Rule system, tag presets, translation

Nodes are split across:
  nodes_prompt.py  — Prompt building, tag presets, rules, translation
  nodes_llm.py     — LLM chat, character consistency, story splitting, vision
"""

from .nodes_prompt import NODE_CLASS_MAPPINGS as PROMPT_NODES
from .nodes_llm import NODE_CLASS_MAPPINGS as LLM_NODES
from .nodes_prompt import NODE_DISPLAY_NAME_MAPPINGS as PROMPT_DISPLAY
from .nodes_llm import NODE_DISPLAY_NAME_MAPPINGS as LLM_DISPLAY

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(PROMPT_NODES)
NODE_CLASS_MAPPINGS.update(LLM_NODES)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(PROMPT_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(LLM_DISPLAY)

WEB_DIRECTORY = None  # No frontend JS needed for core nodes

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
