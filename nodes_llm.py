"""
PromptForge - LLM-Powered Nodes

Nodes that leverage LLM APIs for intelligent prompt generation, character
consistency tracking, story/scene splitting, and image analysis.

LLM backends supported (via configuration):
  - OpenAI-compatible APIs (OpenAI, Together, Groq, etc.)
  - Ollama (local)
  - Custom endpoints
"""

import json
import logging
import os
import re
import requests
from pathlib import Path

logger = logging.getLogger("PromptForge")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"


def _load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_model_list():
    """Return configured model names from config/models.json or sensible defaults."""
    models_file = _CONFIG_DIR / "models.json"
    data = _load_json(models_file)
    if isinstance(data, dict) and "models" in data:
        return data["models"]
    return ["openai/gpt-4o", "ollama/llama3", "openai/gpt-4o-mini"]


# ---------------------------------------------------------------------------
# LLM Backend (shared helper)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, system: str = "", model: str = "openai/gpt-4o",
              temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """
    Unified LLM call. Parses model string as 'provider/model_name'.
    Supports OpenAI-compatible APIs and Ollama.
    Returns the assistant's response text.
    """
    try:
        import requests
    except ImportError:
        return "[Error] 'requests' package not installed. Run: pip install requests"

    parts = model.split("/", 1)
    provider = parts[0].lower() if len(parts) > 1 else "openai"
    model_name = parts[1] if len(parts) > 1 else parts[0]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except Exception as e:
            return f"[Ollama Error] {e}"

    # Default: OpenAI-compatible
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(f"{base_url}/chat/completions", json=payload,
                             headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[API Error] {e}"


# ---------------------------------------------------------------------------
# Node 1: LLMChatNode
# ---------------------------------------------------------------------------

class LLMChatNode:
    """
    General-purpose LLM chat node. Send a prompt to an LLM and receive
    the response. Useful for prompt expansion, creative writing, or
    any text generation task within a ComfyUI workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _get_model_list()
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": "User prompt to send to the LLM",
                }),
                "model": (models,),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant for AI image generation prompts.",
                    "tooltip": "System instruction for the LLM",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "PromptForge/LLM"

    def chat(self, prompt, model, system_prompt="", temperature=0.7, max_tokens=1024):
        response = _call_llm(prompt, system=system_prompt, model=model,
                             temperature=temperature, max_tokens=max_tokens)
        return (response,)


# ---------------------------------------------------------------------------
# Node 2: CharacterConsistencyNode
# ---------------------------------------------------------------------------

class CharacterConsistencyNode:
    """
    Maintain character consistency across multiple prompt generations.
    Stores a character profile (appearance, style, mood) and injects
    it into prompts to ensure visual consistency.

    Profiles are saved under config/characters/<name>.json.
    """

    @classmethod
    def INPUT_TYPES(cls):
        char_dir = _CONFIG_DIR / "characters"
        char_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(p.stem for p in char_dir.glob("*.json"))
        char_list = existing if existing else ["(no profiles)"]

        return {
            "required": {
                "operation": (["create", "inject", "list"],),
                "character_name": ("STRING", {
                    "default": "character_1",
                    "tooltip": "Name/ID of the character",
                }),
            },
            "optional": {
                "appearance": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Physical description: hair, eyes, build, distinguishing features",
                }),
                "style": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Clothing, accessories, signature style elements",
                }),
                "mood": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Typical expressions, posture, aura",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt to inject character details into (for inject mode)",
                }),
                "existing_profile": (char_list,),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "How strongly to enforce character details",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "character_profile", "info")
    FUNCTION = "manage_character"
    CATEGORY = "PromptForge/Character"

    def manage_character(self, operation, character_name,
                         appearance="", style="", mood="", prompt="",
                         existing_profile="(no profiles)", strength=1.0):
        char_dir = _CONFIG_DIR / "characters"
        char_dir.mkdir(parents=True, exist_ok=True)

        if operation == "list":
            profiles = sorted(p.stem for p in char_dir.glob("*.json"))
            info = "Character profiles:\n" + "\n".join(f"  - {p}" for p in profiles) if profiles else "No profiles yet."
            return ("", "", info)

        if operation == "create":
            profile = {
                "name": character_name,
                "appearance": appearance.strip(),
                "style": style.strip(),
                "mood": mood.strip(),
            }
            _save_json(char_dir / f"{character_name}.json", profile)
            profile_str = json.dumps(profile, ensure_ascii=False, indent=2)
            return ("", profile_str, f"Created profile: {character_name}")

        if operation == "inject":
            # Load profile
            profile = _load_json(char_dir / f"{existing_profile}.json")
            if not profile:
                return (prompt, "{}", f"Profile not found: {existing_profile}")

            # Build character descriptor
            descriptors = []
            if profile.get("appearance"):
                descriptors.append(profile["appearance"])
            if profile.get("style"):
                descriptors.append(profile["style"])
            if profile.get("mood"):
                descriptors.append(profile["mood"])

            char_desc = ", ".join(descriptors)
            if strength != 1.0:
                # Wrap in emphasis weight syntax for SD prompt format
                weight = round(strength, 1)
                char_desc = f"({char_desc}:{weight})"

            # Inject at beginning of prompt
            if prompt.strip():
                enhanced = f"{char_desc}, {prompt.strip()}"
            else:
                enhanced = char_desc

            return (enhanced, json.dumps(profile, ensure_ascii=False, indent=2),
                    f"Injected character: {existing_profile}")

        return ("", "", "Unknown operation")


# ---------------------------------------------------------------------------
# Node 3: StorySplitterNode (scene/shot/beat split, character injection,
#          structured SCENE_LIST output)
# ---------------------------------------------------------------------------

class StorySplitterNode:
    """
    Story scene splitter. Takes a long narrative text and splits it into
    structured visual scenes for image generation using an LLM.

    Supports scene/shot/beat split modes, optional CHARACTER_LIST injection,
    and outputs a structured SCENE_LIST for downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _get_model_list()
        return {
            "required": {
                "story": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Paste narrative/story text to split into visual scenes...",
                }),
                "model": (models,),
                "split_mode": (["scene", "shot", "beat"], {
                    "default": "scene",
                    "tooltip": "scene=by scene/location (recommended), shot=by camera shot (finer), beat=by emotional beat",
                }),
                "max_scenes": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Maximum number of scenes to split into",
                }),
            },
            "optional": {
                "character_list": ("CHARACTER_LIST",),
                "style_hint": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Style hint (optional): e.g., anime style, photorealistic, cyberpunk...",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom system prompt (leave empty for built-in default). Controls how LLM splits scenes and generates prompts.",
                }),
            },
        }

    RETURN_TYPES = ("SCENE_LIST", "STRING")
    RETURN_NAMES = ("scene_list", "scene_summary")
    FUNCTION = "split_story"
    CATEGORY = "PromptForge/LLM"

    def split_story(self, story, model, split_mode="scene", max_scenes=8,
                    character_list=None, style_hint="", system_prompt=""):

        # Build character info
        character_info = ""
        if character_list:
            char_parts = []
            for char in character_list:
                desc = f"- {char['name']}: {char.get('age', '')} {char.get('gender', '')}"
                if char.get('appearance'):
                    desc += f", {char['appearance']}"
                if char.get('clothing'):
                    desc += f", wearing {char['clothing']}"
                if char.get('features'):
                    desc += f", {char['features']}"
                char_parts.append(desc)
            character_info = "\n".join(char_parts)

        # Split mode descriptions
        mode_desc = {
            "scene": "Scene-level split - divide by location/time/atmosphere changes, each image is a complete scene",
            "shot": "Shot-level split - finer granularity, includes close-ups, medium shots, wide shots etc.",
            "beat": "Beat-level split - divide by emotional/action turning points, suitable for dynamic narratives",
        }

        # System prompt: built-in default, user custom appended after
        default_system_prompt = (
            "You are a professional storyboard artist and visual director.\n"
            "Your task is to analyze a story/narrative text and split it into visual scenes for image generation.\n"
            "\n"
            "CRITICAL RULES:\n"
            "- Never use character names directly in visual_prompt. Always replace names with full character descriptions (appearance, age, gender, clothing, etc.).\n"
            "- Use natural language descriptions in visual_prompt, not structured tags.\n"
            "- Each visual_prompt MUST start with \"masterpiece, best quality, score_7, \" if not already present.\n"
            "\n"
            "For each scene, output a JSON object with:\n"
            "- \"scene_id\": scene number (1, 2, 3...)\n"
            "- \"scene_title\": brief title (2-5 words)\n"
            "- \"scene_description\": what's happening in this scene (1-2 sentences)\n"
            "- \"visual_prompt\": detailed image generation prompt in English (replace all character names with their physical descriptions)\n"
            "- \"negative_prompt\": what to avoid in the image\n"
            "- \"camera_angle\": suggested camera angle (close-up, medium shot, wide shot, etc.)\n"
            "- \"mood\": emotional tone (happy, tense, peaceful, etc.)\n"
            "\n"
            "Output a JSON array of scenes. Example:\n"
            "[\n"
            "  {\n"
            '    "scene_id": 1,\n'
            '    "scene_title": "Morning Coffee",\n'
            '    "scene_description": "Character sits at a cafe table, looking out the window.",\n'
            '    "visual_prompt": "masterpiece, best quality, score_7, a young woman with black hair sitting at a wooden cafe table, morning sunlight through window, coffee cup, thoughtful expression, warm lighting",\n'
            '    "negative_prompt": "worst quality, low quality, blurry, deformed",\n'
            '    "camera_angle": "medium shot",\n'
            '    "mood": "peaceful"\n'
            "  }\n"
            "]"
        )

        # User custom prompt appended to built-in default
        if system_prompt and system_prompt.strip():
            final_system = default_system_prompt + "\n\nAdditional instructions:\n" + system_prompt.strip()
        else:
            final_system = default_system_prompt

        # Build user prompt
        char_section = f"\nCharacters in the story:\n{character_info}\n" if character_info else ""
        style_section = f"\nStyle hint: {style_hint}\n" if style_hint else ""

        user_prompt = (
            f"Please split the following story into {max_scenes} or fewer visual scenes.\n\n"
            f"Split mode: {split_mode} - {mode_desc.get(split_mode, '')}\n"
            f"{char_section}"
            f"{style_section}"
            f"Story text:\n---\n{story}\n---\n\n"
            f"Output ONLY the JSON array, no other text."
        )

        try:
            response = _call_llm(user_prompt, system=final_system, model=model,
                                 temperature=0.7, max_tokens=4096)

            logger.info(f"[Story Splitter] LLM raw response ({len(response)} chars):\n{response[:2000]}")

            # Extract JSON
            scenes = self._extract_json(response)
            logger.info(f"[Story Splitter] Extracted {len(scenes)} scenes")

            # Normalize field names
            scenes = [self._normalize_scene(s) for s in scenes]

            if not scenes:
                scenes = [{"scene_id": 1, "scene_title": "Error",
                           "scene_description": "Failed to parse AI response",
                           "visual_prompt": story[:500],
                           "camera_angle": "medium shot", "mood": "neutral"}]
                summary = f"[Error] AI response parse failed, raw:\n{response[:1000]}"
            else:
                # Generate summary
                summary_lines = [f"=== Story Split ({len(scenes)} scenes) ===\n"]
                for scene in scenes:
                    summary_lines.append(f"Scene {scene.get('scene_id', '?')}: {scene.get('scene_title', '?')}")
                    summary_lines.append(f"  Description: {scene.get('scene_description', '?')}")
                    summary_lines.append(f"  Camera: {scene.get('camera_angle', '?')} | Mood: {scene.get('mood', '?')}")
                    summary_lines.append(f"  Prompt: {scene.get('visual_prompt', '?')[:80]}...")
                    neg = scene.get('negative_prompt', '')
                    if neg:
                        summary_lines.append(f"  Negative: {neg[:60]}...")
                    summary_lines.append("")
                summary = "\n".join(summary_lines)

        except Exception as e:
            logger.error(f"[Story Splitter] Error: {e}")
            scenes = [{"scene_id": 1, "scene_title": "Error",
                       "scene_description": str(e),
                       "visual_prompt": story[:500],
                       "camera_angle": "medium shot", "mood": "neutral"}]
            summary = f"[API Error] {str(e)}"

        return (scenes, summary)

    def _extract_json(self, text: str):
        """Extract JSON array from AI response, compatible with multiple formats."""

        def _normalize(obj):
            """Normalize various JSON structures to [{}, {}, ...] format."""
            if isinstance(obj, list):
                dicts = [item for item in obj if isinstance(item, dict)]
                if dicts:
                    return dicts
                strings = [item for item in obj if isinstance(item, str)]
                if strings:
                    return [
                        {
                            "scene_id": i + 1,
                            "scene_title": f"Scene {i + 1}",
                            "scene_description": s[:100],
                            "visual_prompt": s,
                            "camera_angle": "medium shot",
                            "mood": "neutral",
                        }
                        for i, s in enumerate(strings)
                    ]
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, list):
                        return _normalize(v)
            return []

        # 1. Try parsing entire text
        try:
            result = json.loads(text.strip())
            normalized = _normalize(result)
            if normalized:
                return normalized
        except Exception:
            pass

        # 2. Find ```json ... ``` blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                normalized = _normalize(result)
                if normalized:
                    return normalized
            except Exception:
                pass

        # 3. Find [ ... ] blocks
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                normalized = _normalize(result)
                if normalized:
                    return normalized
            except Exception:
                pass

        return []

    # Field name normalization map (LLMs use various field names)
    _KEY_MAP = {
        "scene_id": "scene_id", "id": "scene_id", "number": "scene_id",
        "scene_number": "scene_id", "no": "scene_id", "index": "scene_id",
        "scene_title": "scene_title", "title": "scene_title",
        "name": "scene_title", "heading": "scene_title",
        "scene_description": "scene_description", "description": "scene_description",
        "desc": "scene_description", "summary": "scene_description",
        "scene_desc": "scene_description", "narrative": "scene_description",
        "visual_prompt": "visual_prompt", "prompt": "visual_prompt",
        "image_prompt": "visual_prompt", "img_prompt": "visual_prompt",
        "sd_prompt": "visual_prompt", "generation_prompt": "visual_prompt",
        "positive_prompt": "visual_prompt", "pos_prompt": "visual_prompt",
        "camera_angle": "camera_angle", "camera": "camera_angle",
        "shot": "camera_angle", "angle": "camera_angle",
        "shot_type": "camera_angle", "framing": "camera_angle",
        "mood": "mood", "emotion": "mood", "tone": "mood",
        "atmosphere": "mood", "feeling": "mood",
        "negative_prompt": "negative_prompt", "neg_prompt": "negative_prompt",
        "negative": "negative_prompt",
    }

    def _normalize_scene(self, scene: dict) -> dict:
        """Normalize LLM field names to standard format."""
        result = {}
        for k, v in scene.items():
            key = self._KEY_MAP.get(k.lower().strip(), k.lower().strip())
            result[key] = v

        defaults = {
            "scene_id": "?",
            "scene_title": "?",
            "scene_description": "?",
            "visual_prompt": "",
            "negative_prompt": "",
            "camera_angle": "medium shot",
            "mood": "neutral",
        }
        for field, fallback in defaults.items():
            if field not in result or not result[field]:
                result[field] = fallback

        return result


# ---------------------------------------------------------------------------
# Node 3b: SceneSelectorNode - select a single scene from the list
# ---------------------------------------------------------------------------

class SceneSelectorNode:
    """
    Scene selector. Pick a specific scene from a SCENE_LIST by index,
    useful for generating images one at a time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_list": ("SCENE_LIST",),
                "scene_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Which scene to select (0-indexed)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("visual_prompt", "negative_prompt", "scene_title",
                    "scene_description", "mood", "total_scenes")
    FUNCTION = "select_scene"
    CATEGORY = "PromptForge/LLM"

    def select_scene(self, scene_list, scene_index=0):
        total = len(scene_list)

        if total == 0:
            return ("", "", "Empty", "No scenes", "neutral", 0)

        if scene_index >= total:
            scene_index = total - 1

        scene = scene_list[scene_index]

        visual_prompt = scene.get("visual_prompt", "")
        negative_prompt = scene.get("negative_prompt", "")
        scene_title = scene.get("scene_title", f"Scene {scene_index + 1}")
        scene_description = scene.get("scene_description", "")
        mood = scene.get("mood", "neutral")

        return (visual_prompt, negative_prompt, scene_title,
                scene_description, mood, total)


# ---------------------------------------------------------------------------
# Node 3c: SceneListNode - view all scenes as formatted text
# ---------------------------------------------------------------------------

class SceneListNode:
    """
    Scene list viewer. Displays the full SCENE_LIST as readable text
    with all scene details.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_list": ("SCENE_LIST",),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("scenes_text", "total_scenes")
    FUNCTION = "view_scenes"
    CATEGORY = "PromptForge/LLM"

    def view_scenes(self, scene_list):
        total = len(scene_list)

        lines = [f"=== Scene List ({total} scenes) ===\n"]

        for scene in scene_list:
            sid = scene.get("scene_id", "?")
            title = scene.get("scene_title", "?")
            desc = scene.get("scene_description", "")
            prompt = scene.get("visual_prompt", "")
            angle = scene.get("camera_angle", "?")
            mood = scene.get("mood", "?")

            lines.append(f"[Scene {sid}] {title}")
            lines.append(f"  Description: {desc}")
            lines.append(f"  Camera: {angle} | Mood: {mood}")
            lines.append(f"  Prompt: {prompt}")
            neg = scene.get("negative_prompt", "")
            if neg:
                lines.append(f"  Negative: {neg}")
            lines.append("")

        return ("\n".join(lines), total)


# ---------------------------------------------------------------------------
# Node 3d: SceneBatchOutputNode - batch output all prompts at once
# ---------------------------------------------------------------------------

class SceneBatchOutputNode:
    """
    Batch output all scene prompts. Uses separators between scenes
    for easy copy-paste to other tools for batch generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_list": ("SCENE_LIST",),
                "separator": ("STRING", {
                    "default": "---",
                    "multiline": False,
                    "tooltip": "Separator between scenes",
                }),
                "include_metadata": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "Include scene title and other metadata",
                }),
            },
            "optional": {
                "character_list": ("CHARACTER_LIST",),
                "prepend_character": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "Prepend character descriptions to each prompt",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("all_prompts", "prompts_only")
    FUNCTION = "batch_output"
    CATEGORY = "PromptForge/LLM"

    def batch_output(self, scene_list, separator="---", include_metadata="yes",
                     character_list=None, prepend_character="yes"):

        # Build character prefix
        char_prefix = ""
        if character_list and prepend_character == "yes":
            char_parts = []
            for char in character_list:
                desc = f"{char['name']}: {char.get('age', '')} {char.get('gender', '')}"
                if char.get('appearance'):
                    desc += f", {char['appearance']}"
                if char.get('clothing'):
                    desc += f", wearing {char['clothing']}"
                if char.get('features'):
                    desc += f", {char['features']}"
                char_parts.append(desc)
            char_prefix = "Characters: " + "; ".join(char_parts) + "\n\n"

        all_prompts_lines = []
        prompts_only_lines = []

        for i, scene in enumerate(scene_list):
            sid = scene.get("scene_id", i + 1)
            title = scene.get("scene_title", f"Scene {sid}")
            desc = scene.get("scene_description", "")
            prompt = scene.get("visual_prompt", "")
            angle = scene.get("camera_angle", "")
            mood = scene.get("mood", "")

            # Full output with metadata
            if include_metadata == "yes":
                all_prompts_lines.append(f"[Scene {sid}] {title}")
                all_prompts_lines.append(f"Description: {desc}")
                all_prompts_lines.append(f"Camera: {angle} | Mood: {mood}")
                all_prompts_lines.append(f"Prompt: {char_prefix}{prompt}")
            else:
                all_prompts_lines.append(f"{char_prefix}{prompt}")

            # Prompts-only output
            prompts_only_lines.append(f"{char_prefix}{prompt}")

            if i < len(scene_list) - 1:
                all_prompts_lines.append(separator)
                prompts_only_lines.append(separator)

        return ("\n".join(all_prompts_lines), "\n".join(prompts_only_lines))


# ---------------------------------------------------------------------------
# Node 4: ImageAnalyzerNode
# ---------------------------------------------------------------------------

class ImageAnalyzerNode:
    """
    Analyze an input image and generate a text description or prompt
    suitable for reproducing a similar image. Uses multimodal LLM APIs.

    Note: This node requires a VAE encode input for the image tensor,
    which is converted to a base64 representation for the API call.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _get_model_list()
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to analyze",
                }),
                "model": (models,),
            },
            "optional": {
                "analysis_type": ([
                    "describe",
                    "prompt_generation",
                    "style_extraction",
                    "character_extraction",
                    "custom",
                ],),
                "custom_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom analysis instruction (used when analysis_type=custom)",
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 4096,
                    "step": 128,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("analysis", "generated_prompt")
    FUNCTION = "analyze"
    CATEGORY = "PromptForge/LLM"

    def analyze(self, image, model, analysis_type="describe",
                custom_instruction="", max_tokens=1024):
        # Convert image tensor to base64 for API
        try:
            import base64
            import torch
            from io import BytesIO
            from PIL import Image as PILImage

            # ComfyUI images are (B, H, W, C) float32 tensors [0, 1]
            img_tensor = image[0]  # Take first image from batch
            img_np = (img_tensor.cpu().numpy() * 255).astype("uint8")
            pil_img = PILImage.fromarray(img_np)

            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            img_data_url = f"data:image/png;base64,{img_b64}"
        except Exception as e:
            return (f"[Image Processing Error] {e}", "")

        # Build analysis prompt
        system_prompts = {
            "describe": "Describe this image in detail. Focus on visual elements, composition, colors, and mood.",
            "prompt_generation": (
                "Analyze this image and generate a detailed Stable Diffusion / FLUX prompt "
                "that would recreate a similar image. Include subject, style, lighting, composition, "
                "and quality tags. Output the prompt only, no explanation."
            ),
            "style_extraction": (
                "Analyze the artistic style of this image. Describe: art medium, color palette, "
                "lighting style, composition technique, and any distinctive artistic signatures."
            ),
            "character_extraction": (
                "Analyze the main character/subject in this image. Describe in detail: "
                "physical features, clothing, accessories, pose, expression, and distinguishing characteristics."
            ),
        }

        system = system_prompts.get(analysis_type, "")
        if analysis_type == "custom" and custom_instruction:
            system = custom_instruction
        if not system:
            system = "Analyze this image and provide a detailed description."

        # Build multimodal message
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze this image."},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ],
            },
        ]

        # Make API call (OpenAI-compatible multimodal)
        try:
            import requests as req

            parts = model.split("/", 1)
            provider = parts[0].lower() if len(parts) > 1 else "openai"
            model_name = parts[1] if len(parts) > 1 else parts[0]

            if provider == "ollama":
                base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                }
                resp = req.post(f"{base_url}/api/chat", json=payload, timeout=120)
                resp.raise_for_status()
                analysis = resp.json().get("message", {}).get("content", "")
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
                resp = req.post(f"{base_url}/chat/completions", json=payload,
                                headers=headers, timeout=120)
                resp.raise_for_status()
                analysis = resp.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return (f"[API Error] {e}", "")

        # If prompt_generation mode, the analysis itself is the prompt
        if analysis_type == "prompt_generation":
            return (analysis, analysis)

        return (analysis, "")


# ---------------------------------------------------------------------------
# Node 5: PromptEnhancerNode
# ---------------------------------------------------------------------------

class PromptEnhancerNode:
    """
    Enhance/expand a basic prompt into a more detailed, effective prompt
    for image generation using an LLM. Adds quality tags, composition
    details, and stylistic refinements.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _get_model_list()
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": "Basic prompt to enhance",
                }),
                "model": (models,),
            },
            "optional": {
                "enhancement_level": (["subtle", "moderate", "detailed"], {
                    "tooltip": "How much detail to add",
                }),
                "target_model": (["SDXL", "SD1.5", "FLUX", "Midjourney", "DALL-E", "general"], {
                    "tooltip": "Target generation model for optimized prompting",
                }),
                "negative_hint": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional negative prompt suggestions to include",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_positive", "enhanced_negative")
    FUNCTION = "enhance"
    CATEGORY = "PromptForge/LLM"

    def enhance(self, prompt, model, enhancement_level="moderate",
                target_model="general", negative_hint=""):
        system = (
            "You are an expert AI image generation prompt engineer. "
            "Enhance the given prompt to produce better results. "
            f"Target model: {target_model}. "
            f"Enhancement level: {enhancement_level}. "
            "Output exactly two sections separated by '---NEGATIVE---': "
            "first the enhanced positive prompt, then the enhanced negative prompt."
        )

        user = f"Enhance this prompt:\n{prompt}"
        if negative_hint:
            user += f"\n\nAdditional negative considerations: {negative_hint}"

        response = _call_llm(user, system=system, model=model,
                             temperature=0.5, max_tokens=1024)

        # Split response
        if "---NEGATIVE---" in response:
            pos, neg = response.split("---NEGATIVE---", 1)
            return (pos.strip(), neg.strip())

        # Fallback: return full response as positive
        return (response.strip(), negative_hint.strip() if negative_hint else "")


class APIConfigNode:
    """API配置节点 - 存储API URL和API Key，输出默认模型名"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": "https://api.deepseek.com",
                    "placeholder": "https://api.deepseek.com"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "sk-..."
                }),
            },
            "optional": {
                "default_model": ("STRING", {
                    "multiline": False,
                    "default": "deepseek-chat",
                    "placeholder": "deepseek-chat"
                }),
            }
        }

    RETURN_TYPES = ("API_CONFIG", "STRING")
    RETURN_NAMES = ("api_config", "default_model")
    FUNCTION = "configure"
    CATEGORY = "PromptForge/Config"

    def configure(self, api_url, api_key, default_model="deepseek-chat"):
        api_url = api_url.rstrip("/")
        config = {
            "api_url": api_url,
            "api_key": api_key,
            "default_model": default_model
        }
        return (config, default_model)


class APITestNode:
    """测试API连通性和获取可用模型列表"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_config": ("API_CONFIG",),
                "test_mode": (["connectivity", "list_models", "both"], {
                    "default": "both"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("test_result", "models_list")
    FUNCTION = "test_api"
    CATEGORY = "PromptForge/Config"

    def test_api(self, api_config, test_mode="both"):
        api_url = api_config["api_url"]
        api_key = api_config["api_key"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        result_lines = []
        models_list = ""

        if test_mode in ["connectivity", "both"]:
            try:
                test_url = f"{api_url}/v1/models"
                resp = requests.get(test_url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    result_lines.append("[OK] API连接成功!")
                    result_lines.append(f"状态码: {resp.status_code}")
                else:
                    result_lines.append(f"[FAIL] API返回状态码: {resp.status_code}")
                    try:
                        err = resp.json()
                        result_lines.append(f"错误信息: {json.dumps(err, ensure_ascii=False)}")
                    except:
                        result_lines.append(f"响应内容: {resp.text[:500]}")
            except requests.exceptions.ConnectionError:
                result_lines.append(f"[FAIL] 连接失败: 无法连接到 {api_url}")
            except requests.exceptions.Timeout:
                result_lines.append(f"[FAIL] 连接超时: {api_url} 响应超时")
            except Exception as e:
                result_lines.append(f"[FAIL] 测试失败: {str(e)}")

        if test_mode in ["list_models", "both"]:
            try:
                models_url = f"{api_url}/v1/models"
                resp = requests.get(models_url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if "data" in data:
                        model_ids = [m.get("id", "unknown") for m in data["data"]]
                        models_list = "\n".join(model_ids)
                        result_lines.append(f"\n[OK] 获取到 {len(model_ids)} 个模型:")
                        for mid in model_ids:
                            result_lines.append(f"  - {mid}")
                    else:
                        result_lines.append("[WARN] 响应中没有data字段")
                else:
                    result_lines.append(f"[FAIL] 获取模型列表失败: {resp.status_code}")
            except Exception as e:
                result_lines.append(f"[FAIL] 获取模型列表失败: {str(e)}")

        if not models_list:
            models_list = "(无法获取模型列表)"

        return ("\n".join(result_lines), models_list)


class CharacterAnchorNode:
    """
    人物外貌锚点节点
    定义角色外貌描述，用于保持多张图片生成时人物外观的一致性
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_name": ("STRING", {
                    "multiline": False,
                    "default": "主角",
                    "placeholder": "角色名称，如：小红、Alice"
                }),
                "gender": (["female", "male", "other"],),
                "age_range": (["child", "teen", "young_adult", "adult", "middle_aged", "elderly"], {
                    "default": "young_adult"
                }),
                "appearance_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "详细外貌描述，如：黑色长发，大眼睛，瓜子脸，皮肤白皙，身材纤细"
                }),
            },
            "optional": {
                "clothing_style": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "服装风格描述（可选）"
                }),
                "distinctive_features": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "标志性特征，如：左脸有酒窝、戴红色发卡"
                }),
                "existing_characters": ("CHARACTER_LIST",),
            }
        }

    RETURN_TYPES = ("CHARACTER_LIST", "STRING")
    RETURN_NAMES = ("character_list", "character_prompt")
    FUNCTION = "add_character"
    CATEGORY = "PromptForge/Character"

    AGE_MAP = {
        "child": "child (6-12 years old)",
        "teen": "teenager (13-17 years old)",
        "young_adult": "young adult (18-25 years old)",
        "adult": "adult (26-40 years old)",
        "middle_aged": "middle-aged (41-60 years old)",
        "elderly": "elderly (60+ years old)"
    }

    GENDER_MAP = {
        "female": "female",
        "male": "male",
        "other": "androgynous"
    }

    def add_character(self, character_name, gender, age_range, appearance_description,
                      clothing_style="", distinctive_features="", existing_characters=None):

        character = {
            "name": character_name,
            "gender": self.GENDER_MAP.get(gender, gender),
            "age": self.AGE_MAP.get(age_range, age_range),
            "appearance": appearance_description,
            "clothing": clothing_style,
            "features": distinctive_features
        }

        characters = []
        if existing_characters is not None:
            characters = existing_characters.copy()
        characters.append(character)

        character_prompt = self._build_character_prompt(characters)
        return (characters, character_prompt)

    def _build_character_prompt(self, characters: list) -> str:
        if not characters:
            return ""

        parts = []
        for char in characters:
            char_parts = []
            if char["name"]:
                char_parts.append(f"Character '{char['name']}'")
            char_parts.append(f"{char['age']} {char['gender']}")
            if char["appearance"]:
                char_parts.append(char["appearance"])
            if char["clothing"]:
                char_parts.append(f"wearing {char['clothing']}")
            if char["features"]:
                char_parts.append(char["features"])
            parts.append(", ".join(char_parts))

        return "Characters: " + "; ".join(parts)


class CharacterMergeNode:
    """合并多个角色列表，用于多人物场景"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_list_1": ("CHARACTER_LIST",),
                "character_list_2": ("CHARACTER_LIST",),
            },
            "optional": {
                "character_list_3": ("CHARACTER_LIST",),
                "character_list_4": ("CHARACTER_LIST",),
            }
        }

    RETURN_TYPES = ("CHARACTER_LIST", "STRING")
    RETURN_NAMES = ("merged_characters", "merged_prompt")
    FUNCTION = "merge_characters"
    CATEGORY = "PromptForge/Character"

    def merge_characters(self, character_list_1, character_list_2,
                         character_list_3=None, character_list_4=None):
        merged = character_list_1.copy() + character_list_2.copy()
        if character_list_3:
            merged += character_list_3.copy()
        if character_list_4:
            merged += character_list_4.copy()

        prompt_parts = []
        for char in merged:
            char_desc = f"{char['name']}: {char['age']} {char['gender']}"
            if char['appearance']:
                char_desc += f", {char['appearance']}"
            if char['clothing']:
                char_desc += f", wearing {char['clothing']}"
            if char['features']:
                char_desc += f", {char['features']}"
            prompt_parts.append(char_desc)

        prompt = "Characters in scene: " + "; ".join(prompt_parts)
        return (merged, prompt)


class HistoryClearNode:
    """清空对话历史，开始新对话"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("STRING", {"default": "clear", "multiline": False}),
            },
        }

    RETURN_TYPES = ("CHAT_HISTORY",)
    RETURN_NAMES = ("empty_history",)
    FUNCTION = "clear_history"
    CATEGORY = "PromptForge/LLM"

    def clear_history(self, trigger):
        return ({"messages": []},)


class HistoryViewNode:
    """查看对话历史内容"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chat_history": ("CHAT_HISTORY",),
            },
            "optional": {
                "max_display": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("history_text", "total_messages")
    FUNCTION = "view_history"
    CATEGORY = "PromptForge/LLM"

    def view_history(self, chat_history, max_display=20):
        messages = chat_history.get("messages", [])
        total = len(messages)

        lines = [f"=== 对话历史 ({total} 条消息) ===\n"]

        display_msgs = messages[-max_display:] if len(messages) > max_display else messages
        if len(messages) > max_display:
            lines.append(f"... (隐藏了 {len(messages) - max_display} 条早期消息) ...\n")

        for msg in display_msgs:
            role = msg["role"]
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            
            role_cn = {"system": "系统", "user": "用户", "assistant": "助手"}.get(role, role)
            lines.append(f"[{role_cn}]")
            lines.append(content)
            lines.append("")

        return ("\n".join(lines), total)


NODE_CLASS_MAPPINGS = {
    "LLMChatNode": LLMChatNode,
    "CharacterConsistencyNode": CharacterConsistencyNode,
    "StorySplitterNode": StorySplitterNode,
    "SceneSelectorNode": SceneSelectorNode,
    "SceneListNode": SceneListNode,
    "SceneBatchOutputNode": SceneBatchOutputNode,
    "ImageAnalyzerNode": ImageAnalyzerNode,
    "PromptEnhancerNode": PromptEnhancerNode,
    "APIConfigNode": APIConfigNode,
    "APITestNode": APITestNode,
    "CharacterAnchorNode": CharacterAnchorNode,
    "CharacterMergeNode": CharacterMergeNode,
    "HistoryClearNode": HistoryClearNode,
    "HistoryViewNode": HistoryViewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMChatNode": "LLM Chat",
    "CharacterConsistencyNode": "Character Consistency",
    "StorySplitterNode": "Story Splitter",
    "SceneSelectorNode": "Scene Selector",
    "SceneListNode": "Scene List View",
    "SceneBatchOutputNode": "Scene Batch Output",
    "ImageAnalyzerNode": "Image Analyzer",
    "PromptEnhancerNode": "Prompt Enhancer",
    "APIConfigNode": "API Config",
    "APITestNode": "API Test",
    "CharacterAnchorNode": "Character Anchor",
    "CharacterMergeNode": "Character Merge",
    "HistoryClearNode": "History Clear",
    "HistoryViewNode": "History View",
}
