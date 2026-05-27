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
import os
import re
from pathlib import Path

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
# Node 3: StorySplitterNode
# ---------------------------------------------------------------------------

class StorySplitterNode:
    """
    Split a story or narrative into individual scene descriptions suitable
    for image generation. Uses LLM to parse narrative text into discrete
    visual scenes, each becoming a separate prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _get_model_list()
        return {
            "required": {
                "story": ("STRING", {
                    "multiline": True,
                    "tooltip": "Full story/narrative text to split into scenes",
                }),
                "model": (models,),
            },
            "optional": {
                "num_scenes": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Target number of scenes",
                }),
                "style_hint": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Style instruction appended to each scene prompt",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "You are a visual storytelling assistant. "
                        "Split the given story into discrete visual scenes. "
                        "For each scene, provide a detailed image generation prompt. "
                        "Output as a JSON array of objects with 'scene_number' and 'prompt' keys."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("scene_1", "scene_2", "scene_count")
    FUNCTION = "split_story"
    CATEGORY = "PromptForge/LLM"

    def split_story(self, story, model, num_scenes=4, style_hint="",
                    system_prompt=""):
        user_prompt = (
            f"Split this story into exactly {num_scenes} visual scenes.\n"
            f"Each scene should be a detailed image generation prompt.\n"
            f"{'Style: ' + style_hint if style_hint else ''}\n\n"
            f"Story:\n{story}"
        )

        response = _call_llm(user_prompt, system=system_prompt, model=model,
                             temperature=0.7, max_tokens=4096)

        # Try to parse JSON response
        scenes = []
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    scenes = [item.get("prompt", str(item)) for item in data]
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: split by scene markers
        if not scenes:
            parts = re.split(r'(?:Scene\s+\d+|#{1,3}\s*Scene)', response, flags=re.IGNORECASE)
            scenes = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]

        # Fallback: just split by newlines into chunks
        if not scenes:
            scenes = [s.strip() for s in response.split("\n\n") if s.strip() and len(s.strip()) > 10]

        # Pad to at least 2 (for the two RETURN_TYPES)
        while len(scenes) < 2:
            scenes.append("")

        # Append style hint if present
        if style_hint.strip():
            scenes = [f"{s}, {style_hint.strip()}" if s else s for s in scenes]

        scene_count = min(len(scenes), 20)  # Cap for return types
        return (scenes[0] if len(scenes) > 0 else "",
                scenes[1] if len(scenes) > 1 else "",
                scene_count)


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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LLMChatNode": LLMChatNode,
    "CharacterConsistencyNode": CharacterConsistencyNode,
    "StorySplitterNode": StorySplitterNode,
    "ImageAnalyzerNode": ImageAnalyzerNode,
    "PromptEnhancerNode": PromptEnhancerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMChatNode": "LLM Chat",
    "CharacterConsistencyNode": "Character Consistency",
    "StorySplitterNode": "Story Splitter",
    "ImageAnalyzerNode": "Image Analyzer",
    "PromptEnhancerNode": "Prompt Enhancer",
}
