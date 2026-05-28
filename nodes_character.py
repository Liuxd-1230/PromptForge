"""
PromptForge - Character Nodes
人物锚点和合并
"""
from typing import List, Dict


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

    def _build_character_prompt(self, characters: List[Dict]) -> str:
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


# ============================================================
# 4. 人物列表合并节点
# ============================================================


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


# ============================================================
# 5. 改进的LLM对话节点 - 修复历史记录 + 思考模式
# ============================================================


NODE_CLASS_MAPPINGS = {
    "CharacterAnchorNode": CharacterAnchorNode,
    "CharacterMergeNode": CharacterMergeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterAnchorNode": "PromptForge 人物锚点",
    "CharacterMergeNode": "PromptForge 人物合并",
}
