"""
PromptForge - Config Nodes
API 配置和测试
"""
import requests


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

    # 关键修改：default_model 也作为输出
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


# ============================================================
# 2. API测试节点
# ============================================================
class


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


# ============================================================
# 3. 人物外貌锚点节点
# ============================================================
class


NODE_CLASS_MAPPINGS = {
    "APIConfigNode": APIConfigNode,
    "APITestNode": APITestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIConfigNode": "PromptForge API 配置",
    "APITestNode": "PromptForge API 测试",
}
