import base64
import json
import math
import time
from io import BytesIO

import comfy.utils
import numpy as np
import requests
import torch
from PIL import Image
from comfy.comfy_types import IO


def tensor2pil(image):
    return [
        Image.fromarray(
            np.clip(255.0 * img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
        for img in image
    ]


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class NewApiBanana2Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🔑 API密钥": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "请输入您的 API Key"},
                ),
                "🌐 API地址": (
                    "STRING",
                    {
                        "default": "https://api.llyapps.com",
                        "multiline": False,
                        "tooltip": "New API 服务地址（填域名/端口即可；如果你粘贴了 /v1 或 /v1/images/generations 也会自动纠正）",
                    },
                ),
                "📝 提示词": (
                    "STRING",
                    {"multiline": True, "default": "一个美女坐在深林里面，真实摄影", "tooltip": "提示词"},
                ),
                "🎨 生成模式": (
                    ["文生图", "图像编辑"],
                    {"default": "文生图", "tooltip": "模式：文生图 或 图像编辑"},
                ),
                "🤖 模型版本": (
                    "STRING",
                    {
                        "default": "gemini-3-pro-image-preview",
                        "multiline": False,
                        "tooltip": "模型名称/ID（可直接粘贴 New API 面板里的模型名）",
                    },
                ),
                "📐 宽高比": (
                    [
                        "auto",
                        "16:9",
                        "4:3",
                        "4:5",
                        "3:2",
                        "1:1",
                        "2:3",
                        "3:4",
                        "5:4",
                        "9:16",
                        "21:9",
                    ],
                    {"default": "auto", "tooltip": "宽高比"},
                ),
                "🖼️ 图片分辨率": (
                    ["1K", "2K", "4K"],
                    {"default": "2K", "tooltip": "图片分辨率"},
                ),
                "🎲 随机种子": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"},
                ),
                "🖼️ 出图数量": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4, "tooltip": "一次请求生成图片数量"},
                ),
            },
            "optional": {
                "任务ID": (
                    "STRING",
                    {"default": "", "tooltip": "仅用于透传/记录；New API Image 接口一般为同步返回"},
                ),
                "返回格式": (
                    ["url", "b64_json"],
                    {"default": "url", "tooltip": "返回格式：URL链接 或 Base64编码"},
                ),
                "参考图1": ("IMAGE",),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "参考图4": ("IMAGE",),
                "参考图5": ("IMAGE",),
                "参考图6": ("IMAGE",),
                "参考图7": ("IMAGE",),
                "参考图8": ("IMAGE",),
                "参考图9": ("IMAGE",),
                "参考图10": ("IMAGE",),
                "参考图11": ("IMAGE",),
                "参考图12": ("IMAGE",),
                "参考图13": ("IMAGE",),
                "参考图14": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图像", "图片链接", "任务ID", "响应信息")
    FUNCTION = "generate_image"
    CATEGORY = "NIUNIUAPI"

    def __init__(self):
        self.timeout = 600

    def _normalize_base_url(self, base_url: str) -> str:
        url = str(base_url or "").strip()
        if not url:
            return ""
        url = url.strip("`").strip().strip('"').strip("'")
        url = url.split("?", 1)[0].rstrip("/")

        endpoint_suffixes = [
            "/v1/images/generations",
            "/v1/images/edits",
            "/images/generations",
            "/images/edits",
        ]
        for s in endpoint_suffixes:
            if url.endswith(s):
                url = url[: -len(s)].rstrip("/")
                break

        if url.endswith("/v1"):
            return url
        return f"{url}/v1"

    def _infer_long_side(self, image_size: str, model: str) -> int:
        long_side_map = {"1K": 1024, "2K": 2048, "4K": 4096}
        requested = int(long_side_map.get(image_size, 0))
        if requested <= 0:
            return 0

        m = str(model or "").lower()
        is_image_preview = "image-preview" in m
        cap = 0
        if "4k" in m:
            cap = 4096
        elif "2k" in m:
            cap = 2048
        elif "1k" in m:
            cap = 1024
        elif "hd" in m:
            cap = 2048
        elif is_image_preview:
            cap = 1024

        if cap > 0:
            return min(requested, cap)
        return requested

    def _calc_size(self, aspect_ratio: str, image_size: str, model: str):
        long_side = self._infer_long_side(image_size, model)
        if long_side <= 0:
            return None

        ratio = (aspect_ratio or "").strip()
        if not ratio or ratio == "auto" or ":" not in ratio:
            if "image-preview" not in str(model or "").lower():
                return None
            w = max(64, int(math.floor(long_side / 8) * 8))
            h = w
            return f"{w}x{h}"
        try:
            a, b = ratio.split(":", 1)
            rw = float(a)
            rh = float(b)
            if rw <= 0 or rh <= 0:
                return None
        except Exception:
            return None

        if rw >= rh:
            w = long_side
            h = int(round(long_side * (rh / rw)))
        else:
            h = long_side
            w = int(round(long_side * (rw / rh)))

        w = max(64, int(math.floor(w / 8) * 8))
        h = max(64, int(math.floor(h / 8) * 8))
        return f"{w}x{h}"

    def _safe_resp_text(self, resp):
        try:
            t = (resp.text or "").strip()
            if t:
                return t
        except Exception:
            pass
        try:
            raw = resp.content
            if raw:
                return raw.decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        return ""

    def _poll_task(self, base_url_v1: str, api_key: str, task_id: str, pbar):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        max_attempts = 60
        attempt = 0
        while attempt < max_attempts:
            time.sleep(5)
            attempt += 1
            try:
                resp = requests.get(
                    f"{base_url_v1}/images/tasks/{task_id}",
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    continue
                result = resp.json()
                data_obj = result.get("data") if isinstance(result, dict) else None
                if not isinstance(data_obj, dict):
                    continue
                status = str(data_obj.get("status", "unknown"))
                payload = data_obj.get("data")
                if status in ["completed", "success", "done", "finished", "SUCCESS"]:
                    pbar.update_absolute(100)
                    return self._process_success_data(payload, task_id, api_key=api_key)
                if status in ["failed", "error", "FAILURE"]:
                    blank_image = Image.new("RGB", (512, 512), color="red")
                    return (
                        pil2tensor(blank_image),
                        "",
                        task_id,
                        json.dumps(
                            {
                                "status": "failed",
                                "message": data_obj.get("error", "Unknown error"),
                                "raw": result,
                            },
                            ensure_ascii=False,
                        ),
                    )
                pbar.update_absolute(50 + (attempt % 40))
            except Exception:
                continue
        blank_image = Image.new("RGB", (512, 512), color="yellow")
        return (
            pil2tensor(blank_image),
            "",
            task_id,
            json.dumps(
                {"status": "timeout", "message": "Task polling timed out", "task_id": task_id},
                ensure_ascii=False,
            ),
        )

    def _process_success_data(self, data, task_id, api_key: str = ""):
        generated_tensors = []
        image_urls = []

        data_items = data.get("data", []) if isinstance(data, dict) else data
        if not isinstance(data_items, list):
            data_items = [data_items]

        error_message = ""
        for item in data_items:
            if isinstance(item, dict):
                for k in ("error", "message", "detail", "msg", "reason", "code"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        error_message = v.strip()
                        break
                if error_message:
                    break
                for v in item.values():
                    if not isinstance(v, str):
                        continue
                    s = v.strip()
                    if not s:
                        continue
                    if "PROHIBITED_CONTENT" in s or "blocked by" in s.lower():
                        error_message = s
                        break
            elif isinstance(item, str):
                s = item.strip()
                if "PROHIBITED_CONTENT" in s or "blocked by" in s.lower():
                    error_message = s
                    break
            if error_message:
                break

        if error_message:
            blank_image = Image.new("RGB", (512, 512), color="red")
            return (
                pil2tensor(blank_image),
                "",
                task_id,
                json.dumps({"status": "failed", "message": error_message, "raw": data}, ensure_ascii=False),
            )

        def _pick_url(obj):
            if isinstance(obj, str) and obj.startswith(("http://", "https://")):
                return obj
            if isinstance(obj, dict):
                v = obj.get("url")
                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    return v
                v = obj.get("image_url")
                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    return v
                if isinstance(v, dict):
                    u = v.get("url")
                    if isinstance(u, str) and u.startswith(("http://", "https://")):
                        return u
            return ""

        def _pick_b64(obj):
            if isinstance(obj, dict):
                for k in ("b64_json", "b64", "base64", "image_base64"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                v = obj.get("image")
                if isinstance(v, str) and v.strip():
                    return v.strip()
            if isinstance(obj, str):
                s = obj.strip()
                if len(s) > 128 and all(c.isalnum() or c in "+/=\n\r" for c in s[:256]):
                    return s
            return ""

        for item in data_items:
            try:
                img_tensor = None
                img_url = _pick_url(item)
                img_b64 = _pick_b64(item)

                if img_b64:
                    b64_payload = img_b64
                    if "," in b64_payload and "base64" in b64_payload[:64].lower():
                        b64_payload = b64_payload.split(",", 1)[1]
                    image_data = base64.b64decode(b64_payload)
                    img = Image.open(BytesIO(image_data))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_tensor = pil2tensor(img)
                    img_url = "base64_image"

                elif img_url:
                    image_urls.append(img_url)
                    url_headers = {}
                    if (api_key or "").strip():
                        url_headers["Authorization"] = f"Bearer {api_key}"
                    resp = requests.get(img_url, headers=url_headers or None, timeout=self.timeout)
                    resp.raise_for_status()
                    img = Image.open(BytesIO(resp.content))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_tensor = pil2tensor(img)

                if img_tensor is not None:
                    generated_tensors.append(img_tensor)
            except Exception:
                continue

        if generated_tensors:
            combined_tensor = torch.cat(generated_tensors, dim=0)
            first_url = image_urls[0] if image_urls else ""
            result_info = {
                "status": "success",
                "task_id": task_id,
                "images_count": len(generated_tensors),
                "image_url": first_url,
                "all_urls": image_urls,
            }
            return (
                combined_tensor,
                first_url,
                task_id,
                json.dumps(result_info, ensure_ascii=False),
            )

        blank_image = Image.new("RGB", (512, 512), color="white")
        return (
            pil2tensor(blank_image),
            "",
            task_id,
            json.dumps({"status": "empty", "message": "No valid images found", "raw": data}, ensure_ascii=False),
        )

    def generate_image(self, **kwargs):
        api_key = kwargs.get("🔑 API密钥", "")
        api_base = kwargs.get("🌐 API地址", "")
        prompt = kwargs.get("📝 提示词", "")
        mode_raw = kwargs.get("🎨 生成模式", "文生图")
        model = str(kwargs.get("🤖 模型版本", "nano-banana-2") or "").strip()
        aspect_ratio = kwargs.get("📐 宽高比", "auto")
        image_size = kwargs.get("🖼️ 图片分辨率", "2K")
        seed = int(kwargs.get("🎲 随机种子", 0) or 0)
        n_images = int(kwargs.get("🖼️ 出图数量", 1) or 1)
        task_id_input = (kwargs.get("任务ID", "") or "").strip()
        response_format = kwargs.get("返回格式", "url")

        n_images = max(1, min(4, int(n_images)))

        if not (api_key or "").strip():
            blank_image = Image.new("RGB", (512, 512), color="black")
            return (
                pil2tensor(blank_image),
                "",
                "",
                json.dumps(
                    {"status": "failed", "message": "❌ API Key 为空，请在节点中填写 🔑 API密钥"},
                    ensure_ascii=False,
                ),
            )

        base_url_v1 = self._normalize_base_url(api_base)
        if not base_url_v1:
            blank_image = Image.new("RGB", (512, 512), color="black")
            return (
                pil2tensor(blank_image),
                "",
                "",
                json.dumps(
                    {"status": "failed", "message": "❌ API地址 为空，请填写 🌐 API地址"},
                    ensure_ascii=False,
                ),
            )

        images = [kwargs.get(f"参考图{i}") for i in range(1, 15)]
        has_any_image = any(img is not None for img in images)

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        size_str = self._calc_size(aspect_ratio, image_size, model)

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            params = {"async": "true"}

            if mode_raw == "图像编辑":
                if not has_any_image:
                    blank_image = Image.new("RGB", (512, 512), color="black")
                    return (
                        pil2tensor(blank_image),
                        "",
                        "",
                        json.dumps(
                            {"status": "failed", "message": "❌ 图像编辑模式至少需要 1 张参考图"},
                            ensure_ascii=False,
                        ),
                    )

                files = []
                image_count = 0
                for img in images:
                    if img is None:
                        continue
                    pil_img = tensor2pil(img)[0]
                    buffered = BytesIO()
                    pil_img.save(buffered, format="PNG")
                    buffered.seek(0)
                    files.append(("image", (f"image_{image_count}.png", buffered, "image/png")))
                    image_count += 1

                data = {"prompt": prompt, "model": model, "n": 1, "response_format": response_format}
                data["n"] = n_images
                if size_str:
                    data["size"] = size_str
                if seed > 0:
                    data["seed"] = str(seed)
                if task_id_input:
                    data["task_id"] = task_id_input

                resp = requests.post(
                    f"{base_url_v1}/images/edits",
                    headers=headers,
                    params=params,
                    data=data,
                    files=files,
                    timeout=self.timeout,
                )
            else:
                headers["Content-Type"] = "application/json"
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "n": n_images,
                    "response_format": response_format,
                }
                if size_str:
                    payload["size"] = size_str
                if seed > 0:
                    payload["seed"] = seed
                if task_id_input:
                    payload["task_id"] = task_id_input

                resp = requests.post(
                    f"{base_url_v1}/images/generations",
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=self.timeout,
                )

            pbar.update_absolute(40)

            if resp.status_code != 200:
                blank_image = Image.new("RGB", (512, 512), color="red")
                return (
                    pil2tensor(blank_image),
                    "",
                    "",
                    json.dumps(
                        {
                            "status": "failed",
                            "message": f"API Error: {resp.status_code} - {self._safe_resp_text(resp)}",
                            "endpoint": str(resp.url),
                        },
                        ensure_ascii=False,
                    ),
                )

            result = resp.json()
            task_id = str(result.get("task_id") or result.get("id") or task_id_input or "")
            data = result.get("data")
            if not data and isinstance(result, dict) and result.get("task_id"):
                pbar.update_absolute(50)
                return self._poll_task(base_url_v1, api_key, task_id, pbar)
            if not data:
                blank_image = Image.new("RGB", (512, 512), color="gray")
                return (
                    pil2tensor(blank_image),
                    "",
                    task_id,
                    json.dumps({"status": "failed", "message": "未在响应中找到 data", "raw": result}, ensure_ascii=False),
                )

            pbar.update_absolute(80)
            return self._process_success_data(data, task_id, api_key=api_key)
        except Exception as e:
            blank_image = Image.new("RGB", (512, 512), color="red")
            return (
                pil2tensor(blank_image),
                "",
                "",
                json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False),
            )


class NewApiJimeng45Node(NewApiBanana2Node):
    @classmethod
    def INPUT_TYPES(cls):
        base = NewApiBanana2Node.INPUT_TYPES()
        required = dict(base.get("required", {}))

        api_addr = required.get("🌐 API地址")
        if isinstance(api_addr, tuple) and len(api_addr) == 2 and isinstance(api_addr[1], dict):
            api_cfg = dict(api_addr[1])
            api_cfg["default"] = "https://api.llyapps.com"
            required["🌐 API地址"] = (api_addr[0], api_cfg)

        model_field = required.get("🤖 模型版本")
        if isinstance(model_field, tuple) and len(model_field) == 2 and isinstance(model_field[1], dict):
            model_cfg = dict(model_field[1])
            model_cfg["default"] = "jimeng-4.5"
            required["🤖 模型版本"] = (model_field[0], model_cfg)

        base["required"] = required
        return base

    def generate_image(self, **kwargs):
        patched = dict(kwargs or {})
        patched["🤖 模型版本"] = "jimeng-4.5"
        image, url, task_id, info = super().generate_image(**patched)
        try:
            max_out = int(patched.get("🖼️ 出图数量", 1) or 1)
            max_out = max(1, min(4, int(max_out)))
            if isinstance(image, torch.Tensor) and image.dim() == 4 and image.shape[0] > 1:
                image = image[:max_out]
        except Exception:
            pass
        return image, url, task_id, info


class _NiuNiuVideoAdapter:
    def __init__(self, video_url: str, api_key: str = "", fallback_size: str = "1280x720"):
        self.video_url = str(video_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.fallback_size = str(fallback_size or "1280x720")

    def get_dimensions(self):
        try:
            w, h = self.fallback_size.split("x", 1)
            return int(w), int(h)
        except Exception:
            return 1280, 720

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        if not self.video_url:
            return False
        if self.video_url.startswith(("http://", "https://")):
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            try:
                resp = requests.get(self.video_url, headers=headers or None, stream=True, timeout=600)
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                return True
            except Exception:
                return False
        return False


class NiuNiuSora2VideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📝 提示词": ("STRING", {"multiline": True, "default": ""}),
                "🤖 模型": ("STRING", {"default": "sora-2", "multiline": False}),
                "🌐 API地址": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "如 https://api.newapi.pro；自动补全 /v1"},
                ),
                "🔑 API密钥": ("STRING", {"default": "", "multiline": False}),
                "📐 宽高比": (["16:9", "9:16"], {"default": "16:9"}),
                "⏱️ 视频时长(秒)": ("INT", {"default": 8, "min": 1, "max": 60}),
                "🎬 高清模式": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "图像1": ("IMAGE",),
                "图像2": ("IMAGE",),
                "图像3": ("IMAGE",),
                "图像4": ("IMAGE",),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("🎬 视频", "🎥 视频URL", "📋 响应信息")
    FUNCTION = "generate_video"
    CATEGORY = "NIUNIUAPI"

    def __init__(self):
        self.timeout = 900

    def _normalize_base_url(self, base_url: str) -> str:
        url = str(base_url or "").strip()
        if not url:
            return ""
        url = url.strip("`").strip().strip('"').strip("'")
        url = url.split("?", 1)[0].rstrip("/")
        if url.endswith("/v1"):
            return url
        return f"{url}/v1"

    def _pick_size(self, aspect_ratio: str, hd: bool, model: str) -> str:
        m = str(model or "").lower()
        is_pro = "pro" in m
        use_hd = bool(hd and is_pro)
        if aspect_ratio == "9:16":
            return "1024x1792" if use_hd else "720x1280"
        return "1792x1024" if use_hd else "1280x720"

    def _first_input_reference(self, images):
        for img in images:
            if img is None:
                continue
            pil_img = tensor2pil(img)[0]
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            buffered.seek(0)
            return ("input_reference", ("input.png", buffered, "image/png"))
        return None

    def generate_video(self, **kwargs):
        prompt = kwargs.get("📝 提示词", "") or ""
        model = str(kwargs.get("🤖 模型", "sora-2") or "").strip()
        api_base = kwargs.get("🌐 API地址", "")
        api_key = kwargs.get("🔑 API密钥", "")
        aspect_ratio = kwargs.get("📐 宽高比", "16:9")
        seconds = int(kwargs.get("⏱️ 视频时长(秒)", 8) or 8)
        hd = bool(kwargs.get("🎬 高清模式", False))

        if not (api_key or "").strip():
            raise ValueError("❌ API密钥 为空，请在节点中填写 🔑 API密钥")

        base_url_v1 = self._normalize_base_url(api_base)
        if not base_url_v1:
            raise ValueError("❌ API地址 为空，请填写 🌐 API地址")

        size = self._pick_size(aspect_ratio, hd, model)
        video_url = ""

        images = [kwargs.get(f"图像{i}") for i in range(1, 5)]
        input_ref = self._first_input_reference(images)

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        headers = {"Authorization": f"Bearer {api_key}"}
        form_data = {
            "model": model or "sora-2",
            "prompt": prompt,
            "size": size,
            "seconds": str(max(1, seconds)),
        }

        files = []
        if input_ref is not None:
            files.append(input_ref)

        resp = requests.post(
            f"{base_url_v1}/videos",
            headers=headers,
            data=form_data,
            files=files or None,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise ValueError(f"API Error: {resp.status_code} - {resp.text}")

        result = resp.json()
        video_id = str(result.get("id") or result.get("video_id") or "").strip()
        if not video_id:
            raise ValueError(f"API 响应中缺少 id: {json.dumps(result, ensure_ascii=False)}")

        status = str(result.get("status") or "queued")
        progress = int(result.get("progress", 0) or 0)
        pbar.update_absolute(min(30, 10 + progress))

        meta = result
        max_attempts = 240
        attempts = 0
        while status not in ["completed", "succeeded", "failed", "error", "cancelled"] and attempts < max_attempts:
            time.sleep(10)
            attempts += 1
            poll = requests.get(f"{base_url_v1}/videos/{video_id}", headers=headers, timeout=self.timeout)
            if poll.status_code != 200:
                continue
            meta = poll.json()
            status = str(meta.get("status") or status)
            try:
                progress = int(meta.get("progress", progress) or progress)
            except Exception:
                progress = progress
            pbar.update_absolute(min(95, 30 + int(progress * 0.65)))

        if status not in ["completed", "succeeded"]:
            raise ValueError(json.dumps({"status": "failed", "id": video_id, "raw": meta}, ensure_ascii=False))

        video_url = str(meta.get("url") or meta.get("video_url") or meta.get("download_url") or "").strip()
        if not video_url:
            video_url = f"{base_url_v1}/videos/{video_id}/content"

        pbar.update_absolute(100)

        info = {
            "status": "success",
            "id": video_id,
            "model": model or "sora-2",
            "size": size,
            "seconds": str(max(1, seconds)),
            "video_url": video_url,
            "raw": meta,
        }

        return _NiuNiuVideoAdapter(video_url, api_key=api_key, fallback_size=size), video_url, json.dumps(
            info, ensure_ascii=False
        )


NODE_CLASS_MAPPINGS = {
    "NewApiBanana2Node": NewApiBanana2Node,
    "NewApiJimeng45Node": NewApiJimeng45Node,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NewApiBanana2Node": "NIUNIU API-大香蕉🍌2",
    "NewApiJimeng45Node": "🫎NIUNIU API-即梦4.5",
}
