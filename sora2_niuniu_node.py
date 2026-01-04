import base64
import json
import random
import tempfile
import time
from io import BytesIO
from typing import Optional, Tuple

import requests
from PIL import Image
import torch
import comfy.utils
try:
    from comfy.comfy_types import IO
    VIDEO_TYPE = getattr(IO, "VIDEO", "VIDEO")
except Exception:
    VIDEO_TYPE = "VIDEO"


def tensor2pil(image_tensor: torch.Tensor) -> Optional[Image.Image]:
    if image_tensor is None:
        return None
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    image_np = (image_tensor.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(image_np)


class ComfyVideoAdapter:
    def __init__(
        self,
        video_path_or_url: str,
        width: int = 1280,
        height: int = 720,
        headers: Optional[dict] = None,
    ):
        video_path_or_url = str(video_path_or_url or "").strip()
        self.is_url = video_path_or_url.startswith("http")
        self.video_url = video_path_or_url if self.is_url else None
        self.video_path = video_path_or_url if (video_path_or_url and not self.is_url) else None
        self.width = int(width or 1280)
        self.height = int(height or 720)
        self.headers = headers or None

    def get_dimensions(self):
        return self.width, self.height

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        if self.is_url:
            if not self.video_url:
                return False
            response = requests.get(
                self.video_url,
                stream=True,
                timeout=900,
                headers=self.headers or None,
            )
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True

        if not self.video_path:
            return False
        with open(self.video_path, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())
        return True


class NiuNiuSora2VideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📝 提示词": ("STRING", {"multiline": True, "default": "女人在天上飞"}),
                "🤖 模型": ("STRING", {"default": "sora-2-private", "multiline": False}),
                "🌐 API地址": (
                    "STRING",
                    {
                        "default": "https://api.llyapps.com",
                        "multiline": False,
                        "tooltip": "如 https://api.newapi.pro（也可粘贴到 /v1 或 /v1/videos；会自动纠正）",
                    },
                ),
                "🔑 API密钥": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "OpenAI / New API Key"},
                ),
                "📐 宽高比": (
                    ["16:9", "9:16"],
                    {"default": "9:16"},
                ),
                "⏱️ 视频时长": (
                    ["10", "15", "25"],
                    {"default": "15"},
                ),
                "🎬 高清模式": ("BOOLEAN", {"default": False}),
                "🎰 随机种子": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "control_after_generate": "randomize",
                    },
                ),
                "🎯 种子控制": (["随机", "固定", "递增"], {"default": "随机"}),
                "🔐 隐私模式": ("BOOLEAN", {"default": False}),
                "⏳ 超时等待(秒)": (
                    "INT",
                    {"default": 800, "min": 1, "max": 86400, "step": 1},
                ),
            },
            "optional": {
                "图像1": ("IMAGE",),
                "图像2": ("IMAGE",),
                "图像3": ("IMAGE",),
                "图像4": ("IMAGE",),
            },
        }

    RETURN_TYPES = (VIDEO_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("视频", "视频URL", "响应信息")
    FUNCTION = "generate_video"
    CATEGORY = "NIUNIUAPI"

    def __init__(self):
        self.timeout = 900
        self.last_seed = -1

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed_control = kwargs.get("🎯 种子控制", "随机")
        seed = kwargs.get("🎰 随机种子", -1)

        if seed_control in ["随机", "递增"]:
            return float("nan")

        return seed

    def _normalize_base_url(self, base_url: str) -> str:
        url = str(base_url or "").strip()
        if not url:
            return ""
        url = url.strip("`").strip().strip('"').strip("'")
        url = url.split("?", 1)[0].rstrip("/")
        lower = url.lower()
        v1_pos = lower.find("/v1")
        if v1_pos >= 0:
            return url[: v1_pos + 3]

        if lower.endswith("/videos"):
            url = url[: -len("/videos")].rstrip("/")

        return f"{url}/v1"

    def _normalize_api_key(self, api_key: str) -> str:
        k = str(api_key or "").strip()
        if not k:
            return ""
        k = k.strip("`").strip().strip('"').strip("'")
        if ":" in k and k.lower().startswith("authorization"):
            k = k.split(":", 1)[1].strip()
        if k.lower().startswith("bearer "):
            k = k[7:].strip()
        return k

    def _build_size(self, aspect_ratio: str) -> str:
        if aspect_ratio == "9:16":
            return "720x1280"
        return "1280x720"

    def _parse_size(self, size: str) -> Tuple[int, int]:
        s = str(size or "").lower().strip()
        if "x" not in s:
            return 1280, 720
        a, b = s.split("x", 1)
        try:
            w = int(a.strip())
            h = int(b.strip())
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        return 1280, 720

    def _image_to_file(self, img_tensor: torch.Tensor, filename: str):
        pil_img = tensor2pil(img_tensor)
        if pil_img is None:
            return None
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return ("input_reference", (filename, buf, "image/png"))

    def _image_to_base64_data_url(self, img_tensor: torch.Tensor) -> Optional[str]:
        pil_img = tensor2pil(img_tensor)
        if pil_img is None:
            return None
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _compute_seed(self, seed: int, seed_control: str) -> Optional[int]:
        seed_control = str(seed_control or "随机").strip()
        try:
            seed = int(seed)
        except Exception:
            seed = -1

        if seed_control == "随机":
            return None

        if seed_control == "固定":
            if seed == -1:
                return random.randint(0, 2147483647)
            return max(0, seed)

        if seed_control == "递增":
            if self.last_seed == -1:
                if seed == -1:
                    self.last_seed = random.randint(0, 2147483647)
                else:
                    self.last_seed = max(0, seed)
            else:
                self.last_seed = self.last_seed + 1
            return self.last_seed

        return None

    def _safe_json(self, resp: requests.Response, context: str) -> dict:
        text = resp.text or ""
        if not text.strip():
            raise ValueError(f"{context}：API响应为空（HTTP {resp.status_code}）")
        try:
            data = resp.json()
            if isinstance(data, dict):
                return data
            return {"data": data}
        except Exception:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    return data
                return {"data": data}
            except Exception:
                snippet = text.strip().replace("\r", " ").replace("\n", " ")
                if len(snippet) > 800:
                    snippet = snippet[:800] + "..."
                raise ValueError(f"{context}：API响应不是JSON（HTTP {resp.status_code}）{snippet}")

    def _looks_like_base64(self, s: str) -> bool:
        t = (s or "").strip()
        if len(t) < 256:
            return False
        t = t.replace("\r", "").replace("\n", "")
        for ch in t[:512]:
            if ch.isalnum() or ch in "+/=":
                continue
            return False
        return True

    def _save_base64_mp4(self, b64_or_data_url: str) -> str:
        s = (b64_or_data_url or "").strip()
        if s.startswith("data:"):
            pos = s.find("base64,")
            if pos >= 0:
                s = s[pos + 7 :]
        s = s.replace("\r", "").replace("\n", "").strip()
        if not s:
            raise ValueError("视频base64为空")
        pad = (-len(s)) % 4
        if pad:
            s = s + ("=" * pad)
        raw = base64.b64decode(s)
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.write(raw)
        f.flush()
        f.close()
        return f.name

    def _pick_first_str(self, *vals) -> str:
        for v in vals:
            if v is None:
                continue
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _extract_video_value(self, payload) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, dict):
            data = payload.get("data") if isinstance(payload.get("data"), dict) else None
            return self._pick_first_str(
                payload.get("output"),
                payload.get("url"),
                payload.get("video_url"),
                payload.get("download_url"),
                payload.get("b64_json"),
                payload.get("base64"),
                payload.get("video_base64"),
                (data or {}).get("output"),
                (data or {}).get("url"),
                (data or {}).get("video_url"),
                (data or {}).get("download_url"),
                (data or {}).get("b64_json"),
                (data or {}).get("base64"),
                (data or {}).get("video_base64"),
            )
        return ""

    def _resolve_video_to_path_or_url(self, value: str) -> Tuple[str, Optional[dict]]:
        s = (value or "").strip()
        if not s:
            return "", None
        if s.startswith("http"):
            return s, None
        if s.startswith("data:video") or s.startswith("data:application") or self._looks_like_base64(s):
            return self._save_base64_mp4(s), None
        return s, None

    def generate_video(self, **kwargs):
        prompt = kwargs.get("📝 提示词", "") or ""
        model = str(kwargs.get("🤖 模型", "sora-2") or "").strip()
        api_base = kwargs.get("🌐 API地址", "")
        api_key = self._normalize_api_key(kwargs.get("🔑 API密钥", ""))
        aspect_ratio = kwargs.get("📐 宽高比", "16:9")
        seconds = str(kwargs.get("⏱️ 视频时长", "4") or "4")
        hd = bool(kwargs.get("🎬 高清模式", False))
        seed = kwargs.get("🎰 随机种子", -1)
        seed_control = kwargs.get("🎯 种子控制", "随机")
        private = bool(kwargs.get("🔐 隐私模式", False))
        max_wait_seconds = int(kwargs.get("⏳ 超时等待(秒)", 600) or 600)

        if not (api_key or "").strip():
            raise ValueError("API密钥不能为空")

        if not str(prompt).strip():
            raise ValueError("提示词不能为空")

        base_url_v1 = self._normalize_base_url(api_base)
        if not base_url_v1:
            raise ValueError("API地址不能为空")

        size = self._build_size(aspect_ratio)

        images = [kwargs.get(f"图像{i}") for i in range(1, 5)]
        has_image = any(img is not None for img in images)

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        base_headers = {"Authorization": f"Bearer {api_key}"}
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "seconds": seconds,
        }

        effective_seed = self._compute_seed(seed, seed_control)
        if effective_seed is not None:
            data["seed"] = int(effective_seed)

        if hd:
            data["quality"] = "high"

        if private:
            data["private"] = True

        image_files = []
        if has_image:
            for idx, img in enumerate(images, 1):
                if img is None:
                    continue
                f = self._image_to_file(img, f"input_{idx}.png")
                if f is not None:
                    image_files.append(f)

        if has_image:
            try:
                root_url = base_url_v1[:-3] if base_url_v1.lower().endswith("/v1") else base_url_v1
                aspect_ratio_for_v2 = "9:16" if str(aspect_ratio) == "9:16" else "16:9"
                v2_payload = {
                    "prompt": prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio_for_v2,
                    "duration": seconds,
                    "hd": hd,
                    "private": private,
                }
                v2_images = []
                for img in images:
                    if img is None:
                        continue
                    s = self._image_to_base64_data_url(img)
                    if s:
                        v2_images.append(s)
                if not v2_images:
                    raise ValueError("输入图像处理失败")
                v2_payload["images"] = v2_images
                if effective_seed is not None:
                    v2_payload["seed"] = int(effective_seed)

                pbar.update_absolute(20)
                create_resp = requests.post(
                    f"{root_url}/v2/videos/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=v2_payload,
                    timeout=self.timeout,
                )
                if create_resp.status_code not in (200, 201, 202):
                    if create_resp.status_code == 401:
                        raise ValueError(
                            f"API鉴权失败(401)：API密钥无效/已过期/无权限，或API地址填错。{create_resp.text}"
                        )
                    raise ValueError(f"API Error: {create_resp.status_code} - {create_resp.text}")

                create_data = self._safe_json(create_resp, "创建视频任务失败")
                task_id = str(
                    create_data.get("task_id")
                    or create_data.get("id")
                    or (create_data.get("data") or {}).get("task_id")
                    or (create_data.get("data") or {}).get("id")
                    or ""
                ).strip()
                if not task_id:
                    raise ValueError(f"响应中缺少task_id：{create_resp.text}")

                status = "queued"
                meta = {}
                start_ts = time.monotonic()
                while True:
                    if time.monotonic() - start_ts >= max_wait_seconds:
                        break
                    time.sleep(10)
                    poll = requests.get(
                        f"{root_url}/v2/videos/generations/{task_id}",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        timeout=self.timeout,
                    )
                    if poll.status_code not in (200, 201, 202):
                        continue
                    meta = self._safe_json(poll, "轮询视频任务失败")
                    status = str(meta.get("status") or status)
                    progress_text = str(meta.get("progress") or "").strip()
                    if progress_text.endswith("%"):
                        try:
                            p = int(progress_text[:-1])
                            pbar.update_absolute(min(95, 30 + int(p * 0.65)))
                        except Exception:
                            pass
                    status_upper = status.upper()
                    if status_upper in ("SUCCESS", "SUCCEEDED", "COMPLETED", "DONE"):
                        break
                    if status_upper in ("FAILURE", "FAILED", "ERROR"):
                        break

                if str(status).upper() not in ("SUCCESS", "SUCCEEDED", "COMPLETED", "DONE"):
                    elapsed = int(time.monotonic() - start_ts)
                    raise ValueError(f"视频任务失败或超时，状态：{status}，已等待：{elapsed}s，task_id：{task_id}")

                raw_value = self._extract_video_value(meta)
                path_or_url, download_headers = self._resolve_video_to_path_or_url(raw_value)
                if not path_or_url:
                    raise ValueError(f"视频URL为空：{json.dumps(meta, ensure_ascii=False)}")

                width, height = self._parse_size(size)
                info = {
                    "status": status,
                    "task_id": task_id,
                    "aspect_ratio": aspect_ratio_for_v2,
                    "duration": seconds,
                    "model": model,
                    "hd": hd,
                    "seed": effective_seed,
                    "private": private,
                    "video_url": path_or_url if str(path_or_url).startswith("http") else "",
                    "raw": meta or create_data,
                }
                pbar.update_absolute(100)
                return (
                    ComfyVideoAdapter(path_or_url, width=width, height=height, headers=download_headers),
                    path_or_url,
                    json.dumps(info, ensure_ascii=False),
                )
            except Exception as e:
                msg = str(e)
                if "API响应不是JSON" not in msg and "API响应为空" not in msg:
                    raise

        root_for_v1 = base_url_v1[:-3] if base_url_v1.lower().endswith("/v1") else base_url_v1
        base_candidates_v1 = [
            base_url_v1,
            f"{root_for_v1}/api/v1",
            f"{root_for_v1}/openai/v1",
            f"{root_for_v1}/api/openai/v1",
        ]

        multipart_files = [(k, (None, str(v))) for k, v in data.items()]
        multipart_files.extend(image_files)

        resp = None
        result = None
        used_base_v1 = ""
        last_err = None
        for cand_base in base_candidates_v1:
            url = f"{cand_base}/videos"
            resp = requests.post(url, headers=base_headers, files=multipart_files, timeout=self.timeout)
            if resp.status_code != 200:
                if resp.status_code == 401:
                    raise ValueError(
                        f"API鉴权失败(401)：API密钥无效/已过期/无权限，或API地址填错。{resp.text}"
                    )
                last_err = ValueError(f"API Error: {resp.status_code} - {resp.text}")
                continue
            try:
                result = self._safe_json(resp, "创建视频任务失败(v1)")
                used_base_v1 = cand_base
                break
            except Exception as e:
                text_low = (resp.text or "").lstrip().lower()
                if "<!doctype html" in text_low or "<html" in text_low:
                    last_err = e
                    continue
                raise
        if not used_base_v1 or result is None:
            raise last_err or ValueError("创建视频任务失败(v1)：未找到可用的API端点")

        job = result.get("data") if isinstance(result.get("data"), dict) else result
        video_id = job.get("id") or job.get("video_id") or ""
        status = str(job.get("status", "queued"))
        pbar.update_absolute(30)

        if not video_id:
            raise ValueError("响应中缺少视频ID")

        video_url = ""
        meta = {}
        start_ts = time.monotonic()
        attempts = 0

        while status not in ["completed", "succeeded", "failed", "error"]:
            if time.monotonic() - start_ts >= max_wait_seconds:
                break
            time.sleep(10)
            attempts += 1
            poll = requests.get(
                f"{used_base_v1}/videos/{video_id}",
                headers=base_headers,
                timeout=self.timeout,
            )
            if poll.status_code != 200:
                continue
            meta = self._safe_json(poll, "轮询视频任务失败(v1)")
            meta_job = meta.get("data") if isinstance(meta.get("data"), dict) else meta
            status = str(meta_job.get("status", status))
            status_lower = status.lower()
            progress_raw = meta_job.get("progress", 0)
            progress = 0
            try:
                if isinstance(progress_raw, str) and progress_raw.strip().endswith("%"):
                    progress = int(progress_raw.strip()[:-1])
                else:
                    progress = int(progress_raw or 0)
            except Exception:
                progress = 0
            pbar.update_absolute(min(95, 30 + max(0, min(100, progress))))
            if status_lower in ["completed", "succeeded", "success", "done"]:
                meta = meta_job
                break
            if status_lower in ["failed", "error", "failure"]:
                meta = meta_job
                break

        # 尝试提取视频地址
        raw_value = self._extract_video_value(meta)
        download_headers = None

        # 特殊处理：检查fail_reason是否包含URL
        if not raw_value:
            fail_reason = str(meta.get("fail_reason") or "").strip()
            if len(fail_reason) > 10 and "http" in fail_reason:
                raw_value = fail_reason

        # 清理URL
        raw_value = str(raw_value or "").strip().strip("`").strip("'").strip('"').strip()

        if not raw_value:
            # 如果真的没找到URL，且状态是失败，才抛出异常
            if str(status).lower() not in ["completed", "succeeded", "success", "done"]:
                elapsed = int(time.monotonic() - start_ts)
                raise ValueError(f"视频任务失败或超时，状态：{status}，已等待：{elapsed}s，id：{video_id}")
            
            # 状态成功但无URL，尝试默认路径
            raw_value = f"{used_base_v1}/videos/{video_id}/content"
            download_headers = base_headers

        path_or_url, download_headers2 = self._resolve_video_to_path_or_url(str(raw_value))
        if download_headers2 is not None:
            download_headers = download_headers2
        width, height = self._parse_size(size)

        info = {
            "status": status,
            "id": video_id,
            "size": size,
            "seconds": seconds,
            "model": model,
            "hd": hd,
            "seed": effective_seed,
            "seed_control": seed_control,
            "private": private,
            "video_url": path_or_url if str(path_or_url).startswith("http") else "",
            "raw": meta or result,
        }

        pbar.update_absolute(100)
        return (
            ComfyVideoAdapter(path_or_url, width=width, height=height, headers=download_headers),
            path_or_url,
            json.dumps(info, ensure_ascii=False),
        )


class NiuNiuVeo31VideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "📝 提示词": ("STRING", {"multiline": True, "default": "女人在天上飞"}),
                "🤖 模型": ("STRING", {"default": "veo_3_1-fast", "multiline": False}),
                "🌐 API地址": (
                    "STRING",
                    {
                        "default": "https://api.llyapps.com",
                        "multiline": False,
                        "tooltip": "如 https://api.llyapps.com（也可粘贴到 /v1 或 /v1/videos；会自动纠正）",
                    },
                ),
                "🔑 API密钥": ("STRING", {"default": "", "multiline": False}),
                "📐 宽高比": (["16:9", "9:16", "Auto"], {"default": "Auto"}),
                "⏱️ 视频时长(秒)": ("INT", {"default": 8, "min": 1, "max": 60, "step": 1}),
                "🎬 高清模式": ("BOOLEAN", {"default": False}),
                "🎰 随机种子": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "control_after_generate": "randomize",
                    },
                ),
                "⏳ 超时等待(秒)": (
                    "INT",
                    {"default": 600, "min": 1, "max": 86400, "step": 1},
                ),
            },
            "optional": {
                "参考图": ("IMAGE",),
                "首帧图": ("IMAGE",),
                "尾帧图": ("IMAGE",),
            },
        }

    RETURN_TYPES = (VIDEO_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("视频", "视频URL", "响应信息")
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
        lower = url.lower()
        v1_pos = lower.find("/v1")
        if v1_pos >= 0:
            return url[: v1_pos + 3]
        if lower.endswith("/videos"):
            url = url[: -len("/videos")].rstrip("/")
        return f"{url}/v1"

    def _normalize_api_key(self, api_key: str) -> str:
        k = str(api_key or "").strip()
        if not k:
            return ""
        k = k.strip("`").strip().strip('"').strip("'")
        if ":" in k and k.lower().startswith("authorization"):
            k = k.split(":", 1)[1].strip()
        if k.lower().startswith("bearer "):
            k = k[7:].strip()
        return k

    def _build_size(self, aspect_ratio: str) -> str:
        if aspect_ratio == "9:16":
            return "720x1280"
        return "1280x720"

    def _parse_size(self, size: str) -> Tuple[int, int]:
        s = str(size or "").lower().strip()
        if "x" not in s:
            return 1280, 720
        a, b = s.split("x", 1)
        try:
            w = int(a.strip())
            h = int(b.strip())
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        return 1280, 720

    def _image_to_file(self, field: str, img_tensor: torch.Tensor, filename: str):
        pil_img = tensor2pil(img_tensor)
        if pil_img is None:
            return None
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return (field, (filename, buf, "image/png"))

    def _auto_aspect_ratio(self, image_tensor: Optional[torch.Tensor]) -> str:
        if image_tensor is None:
            return "16:9"
        pil_img = tensor2pil(image_tensor)
        if pil_img is None:
            return "16:9"
        w, h = pil_img.size
        if h >= w:
            return "9:16"
        return "16:9"

    def generate_video(self, **kwargs):
        prompt = kwargs.get("📝 提示词", "") or ""
        model = str(kwargs.get("🤖 模型", "veo_3_1-fast") or "").strip()
        api_base = kwargs.get("🌐 API地址", "")
        api_key = self._normalize_api_key(kwargs.get("🔑 API密钥", ""))
        aspect_ratio = str(kwargs.get("📐 宽高比", "Auto") or "Auto").strip()
        seconds = int(kwargs.get("⏱️ 视频时长(秒)", 8) or 8)
        hd = bool(kwargs.get("🎬 高清模式", False))
        seed = int(kwargs.get("🎰 随机种子", 0) or 0)
        max_wait_seconds = int(kwargs.get("⏳ 超时等待(秒)", 600) or 600)

        first_frame = kwargs.get("首帧图")
        last_frame = kwargs.get("尾帧图")
        ref_image = kwargs.get("参考图")

        if not (api_key or "").strip():
            raise ValueError("API密钥不能为空")

        if not str(prompt).strip():
            raise ValueError("提示词不能为空")

        base_url_v1 = self._normalize_base_url(api_base)
        if not base_url_v1:
            raise ValueError("API地址不能为空")

        if aspect_ratio == "Auto":
            candidate_image = first_frame
            if candidate_image is None:
                candidate_image = ref_image
            if candidate_image is None:
                candidate_image = last_frame
            aspect_ratio = self._auto_aspect_ratio(candidate_image)
        size = self._build_size(aspect_ratio)

        if first_frame is not None and last_frame is not None:
            generation_type = "FIRST_AND_LAST_FRAMES_2_VIDEO"
            input_images = [("input_reference", first_frame, "first_frame.png"), ("input_reference", last_frame, "last_frame.png")]
        elif ref_image is not None:
            generation_type = "FIRST_AND_LAST_FRAMES_2_VIDEO"
            input_images = [("input_reference", ref_image, "reference.png")]
        elif first_frame is not None:
            generation_type = "FIRST_AND_LAST_FRAMES_2_VIDEO"
            input_images = [("input_reference", first_frame, "first_frame.png")]
        elif last_frame is not None:
            generation_type = "FIRST_AND_LAST_FRAMES_2_VIDEO"
            input_images = [("input_reference", last_frame, "last_frame.png")]
        else:
            generation_type = "TEXT_2_VIDEO"
            input_images = []

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        base_headers = {"Authorization": f"Bearer {api_key}"}
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "seconds": str(max(1, seconds)),
            "generationType": generation_type,
        }

        if hd:
            data["quality"] = "high"

        if seed > 0:
            data["seed"] = int(seed)

        image_files = []
        for field, img, name in input_images:
            f = self._image_to_file(field, img, name)
            if f is not None:
                image_files.append(f)

        url = f"{base_url_v1}/videos"
        multipart_files = [(k, (None, str(v))) for k, v in data.items()]
        multipart_files.extend(image_files)
        resp = requests.post(url, headers=base_headers, files=multipart_files, timeout=self.timeout)

        if resp.status_code != 200:
            if resp.status_code == 401:
                raise ValueError(
                    f"API鉴权失败(401)：API密钥无效/已过期/无权限，或API地址填错。{resp.text}"
                )
            raise ValueError(f"API Error: {resp.status_code} - {resp.text}")

        result = resp.json()
        job = result.get("data") if isinstance(result.get("data"), dict) else result
        video_id = job.get("id") or job.get("video_id") or ""
        status = str(job.get("status", "queued"))
        pbar.update_absolute(30)

        if not video_id:
            raise ValueError("响应中缺少视频ID")

        meta = {}
        start_ts = time.monotonic()

        while status not in ["completed", "succeeded", "failed", "error"]:
            if time.monotonic() - start_ts >= max_wait_seconds:
                break
            time.sleep(5)
            poll = None
            for _ in range(3):
                try:
                    poll = requests.get(
                        f"{base_url_v1}/videos/{video_id}",
                        headers=base_headers,
                        timeout=self.timeout,
                    )
                    break
                except requests.exceptions.RequestException:
                    time.sleep(2)
            if poll.status_code != 200:
                continue
            meta = poll.json()
            meta_job = meta.get("data") if isinstance(meta.get("data"), dict) else meta
            status = str(meta_job.get("status", status))
            progress = int(meta_job.get("progress", 0) or 0)
            pbar.update_absolute(min(95, 30 + progress))
            if status in ["completed", "succeeded"]:
                meta = meta_job
                break

        if status not in ["completed", "succeeded"]:
            elapsed = int(time.monotonic() - start_ts)
            raise ValueError(f"视频任务失败或超时，状态：{status}，已等待：{elapsed}s")

        video_url = meta.get("url") or meta.get("video_url") or meta.get("download_url") or ""
        download_headers = None
        if not str(video_url).strip():
            video_url = f"{base_url_v1}/videos/{video_id}/content"
            download_headers = base_headers
        width, height = self._parse_size(size)

        info = {
            "status": status,
            "id": video_id,
            "size": size,
            "seconds": str(max(1, seconds)),
            "model": model,
            "hd": hd,
            "seed": seed if seed > 0 else None,
            "generationType": generation_type,
            "video_url": video_url,
            "raw": meta or result,
        }

        pbar.update_absolute(100)
        return (
            ComfyVideoAdapter(video_url, width=width, height=height, headers=download_headers),
            video_url,
            json.dumps(info, ensure_ascii=False),
        )


NODE_CLASS_MAPPINGS = {
    "NiuNiuSora2VideoNode": NiuNiuSora2VideoNode,
    "NiuNiuVeo31VideoNode": NiuNiuVeo31VideoNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NiuNiuSora2VideoNode": "🎨SORA2视频生成 NIUNIU",
    "NiuNiuVeo31VideoNode": "🫎NIUNIU API-Veo3.1视频生成",
}
