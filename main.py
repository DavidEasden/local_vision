
import os
import sys
import json


# 如果不是在指定的 Conda 环境中启动，就自动切到该环境的 Python。
REQUIRED_CONDA_ENV = os.environ.get("MCP_CONDA_ENV", "mlx")
REEXEC_FLAG = "LOCAL_VISION_MCP_ENV_READY"


def _find_conda_env_python() -> str | None:
    candidates = []

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = os.path.dirname(os.path.dirname(conda_exe))
        candidates.append(os.path.join(conda_root, "envs", REQUIRED_CONDA_ENV, "bin", "python"))

    current_root = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
    candidates.append(os.path.join(current_root, "envs", REQUIRED_CONDA_ENV, "bin", "python"))
    candidates.append(os.path.expanduser(f"~/miniconda3/envs/{REQUIRED_CONDA_ENV}/bin/python"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def _ensure_conda_env() -> None:
    if __name__ != "__main__":
        return

    env_python = _find_conda_env_python()
    if not env_python:
        raise RuntimeError(f"未找到 Conda 环境 {REQUIRED_CONDA_ENV} 的 Python，请检查环境是否存在")

    current_python = os.path.realpath(sys.executable)
    target_python = os.path.realpath(env_python)

    if current_python == target_python:
        return

    if os.environ.get(REEXEC_FLAG) == "1":
        raise RuntimeError(
            f"当前 Python 为 {current_python}，但期望使用 {target_python}，请检查 PyCharm 的解释器配置"
        )

    os.environ[REEXEC_FLAG] = "1"
    os.execv(target_python, [target_python, os.path.abspath(__file__), *sys.argv[1:]])


_ensure_conda_env()

import httpx
from typing import Any

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# 配置你的 LM Studio 服务地址
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:11434")
# 使用的视觉模型名称
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3.5:2b-bf16")

server = Server("local-vision-mcp")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """列出可用工具"""
    return [
        types.Tool(
            name="analyze_image",
            description="分析本地图片文件，返回图像描述。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "图片文件的本地绝对路径",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "可选，对图像提出的具体问题；未提供时使用默认图片分析提示词",
                    },
                },
                "required": ["image_path"],
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """处理工具调用"""
    if name != "analyze_image":
        raise ValueError(f"Unknown tool: {name}")

    if not arguments:
        raise ValueError("Missing arguments")

    image_path = arguments.get("image_path")
    if not image_path:
        raise ValueError("Missing image_path argument")

    # 安全检查：只允许访问本地存在的文件，且必须是图片格式
    if not os.path.exists(image_path):
        return [types.TextContent(type="text", text=f"错误：文件不存在 - {image_path}")]

    allowed_ext = (".png", ".jpg", ".jpeg", ".webp")
    if not image_path.lower().endswith(allowed_ext):
        return [types.TextContent(type="text", text=f"错误：不支持的文件格式，请使用 {', '.join(allowed_ext)}")]

    prompt = arguments.get(
        "prompt",
        "请严格根据图片中可见的信息回答，不要猜测。先概述图片，再提取可见文字，再说明关键元素和位置关系，最后直接回答我的问题。看不清的地方请明确写“不确定”。我的问题：请描述这张图片",
    )

    # 调用 LM Studio
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 读取图片并转为 base64
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }.get(ext, "application/octet-stream")

            # 构造 LM Studio OpenAI Responses 请求
            payload = {
                "model": VISION_MODEL,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{image_data}",
                            },
                        ],
                    }
                ],
                "stream": False,
            }

            response = await client.post(
                f"{LM_STUDIO_URL}/v1/responses",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            content_parts = []
            for item in result.get("output", []):
                if item.get("type") != "message":
                    continue
                for part in item.get("content", []):
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        content_parts.append(part["text"])

            if not content_parts:
                return [types.TextContent(type="text", text="调用视觉模型失败：响应中没有可读取的文本内容")]

            content = "\n".join(content_parts)
            if isinstance(content, str):
                content = content.replace("<|im_end|>", "").strip()
            return [types.TextContent(type="text", text=content)]

    except Exception as e:
        return [types.TextContent(type="text", text=f"调用视觉模型失败：{str(e)}")]

async def main():
    """启动 MCP 服务器"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="local-vision-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
