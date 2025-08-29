# 使用官方 Python 3.11.2 slim 镜像
FROM python:3.11.2-slim

# 设置工作目录
WORKDIR /ComfyUI

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# 创建 Python 虚拟环境
RUN python3 -m venv /ComfyUI/venv

# 设置默认使用虚拟环境的 Python
ENV PATH="/ComfyUI/venv/bin:$PATH"

# 复制依赖文件先安装（利用 Docker 缓存层）
COPY requirements.txt /ComfyUI/

# 安装 Python 依赖并清理缓存
RUN pip install --upgrade pip --no-cache-dir \
    && pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
       --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir \
    && pip install -r requirements.txt --no-cache-dir \
    && find /ComfyUI/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /ComfyUI/venv -name "*.pyc" -delete

# 复制 ComfyUI 项目文件到容器（假设你本地有 ComfyUI 文件夹）
COPY . /ComfyUI

# 清理不必要的文件
RUN rm -rf .git* \
    && rm -rf tests \
    && rm -rf docs \
    && rm -rf *.md \
    && rm -rf .pytest_cache \
    && find . -name "*.pyc" -delete \
    && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# 设置 ComfyUI 环境变量
ENV COMFYUI_HOST=0.0.0.0
ENV COMFYUI_PORT=8188

# 暴露端口
EXPOSE ${COMFYUI_PORT}

# 创建非 root 用户（安全性考虑）
# RUN useradd --create-home --shell /bin/bash comfyui \
#     && chown -R comfyui:comfyui /ComfyUI
# USER comfyui

# 启动 ComfyUI（使用环境变量传参）
CMD ["sh", "-c", "python main.py --listen $COMFYUI_HOST --port $COMFYUI_PORT"]
