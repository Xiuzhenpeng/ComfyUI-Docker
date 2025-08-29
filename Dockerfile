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
    && rm -rf /var/lib/apt/lists/*

# 创建 Python 虚拟环境
RUN python3 -m venv /ComfyUI/venv

# 激活虚拟环境并升级 pip
RUN /ComfyUI/venv/bin/pip install --upgrade pip

# 复制 ComfyUI 项目文件到容器（假设你本地有 ComfyUI 文件夹）
COPY . /ComfyUI

# 安装 Python 依赖
RUN /ComfyUI/venv/bin/pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN /ComfyUI/venv/bin/pip install -r requirements.txt

# 设置默认使用虚拟环境的 Python
ENV PATH="/ComfyUI/venv/bin:$PATH"

# 设置 ComfyUI 环境变量
ENV COMFYUI_HOST=0.0.0.0
ENV COMFYUI_PORT=8188

# 暴露端口
EXPOSE ${COMFYUI_PORT}

# 启动 ComfyUI（使用环境变量传参）
CMD ["sh", "-c", "python main.py --listen $COMFYUI_HOST --port $COMFYUI_PORT"]
