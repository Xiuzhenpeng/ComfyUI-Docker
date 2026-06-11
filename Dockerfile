FROM ghcr.io/astral-sh/uv:python3.11-trixie

# 设置工作目录
WORKDIR /ComfyUI

# 复制依赖文件先安装（利用 Docker 缓存层）
COPY requirements.txt /ComfyUI/
COPY requirements-swarmui.txt /ComfyUI/

# 复制 ComfyUI 项目文件到容器
COPY . /ComfyUI

RUN uv venv

# 安装 Python 依赖并清理缓存
RUN uv pip install --upgrade uv pip --no-cache-dir \
    && uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
       --index-url https://download.pytorch.org/whl/cu130 --no-cache-dir \
    && uv pip install -r requirements.txt --no-cache-dir \
    && uv pip install -r requirements-swarmui.txt --no-cache-dir \
    && for req in $(find custom_nodes -maxdepth 2 -mindepth 2 -type f -name "requirements.txt"); do \
           echo "Installing requirements from $req"; \
           uv pip install -r "$req" --no-cache-dir || true; \
       done

# 清理不必要的文件
RUN rm -rf .git* \
    && rm -rf tests \
    && rm -rf docs \
    && rm -rf *.md \
    && rm -rf .pytest_cache \
    && find . -name "*.pyc" -delete \
    && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# 创建软链接
RUN ln -s /SwarmUI/Models/LLM /ComfyUI/models/LLM \
    && ln -s /SwarmUI/Models/clip/siglip-so400m-patch14-384 /ComfyUI/models/clip/siglip-so400m-patch14-384 \
    && ln -s /SwarmUI/Models/rembg /ComfyUI/models/rembg \
    && ln -s /SwarmUI/Models/SEEDVR2 /ComfyUI/models/SEEDVR2

# 设置 ComfyUI 环境变量
ENV COMFYUI_HOST=0.0.0.0
ENV COMFYUI_PORT=8188

# 暴露端口
EXPOSE ${COMFYUI_PORT}

# 启动 ComfyUI（使用环境变量传参）
CMD ["sh", "-c", "python main.py --listen $COMFYUI_HOST --port $COMFYUI_PORT"]
