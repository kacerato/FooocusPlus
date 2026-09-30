"""Run FooocusPlus on Modal, with models and user files in a persistent Volume.

Deploy from this directory: modal deploy modal_app.py
"""
from __future__ import annotations

import modal

app = modal.App("fooocusplus-studio")
storage = modal.Volume.from_name("fooocusplus-studio-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "cmake", "build-essential", "libgl1", "libglib2.0-0", "libzbar0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1")
    .pip_install_from_requirements("requirements_versions.txt")
    .pip_install_from_requirements("requirements_linux.txt")
    .env({"HF_HOME": "/data/huggingface", "GRADIO_ANALYTICS_ENABLED": "False"})
    .add_local_dir(
        ".", "/app/FooocusPlus",
        ignore=[".git", "UserDir", "outputs", "__pycache__", "*.pyc"],
    )
)


@app.function(
    image=image,
    gpu=["L40S", "A100-40GB", "A100-80GB"],
    cpu=2,
    memory=49152,
    timeout=1200,
    scaledown_window=600,
    max_containers=1,
    volumes={"/data": storage},
    secrets=[modal.Secret.from_name("aero-canvas-web-auth")],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(7860, startup_timeout=900)
def studio():
    import os
    import subprocess
    import sys
    import shutil
    from pathlib import Path

    os.makedirs("/data/models", exist_ok=True)
    os.makedirs("/data/user", exist_ok=True)
    # --models-root bypasses UserDir/models, where upstream seeds these assets.
    # Include the vocabulary/config files shipped with the fork in the real root.
    bundled = Path("/app/FooocusPlus/masters/models")
    for source in bundled.rglob("*"):
        if source.is_file():
            target = Path("/data/models") / source.relative_to(bundled)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    storage.commit()
    command = [
        sys.executable, "-u", "entry_without_update.py",
        "--gpu-type", "none", "--attention-pytorch", "--disable-xformers",
        "--disable-in-browser", "--listen", "0.0.0.0", "--port", "7860",
        "--models-root", "/data/models", "--user-dir", "/data/user",
        "--temp-path", "/tmp/fooocusplus",
        "--preset", "ZImage_Fun_Inpaint",
    ]
    subprocess.Popen(command, cwd="/app/FooocusPlus", env=os.environ.copy())
