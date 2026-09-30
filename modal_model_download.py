"""Install a Civitai checkpoint directly into the FooocusPlus Modal volume.

Example: modal run modal_model_download.py::main --version-id 2739037 --file-id 2625456
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError
from urllib.parse import urlparse

import modal


app = modal.App("fooocusplus-model-download")
storage = modal.Volume.from_name("fooocusplus-studio-data")
image = modal.Image.debian_slim(python_version="3.10")
CHUNK = 8 * 1024 * 1024


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        redirected = super().redirect_request(request, fp, code, message, headers, new_url)
        if redirected and urlparse(request.full_url).netloc != urlparse(new_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def download_verified(url: str, target: Path, expected_hash: str) -> str:
    name = target.name
    partial = target.with_name(name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if target.exists() and sha256(target) == expected_hash:
        print(f"Already verified: {target}", flush=True)
        return str(target)

    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0", **({"Range": f"bytes={downloaded}-"} if downloaded else {})}
    if urlparse(url).hostname in ("civitai.com", "civitai.red") and os.environ.get("CIVITAI_API_KEY"):
        url = url.replace("https://civitai.red/", "https://civitai.com/")
        headers["Authorization"] = "Bearer " + os.environ["CIVITAI_API_KEY"]
    try:
        response = build_opener(SafeRedirect()).open(Request(url, headers=headers), timeout=120)
    except HTTPError as error:
        raise RuntimeError(f"Download denied for {name}: HTTP {error.code}. Civitai may require an API key.") from None
    with response:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError(f"Download returned a login page for {name}; a Civitai API key is required.")
        if downloaded and response.status != 206:
            downloaded = 0
        mode = "ab" if downloaded else "wb"
        next_report = downloaded + 512 * 1024 * 1024
        with partial.open(mode) as output:
            while chunk := response.read(CHUNK):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"Downloaded {downloaded / 1e9:.2f} GB: {name}", flush=True)
                    next_report += 512 * 1024 * 1024

    actual_hash = sha256(partial)
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {name}: {actual_hash} != {expected_hash}")
    os.replace(partial, target)
    storage.commit()
    print(f"Verified and installed: {target} ({downloaded / 1e9:.2f} GB)", flush=True)
    return str(target)


@app.function(image=image, volumes={"/data": storage}, timeout=3600, cpu=2)
def download(version_id: int, file_id: int, zimage_preset: dict) -> str:
    metadata_url = f"https://civitai.red/api/v1/model-versions/{version_id}"
    with urlopen(Request(metadata_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
        metadata = json.load(response)
    if metadata.get("model", {}).get("type") != "Checkpoint":
        raise ValueError("The requested file is not a checkpoint")
    entry = next((item for item in metadata.get("files", []) if item.get("id") == file_id), None)
    if entry is None:
        raise ValueError(f"File {file_id} is not in model version {version_id}")
    name = Path(entry["name"]).name
    if not name.endswith(".safetensors"):
        raise ValueError(f"Unexpected checkpoint filename: {name}")
    is_zimage = metadata.get("baseModel") == "ZImageTurbo"
    relative = f"Z-Image/{name}" if is_zimage else name
    url = f"https://civitai.red/api/download/models/{version_id}?fileId={file_id}"
    result = download_verified(url, Path("/data/models/checkpoints") / relative, entry["hashes"]["SHA256"].lower())
    if is_zimage:
        download_verified(
            "https://huggingface.co/DavidDragonsage/FooocusPlus/resolve/main/support/Qwen_3_4b-Q6_K.gguf",
            Path("/data/models/clip/Qwen_3_4b-Q6_K.gguf"),
            "f53c054dcefa2d5b78a4a2a7f79460c6ea56bb25128ad87421e2c8c40add6abd",
        )
        download_verified(
            "https://huggingface.co/Owen777/UltraFlux-v1/resolve/main/vae/diffusion_pytorch_model.safetensors",
            Path("/data/models/vae/UltraFlux.safetensors"),
            "2bf9ad685686b480b03651a8d8595951e4a5578016b8ead4af5e22d3dc9b3409",
        )
        zimage_preset["default_model"] = relative
        zimage_preset["checkpoint_downloads"] = {relative: url}
        for category in ("Z-Image Turbo", "Favorite"):
            preset = Path("/data/user/user_presets") / category / f"{Path(name).stem}.json"
            preset.parent.mkdir(parents=True, exist_ok=True)
            preset.write_text(json.dumps(zimage_preset, indent=4), encoding="utf-8")
        storage.commit()
        print("Z-Image preset and its text encoder/VAE are installed.", flush=True)
    return result


@app.local_entrypoint()
def main(version_id: int, file_id: int, authenticated: bool = False):
    template = json.loads(Path("masters/master_presets/Z-Image Turbo/ZI-Turbo.json").read_text(encoding="utf-8"))
    job = download.with_options(secrets=[modal.Secret.from_name("civitai-download")]) if authenticated else download
    job.remote(version_id, file_id, template)


@app.function(image=image, volumes={"/data": storage}, timeout=1800, cpu=2)
def prepare_inpaint():
    for name, digest in (
        ("fooocus_inpaint_head.pth", "32f7f838e0c6d8f13437ba8411e77a4688d77a2e34df8857e4ef4d51f6b97692"),
        ("inpaint_v26.fooocus.patch", "f8657a025104e22d70f9c060635d8e8c2196f433871a2f68dc40abd2171f0d59"),
    ):
        download_verified(f"https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/{name}", Path("/data/models/inpaint") / name, digest)
    config_path = Path("/data/user/config.txt")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"default_model": "babesIllustriousBy_v55FP16.safetensors", "default_input_image_checkbox": True, "default_selected_image_input_tab_id": "inpaint_tab", "default_advanced_checkbox": True})
    config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")
    preset = {key: config[key] for key in (
        "default_model", "default_loras", "default_cfg_scale", "default_sampler", "default_scheduler", "default_styles", "default_performance", "default_vae", "default_clip_skip",
    )}
    preset.update({"default_engine": {}, "default_refiner": "None", "default_loras": [[True, "None", 1.0]] * 5, "checkpoint_downloads": {}, "lora_downloads": {}, "default_inpaint_engine_version": "v2.6"})
    for category in ("Favorite", "SDXL_Favorite"):
        path = Path("/data/user/user_presets") / category / "BabesIllustriousV55.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preset, indent=4), encoding="utf-8")
    storage.commit()
    print("Inpaint v2.6 installed; BabesIllustriousV55 preset and input-image defaults configured.", flush=True)


@app.local_entrypoint()
def inpaint():
    prepare_inpaint.remote()


@app.function(image=image, volumes={"/data": storage}, timeout=600, cpu=2)
def configure_native_inpaint():
    import struct

    name = "realvisxlV50_v30InpaintBakedvae.safetensors"
    checkpoint = Path("/data/models/checkpoints") / name
    with checkpoint.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    shape = header["model.diffusion_model.input_blocks.0.0.weight"]["shape"]
    if len(shape) != 4 or shape[1] != 9:
        raise ValueError(f"Expected native nine-channel inpainting UNet, found {shape}")
    preset = {
        "default_engine": {}, "default_model": name, "default_refiner": "None",
        "default_loras": [[True, "None", 1.0]] * 5,
        "default_cfg_scale": 5.0, "default_sampler": "dpmpp_2m_sde_gpu",
        "default_scheduler": "karras", "default_styles": [],
        "default_performance": "Speed", "default_clip_skip": 1,
        "default_vae": "Default (model)", "default_sample_sharpness": 0.0,
        "default_inpaint_engine_version": "None",
        "default_input_image_checkbox": True,
        "default_selected_image_input_tab_id": "inpaint_tab",
        "default_advanced_checkbox": True,
        "checkpoint_downloads": {}, "lora_downloads": {}, "embeddings_downloads": {},
    }
    for category in ("Favorite", "SDXL_Favorite"):
        path = Path("/data/user/user_presets") / category / "RealVisXL_Inpaint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preset, indent=4), encoding="utf-8")
    config_path = Path("/data/user/config.txt")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(preset)
    config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")
    # Keep the rejected download recoverable, outside the selectable model root.
    for rejected_name in ("cyberrealisticXL_v80-inpainting.safetensors",
                          "pornmasterAnimeXLV1_xlV1VAE-inpainting.safetensors"):
        rejected = Path("/data/models/checkpoints") / rejected_name
        if rejected.exists():
            archive = Path("/data/unused-checkpoints")
            archive.mkdir(exist_ok=True)
            rejected.rename(archive / rejected.name)
    storage.commit()
    print(f"Installed native inpaint preset RealVisXL_Inpaint; UNet input shape {shape}.", flush=True)


@app.local_entrypoint()
def native_inpaint():
    configure_native_inpaint.remote()


@app.function(image=image, volumes={"/data": storage}, timeout=3600, cpu=2)
def prepare_zimage_inpaint(template: dict):
    assets = (
        ("https://huggingface.co/Comfy-Org/z_image_turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
         "checkpoints/Z-Image/z_image_turbo_bf16.safetensors",
         "2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6"),
        ("https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/5155fc56d17821007d6f62ac192c09e0f0e72016/Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors",
         "model_patches/Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors",
         "d1251cc7bc3486bc61d25c3be498ef394c31c85ddf4ee9137d2e933411f4a689"),
        ("https://huggingface.co/Comfy-Org/z_image_turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/vae/ae.safetensors",
         "vae/z_image_ae.safetensors",
         "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"),
        ("https://huggingface.co/DavidDragonsage/FooocusPlus/resolve/main/support/Qwen_3_4b-Q6_K.gguf",
         "clip/Qwen_3_4b-Q6_K.gguf",
         "f53c054dcefa2d5b78a4a2a7f79460c6ea56bb25128ad87421e2c8c40add6abd"),
    )
    for url, relative, digest in assets:
        download_verified(url, Path("/data/models") / relative, digest)
    preset = template
    preset["default_engine"]["disinteractive"] = ["enhance_checkbox", "refiner_model"]
    preset["default_engine"]["backend_params"].update({
        "task_method": "ZIT_inpaint", "base_model_dtype": "default",
        "control_strength": 0.9,
    })
    preset.update({
        "default_model": "Z-Image/z_image_turbo_bf16.safetensors",
        "default_vae": "z_image_ae.safetensors", "default_styles": [],
        "default_input_image_checkbox": True, "default_advanced_checkbox": True,
        "default_selected_image_input_tab_id": "inpaint_tab",
        "default_inpaint_engine_version": "None", "default_image_quantity": 1,
        "default_aspect_ratio": "1024*1024", "default_overwrite_step": 8,
        "checkpoint_downloads": {}, "clip_downloads": {}, "vae_downloads": {},
    })
    for category in ("Favorite", "Z-Image Turbo"):
        path = Path("/data/user/user_presets") / category / "ZImage_Fun_Inpaint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preset, indent=4), encoding="utf-8")
    config_path = Path("/data/user/config.txt")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(preset)
    config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")
    storage.commit()
    print("ZImage_Fun_Inpaint installed with BF16 base, Union 2.1-2602-8steps, matching VAE, and Qwen encoder.", flush=True)


@app.local_entrypoint()
def zimage_inpaint():
    template = json.loads(Path("masters/master_presets/Z-Image Turbo/ZI-Turbo.json").read_text(encoding="utf-8"))
    prepare_zimage_inpaint.remote(template)
