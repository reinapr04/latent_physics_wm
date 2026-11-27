"""Run any repo command on a Modal GPU, with runs/ on a persistent volume.

    modal run --detach modal_app.py --name smoke --cmd "python train_dynamics.py --run-dir runs/quick ..."

The command runs from the repo root; its output goes to the Modal logs and to
runs/logs/<name>.log on the volume (committed every few minutes and at the end). Fetch results:

    modal volume get latent-physics-runs logs/<name>.log .
    modal volume get latent-physics-runs com015/checkpoint.pt runs/com015/
"""

import subprocess
import threading
import time

import modal

VOLUME = modal.Volume.from_name("latent-physics-runs", create_if_missing=True)
REPO = "/root/repo"

# ManiSkill renders point clouds with Vulkan: point the loader at the NVIDIA driver
NVIDIA_ICD = '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.2.155"}}'

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libvulkan1", "vulkan-tools", "libglvnd0", "libgl1", "libegl1", "libxext6", "libx11-6",
                 "ffmpeg", "fonts-dejavu-core")
    .run_commands("mkdir -p /usr/share/vulkan/icd.d",
                  f"echo '{NVIDIA_ICD}' > /usr/share/vulkan/icd.d/nvidia_icd.json")
    .pip_install("mani_skill==3.0.1", "torch", "tensorboard", "imageio[ffmpeg]", "matplotlib")
    # SAPIEN otherwise downloads its PhysX GPU library at the start of every job
    .run_commands("mkdir -p /root/.sapien/physx/105.1-physx-5.3.1.patch0 && cd /root/.sapien/physx/105.1-physx-5.3.1.patch0"
                  " && python -c \"import urllib.request; urllib.request.urlretrieve('https://github.com/sapien-sim/"
                  "physx-precompiled/releases/download/105.1-physx-5.3.1.patch0/linux-so.zip', 'l.zip')\""
                  " && python -m zipfile -e l.zip . && rm l.zip")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(".", REPO, ignore=[".git", ".git/**", "runs", "runs/**", "demos", "demos/**",
                                      "**/__pycache__", "context", "context/**"])
)
app = modal.App("latent-physics-com", image=image)


@app.function(gpu=["L4", "A10G", "T4"], timeout=24 * 3600, volumes={f"{REPO}/runs": VOLUME})
def run(cmd: str, name: str):
    import os

    os.makedirs(f"{REPO}/runs/logs", exist_ok=True)
    log_path = f"{REPO}/runs/logs/{name}.log"
    done = threading.Event()

    def committer():  # keep partial results visible while a long job runs
        while not done.wait(300):
            VOLUME.commit()

    threading.Thread(target=committer, daemon=True).start()
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write(f"$ {cmd}\n")
        proc = subprocess.Popen(cmd, shell=True, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
        log.write(f"\n[exit {rc} after {(time.time() - t0) / 60:.1f} min]\n")
    done.set()
    VOLUME.commit()
    return rc


@app.local_entrypoint()
def main(cmd: str, name: str = "job"):
    rc = run.remote(cmd, name)
    print(f"[{name}] exit code {rc}")
