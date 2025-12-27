import modal
import os
import json
import time
import base64
from pathlib import Path
from typing import Dict, Any

app = modal.App("jaide-v40-training")

jaide_source_mount = modal.Mount.from_local_dir(
    "src",
    remote_path="/jaide_src"
)

dataset_mount = modal.Mount.from_local_file(
    "arxiv_hungarian_dataset_2.jsonl",
    remote_path="/dataset/arxiv_hungarian_dataset_2.jsonl"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential",
        "wget",
        "curl",
        "git",
        "ca-certificates",
        "xz-utils"
    )
    .run_commands(
        "wget -q https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz -O /tmp/zig.tar.xz",
        "tar -xJf /tmp/zig.tar.xz -C /usr/local",
        "ln -s /usr/local/zig-linux-x86_64-0.11.0/zig /usr/local/bin/zig",
        "rm /tmp/zig.tar.xz",
        "zig version"
    )
)

volume = modal.Volume.from_name("jaide-training-data", create_if_missing=True)

@app.function(
    image=image,
    gpu=modal.gpu.B200(count=8),
    timeout=86400,
    volumes={"/models": volume},
    mounts=[jaide_source_mount, dataset_mount],
    cpu=64,
    memory=524288,
    secrets=[modal.Secret.from_name("jaide-secrets")]
)
def train_jaide_rsf(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dim: int,
    layers: int,
    sample_limit: int,
    noise_level: float,
    gradient_clip: float
) -> Dict[str, Any]:
    import subprocess
    import shutil

    work_dir = Path("/workspace")
    work_dir.mkdir(exist_ok=True)

    src_target = work_dir / "jaide40" / "jaide" / "src"
    src_target.mkdir(parents=True, exist_ok=True)

    src_mount = Path("/jaide_src")
    for item in src_mount.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src_mount)
            target_path = src_target / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_path)

    os.chdir(work_dir)

    build_cmd = [
        "zig", "build-exe",
        str(src_target / "main.zig"),
        "-O", "ReleaseFast",
        "-fstrip",
        "-target", "x86_64-linux-gnu"
    ]

    build_result = subprocess.run(
        build_cmd,
        capture_output=True,
        text=True,
        cwd=work_dir
    )

    if build_result.returncode != 0:
        return {
            "status": "build_failed",
            "build_stdout": build_result.stdout,
            "build_stderr": build_result.stderr,
            "exit_code": build_result.returncode
        }

    binary_path = work_dir / "main"
    if not binary_path.exists():
        return {
            "status": "binary_not_found",
            "error": "Compiled binary not found after build"
        }

    dataset_path = Path("/dataset/arxiv_hungarian_dataset_2.jsonl")
    model_output = Path("/models") / f"rsf_trained_8xb200_{int(time.time())}.bin"

    train_args = [
        str(binary_path),
        "--mode", "train",
        "--dataset", str(dataset_path),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--dim", str(dim),
        "--layers", str(layers),
        "--sample-limit", str(sample_limit),
        "--noise-level", str(noise_level),
        "--model-output", str(model_output)
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env["JAIDE_GPU_COUNT"] = "8"
    env["JAIDE_GPU_TYPE"] = "B200"

    start_time = time.time()

    train_result = subprocess.run(
        train_args,
        capture_output=True,
        text=True,
        env=env,
        cwd=work_dir
    )

    end_time = time.time()
    duration = end_time - start_time

    volume.commit()

    training_log = {
        "status": "completed" if train_result.returncode == 0 else "failed",
        "exit_code": train_result.returncode,
        "duration_seconds": duration,
        "gpu_config": "8x B200",
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dim": dim,
            "layers": layers,
            "sample_limit": sample_limit,
            "noise_level": noise_level,
            "gradient_clip": gradient_clip
        },
        "model_path": str(model_output) if train_result.returncode == 0 else None,
        "stdout": train_result.stdout,
        "stderr": train_result.stderr,
        "timestamp": end_time
    }

    log_path = Path("/models") / f"training_log_{int(end_time)}.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    volume.commit()

    return training_log

@app.function(
    image=image,
    volumes={"/models": volume}
)
def list_models():
    models_dir = Path("/models")
    models = []

    for f in models_dir.glob("rsf_trained_*.bin"):
        stat = f.stat()
        models.append({
            "filename": f.name,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime,
            "path": str(f)
        })

    logs = []
    for f in models_dir.glob("training_log_*.json"):
        try:
            with open(f) as log_file:
                log_data = json.load(log_file)
                logs.append(log_data)
        except:
            pass

    return {
        "models": sorted(models, key=lambda x: x["modified"], reverse=True),
        "training_logs": sorted(logs, key=lambda x: x["timestamp"], reverse=True)
    }

@app.function(
    image=image,
    volumes={"/models": volume}
)
def get_model_bytes(model_filename: str) -> bytes:
    model_path = Path("/models") / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_filename}")
    return model_path.read_bytes()

@app.local_entrypoint()
def main(
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    dim: int = 128,
    layers: int = 4,
    sample_limit: int = 100,
    noise_level: float = 0.05,
    gradient_clip: float = 5.0
):
    print("="*70)
    print("JAIDE v40 - Root-Level AGI Training on 8x B200 GPUs")
    print("="*70)
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Embedding Dimension: {dim}")
    print(f"  RSF Layers: {layers}")
    print(f"  Sample Limit: {sample_limit}")
    print(f"  Noise Level: {noise_level}")
    print(f"  Gradient Clip: {gradient_clip}")
    print(f"  GPU: 8x NVIDIA B200")
    print("="*70)

    result = train_jaide_rsf.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dim=dim,
        layers=layers,
        sample_limit=sample_limit,
        noise_level=noise_level,
        gradient_clip=gradient_clip
    )

    print("\n" + "="*70)
    print("TRAINING RESULTS")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Duration: {result['duration_seconds']:.2f} seconds ({result['duration_seconds']/60:.2f} minutes)")
    print(f"GPU Configuration: {result['gpu_config']}")

    if result['status'] == 'completed':
        print(f"Model saved to: {result['model_path']}")

    print("\n--- Training Output ---")
    print(result['stdout'])

    if result['stderr']:
        print("\n--- Errors/Warnings ---")
        print(result['stderr'])

    print("="*70)

    if result['status'] == 'completed':
        print("\nListing all trained models...")
        models_info = list_models.remote()

        print(f"\nAvailable models: {len(models_info['models'])}")
        for model in models_info['models'][:5]:
            print(f"  - {model['filename']} ({model['size_mb']:.2f} MB)")