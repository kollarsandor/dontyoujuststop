import modal
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

app = modal.App("jaide-v40-distributed")

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
        "xz-utils",
        "libopenmpi-dev",
        "openmpi-bin"
    )
    .pip_install("mpi4py==3.1.5")
    .run_commands(
        "wget -q https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz -O /tmp/zig.tar.xz",
        "tar -xJf /tmp/zig.tar.xz -C /usr/local",
        "ln -s /usr/local/zig-linux-x86_64-0.11.0/zig /usr/local/bin/zig",
        "rm /tmp/zig.tar.xz"
    )
)

volume = modal.Volume.from_name("jaide-training-data", create_if_missing=True)

@app.function(
    image=image,
    gpu=modal.gpu.B200(count=8),
    timeout=172800,
    volumes={"/models": volume, "/checkpoints": volume},
    mounts=[jaide_source_mount, dataset_mount],
    cpu=64,
    memory=524288,
    secrets=[modal.Secret.from_name("jaide-secrets")]
)
def train_epoch_distributed(
    epoch_num: int,
    total_epochs: int,
    batch_size: int,
    learning_rate: float,
    dim: int,
    layers: int,
    sample_limit: int,
    noise_level: float,
    checkpoint_input: str = None
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

    build_result = subprocess.run(
        [
            "zig", "build-exe",
            str(src_target / "main.zig"),
            "-O", "ReleaseFast",
            "-fstrip"
        ],
        capture_output=True,
        text=True
    )

    if build_result.returncode != 0:
        return {
            "epoch": epoch_num,
            "status": "build_failed",
            "error": build_result.stderr
        }

    binary_path = work_dir / "main"
    dataset_path = Path("/dataset/arxiv_hungarian_dataset_2.jsonl")
    checkpoint_output = Path("/checkpoints") / f"epoch_{epoch_num}.bin"

    train_args = [
        str(binary_path),
        "--mode", "train",
        "--dataset", str(dataset_path),
        "--epochs", "1",
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--dim", str(dim),
        "--layers", str(layers),
        "--sample-limit", str(sample_limit),
        "--noise-level", str(noise_level),
        "--model-output", str(checkpoint_output)
    ]

    if checkpoint_input and Path(checkpoint_input).exists():
        train_args.extend(["--model-input", checkpoint_input])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env["JAIDE_GPU_COUNT"] = "8"
    env["JAIDE_EPOCH"] = str(epoch_num)
    env["JAIDE_TOTAL_EPOCHS"] = str(total_epochs)

    start_time = time.time()

    train_result = subprocess.run(
        train_args,
        capture_output=True,
        text=True,
        env=env
    )

    end_time = time.time()

    volume.commit()

    return {
        "epoch": epoch_num,
        "total_epochs": total_epochs,
        "status": "completed" if train_result.returncode == 0 else "failed",
        "duration_seconds": end_time - start_time,
        "checkpoint_path": str(checkpoint_output) if train_result.returncode == 0 else None,
        "stdout": train_result.stdout,
        "stderr": train_result.stderr,
        "exit_code": train_result.returncode
    }

@app.function(
    image=image,
    volumes={"/models": volume, "/checkpoints": volume},
    timeout=259200
)
def orchestrate_distributed_training(
    total_epochs: int,
    batch_size: int,
    learning_rate: float,
    dim: int,
    layers: int,
    sample_limit: int,
    noise_level: float
) -> Dict[str, Any]:
    epoch_results = []
    last_checkpoint = None

    training_start = time.time()

    for epoch in range(1, total_epochs + 1):
        print(f"Starting epoch {epoch}/{total_epochs}")

        result = train_epoch_distributed.remote(
            epoch_num=epoch,
            total_epochs=total_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dim=dim,
            layers=layers,
            sample_limit=sample_limit,
            noise_level=noise_level,
            checkpoint_input=last_checkpoint
        )

        epoch_results.append(result)

        print(f"Epoch {epoch} {result['status']}: {result['duration_seconds']:.2f}s")

        if result['status'] != 'completed':
            print(f"Training failed at epoch {epoch}")
            break

        last_checkpoint = result['checkpoint_path']

    training_end = time.time()

    final_model_path = None
    if last_checkpoint and Path(last_checkpoint).exists():
        final_model_path = Path("/models") / f"rsf_distributed_8xb200_{int(training_end)}.bin"
        import shutil
        shutil.copy2(last_checkpoint, final_model_path)
        volume.commit()

    summary = {
        "total_epochs": total_epochs,
        "completed_epochs": sum(1 for r in epoch_results if r['status'] == 'completed'),
        "failed_epochs": sum(1 for r in epoch_results if r['status'] != 'completed'),
        "total_duration_seconds": training_end - training_start,
        "gpu_config": "8x B200 Distributed",
        "final_model": str(final_model_path) if final_model_path else None,
        "epoch_details": epoch_results,
        "parameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dim": dim,
            "layers": layers,
            "sample_limit": sample_limit,
            "noise_level": noise_level
        }
    }

    summary_path = Path("/models") / f"distributed_training_summary_{int(training_end)}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    return summary

@app.local_entrypoint()
def main(
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    dim: int = 128,
    layers: int = 4,
    sample_limit: int = 1000,
    noise_level: float = 0.05
):
    print("="*70)
    print("JAIDE v40 Distributed Training - 8x B200 GPUs (Epoch-by-Epoch)")
    print("="*70)
    print(f"Total Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Dimension: {dim}")
    print(f"Layers: {layers}")
    print(f"Sample Limit: {sample_limit}")
    print(f"Noise Level: {noise_level}")
    print("="*70)

    summary = orchestrate_distributed_training.remote(
        total_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dim=dim,
        layers=layers,
        sample_limit=sample_limit,
        noise_level=noise_level
    )

    print("\n" + "="*70)
    print("DISTRIBUTED TRAINING COMPLETE")
    print("="*70)
    print(f"Total Epochs: {summary['total_epochs']}")
    print(f"Completed: {summary['completed_epochs']}")
    print(f"Failed: {summary['failed_epochs']}")
    print(f"Total Duration: {summary['total_duration_seconds']:.2f}s ({summary['total_duration_seconds']/60:.2f} min)")
    print(f"GPU Config: {summary['gpu_config']}")
    print(f"Final Model: {summary['final_model']}")
    print("="*70)

    print("\nPer-Epoch Results:")
    for epoch_result in summary['epoch_details']:
        status_symbol = "✓" if epoch_result['status'] == 'completed' else "✗"
        print(f"  {status_symbol} Epoch {epoch_result['epoch']}: {epoch_result['duration_seconds']:.2f}s")