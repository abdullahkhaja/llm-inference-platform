#!/usr/bin/env python3
"""Download a private HF adapter subfolder to a local directory for vLLM LoRA.

Never prints HF_TOKEN. Intended to run on the GPU pod before vLLM starts.
Does not load CUDA or the base LLM — hub download only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Resolve private HF LoRA adapter to disk")
    p.add_argument("--repo-id", required=True, help="HF model repo id")
    p.add_argument("--subfolder", default="sft/final")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF token (prefer env HF_TOKEN; never commit)",
    )
    args = p.parse_args()

    if not args.token:
        print("ERROR: HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required for private repos", file=sys.stderr)
        return 2

    from huggingface_hub import snapshot_download

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"{args.subfolder.strip('/') }/*"
    print(f"Downloading adapter files from repo={args.repo_id} pattern={pattern}")
    snapshot = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        token=args.token,
        allow_patterns=[pattern, f"{args.subfolder.strip('/')}/**"],
        local_dir=str(args.out_dir.parent / "_hf_snap"),
    )
    src = Path(snapshot) / args.subfolder
    if not (src / "adapter_config.json").exists():
        # Some snapshots flatten; search
        matches = list(Path(snapshot).rglob("adapter_config.json"))
        if not matches:
            print(f"ERROR: adapter_config.json not found under {snapshot}", file=sys.stderr)
            return 3
        src = matches[0].parent

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    shutil.copytree(src, args.out_dir)

    cfg = args.out_dir / "adapter_config.json"
    weights = list(args.out_dir.glob("adapter_model*")) + list(
        args.out_dir.glob("*.safetensors")
    )
    if not cfg.exists():
        print("ERROR: adapter_config.json missing after copy", file=sys.stderr)
        return 4
    if not weights:
        print(
            "WARNING: no adapter weight files matched common names; "
            "verify the subfolder contents before starting vLLM",
            file=sys.stderr,
        )
    print(f"Adapter ready at {args.out_dir}")
    print(f"  config={cfg.name} weight_files={len(weights)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
