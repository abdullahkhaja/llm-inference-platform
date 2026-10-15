#!/usr/bin/env python3
"""
Llama 3.1 8B Instruct → Spider 1.0 Text-to-SQL (QLoRA SFT + optional GRPO)

Faithful conversion of the Colab notebook:
  Llama31_Spider_QLoRA_GRPO_Colab.ipynb

Preserved from the original experiment:
  - Model: meta-llama/Llama-3.1-8B-Instruct
  - Spider dataset loading / validation
  - QLoRA / LoRA hyperparameters
  - SFT training configuration
  - Checkpoint and private adapter upload logic
  - Official Spider evaluation path
  - Optional GRPO stage (train DBs only)

Colab-only interactive uploads / Drive mounts are replaced by CLI path arguments.
This script is training-only. The inference server must NEVER import or launch it.

Reported experiment results (from the original notebook run, not reproduced here):
  Spider exact-match accuracy: 40.5% → 73.3%
  Spider execution accuracy:   49.9% → 75.7%
  Evaluation set: 1,034 examples

Requires a separate training environment with GPU + TRL / PEFT / bitsandbytes.
Do not install those packages into the Phase 1 serving Docker image.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Defaults preserved from the notebook
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REPO_NAME = "Llama-3.1-8B-Spider-Text-to-SQL"
MAX_LENGTH = 2048
EVAL_MAX_LENGTH = 4096
MAX_NEW_TOKENS = 256
SEED = 42
SAVE_STEPS = 200
SYSTEM = (
    "You translate a question into one read-only SQLite SELECT query. "
    "Return SQL only, no markdown or explanation."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tune Llama 3.1 8B Instruct on Spider 1.0"
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("./spider_project"),
        help="Working directory for unpacked data, results, and local checkpoints",
    )
    p.add_argument(
        "--spider-zip",
        type=Path,
        default=None,
        help="Path to the official Spider 1.0 ZIP (required unless --spider-dir is set)",
    )
    p.add_argument(
        "--spider-dir",
        type=Path,
        default=None,
        help="Path to an already-extracted Spider data directory containing "
        "train_spider.json, dev.json, tables.json",
    )
    p.add_argument(
        "--suite-zip",
        type=Path,
        default=None,
        help="Optional official test-suite databases ZIP",
    )
    p.add_argument(
        "--spider-eval-dir",
        type=Path,
        default=None,
        help="Path to cloned taoyds/spider (for evaluation.py)",
    )
    p.add_argument(
        "--suite-eval-dir",
        type=Path,
        default=None,
        help="Path to cloned taoyds/test-suite-sql-eval",
    )
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Trainer checkpoint root (defaults to <work-dir>/checkpoints)",
    )
    p.add_argument("--repo-name", default=REPO_NAME)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--quick-test", action="store_true", help="Smoke test; do not report as full-dev")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--run-grpo", action="store_true", help="Opt-in GRPO stage (A100 recommended)")
    p.add_argument("--grpo-steps", type=int, default=100)
    p.add_argument("--grpo-train-examples", type=int, default=2000)
    p.add_argument("--upload-adapters", action="store_true", help="Upload adapters to a private HF repo")
    p.add_argument("--max-steps", type=int, default=None, help="Override SFT max_steps (-1 = full epoch)")
    return p.parse_args()


def safe_extract(zpath: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError("Unsafe ZIP member: " + m.filename)
        z.extractall(dest)


def locate_spider_data(work: Path, spider_dir: Optional[Path]) -> Path:
    if spider_dir is not None:
        data = spider_dir
        assert (data / "train_spider.json").is_file()
        assert (data / "dev.json").is_file()
        assert (data / "tables.json").is_file()
        return data
    candidates = [
        p.parent
        for p in work.rglob("train_spider.json")
        if (p.parent / "dev.json").is_file() and (p.parent / "tables.json").is_file()
    ]
    assert len(candidates) == 1, f"Expected one official Spider data directory, found: {candidates}"
    return candidates[0]


def schema_text(table_by_id: dict, db_id: str) -> str:
    s = table_by_id[db_id]
    cols = s["column_names_original"]
    typs = s["column_types"]
    pk = set(s["primary_keys"] if isinstance(s["primary_keys"], list) else [s["primary_keys"]])
    lines = []
    for i, name in enumerate(s["table_names_original"]):
        attrs = [
            f"{cols[j][1]} {typs[j]}" + (" PRIMARY KEY" if j in pk else "")
            for j in range(len(cols))
            if cols[j][0] == i
        ]
        lines.append(f"{name}(" + ", ".join(attrs) + ")")
    fks = [
        f'{s["table_names_original"][cols[a][0]]}.{cols[a][1]} -> '
        f'{s["table_names_original"][cols[b][0]]}.{cols[b][1]}'
        for a, b in s["foreign_keys"]
    ]
    if fks:
        lines.append("Foreign keys: " + "; ".join(fks))
    return "\n".join(lines)


def messages_for(table_by_id: dict, ex: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f'Database schema:\n{schema_text(table_by_id, ex["db_id"])}\n\n'
                f'Question: {ex["question"]}\nSQL:'
            ),
        },
    ]


def clean_sql(text: str) -> str:
    text = re.sub(r"^\s*```(?:sql)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    text = re.split(r"(?i)\n(?:explanation|note):", text, maxsplit=1)[0].strip()
    return text.split(";", 1)[0].strip().rstrip(";") + (";" if text.strip() else "")


def main() -> None:
    args = parse_args()

    # Heavy ML imports stay inside main so importing this module for inspection
    # does not pull TRL / bitsandbytes into the serving process.
    import torch
    import pandas as pd
    from datasets import Dataset
    from huggingface_hub import HfApi, login, snapshot_download, whoami
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from tqdm.auto import tqdm
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    assert torch.cuda.is_available(), "A CUDA GPU is required for training"
    print(torch.cuda.get_device_name(0), "VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))

    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    results = work / "results"
    results.mkdir(exist_ok=True)
    checkpoint_root = args.checkpoint_root or (work / "checkpoints")
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    quick_test = args.quick_test
    max_new_tokens = MAX_NEW_TOKENS
    seed = SEED
    save_steps = SAVE_STEPS

    if args.hf_token:
        login(token=args.hf_token)
    else:
        login()

    api = HfApi()
    hf_user = whoami()["name"]
    repo_id = f"{hf_user}/{args.repo_name}"
    if args.upload_adapters:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        assert api.repo_info(repo_id, repo_type="model").private, "Refusing to push to a public repo"
        print("Private adapter repository:", f"https://huggingface.co/{repo_id}")
    api.model_info(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Data ----------------------------------------------------------------
    if args.spider_zip is not None:
        dest = work / args.spider_zip.name
        if args.spider_zip.resolve() != dest.resolve():
            shutil.copy2(args.spider_zip, dest)
        out = work / ("unpacked_" + dest.stem)
        safe_extract(dest, out)
        print("Extracted", dest.name, "to", out)

    if args.suite_zip is not None:
        dest = work / args.suite_zip.name
        if args.suite_zip.resolve() != dest.resolve():
            shutil.copy2(args.suite_zip, dest)
        out = work / ("unpacked_" + dest.stem)
        safe_extract(dest, out)
        print("Extracted", dest.name, "to", out)

    data = locate_spider_data(work, args.spider_dir)
    db_dir = next((p for p in [data / "database", data.parent / "database"] if p.is_dir()), None)
    assert db_dir is not None, "Original Spider database/ directory missing"
    train = json.loads((data / "train_spider.json").read_text(encoding="utf-8"))
    dev = json.loads((data / "dev.json").read_text(encoding="utf-8"))
    tables = json.loads((data / "tables.json").read_text(encoding="utf-8"))
    table_by_id = {x["db_id"]: x for x in tables}
    assert len(train) > 6000 and len(dev) > 1000, "Unexpected Spider split; check original ZIP"
    assert set(x["db_id"] for x in train).isdisjoint(x["db_id"] for x in dev), "Train/dev database leakage"
    for ex in train + dev:
        assert (db_dir / ex["db_id"] / f'{ex["db_id"]}.sqlite').exists(), ex["db_id"]
    print("Spider train:", len(train), "dev:", len(dev), "DB root:", db_dir)

    def prompt_tokens(ex: dict) -> list[int]:
        return tokenizer.apply_chat_template(
            messages_for(table_by_id, ex), tokenize=True, add_generation_prompt=True
        )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    def load_base(adapter: Optional[str] = None, trainable: bool = False):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            device_map={"": torch.cuda.current_device()},
            torch_dtype=dtype,
        )
        model.config.use_cache = not trainable
        if trainable:
            model = prepare_model_for_kbit_training(model)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
        return model

    def predict_split(model, label: str, examples: list) -> list[dict]:
        path = results / f"{label}.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        assert all(
            row["index"] == i and row["db_id"] == examples[i]["db_id"] for i, row in enumerate(rows)
        ), "Saved predictions don't match this split"
        model.eval()
        for i in tqdm(range(len(rows), len(examples))):
            ex = examples[i]
            input_ids = prompt_tokens(ex)
            assert len(input_ids) + max_new_tokens <= EVAL_MAX_LENGTH, (
                f"Dev prompt {i} exceeds EVAL_MAX_LENGTH; raise it, then restart predictions."
            )
            inputs = torch.tensor([input_ids], device="cuda")
            with torch.inference_mode():
                output = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(output[0, inputs.shape[1] :], skip_special_tokens=True)
            rows.append(
                {
                    "index": i,
                    "db_id": ex["db_id"],
                    "question": ex["question"],
                    "prediction": clean_sql(raw),
                    "raw": raw,
                }
            )
            if (i + 1) % 25 == 0 or i + 1 == len(examples):
                path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
            del inputs, output
        return rows

    def release_gpu() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    class PrivateAdapterMilestones(TrainerCallback):
        def __init__(self, stage: str) -> None:
            self.stage = stage

        def on_save(self, args_, state, control, model=None, **kwargs):
            if not args.upload_adapters or model is None or not state.is_world_process_zero:
                return control
            tmp = work / f"adapter_{self.stage}_{state.global_step}"
            model.save_pretrained(tmp, safe_serialization=True)
            tokenizer.save_pretrained(tmp)
            api.upload_folder(
                folder_path=str(tmp),
                repo_id=repo_id,
                path_in_repo=f"{self.stage}/step-{state.global_step}",
                commit_message=f"{self.stage} adapter step {state.global_step}",
            )
            shutil.rmtree(tmp)
            return control

    def upload_final(model, stage: str) -> None:
        tmp = work / f"{stage}_final_adapter"
        model.save_pretrained(tmp, safe_serialization=True)
        tokenizer.save_pretrained(tmp)
        if args.upload_adapters:
            api.upload_folder(
                folder_path=str(tmp),
                repo_id=repo_id,
                path_in_repo=f"{stage}/final",
                commit_message=f"{stage} final adapter",
            )
            print("Saved private adapter at", repo_id, stage + "/final")
        else:
            print("Final adapter saved locally at", tmp)

    def private_adapter_path(stage: str) -> str:
        local = checkpoint_root / stage / "final"
        if (local / "adapter_config.json").exists():
            return str(local)
        if not args.upload_adapters:
            raise FileNotFoundError(f"Local adapter not found at {local}")
        snapshot = snapshot_download(repo_id=repo_id, allow_patterns=[f"{stage}/final/*"])
        folder = Path(snapshot) / stage / "final"
        assert (folder / "adapter_config.json").exists(), folder
        return str(folder)

    examples = dev[:20] if quick_test else dev
    baseline_rows: list[dict] = []
    sft_rows: list[dict] = []

    # ---- Baseline ------------------------------------------------------------
    if not args.skip_baseline:
        baseline_model = load_base()
        baseline_rows = predict_split(
            baseline_model, "base" + ("_smoke" if quick_test else ""), examples
        )
        del baseline_model
        release_gpu()
        print("Baseline predictions:", len(baseline_rows))

    # ---- QLoRA SFT -----------------------------------------------------------
    if not args.skip_sft:
        sft_records = []
        skipped = 0
        for ex in train:
            sql = ex["query"].strip()
            prompt = messages_for(table_by_id, ex)
            full = tokenizer.apply_chat_template(
                prompt + [{"role": "assistant", "content": sql}], tokenize=True
            )
            if len(full) > MAX_LENGTH:
                skipped += 1
                continue
            sft_records.append(
                {"prompt": prompt, "completion": [{"role": "assistant", "content": sql}]}
            )
        assert len(sft_records) > 5000, f"Too many examples excluded: {skipped}"
        print("SFT rows:", len(sft_records), "Excluded for length:", skipped)
        sft_dataset = Dataset.from_list(sft_records)

        sft_model = load_base(trainable=True)
        # LoRA hyperparameters preserved from notebook
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        sft_dir = checkpoint_root / "sft"
        max_steps = 10 if quick_test else (args.max_steps if args.max_steps is not None else -1)
        sft_args = SFTConfig(
            output_dir=str(sft_dir),
            max_length=MAX_LENGTH,
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=1,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit",
            logging_steps=25,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            report_to="none",
            seed=seed,
            packing=False,
            completion_only_loss=True,
            dataset_num_proc=1,
        )
        sft_trainer = SFTTrainer(
            model=sft_model,
            args=sft_args,
            train_dataset=sft_dataset,
            processing_class=tokenizer,
            peft_config=lora,
        )
        sft_trainer.add_callback(PrivateAdapterMilestones("sft"))
        resume = get_last_checkpoint(str(sft_dir)) if sft_dir.exists() else None
        print("Resume from:", resume)
        sft_trainer.train(resume_from_checkpoint=resume)
        sft_trainer.save_model(str(sft_dir / "final"))
        upload_final(sft_trainer.model, "sft")
        print("SFT complete")

        del sft_trainer, sft_model
        release_gpu()
        sft_eval_model = load_base(adapter=private_adapter_path("sft"))
        sft_rows = predict_split(
            sft_eval_model, "sft" + ("_smoke" if quick_test else ""), examples
        )
        del sft_eval_model
        release_gpu()
        print("SFT predictions:", len(sft_rows))

    # ---- Official evaluation -------------------------------------------------
    def write_eval_files(label: str, rows: list, examples_: list):
        assert len(rows) == len(examples_)
        assert all(
            row["index"] == i and row["db_id"] == examples_[i]["db_id"]
            for i, row in enumerate(rows)
        )
        gold = results / f"{label}.gold"
        pred = results / f"{label}.pred"
        gold.write_text(
            "".join(ex["query"].replace("\n", " ") + "\t" + ex["db_id"] + "\n" for ex in examples_),
            encoding="utf-8",
        )
        pred.write_text(
            "".join((row["prediction"] or "SELECT 1").replace("\n", " ") + "\n" for row in rows),
            encoding="utf-8",
        )
        return gold, pred

    def official_eval(
        label: str,
        rows: list,
        examples_: list,
        script: Path,
        db_root: Path,
    ) -> Path:
        gold, pred = write_eval_files(label, rows, examples_)
        cmd = [
            sys.executable,
            str(script),
            "--gold",
            str(gold),
            "--pred",
            str(pred),
            "--etype",
            "all",
            "--db",
            str(db_root),
            "--table",
            str(data / "tables.json"),
        ]
        proc = subprocess.run(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600
        )
        output = results / f"{label}_{script.parent.name}.txt"
        output.write_text("$ " + " ".join(cmd) + "\n" + proc.stdout, encoding="utf-8")
        print(proc.stdout[-10000:])
        if proc.returncode:
            raise RuntimeError(f"Evaluator failed ({proc.returncode}); see {output}")
        return output

    if not args.skip_eval and args.spider_eval_dir is not None:
        eval_script = args.spider_eval_dir / "evaluation.py"
        if baseline_rows:
            official_eval(
                "base" + ("_smoke" if quick_test else ""),
                baseline_rows,
                examples,
                eval_script,
                db_dir,
            )
        if sft_rows:
            official_eval(
                "sft" + ("_smoke" if quick_test else ""),
                sft_rows,
                examples,
                eval_script,
                db_dir,
            )

        def locate_suite_root() -> Optional[Path]:
            sample = dev[0]["db_id"]
            roots = []
            for p in work.rglob(sample):
                if p.is_dir() and list(p.glob("*.sqlite")) and p.parent.resolve() != db_dir.resolve():
                    roots.append(p.parent)
            return roots[0] if roots else None

        suite_db = locate_suite_root()
        if suite_db and args.suite_eval_dir is not None:
            suite_script = args.suite_eval_dir / "evaluation.py"
            print("Test suite root:", suite_db)
            if baseline_rows:
                official_eval(
                    "base_suite" + ("_smoke" if quick_test else ""),
                    baseline_rows,
                    examples,
                    suite_script,
                    suite_db,
                )
            if sft_rows:
                official_eval(
                    "sft_suite" + ("_smoke" if quick_test else ""),
                    sft_rows,
                    examples,
                    suite_script,
                    suite_db,
                )
        else:
            print("Official test suite databases not available. Test Suite Accuracy: NOT MEASURED.")

        if baseline_rows and sft_rows:
            analysis = []
            for i, (ex, b, s) in enumerate(zip(examples, baseline_rows, sft_rows)):
                gold = ex["query"].upper()
                analysis.append(
                    dict(
                        index=i,
                        db_id=ex["db_id"],
                        difficulty=ex.get("hardness", "computed by evaluator"),
                        question=ex["question"],
                        gold_sql=ex["query"],
                        base_sql=b["prediction"],
                        sft_sql=s["prediction"],
                        joins=" JOIN " in gold,
                        nested="(SELECT" in gold or "( SELECT" in gold,
                        group_by="GROUP BY" in gold,
                        having="HAVING" in gold,
                        order_by="ORDER BY" in gold,
                        aggregate=bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", gold)),
                        multiple_conditions=" AND " in gold or " OR " in gold,
                    )
                )
            pd.DataFrame(analysis).to_csv(results / "error_analysis.csv", index=False)
            print("Error analysis written to", results / "error_analysis.csv")

    # ---- Optional GRPO -------------------------------------------------------
    if args.run_grpo:
        from trl import GRPOConfig, GRPOTrainer

        def execute_readonly(db_id: str, sql: str, deadline: float = 0.3, row_cap: int = 1000):
            sql = clean_sql(sql)
            if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.I) or sql.count(";") > 1:
                return None
            db = db_dir / db_id / f"{db_id}.sqlite"
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            start = time.monotonic()
            con.set_progress_handler(lambda: 1 if time.monotonic() - start > deadline else 0, 2000)
            try:
                cur = con.execute(sql)
                rows = cur.fetchmany(row_cap + 1)
                if len(rows) > row_cap:
                    return None
                return sorted(map(repr, rows))
            except (sqlite3.Error, TypeError, ValueError):
                return None
            finally:
                con.close()

        def sql_reward(completions, db_id, gold_sql, **kwargs):
            rewards = []
            for out, db, gold in zip(completions, db_id, gold_sql):
                text = out[-1]["content"] if isinstance(out, list) else str(out)
                pred = clean_sql(text)
                target = execute_readonly(db, gold)
                actual = execute_readonly(db, pred)
                correct = target is not None and actual is not None and target == actual
                format_ok = bool(re.match(r"^\s*(SELECT|WITH)\b", pred, re.I)) and "```" not in text
                rewards.append(float(correct) + (0.05 if format_ok else 0.0))
            return rewards

        assert (
            sql_reward([train[0]["query"], "DROP TABLE x"], [train[0]["db_id"]] * 2, [train[0]["query"]] * 2)[0]
            >= 1.0
        )
        print("Reward smoke test passed")

        grpo_model = load_base(adapter=private_adapter_path("sft"), trainable=True)
        rng = random.Random(seed)
        sample = rng.sample(train, min(args.grpo_train_examples, len(train)))
        grpo_rows = [
            {"prompt": messages_for(table_by_id, ex), "db_id": ex["db_id"], "gold_sql": ex["query"]}
            for ex in sample
            if len(prompt_tokens(ex)) + max_new_tokens <= MAX_LENGTH
        ]
        grpo_ds = Dataset.from_list(grpo_rows)
        grpo_dir = checkpoint_root / "grpo"
        grpo_args = GRPOConfig(
            output_dir=str(grpo_dir),
            max_steps=args.grpo_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_prompt_length=MAX_LENGTH - max_new_tokens,
            max_completion_length=max_new_tokens,
            learning_rate=5e-6,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit",
            logging_steps=10,
            save_steps=50,
            save_strategy="steps",
            save_total_limit=2,
            report_to="none",
            seed=seed,
            use_vllm=False,
        )
        grpo_trainer = GRPOTrainer(
            model=grpo_model,
            reward_funcs=sql_reward,
            args=grpo_args,
            train_dataset=grpo_ds,
            processing_class=tokenizer,
        )
        grpo_trainer.add_callback(PrivateAdapterMilestones("grpo"))
        grpo_resume = get_last_checkpoint(str(grpo_dir)) if grpo_dir.exists() else None
        grpo_trainer.train(resume_from_checkpoint=grpo_resume)
        grpo_trainer.save_model(str(grpo_dir / "final"))
        upload_final(grpo_trainer.model, "grpo")
        del grpo_trainer, grpo_model
        release_gpu()
        print("GRPO complete")
    else:
        print("GRPO skipped. SFT experiment remains complete.")

    print("Done. Adapter artifacts live under", checkpoint_root)
    print("Serving integration (Phase 2+) will load MODEL_PATH / adapter from this directory.")


if __name__ == "__main__":
    main()
