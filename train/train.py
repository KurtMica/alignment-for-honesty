"""
SFT training for Alignment for Honesty (NeurIPS 2024) using CoLLiE.

Hyperparameters from paper (Section D.4):
  optimizer:     AdamW
  lr:            1e-6
  weight_decay:  0.1
  warmup_ratio:  0.05
  batch_size:    8 total (micro_batch_size = 8 // NUM_GPUS, auto-computed)
  epochs:        2   (use --train_epochs 1 for the MULTISAMPLE method)

Parameters NOT stated in the paper (left at CoLLiE/DeepSpeed defaults):
  - LR scheduler type
  - Gradient clipping
  - Max sequence length   [FLAG: required for tokenisation — defaults to 1024,
                           which matches the data_max_length in process_training_data.py]
  - Gradient accumulation steps
  - Mixed precision (fp16 / bf16)   [FLAG: add to ds_config.json if needed]
  - Tensor / pipeline parallelism (tp_size=1, dp_size=NUM_GPUS, pp_size=1)

Usage:  bash train/train.sh [NUM_GPUS]
"""

import os
import math
import argparse
import torch
from transformers import AutoTokenizer

from collie.config import CollieConfig
from collie.data import CollieDatasetForTraining
from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity
from collie.models.llama.model import LlamaForCausalLM
from collie.module import GPTLMLoss
from collie.metrics import PPLMetric
from collie.utils.monitor import LossMonitor, EvalMonitor
from collie.callbacks import CheckpointCallback


# ── Hyperparameters stated in the paper ───────────────────────────────────────
LR = 1e-6
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.05

# Eval split held out from train.pt for loss monitoring (not in paper)
EVAL_SPLIT_SIZE = 500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Llama-2-13b-chat-hf")
    p.add_argument("--train_data_path",
                   default="data/processed/triviaqa_13b/confidence-verb_p3/train.pt")
    p.add_argument("--output_dir", default="checkpoints/confidence-verb_13b")
    p.add_argument("--ds_config", default=None,
                   help="Path to a DeepSpeed config JSON.")
    # ── From paper ─────────────────────────────────────────────────────────────
    p.add_argument("--train_epochs", type=int, default=2,
                   help="Paper: 2 for most methods, 1 for MULTISAMPLE")
    p.add_argument("--total_batch_size", type=int, default=8,
                   help="Effective total batch size across all GPUs (paper: 8). "
                        "micro_batch_size is derived as total_batch_size // num_gpus.")
    # ── Not stated in paper — required by CoLLiE ───────────────────────────────
    p.add_argument("--max_seq_length", type=int, default=1024,
                   help="Required for tokenisation.")
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_per_n_steps", type=int, default=200)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Truncate training set to this many examples. Use for smoke tests.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=0,
                   help="Injected automatically by the DeepSpeed launcher.")
    return p.parse_args()


def load_data(path: str):
    """Load processed .pt file and split off an eval subset."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    dicts = [{"input": d["input"], "output": d["output"]} for d in raw]
    return dicts[:-EVAL_SPLIT_SIZE], dicts[-EVAL_SPLIT_SIZE:]


def main():
    args = parse_args()

    # ── Data ───────────────────────────────────────────────────────────────────
    train_raw, eval_raw = load_data(args.train_data_path)
    if args.max_train_samples is not None and args.max_train_samples > 0:
        train_raw = train_raw[:args.max_train_samples]
    print(f"Train: {len(train_raw)} examples  |  Eval: {len(eval_raw)} examples")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── CoLLiE config ─────────────────────────────────────────────────────────
    config = CollieConfig.from_pretrained(args.model_name, trust_remote_code=True)

    # Parallelism: tp=1, pp=1; dp fills remaining GPUs [NOT IN PAPER]
    config.tp_size = 1
    config.pp_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.dp_size = world_size  # tp=1, pp=1 → all GPUs do data parallelism

    # micro_batch_size = total_batch_size / num_gpus  (paper: total=8)
    micro_batch_size = max(1, args.total_batch_size // world_size)
    print(f"GPUs: {world_size}  |  micro_batch_size: {micro_batch_size}  "
          f"(effective total: {micro_batch_size * world_size})")

    # Training params from paper
    config.train_epochs = args.train_epochs
    config.train_micro_batch_size = micro_batch_size
    config.eval_batch_size = args.eval_batch_size
    config.eval_per_n_steps = args.eval_per_n_steps
    config.seed = args.seed

    # Warmup scheduler: paper gives warmup_ratio=0.05; compute warmup_steps.
    # Scheduler *type* is not stated — leaving scheduler unset (CoLLiE default).
    steps_per_epoch = math.ceil(
        len(train_raw) / (micro_batch_size * config.dp_size)
    )
    total_steps = steps_per_epoch * args.train_epochs
    warmup_steps = max(1, round(WARMUP_RATIO * total_steps))
    print(f"Total steps: {total_steps}  |  Warmup steps: {warmup_steps}")
    # [FLAG] To add a scheduler, set config.ds_config["scheduler"] here,
    # e.g. WarmupLR with warmup_num_steps=warmup_steps.

    if args.ds_config is not None:
        import json
        with open(args.ds_config) as f:
            config.ds_config = json.load(f)

    # ── Datasets ───────────────────────────────────────────────────────────────
    # CollieDatasetForTraining computes loss only on `output` tokens (not `input`).
    train_dataset = CollieDatasetForTraining(
        train_raw, tokenizer=tokenizer, max_length=args.max_seq_length
    )
    eval_dataset = CollieDatasetForTraining(
        eval_raw, tokenizer=tokenizer, max_length=args.max_seq_length
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, config=config, trust_remote_code=True
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Paper: AdamW, lr=1e-6, weight_decay=0.1.
    # betas and eps are not stated — using torch defaults (0.9, 0.999), 1e-8.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Evaluator (perplexity on held-out eval split) ─────────────────────────
    evaluator = EvaluatorForPerplexity(
        model=model,
        config=config,
        dataset=eval_dataset,
        metrics={"ppl": PPLMetric()},
        monitors=[EvalMonitor(config)],
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        loss_fn=GPTLMLoss(ignore_index=-100),
        optimizer=optimizer,
        train_dataset=train_dataset,
        callbacks=[
            CheckpointCallback(folder=args.output_dir, every_n_epochs=1, last=True)
        ],
        monitors=[LossMonitor(config)],
        evaluators=[evaluator],
    )

    trainer.train()


if __name__ == "__main__":
    main()
