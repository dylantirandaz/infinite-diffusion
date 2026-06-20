#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_hf_masked_diffusion import (
    blank_token_mask,
    choose_device,
    generate_once,
    piece_logit_penalties,
    quality_score,
    repeated_ngram_count,
    special_token_mask,
    suffix_after_prompt,
    valid_uniform_noise_ids,
)


SETTING_PRESETS = {
    "mask-ancestral": {
        "sampler": "ancestral",
        "initial_noise": "mask",
        "renoise": "mask",
        "self_conditioning_strength": 0.0,
    },
    "mask-refine": {
        "sampler": "refine",
        "initial_noise": "mask",
        "renoise": "mask",
        "self_conditioning_strength": 0.0,
    },
    "uniform-refine": {
        "sampler": "refine",
        "initial_noise": "uniform",
        "renoise": "uniform",
        "self_conditioning_strength": 0.0,
    },
    "uniform-selfcond": {
        "sampler": "refine",
        "initial_noise": "uniform",
        "renoise": "uniform",
        "self_conditioning_strength": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled ablations for the HF diffusion sampler.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="outputs/evals/hf-diffusion-ablations.json")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--seeds", default="101,10074,20047")
    parser.add_argument("--settings", default="mask-refine,uniform-refine,uniform-selfcond")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--final-temperature", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--unmask-schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--remask-strategy", choices=("confidence", "entropy", "hybrid"), default="entropy")
    parser.add_argument("--blank-commit-penalty", type=float, default=6.0)
    parser.add_argument("--repeat-penalty", type=float, default=3.5)
    parser.add_argument("--max-token-repeat-fraction", type=float, default=0.12)
    parser.add_argument("--entropy-bound", type=float, default=0.1)
    parser.add_argument("--early-stop-entropy", type=float, default=0.005)
    parser.add_argument("--piece-logit-penalty", type=float, default=1.2)
    parser.add_argument("--self-conditioning-strength", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    seeds = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one integer")
    return seeds


def parse_csv_strings(value: str) -> list[str]:
    items = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not items:
        raise ValueError("expected at least one comma-separated value")
    return items


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt)
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        prompts.extend(line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not prompts:
        prompts = [
            "I am seated in an office, surrounded by heads and bodies.",
            "Serious juniors never pick up tennis balls with their hands.",
        ]
    return prompts


def sample_metrics(text: str, prompt: str) -> dict[str, float | int | bool]:
    suffix = suffix_after_prompt(text, prompt)
    words = re.findall(r"[A-Za-z][A-Za-z']*", suffix.lower())
    unique_words = set(words)
    return {
        "quality_score": float(quality_score(text, prompt)),
        "generated_words": len(words),
        "unique_word_fraction": float(len(unique_words) / len(words)) if words else 0.0,
        "repeated_bigram_count": repeated_ngram_count(words, 2),
        "repeated_trigram_count": repeated_ngram_count(words, 3),
        "ends_with_sentence_punctuation": bool(re.search(r"[.!?][\"')\]]?\s*$", suffix)),
        "characters": len(suffix),
    }


def summarize(results: list[dict[str, object]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result["setting"]), []).append(result)

    summaries = []
    for setting, rows in sorted(grouped.items()):
        metrics = [row["metrics"] for row in rows]
        summaries.append(
            {
                "setting": setting,
                "runs": len(rows),
                "mean_quality_score": mean(float(metric["quality_score"]) for metric in metrics),
                "mean_generated_words": mean(int(metric["generated_words"]) for metric in metrics),
                "mean_unique_word_fraction": mean(float(metric["unique_word_fraction"]) for metric in metrics),
                "mean_repeated_bigram_count": mean(int(metric["repeated_bigram_count"]) for metric in metrics),
            }
        )
    return summaries


def build_sample_args(base: argparse.Namespace, prompt: str, setting_name: str) -> argparse.Namespace:
    preset = SETTING_PRESETS[setting_name]
    self_conditioning_strength = preset["self_conditioning_strength"]
    if self_conditioning_strength is None:
        self_conditioning_strength = base.self_conditioning_strength
    return argparse.Namespace(
        prompt=prompt,
        new_tokens=base.new_tokens,
        steps=base.steps,
        temperature=base.temperature,
        final_temperature=base.final_temperature,
        top_k=base.top_k,
        sampler=preset["sampler"],
        initial_noise=preset["initial_noise"],
        renoise=preset["renoise"],
        unmask_schedule=base.unmask_schedule,
        remask_strategy=base.remask_strategy,
        blank_commit_penalty=base.blank_commit_penalty,
        repeat_penalty=base.repeat_penalty,
        max_token_repeat_fraction=base.max_token_repeat_fraction,
        entropy_bound=base.entropy_bound,
        early_stop_entropy=base.early_stop_entropy,
        self_conditioning_strength=float(self_conditioning_strength),
        verbose=False,
    )


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    seeds = parse_csv_ints(args.seeds)
    settings = parse_csv_strings(args.settings)
    unknown = [setting for setting in settings if setting not in SETTING_PRESETS]
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(unknown)}")

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.eval()

    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    piece_penalties = piece_logit_penalties(tokenizer, vocab_size, args.piece_logit_penalty)
    blanks = blank_token_mask(tokenizer, vocab_size) if args.blank_commit_penalty > 0 else None
    specials = special_token_mask(tokenizer, vocab_size)
    noise_token_ids = valid_uniform_noise_ids(tokenizer, vocab_size)

    results: list[dict[str, object]] = []
    prompts = load_prompts(args)
    for setting in settings:
        for prompt_index, prompt in enumerate(prompts):
            for seed in seeds:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                sample_args = build_sample_args(args, prompt, setting)
                text = generate_once(
                    args=sample_args,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    rng=rng,
                    piece_penalties=piece_penalties,
                    blanks=blanks,
                    specials=specials,
                    vocab_size=vocab_size,
                    noise_token_ids=noise_token_ids,
                )
                results.append(
                    {
                        "setting": setting,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "seed": seed,
                        "text": text,
                        "metrics": sample_metrics(text, prompt),
                    }
                )

    summaries = summarize(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "settings": settings,
                "seeds": seeds,
                "prompts": prompts,
                "sample_config": {
                    "new_tokens": args.new_tokens,
                    "steps": args.steps,
                    "temperature": args.temperature,
                    "final_temperature": args.final_temperature,
                    "top_k": args.top_k,
                    "unmask_schedule": args.unmask_schedule,
                    "remask_strategy": args.remask_strategy,
                    "entropy_bound": args.entropy_bound,
                    "self_conditioning_strength": args.self_conditioning_strength,
                },
                "summary": summaries,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for row in summaries:
        print(
            "{setting}\truns={runs}\tquality={mean_quality_score:.3f}\twords={mean_generated_words:.1f}\t"
            "unique={mean_unique_word_fraction:.3f}\trep2={mean_repeated_bigram_count:.2f}".format(**row)
        )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
