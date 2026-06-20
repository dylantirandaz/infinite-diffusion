#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_lm.files import sha256_file
from diffusion_lm.token_diffusion import exact_k_token_mask, low_discrepancy_uniform, mdlm_mask_rates


CORRUPTION_METADATA = {
    "mask": {
        "objective": "variable_rate_masked_lm_diffusion",
        "parameterization": "absorbing_mask_parallel_prefix_continuation_denoising_no_next_token_prediction",
    },
    "uniform": {
        "objective": "variable_rate_uniform_token_diffusion",
        "parameterization": "uniform_state_parallel_prefix_continuation_denoising_no_next_token_prediction",
    },
    "mixed": {
        "objective": "variable_rate_mixed_mask_uniform_token_diffusion",
        "parameterization": "mixed_absorbing_mask_uniform_state_prefix_continuation_denoising_no_next_token_prediction",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained masked LM as a text diffusion denoiser.")
    parser.add_argument("--corpus", default="data/infinite_jest.txt")
    parser.add_argument("--out", default="outputs/roberta-infinite-jest-diffusion")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument(
        "--objective",
        choices=("mlm-ce", "mdlm-subs"),
        default="mdlm-subs",
        help="Denoising objective. mdlm-subs filters invalid clean tokens and uses explicit diffusion weighting.",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=("none", "sequence", "inverse-mask-rate", "mdlm"),
        default="mdlm",
        help="How to weight the denoising loss across sampled noise rates.",
    )
    parser.add_argument(
        "--max-loss-weight",
        type=float,
        default=8.0,
        help="Upper bound for inverse-rate MDLM loss weights.",
    )
    parser.add_argument(
        "--self-conditioning-prob",
        type=float,
        default=0.0,
        help="Probability of doing a no-grad first denoising pass and feeding its expected embeddings back in.",
    )
    parser.add_argument(
        "--self-conditioning-strength",
        type=float,
        default=0.5,
        help="Blend factor for self-conditioned expected-token embeddings.",
    )
    parser.add_argument("--min-mask-ratio", type=float, default=0.01)
    parser.add_argument("--max-mask-ratio", type=float, default=1.0)
    parser.add_argument(
        "--mask-distribution",
        choices=("uniform", "high", "full"),
        default="uniform",
        help="How to sample the continuation corruption level. 'high' rehearses near/full-mask continuation denoising.",
    )
    parser.add_argument(
        "--full-mask-fraction",
        type=float,
        default=0.0,
        help="Fraction of rows forced to mask the full continuation span.",
    )
    parser.add_argument(
        "--high-mask-power",
        type=float,
        default=2.0,
        help="Power used by --mask-distribution high; larger values put more mass near max-mask-ratio.",
    )
    parser.add_argument(
        "--corruption",
        choices=("mask", "uniform", "mixed"),
        default="mask",
        help=(
            "How to corrupt selected continuation tokens. 'mask' is absorbing-mask MLM diffusion; "
            "'uniform' replaces selected tokens with random vocabulary tokens; 'mixed' uses both."
        ),
    )
    parser.add_argument(
        "--uniform-corruption-fraction",
        type=float,
        default=0.5,
        help="For --corruption mixed, fraction of corrupted positions replaced by random tokens instead of [MASK].",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_data(token_ids: np.ndarray, valid_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be between 0 and 0.5")
    split = max(1, int(token_ids.size * (1.0 - valid_fraction)))
    return token_ids[:split].copy(), token_ids[split:].copy()


def make_core_windows(
    token_ids: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    core_len: int,
) -> np.ndarray:
    if token_ids.size <= core_len:
        raise ValueError(f"tokenized corpus must contain more than {core_len} tokens")
    starts = rng.integers(0, token_ids.size - core_len, size=batch_size)
    offsets = np.arange(core_len)
    return token_ids[starts[:, None] + offsets[None, :]].astype(np.int64, copy=True)


def sample_mask_rates(
    batch_size: int,
    rng: np.random.Generator,
    min_mask_ratio: float,
    max_mask_ratio: float,
    mask_distribution: str,
    full_mask_fraction: float,
    high_mask_power: float,
) -> np.ndarray:
    if not 0.0 <= full_mask_fraction <= 1.0:
        raise ValueError("--full-mask-fraction must be between 0 and 1")
    if high_mask_power <= 0:
        raise ValueError("--high-mask-power must be positive")

    if mask_distribution == "uniform":
        rates = mdlm_mask_rates(
            batch_size=batch_size,
            rng=rng,
            min_ratio=min_mask_ratio,
            max_ratio=max_mask_ratio,
            low_discrepancy=True,
        )
    elif mask_distribution == "high":
        unit = low_discrepancy_uniform(batch_size, rng)
        high_biased = 1.0 - np.power(1.0 - unit, high_mask_power)
        rates = min_mask_ratio + (max_mask_ratio - min_mask_ratio) * high_biased
        rates = rates.astype(np.float32)
    elif mask_distribution == "full":
        rates = np.full((batch_size,), max_mask_ratio, dtype=np.float32)
    else:
        raise ValueError(f"unknown mask distribution: {mask_distribution}")

    if full_mask_fraction > 0:
        force_full = rng.random(batch_size) < full_mask_fraction
        rates[force_full] = max_mask_ratio
    return np.clip(rates, min_mask_ratio, max_mask_ratio).astype(np.float32)


def valid_uniform_noise_ids(tokenizer: object) -> np.ndarray:
    vocab_size = len(tokenizer)
    excluded = {int(token_id) for token_id in (tokenizer.all_special_ids or [])}
    if tokenizer.mask_token_id is not None:
        excluded.add(int(tokenizer.mask_token_id))
    return np.array([token_id for token_id in range(vocab_size) if token_id not in excluded], dtype=np.int64)


def forbidden_prediction_ids(tokenizer: object, vocab_size: int) -> torch.Tensor:
    forbidden_set = {int(token_id) for token_id in (tokenizer.all_special_ids or []) if 0 <= int(token_id) < vocab_size}
    if tokenizer.mask_token_id is not None and 0 <= int(tokenizer.mask_token_id) < vocab_size:
        forbidden_set.add(int(tokenizer.mask_token_id))
    forbidden = sorted(forbidden_set)
    return torch.tensor(forbidden, dtype=torch.long)


def filter_for_subs_parameterization(logits: torch.Tensor, forbidden_ids: torch.Tensor, objective: str) -> torch.Tensor:
    if objective != "mdlm-subs" or forbidden_ids.numel() == 0:
        return logits
    filtered = logits.clone()
    ids = forbidden_ids.to(device=filtered.device)
    filtered[..., ids] = torch.finfo(filtered.dtype).min
    return filtered


def label_loss_weights(
    labels: torch.Tensor,
    rates: torch.Tensor,
    loss_weighting: str,
    max_loss_weight: float,
) -> torch.Tensor:
    active = labels.ne(-100).float()
    if not bool(active.any().detach().cpu().item()):
        return active
    if loss_weighting == "none":
        return active

    counts = active.sum(dim=1).clamp_min(1.0)
    row_weight = torch.ones_like(counts)
    if loss_weighting in {"inverse-mask-rate", "mdlm"}:
        if max_loss_weight <= 0:
            raise ValueError("--max-loss-weight must be positive")
        row_weight = torch.clamp(1.0 / rates.float().clamp_min(1e-4), max=max_loss_weight)
    elif loss_weighting != "sequence":
        raise ValueError(f"unknown loss weighting: {loss_weighting}")
    return active * (row_weight / counts)[:, None]


def diffusion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    rates: torch.Tensor,
    forbidden_ids: torch.Tensor,
    objective: str,
    loss_weighting: str,
    max_loss_weight: float,
) -> torch.Tensor:
    logits = filter_for_subs_parameterization(logits, forbidden_ids, objective)
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    weights = label_loss_weights(labels, rates, loss_weighting, max_loss_weight)
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


def self_conditioned_logits(
    model: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    forbidden_ids: torch.Tensor,
    objective: str,
    strength: float,
) -> torch.Tensor:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("--self-conditioning-strength must be between 0 and 1")
    active = labels.ne(-100)
    if not bool(active.any().detach().cpu().item()) or strength == 0.0:
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        first_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        active_logits = filter_for_subs_parameterization(first_logits[active], forbidden_ids, objective)
        probs = torch.softmax(active_logits.float(), dim=-1).to(dtype=embedding_layer.weight.dtype)
        expected_embeddings = probs @ embedding_layer.weight

    input_embeddings = embedding_layer(input_ids)
    input_embeddings = input_embeddings.clone()
    input_embeddings[active] = (1.0 - strength) * input_embeddings[active] + strength * expected_embeddings
    return model(inputs_embeds=input_embeddings, attention_mask=attention_mask).logits


def corrupt_core_tokens(
    clean_core: np.ndarray,
    mask_core: np.ndarray,
    tokenizer: object,
    rng: np.random.Generator,
    corruption: str,
    uniform_corruption_fraction: float,
) -> np.ndarray:
    noisy_core = clean_core.copy()
    corrupted_count = int(np.count_nonzero(mask_core))
    if corrupted_count == 0:
        return noisy_core
    if corruption == "mask":
        noisy_core[mask_core] = int(tokenizer.mask_token_id)
        return noisy_core
    if corruption == "uniform":
        use_uniform = mask_core
    elif corruption == "mixed":
        if not 0.0 <= uniform_corruption_fraction <= 1.0:
            raise ValueError("--uniform-corruption-fraction must be between 0 and 1")
        use_uniform = mask_core & (rng.random(clean_core.shape) < uniform_corruption_fraction)
        use_mask = mask_core & ~use_uniform
        noisy_core[use_mask] = int(tokenizer.mask_token_id)
    else:
        raise ValueError(f"unknown corruption mode: {corruption}")

    noise_ids = valid_uniform_noise_ids(tokenizer)
    if noise_ids.size == 0:
        raise ValueError("tokenizer has no valid non-special token ids for uniform corruption")
    random_ids = rng.choice(noise_ids, size=int(np.count_nonzero(use_uniform)), replace=True)
    noisy_core[use_uniform] = random_ids.astype(np.int64)
    return noisy_core


def core_start_after_specials(row_len: int, core_len: int) -> int:
    special_count = row_len - core_len
    if special_count <= 0:
        return 0
    if special_count >= 2:
        return 1
    return max(0, row_len - core_len)


def make_batch(
    token_ids: np.ndarray,
    tokenizer: object,
    rng: np.random.Generator,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
    mask_distribution: str,
    full_mask_fraction: float,
    high_mask_power: float,
    corruption: str,
    uniform_corruption_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    core_len = seq_len - special_count
    if core_len <= 0:
        raise ValueError("--seq-len is too small for model special tokens")
    if not 0 <= prefix_len < core_len:
        raise ValueError("--prefix-len must be between 0 and the non-special sequence length - 1")

    clean_core = make_core_windows(token_ids, rng, batch_size, core_len)
    rates = sample_mask_rates(
        batch_size=batch_size,
        rng=rng,
        min_mask_ratio=min_mask_ratio,
        max_mask_ratio=max_mask_ratio,
        mask_distribution=mask_distribution,
        full_mask_fraction=full_mask_fraction,
        high_mask_power=high_mask_power,
    )
    mask_core = exact_k_token_mask(clean_core.shape, rates, rng, prefix_len=prefix_len)
    noisy_core = corrupt_core_tokens(
        clean_core=clean_core,
        mask_core=mask_core,
        tokenizer=tokenizer,
        rng=rng,
        corruption=corruption,
        uniform_corruption_fraction=uniform_corruption_fraction,
    )

    input_ids = []
    labels = []
    for clean_row, noisy_row, mask_row in zip(clean_core, noisy_core, mask_core):
        row = tokenizer.build_inputs_with_special_tokens(noisy_row.tolist())
        label_row = np.full((len(row),), -100, dtype=np.int64)
        clean_with_specials = tokenizer.build_inputs_with_special_tokens(clean_row.tolist())
        core_start = core_start_after_specials(row_len=len(row), core_len=core_len)
        core_positions = np.arange(core_start, core_start + core_len)
        label_row[core_positions[mask_row]] = np.array(clean_with_specials, dtype=np.int64)[core_positions[mask_row]]
        input_ids.append(row)
        labels.append(label_row.tolist())

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=device)
    rate_tensor = torch.tensor(rates, dtype=torch.float32, device=device)
    return input_tensor, attention_mask, label_tensor, rate_tensor


@torch.no_grad()
def evaluate(
    model: object,
    token_ids: np.ndarray,
    tokenizer: object,
    rng: np.random.Generator,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
    mask_distribution: str,
    full_mask_fraction: float,
    high_mask_power: float,
    corruption: str,
    uniform_corruption_fraction: float,
    objective: str,
    loss_weighting: str,
    max_loss_weight: float,
    self_conditioning_prob: float,
    self_conditioning_strength: float,
    forbidden_ids: torch.Tensor,
    device: torch.device,
    batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        input_ids, attention_mask, labels, rates = make_batch(
            token_ids,
            tokenizer,
            rng,
            batch_size,
            seq_len,
            prefix_len,
            min_mask_ratio,
            max_mask_ratio,
            mask_distribution,
            full_mask_fraction,
            high_mask_power,
            corruption,
            uniform_corruption_fraction,
            device,
        )
        use_self_conditioning = self_conditioning_prob > 0 and float(rng.random()) < self_conditioning_prob
        if use_self_conditioning:
            logits = self_conditioned_logits(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                forbidden_ids=forbidden_ids,
                objective=objective,
                strength=self_conditioning_strength,
            )
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = diffusion_loss(
            logits=logits,
            labels=labels,
            rates=rates,
            forbidden_ids=forbidden_ids,
            objective=objective,
            loss_weighting=loss_weighting,
            max_loss_weight=max_loss_weight,
        )
        losses.append(float(loss.detach().cpu().item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(
    model: object,
    tokenizer: object,
    out_dir: Path,
    metadata: dict,
    step: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "metadata.json").write_text(
        json.dumps({**metadata, "backend": "hf-transformers", "step": step}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved={out_dir} step={step}", flush=True)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not 0.0 <= args.self_conditioning_prob <= 1.0:
        raise ValueError("--self-conditioning-prob must be between 0 and 1")
    if not 0.0 <= args.self_conditioning_strength <= 1.0:
        raise ValueError("--self-conditioning-strength must be between 0 and 1")

    corpus_path = Path(args.corpus)
    out_dir = Path(args.out)
    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.mask_token_id is None:
        raise ValueError("Selected tokenizer has no mask token")
    token_ids = np.array(tokenizer.encode(text, add_special_tokens=False), dtype=np.int64)
    train_data, valid_data = split_data(token_ids, args.valid_fraction)
    if valid_data.size <= args.seq_len:
        valid_data = train_data

    device = choose_device(args.device)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    forbidden_ids = forbidden_prediction_ids(tokenizer, int(getattr(model.config, "vocab_size", len(tokenizer))))

    metadata = {
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_bytes": corpus_path.stat().st_size,
        "source_model": args.model,
        "tokens": int(token_ids.size),
        "train_tokens": int(train_data.size),
        "valid_tokens": int(valid_data.size),
        "seq_len": args.seq_len,
        "prefix_len": args.prefix_len,
        "objective": args.objective,
        "loss_weighting": args.loss_weighting,
        "max_loss_weight": args.max_loss_weight,
        "self_conditioning_prob": args.self_conditioning_prob,
        "self_conditioning_strength": args.self_conditioning_strength,
        "min_mask_ratio": args.min_mask_ratio,
        "max_mask_ratio": args.max_mask_ratio,
        "mask_distribution": args.mask_distribution,
        "full_mask_fraction": args.full_mask_fraction,
        "high_mask_power": args.high_mask_power,
        "corruption": args.corruption,
        "uniform_corruption_fraction": args.uniform_corruption_fraction,
        "seed": args.seed,
        "device": str(device),
        "corruption_objective": CORRUPTION_METADATA[args.corruption]["objective"],
        "parameterization": CORRUPTION_METADATA[args.corruption]["parameterization"],
    }
    print(
        "train_hf_masked_diffusion corpus_bytes={corpus_bytes} tokens={tokens} train={train_tokens} "
        "valid={valid_tokens} model={source_model} device={device}".format(**metadata),
        flush=True,
    )

    start = time.time()
    last_loss = None
    progress = trange(1, args.steps + 1, desc="train_hf_masked_diffusion", dynamic_ncols=True)
    for step in progress:
        input_ids, attention_mask, labels, rates = make_batch(
            train_data,
            tokenizer,
            rng,
            args.batch_size,
            args.seq_len,
            args.prefix_len,
            args.min_mask_ratio,
            args.max_mask_ratio,
            args.mask_distribution,
            args.full_mask_fraction,
            args.high_mask_power,
            args.corruption,
            args.uniform_corruption_fraction,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        use_self_conditioning = args.self_conditioning_prob > 0 and float(rng.random()) < args.self_conditioning_prob
        if use_self_conditioning:
            logits = self_conditioned_logits(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                forbidden_ids=forbidden_ids,
                objective=args.objective,
                strength=args.self_conditioning_strength,
            )
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = diffusion_loss(
            logits=logits,
            labels=labels,
            rates=rates,
            forbidden_ids=forbidden_ids,
            objective=args.objective,
            loss_weighting=args.loss_weighting,
            max_loss_weight=args.max_loss_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        last_loss = float(loss.detach().cpu().item())
        if step == 1 or step % 10 == 0:
            progress.set_postfix(loss=f"{last_loss:.3f}")

        if step % args.eval_every == 0 or step == args.steps:
            valid_loss = evaluate(
                model,
                valid_data,
                tokenizer,
                rng,
                args.batch_size,
                args.seq_len,
                args.prefix_len,
                args.min_mask_ratio,
                args.max_mask_ratio,
                args.mask_distribution,
                args.full_mask_fraction,
                args.high_mask_power,
                args.corruption,
                args.uniform_corruption_fraction,
                args.objective,
                args.loss_weighting,
                args.max_loss_weight,
                args.self_conditioning_prob,
                args.self_conditioning_strength,
                forbidden_ids,
                device,
                args.eval_batches,
            )
            metadata["last_train_loss"] = last_loss
            metadata["last_valid_loss"] = valid_loss
            print(f"step={step} train_loss={last_loss:.4f} valid_loss={valid_loss:.4f}", flush=True)

        if step % args.save_every == 0:
            save_checkpoint(model, tokenizer, out_dir, metadata, step)

    metadata["elapsed_seconds"] = round(time.time() - start, 3)
    metadata["last_train_loss"] = last_loss
    save_checkpoint(model, tokenizer, out_dir, metadata, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
