from __future__ import annotations

import math

import numpy as np


LEGACY_OBJECTIVE = "legacy-masked-ce"
MDLM_OBJECTIVE = "mdlm-subs"
MASK_SAMPLING_BERNOULLI = "bernoulli"
MASK_SAMPLING_EXACT_K = "exact-k"


def legacy_mask_ratios(timesteps: np.ndarray, diffusion_steps: int) -> np.ndarray:
    if diffusion_steps <= 1:
        return np.ones_like(timesteps, dtype=np.float32)
    level = (timesteps.astype(np.float32) - 1.0) / float(diffusion_steps - 1)
    return np.clip(0.10 + 0.90 * level, 0.10, 1.0)


def low_discrepancy_uniform(batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values = (np.arange(batch_size, dtype=np.float32) + float(rng.random())) / float(batch_size)
    rng.shuffle(values)
    return values


def mdlm_mask_rates(
    batch_size: int,
    rng: np.random.Generator,
    min_ratio: float,
    max_ratio: float,
    low_discrepancy: bool,
) -> np.ndarray:
    if not 0.0 <= min_ratio < max_ratio <= 1.0:
        raise ValueError("mask ratios must satisfy 0 <= min < max <= 1")
    unit = low_discrepancy_uniform(batch_size, rng) if low_discrepancy else rng.random(batch_size)
    rates = min_ratio + (max_ratio - min_ratio) * unit
    return rates.astype(np.float32)


def timesteps_from_mask_rates(mask_rates: np.ndarray, diffusion_steps: int) -> np.ndarray:
    if diffusion_steps <= 0:
        raise ValueError("diffusion_steps must be positive")
    clipped = np.clip(mask_rates.astype(np.float32), 0.0, 1.0)
    timesteps = np.rint(1.0 + clipped * float(diffusion_steps - 1)).astype(np.int32)
    return np.clip(timesteps, 1, diffusion_steps)


def exact_mask_counts(mask_rates: np.ndarray, maskable_tokens: int) -> np.ndarray:
    if maskable_tokens <= 0:
        raise ValueError("maskable_tokens must be positive")
    counts = np.rint(mask_rates.astype(np.float32) * float(maskable_tokens)).astype(np.int32)
    return np.clip(counts, 1, maskable_tokens)


def exact_k_token_mask(
    shape: tuple[int, int],
    mask_rates: np.ndarray,
    rng: np.random.Generator,
    prefix_len: int = 0,
) -> np.ndarray:
    batch_size, seq_len = shape
    if mask_rates.shape != (batch_size,):
        raise ValueError("mask_rates must have one value per batch row")
    if not 0 <= prefix_len < seq_len:
        raise ValueError("prefix_len must be between 0 and seq_len - 1")

    mask = np.zeros(shape, dtype=bool)
    maskable_tokens = seq_len - prefix_len
    counts = exact_mask_counts(mask_rates, maskable_tokens)
    for row, count in enumerate(counts.tolist()):
        cols = rng.choice(maskable_tokens, size=count, replace=False) + prefix_len
        mask[row, cols] = True
    return mask


def target_masked_after_step(reverse_step: int, steps: int, total: int, schedule: str = "linear") -> int:
    if total <= 0 or steps <= 1 or reverse_step <= 1:
        return 0
    remaining = max(0.0, min(1.0, float(reverse_step - 1) / float(steps)))
    if schedule == "linear":
        fraction = remaining
    elif schedule == "cosine":
        fraction = math.sin(remaining * math.pi / 2.0) ** 2
    else:
        raise ValueError(f"unknown unmask schedule: {schedule}")
    return int(round(float(total) * fraction))
