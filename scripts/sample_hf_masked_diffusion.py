#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import re
import string
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from diffusion_lm.token_diffusion import target_masked_after_step


@dataclass(frozen=True)
class TraceFrame:
    frame_index: int
    noise_percent: int
    tokens: np.ndarray
    visible_generated: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a pretrained masked LM as an absorbing-mask text diffusion model.")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--prompt", default="I am seated in an office,")
    parser.add_argument("--new-tokens", type=int, default=96)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--final-temperature", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--sampler", choices=("ancestral", "refine"), default="ancestral")
    parser.add_argument(
        "--initial-noise",
        choices=("mask", "uniform"),
        default="mask",
        help="How to initialize the generated canvas before denoising.",
    )
    parser.add_argument(
        "--renoise",
        choices=("same", "mask", "uniform"),
        default="same",
        help="How to corrupt rejected positions between denoising passes. 'same' follows --initial-noise.",
    )
    parser.add_argument("--unmask-schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--remask-strategy", choices=("confidence", "entropy", "hybrid"), default="hybrid")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--blank-commit-penalty", type=float, default=6.0)
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=0.0,
        help="Extra remask uncertainty for over-repeated decoded token pieces; useful for collapse diagnostics.",
    )
    parser.add_argument("--max-token-repeat-fraction", type=float, default=0.18)
    parser.add_argument(
        "--entropy-bound",
        type=float,
        default=0.0,
        help="If >0 with --sampler refine, keep low-entropy tokens under this average entropy bound.",
    )
    parser.add_argument(
        "--early-stop-entropy",
        type=float,
        default=-1.0,
        help="If >=0 with --sampler refine, stop when mean entropy is below this and argmax tokens are stable.",
    )
    parser.add_argument(
        "--piece-logit-penalty",
        type=float,
        default=0.0,
        help="Subtract this from blank or pathological punctuation token logits before sampling.",
    )
    parser.add_argument(
        "--self-conditioning-strength",
        type=float,
        default=0.0,
        help="Blend previous denoising predictions into the next pass as expected-token embeddings.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=1,
        help="Generate this many diffusion trajectories and print the highest-scoring continuation.",
    )
    parser.add_argument("--candidate-seed-stride", type=int, default=9973)
    parser.add_argument("--selection-out", default="", help="Optional JSON path for candidate scores and selected seed.")
    parser.add_argument("--keep-specials", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def temperature_for_step(step: int, steps: int, start: float, end: float) -> float:
    if steps <= 1:
        return end
    progress = 1.0 - max(0.0, min(1.0, float(step - 1) / float(steps - 1)))
    return start + (end - start) * progress


def sample_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if temperature <= 0:
        tokens = logits.argmax(axis=-1)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        confidence = probs[np.arange(tokens.size), tokens]
        entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=-1)
        return tokens.astype(np.int64), confidence, entropy

    scaled = logits.astype(np.float64) / temperature
    if 0 < top_k < scaled.shape[-1]:
        keep = np.argpartition(scaled, -top_k, axis=-1)[:, -top_k:]
        filtered = np.full_like(scaled, -np.inf)
        rows = np.arange(scaled.shape[0])[:, None]
        filtered[rows, keep] = scaled[rows, keep]
        scaled = filtered
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    tokens = np.array([rng.choice(probs.shape[-1], p=row) for row in probs], dtype=np.int64)
    confidence = probs[np.arange(tokens.size), tokens]
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=-1)
    return tokens, confidence, entropy


def uncertainty_scores(
    confidence: np.ndarray,
    entropy: np.ndarray,
    strategy: str,
    vocab_size: int,
) -> np.ndarray:
    norm_entropy = entropy / max(float(np.log(max(2, vocab_size))), 1e-6)
    confidence_uncertainty = -np.log(np.maximum(confidence, 1e-12))
    if strategy == "confidence":
        return confidence_uncertainty
    if strategy == "entropy":
        return norm_entropy
    if strategy == "hybrid":
        return confidence_uncertainty + norm_entropy
    raise ValueError(f"unknown remask strategy: {strategy}")


def blank_token_mask(tokenizer: object, vocab_size: int) -> np.ndarray:
    blanks = []
    for token_id in range(vocab_size):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        blanks.append(bool(text) and text.strip() == "")
    return np.array(blanks, dtype=bool)


def special_token_mask(tokenizer: object, vocab_size: int) -> np.ndarray:
    mask = np.zeros((vocab_size,), dtype=bool)
    for token_id in set(tokenizer.all_special_ids or []):
        if 0 <= int(token_id) < vocab_size:
            mask[int(token_id)] = True
    return mask


def valid_uniform_noise_ids(tokenizer: object, vocab_size: int) -> np.ndarray:
    excluded = {int(token_id) for token_id in (tokenizer.all_special_ids or [])}
    if tokenizer.mask_token_id is not None:
        excluded.add(int(tokenizer.mask_token_id))
    return np.array([token_id for token_id in range(vocab_size) if token_id not in excluded], dtype=np.int64)


def draw_noise(
    tokenizer: object,
    rng: np.random.Generator,
    count: int,
    mode: str,
    noise_token_ids: np.ndarray,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if mode == "mask":
        return np.full((count,), int(tokenizer.mask_token_id), dtype=np.int64)
    if mode == "uniform":
        if noise_token_ids.size == 0:
            raise ValueError("tokenizer has no valid non-special token ids for uniform noising")
        return rng.choice(noise_token_ids, size=count, replace=True).astype(np.int64)
    raise ValueError(f"unknown noise mode: {mode}")


def resolve_renoise_mode(initial_noise: str, renoise: str) -> str:
    return initial_noise if renoise == "same" else renoise


def piece_logit_penalties(tokenizer: object, vocab_size: int, penalty: float) -> np.ndarray | None:
    if penalty <= 0:
        return None
    penalties = np.zeros((vocab_size,), dtype=np.float64)
    allowed_punctuation = {".", ",", "?", "!", ";", ":", "'", '"', "(", ")", "-"}
    for token_id in range(vocab_size):
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        stripped = piece.strip()
        if not stripped:
            penalties[token_id] = penalty
            continue
        has_alnum = any(ch.isalnum() for ch in stripped)
        if has_alnum or stripped in allowed_punctuation:
            continue
        penalties[token_id] = penalty
        if len(stripped) > 1:
            penalties[token_id] += 0.5 * penalty
    return penalties


def normalized_piece(tokenizer: object, token_id: int) -> str:
    piece = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    piece = piece.strip().lower()
    return piece.strip(string.whitespace)


def repetition_penalty_scores(
    tokenizer: object,
    sampled: np.ndarray,
    repeat_penalty: float,
    max_fraction: float,
) -> np.ndarray:
    penalties = np.zeros((sampled.size,), dtype=np.float64)
    if repeat_penalty <= 0 or sampled.size == 0:
        return penalties
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError("--max-token-repeat-fraction must be in (0, 1]")

    pieces = [normalized_piece(tokenizer, int(token_id)) for token_id in sampled.tolist()]
    counts = Counter(piece for piece in pieces if piece)
    if not counts:
        return penalties
    total = float(len(pieces))
    for index, piece in enumerate(pieces):
        if not piece:
            continue
        fraction = counts[piece] / total
        if fraction > max_fraction:
            penalties[index] += repeat_penalty * (fraction / max_fraction)
    return penalties


def entropy_bound_commit_count(entropy: np.ndarray, entropy_bound: float, min_commit: int) -> int:
    if entropy_bound <= 0 or entropy.size == 0:
        return max(0, min(min_commit, entropy.size))
    order = np.argsort(entropy)
    ordered_entropy = entropy[order]
    running_mean = np.cumsum(ordered_entropy) / np.arange(1, entropy.size + 1)
    eligible = int(np.count_nonzero(running_mean <= entropy_bound))
    commit_count = max(min_commit, eligible)
    return max(0, min(commit_count, entropy.size))


def lowest_uncertainty_indices(uncertainty: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if count >= uncertainty.size:
        return np.arange(uncertainty.size)
    return np.argpartition(uncertainty, count - 1)[:count]


def highest_uncertainty_indices(uncertainty: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if count >= uncertainty.size:
        return np.arange(uncertainty.size)
    start = uncertainty.size - count
    return np.argpartition(uncertainty, start)[start:]


def apply_logit_filters(
    logits: np.ndarray,
    specials: np.ndarray | None,
    piece_penalties: np.ndarray | None,
) -> np.ndarray:
    if specials is not None:
        logits[:, specials] = -np.inf
    if piece_penalties is not None:
        logits -= piece_penalties[None, :]
    return logits


def apply_logit_filters_torch(
    logits: object,
    special_indices: object | None,
    piece_penalties: object | None,
) -> object:
    import torch

    filtered = logits
    if special_indices is not None and special_indices.numel() > 0:
        filtered = filtered.clone()
        filtered[:, special_indices] = torch.finfo(filtered.dtype).min
    if piece_penalties is not None:
        if filtered is logits:
            filtered = filtered.clone()
        filtered = filtered - piece_penalties[None, :]
    return filtered


def renoise_positions(
    tokens: np.ndarray,
    positions: np.ndarray,
    tokenizer: object,
    rng: np.random.Generator,
    mode: str,
    noise_token_ids: np.ndarray,
) -> None:
    if positions.size == 0:
        return
    tokens[positions] = draw_noise(
        tokenizer=tokenizer,
        rng=rng,
        count=positions.size,
        mode=mode,
        noise_token_ids=noise_token_ids,
    )


def core_start_after_specials(row_len: int, core_len: int) -> int:
    special_count = row_len - core_len
    if special_count <= 0:
        return 0
    if special_count >= 2:
        return 1
    return 0


def build_input(
    tokenizer: object,
    prompt: str,
    new_tokens: int,
    initial_noise: str,
    rng: np.random.Generator,
    noise_token_ids: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask token; this fallback requires a masked language model.")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated_noise = draw_noise(
        tokenizer=tokenizer,
        rng=rng,
        count=new_tokens,
        mode=initial_noise,
        noise_token_ids=noise_token_ids,
    ).tolist()
    sequence = prompt_ids + generated_noise
    input_ids = tokenizer.build_inputs_with_special_tokens(sequence)
    core_start = core_start_after_specials(row_len=len(input_ids), core_len=len(sequence))
    generated_positions = np.arange(core_start + len(prompt_ids), core_start + len(prompt_ids) + new_tokens)
    return input_ids, generated_positions


def suffix_after_prompt(text: str, prompt: str) -> str:
    marker = text.find(prompt)
    if marker < 0:
        return text
    return text[marker + len(prompt) :]


def repeated_ngram_count(words: list[str], n: int) -> int:
    if len(words) < n:
        return 0
    counts = Counter(tuple(words[index : index + n]) for index in range(len(words) - n + 1))
    return sum(count - 1 for count in counts.values() if count > 1)


def quality_score(text: str, prompt: str) -> float:
    suffix = suffix_after_prompt(text, prompt)
    words = re.findall(r"[A-Za-z][A-Za-z']*", suffix.lower())
    if not words:
        return -1_000.0
    word_counts = Counter(words)
    unique_fraction = len(word_counts) / float(len(words))
    top_fraction = max(word_counts.values()) / float(len(words))
    punctuation = re.findall(r"[^\w\s]", suffix)
    punctuation_fraction = len(punctuation) / max(1, len(suffix))
    bad_punctuation = len(re.findall(r"(?:\.{2,}|,{2,}|—|–|--|\s-[\s.,;:!?])", suffix))
    repeated_words = len(re.findall(r"\b([A-Za-z']+)(?:\s+\1\b)+", suffix.lower()))

    score = 0.8 * min(len(words), 80)
    score += 30.0 * unique_fraction
    score -= 35.0 * top_fraction
    score -= 120.0 * punctuation_fraction
    score -= 8.0 * bad_punctuation
    score -= 6.0 * repeated_words
    score -= 5.0 * repeated_ngram_count(words, 2)
    score -= 3.0 * repeated_ngram_count(words, 3)
    stripped = suffix.strip()
    if len(words) < 16:
        score -= 30.0
    if stripped and not re.search(r"[.!?][\"')\]]?$", stripped):
        score -= 4.0
    return score


def apply_uncertainty_penalties(
    uncertainty: np.ndarray,
    tokenizer: object,
    sampled: np.ndarray,
    blanks: np.ndarray | None,
    blank_commit_penalty: float,
    repeat_penalty: float,
    max_token_repeat_fraction: float,
    step: int,
) -> np.ndarray:
    adjusted = uncertainty.astype(np.float64, copy=True)
    if blanks is not None and step > 1:
        adjusted[blanks[sampled]] += blank_commit_penalty
    if repeat_penalty > 0:
        adjusted += repetition_penalty_scores(
            tokenizer=tokenizer,
            sampled=sampled,
            repeat_penalty=repeat_penalty,
            max_fraction=max_token_repeat_fraction,
        )
    return adjusted


def make_trace_frame(
    frame_index: int,
    tokens: np.ndarray,
    visible_generated: np.ndarray,
) -> TraceFrame:
    noise_percent = int(round(100.0 * (1.0 - visible_generated.mean())))
    return TraceFrame(
        frame_index=frame_index,
        noise_percent=noise_percent,
        tokens=tokens.copy(),
        visible_generated=visible_generated.copy(),
    )


def run_denoising(
    args: argparse.Namespace,
    tokenizer: object,
    model: object,
    device: str,
    rng: np.random.Generator,
    piece_penalties: np.ndarray | None,
    blanks: np.ndarray | None,
    specials: np.ndarray | None,
    vocab_size: int,
    noise_token_ids: np.ndarray,
    trace_passes: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[TraceFrame]]:
    import torch

    self_conditioning_strength = float(getattr(args, "self_conditioning_strength", 0.0))
    if not 0.0 <= self_conditioning_strength <= 1.0:
        raise ValueError("--self-conditioning-strength must be between 0 and 1")
    renoise_mode = resolve_renoise_mode(args.initial_noise, args.renoise)
    input_ids, generated_positions = build_input(
        tokenizer=tokenizer,
        prompt=args.prompt,
        new_tokens=args.new_tokens,
        initial_noise=args.initial_noise,
        rng=rng,
        noise_token_ids=noise_token_ids,
    )
    tokens = np.array(input_ids, dtype=np.int64)
    active_generated = np.ones((generated_positions.size,), dtype=bool)
    visible_generated = np.zeros((generated_positions.size,), dtype=bool)
    trace_frames: list[TraceFrame] = []
    if trace_passes is not None:
        trace_frames.append(make_trace_frame(0, tokens, visible_generated))

    with torch.no_grad():
        previous_argmax: np.ndarray | None = None
        embedding_layer = model.get_input_embeddings() if self_conditioning_strength > 0 else None
        previous_expected = None
        previous_expected_mask = None
        special_indices = None
        piece_penalties_tensor = None
        if specials is not None:
            special_indices = torch.tensor(np.flatnonzero(specials), dtype=torch.long, device=device)
        if piece_penalties is not None:
            piece_penalties_tensor = torch.tensor(piece_penalties, dtype=torch.float32, device=device)
        for step in range(args.steps, 0, -1):
            current_temperature = temperature_for_step(step, args.steps, args.temperature, args.final_temperature)
            if args.sampler == "ancestral":
                active_positions = generated_positions[active_generated]
                active_count = active_positions.size
                if active_count == 0:
                    break
            else:
                active_positions = generated_positions
                active_count = active_positions.size

            batch = torch.tensor(tokens[None, :], dtype=torch.long, device=device)
            if (
                embedding_layer is not None
                and previous_expected is not None
                and bool(previous_expected_mask.any().detach().cpu().item())
            ):
                input_embeddings = embedding_layer(batch)
                conditioned = previous_expected_mask[None, :, None]
                blended = (1.0 - self_conditioning_strength) * input_embeddings + (
                    self_conditioning_strength * previous_expected[None, :, :]
                )
                input_embeddings = torch.where(conditioned, blended, input_embeddings)
                outputs = model(inputs_embeds=input_embeddings)
            else:
                outputs = model(input_ids=batch)

            active_positions_tensor = torch.tensor(active_positions, dtype=torch.long, device=device)
            filtered_logits_tensor = apply_logit_filters_torch(
                outputs.logits[0, active_positions_tensor].float(),
                special_indices=special_indices,
                piece_penalties=piece_penalties_tensor,
            )
            if embedding_layer is not None:
                probs_tensor = torch.softmax(filtered_logits_tensor, dim=-1).to(dtype=embedding_layer.weight.dtype)
                expected = probs_tensor @ embedding_layer.weight
                if previous_expected is None:
                    previous_expected = torch.zeros(
                        (tokens.size, embedding_layer.weight.shape[-1]),
                        dtype=embedding_layer.weight.dtype,
                        device=device,
                    )
                    previous_expected_mask = torch.zeros((tokens.size,), dtype=torch.bool, device=device)
                previous_expected[active_positions_tensor] = expected
                previous_expected_mask[active_positions_tensor] = True
            logits = filtered_logits_tensor.detach().float().cpu().numpy()
            argmax_tokens = logits.argmax(axis=-1).astype(np.int64)
            sampled, confidence, entropy = sample_logits(
                logits,
                rng,
                temperature=current_temperature,
                top_k=args.top_k,
            )
            target_remaining = target_masked_after_step(
                reverse_step=step,
                steps=args.steps,
                total=generated_positions.size,
                schedule=args.unmask_schedule,
            )
            if args.sampler == "ancestral":
                target_remaining = min(target_remaining, active_count)
                commit_count = active_count - target_remaining
                if step == 1:
                    commit_count = active_count
                if commit_count > 0:
                    uncertainty = uncertainty_scores(
                        confidence,
                        entropy,
                        strategy=args.remask_strategy,
                        vocab_size=vocab_size,
                    )
                    uncertainty = apply_uncertainty_penalties(
                        uncertainty=uncertainty,
                        tokenizer=tokenizer,
                        sampled=sampled,
                        blanks=blanks,
                        blank_commit_penalty=args.blank_commit_penalty,
                        repeat_penalty=args.repeat_penalty,
                        max_token_repeat_fraction=args.max_token_repeat_fraction,
                        step=step,
                    )
                    commit_local = lowest_uncertainty_indices(uncertainty, commit_count)
                    tokens[active_positions[commit_local]] = sampled[commit_local]
                    active_local = np.flatnonzero(active_generated)
                    active_generated[active_local[commit_local]] = False
                    visible_generated = ~active_generated
                    if step > 1:
                        remaining_local = np.flatnonzero(active_generated)
                        renoise_positions(
                            tokens=tokens,
                            positions=generated_positions[remaining_local],
                            tokenizer=tokenizer,
                            rng=rng,
                            mode=renoise_mode,
                            noise_token_ids=noise_token_ids,
                        )
            else:
                if (
                    args.early_stop_entropy >= 0
                    and previous_argmax is not None
                    and float(entropy.mean()) <= args.early_stop_entropy
                    and np.array_equal(previous_argmax, argmax_tokens)
                ):
                    tokens[generated_positions] = argmax_tokens
                    visible_generated = np.ones_like(visible_generated, dtype=bool)
                    if trace_passes is not None:
                        trace_frames.append(make_trace_frame(len(trace_frames), tokens, visible_generated))
                    if getattr(args, "verbose", False):
                        print(
                            f"step={step} sampler={args.sampler} adaptive_stop_entropy={float(entropy.mean()):.6f}",
                            file=sys.stderr,
                        )
                    break
                previous_argmax = argmax_tokens.copy()
                tokens[generated_positions] = sampled
                visible_generated = np.ones((generated_positions.size,), dtype=bool)
                scheduled_commit = generated_positions.size - min(target_remaining, generated_positions.size)
                if args.entropy_bound > 0:
                    commit_count = entropy_bound_commit_count(
                        entropy=entropy,
                        entropy_bound=args.entropy_bound,
                        min_commit=scheduled_commit,
                    )
                    remask_count = generated_positions.size - commit_count
                else:
                    remask_count = min(target_remaining, generated_positions.size)
                if step > 1 and remask_count > 0:
                    uncertainty = uncertainty_scores(
                        confidence,
                        entropy,
                        strategy=args.remask_strategy,
                        vocab_size=vocab_size,
                    )
                    uncertainty = apply_uncertainty_penalties(
                        uncertainty=uncertainty,
                        tokenizer=tokenizer,
                        sampled=sampled,
                        blanks=blanks,
                        blank_commit_penalty=args.blank_commit_penalty,
                        repeat_penalty=args.repeat_penalty,
                        max_token_repeat_fraction=args.max_token_repeat_fraction,
                        step=step,
                    )
                    remask_local = highest_uncertainty_indices(uncertainty, remask_count)
                    visible_generated[remask_local] = False
                    renoise_positions(
                        tokens=tokens,
                        positions=generated_positions[remask_local],
                        tokenizer=tokenizer,
                        rng=rng,
                        mode=renoise_mode,
                        noise_token_ids=noise_token_ids,
                    )
                commit_count = active_count - remask_count
            if getattr(args, "verbose", False):
                if args.sampler == "ancestral":
                    remaining = int(np.count_nonzero(active_generated))
                else:
                    remaining = int((tokens[generated_positions] == tokenizer.mask_token_id).sum())
                print(
                    f"step={step} sampler={args.sampler} unmasked={commit_count} masked_remaining={remaining}",
                    file=sys.stderr,
                )
            if trace_passes is not None:
                pass_index = args.steps - step + 1
                if pass_index in trace_passes or step == 1:
                    trace_frames.append(make_trace_frame(len(trace_frames), tokens, visible_generated))

    if trace_passes is not None and trace_frames and not trace_frames[-1].visible_generated.all():
        visible_generated = np.ones_like(visible_generated, dtype=bool)
        trace_frames.append(make_trace_frame(len(trace_frames), tokens, visible_generated))
    return tokens, generated_positions, trace_frames


def generate_once(
    args: argparse.Namespace,
    tokenizer: object,
    model: object,
    device: str,
    rng: np.random.Generator,
    piece_penalties: np.ndarray | None,
    blanks: np.ndarray | None,
    specials: np.ndarray | None,
    vocab_size: int,
    noise_token_ids: np.ndarray,
) -> str:
    tokens, _generated_positions, _trace_frames = run_denoising(
        args=args,
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
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.eval()

    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    piece_penalties = piece_logit_penalties(tokenizer, vocab_size, args.piece_logit_penalty)
    blanks = blank_token_mask(tokenizer, vocab_size) if args.blank_commit_penalty > 0 else None
    specials = special_token_mask(tokenizer, vocab_size) if not args.keep_specials else None
    noise_token_ids = valid_uniform_noise_ids(tokenizer, vocab_size)

    best_text = ""
    best_score = -math.inf
    best_seed = args.seed
    candidates: list[dict[str, float | int]] = []
    for candidate_index in range(args.candidate_count):
        candidate_seed = args.seed + candidate_index * args.candidate_seed_stride
        rng = np.random.default_rng(candidate_seed)
        text = generate_once(
            args=args,
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
        score = quality_score(text, args.prompt)
        candidates.append({"index": candidate_index, "seed": candidate_seed, "quality_score": float(score)})
        if args.verbose and args.candidate_count > 1:
            print(f"candidate={candidate_index} seed={candidate_seed} quality_score={score:.3f}", file=sys.stderr)
        if score > best_score:
            best_text = text
            best_score = score
            best_seed = candidate_seed

    if args.verbose and args.candidate_count > 1:
        print(f"selected_seed={best_seed} quality_score={best_score:.3f}", file=sys.stderr)
    if args.selection_out:
        selection_path = Path(args.selection_out)
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(
            json.dumps(
                {
                    "candidate_count": args.candidate_count,
                    "seed": args.seed,
                    "candidate_seed_stride": args.candidate_seed_stride,
                    "selected_seed": best_seed,
                    "selected_quality_score": float(best_score),
                    "initial_noise": args.initial_noise,
                    "renoise": args.initial_noise if args.renoise == "same" else args.renoise,
                    "candidates": candidates,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(best_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
