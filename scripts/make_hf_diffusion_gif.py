#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_hf_masked_diffusion import (
    TraceFrame,
    blank_token_mask,
    choose_device,
    piece_logit_penalties,
    quality_score,
    run_denoising,
    special_token_mask,
    valid_uniform_noise_ids,
)


WIDTH = 1120
HEIGHT = 620
MARGIN = 58
FONT_BODY = "/System/Library/Fonts/Supplemental/Iowan Old Style.ttc"
FONT_SANS = "/System/Library/Fonts/Avenir Next.ttc"

PAPER = (249, 248, 243)
INK = (28, 32, 33)
RULE = (42, 55, 54)
MUTED = (99, 111, 108)
LIGHT = (179, 184, 176)
FAINT = (224, 224, 216)
PROGRESS = (72, 94, 86)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal GIF from a Hugging Face masked-LM diffusion trace.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="assets/token-diffusion-hf.gif")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--final-temperature", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--sampler", choices=("ancestral", "refine"), default="refine")
    parser.add_argument("--initial-noise", choices=("mask", "uniform"), default="mask")
    parser.add_argument("--renoise", choices=("same", "mask", "uniform"), default="same")
    parser.add_argument("--unmask-schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--remask-strategy", choices=("confidence", "entropy", "hybrid"), default="entropy")
    parser.add_argument("--blank-commit-penalty", type=float, default=6.0)
    parser.add_argument("--repeat-penalty", type=float, default=3.5)
    parser.add_argument("--max-token-repeat-fraction", type=float, default=0.12)
    parser.add_argument("--entropy-bound", type=float, default=0.1)
    parser.add_argument("--early-stop-entropy", type=float, default=0.005)
    parser.add_argument("--piece-logit-penalty", type=float, default=1.2)
    parser.add_argument("--self-conditioning-strength", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--min-quality-score", type=float, default=0.0)
    parser.add_argument("--min-generated-words", type=int, default=24)
    return parser.parse_args()


def load_font(path: str, size: int, fallback: str = "DejaVuSerif.ttf") -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.truetype(fallback, size)


def capture_passes(steps: int, frames: int) -> set[int]:
    frames = max(1, min(frames, steps))
    return {max(1, min(steps, int(round(index * steps / frames)))) for index in range(1, frames + 1)}


def generate_states(args: argparse.Namespace) -> tuple[object, np.ndarray, np.ndarray, list[TraceFrame], str]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.eval()

    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    noise_token_ids = valid_uniform_noise_ids(tokenizer, vocab_size)
    piece_penalties = piece_logit_penalties(tokenizer, vocab_size, args.piece_logit_penalty)
    blanks = blank_token_mask(tokenizer, vocab_size) if args.blank_commit_penalty > 0 else None
    specials = special_token_mask(tokenizer, vocab_size)

    tokens, generated_positions, frames = run_denoising(
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
        trace_passes=capture_passes(args.steps, args.frames),
    )
    final_text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    validate_final_text(final_text, args.prompt, args.min_quality_score, args.min_generated_words)
    return tokenizer, generated_positions, tokens.copy(), frames, final_text


def validate_final_text(text: str, prompt: str, min_quality_score: float, min_generated_words: int) -> None:
    suffix = text[text.find(prompt) + len(prompt) :] if prompt in text else text
    words = re.findall(r"[A-Za-z][A-Za-z']*", suffix)
    score = quality_score(text, prompt)
    if len(words) < min_generated_words or score < min_quality_score:
        raise ValueError(
            "generated span is not good enough for a GIF: "
            f"words={len(words)} quality_score={score:.3f} min_quality_score={min_quality_score:.3f}"
        )


def printable(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return "".join(ch for ch in text if ch.isprintable())


def masked_piece(tokenizer: object, final_id: int) -> str:
    piece = printable(tokenizer.decode([int(final_id)], clean_up_tokenization_spaces=False))
    leading = " " if piece.startswith(" ") else ""
    body = piece.strip()
    width = max(2, min(9, len(body) if body else len(piece)))
    return leading + ("_" * width)


def state_text(
    tokenizer: object,
    frame: TraceFrame,
    final_tokens: np.ndarray,
    generated_positions: np.ndarray,
) -> str:
    generated_lookup = {int(position): index for index, position in enumerate(generated_positions.tolist())}
    special_ids = {int(token_id) for token_id in (tokenizer.all_special_ids or [])}
    pieces = []
    for position, token_id in enumerate(frame.tokens.tolist()):
        generated_index = generated_lookup.get(position)
        if generated_index is not None and not bool(frame.visible_generated[generated_index]):
            pieces.append(masked_piece(tokenizer, int(final_tokens[position])))
        elif generated_index is None and int(token_id) in special_ids:
            continue
        else:
            pieces.append(printable(tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)))
    return "".join(pieces).strip()


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def draw_frame(
    tokenizer: object,
    frame: TraceFrame,
    final_tokens: np.ndarray,
    generated_positions: np.ndarray,
    total_frames: int,
    sampler: str,
    initial_noise: str,
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((MARGIN, 44), "diffusion trace", font=fonts["label"], fill=MUTED)
    draw.text((WIDTH - 250, 44), f"pass {frame.frame_index:02d}/{total_frames}", font=fonts["label"], fill=MUTED)
    draw.text((WIDTH - 250, 72), f"{frame.noise_percent:03d}% noisy", font=fonts["label"], fill=MUTED)
    draw.line((MARGIN, 112, WIDTH - MARGIN, 112), fill=RULE, width=2)

    text = state_text(tokenizer, frame, final_tokens, generated_positions)
    y = 145
    for line_index, line in enumerate(wrap_text(text, 84)[:12]):
        draw.text((MARGIN, y), line, font=fonts["body"], fill=INK if line_index < 10 else LIGHT)
        y += 33

    progress_left = MARGIN
    progress_top = HEIGHT - 78
    progress_width = WIDTH - 2 * MARGIN
    draw.line((progress_left, progress_top, progress_left + progress_width, progress_top), fill=FAINT, width=4)
    draw.line(
        (
            progress_left,
            progress_top,
            progress_left + int(progress_width * frame.frame_index / max(1, total_frames)),
            progress_top,
        ),
        fill=PROGRESS,
        width=4,
    )
    draw.text((MARGIN, HEIGHT - 50), footer_text(sampler, initial_noise), font=fonts["small"], fill=MUTED)
    return image


def footer_text(sampler: str, initial_noise: str) -> str:
    if initial_noise == "uniform":
        return "the canvas starts as random tokens; uncertain positions are re-noised and revised together"
    if sampler == "ancestral":
        return "masked positions are predicted together, then high-confidence tokens are carried forward"
    return "masked positions are predicted together, then uncertain tokens are masked again"


def main() -> int:
    args = parse_args()
    tokenizer, generated_positions, final_tokens, states, final_text = generate_states(args)
    fonts = {
        "body": load_font(FONT_BODY, 25),
        "label": load_font(FONT_SANS, 15, fallback="DejaVuSans.ttf"),
        "small": load_font(FONT_SANS, 16, fallback="DejaVuSans.ttf"),
    }
    total_frames = len(states) - 1
    images = [
        draw_frame(
            tokenizer=tokenizer,
            frame=frame,
            final_tokens=final_tokens,
            generated_positions=generated_positions,
            total_frames=total_frames,
            sampler=args.sampler,
            initial_noise=args.initial_noise,
            fonts=fonts,
        )
        for frame in states
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [650] * (len(images) - 1) + [1700]
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(out)
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
