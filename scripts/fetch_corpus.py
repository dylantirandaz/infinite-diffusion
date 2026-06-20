#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or copy an authorized text/PDF corpus and extract text.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Authorized URL to a .txt or .pdf corpus.")
    source.add_argument("--input", help="Local .txt or .pdf corpus path.")
    parser.add_argument("--out", default="data/raw/source", help="Raw downloaded/copied output path.")
    parser.add_argument("--text-out", default="data/corpus.txt", help="Extracted UTF-8 text output path.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def infer_suffix(url: Optional[str], input_path: Optional[Path], content_type: str) -> str:
    if input_path and input_path.suffix:
        return input_path.suffix.lower()
    if url:
        suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
        if suffix:
            return suffix
    if "pdf" in content_type.lower():
        return ".pdf"
    return ".txt"


def with_suffix_if_missing(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path
    return path.with_suffix(suffix)


def download(url: str, target_base: Path, force: bool) -> Path:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "diffusion-lm-lab/0.1 (+local research corpus ingest)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_type = response.headers.get("content-type", "")
        suffix = infer_suffix(url, None, content_type)
        target = with_suffix_if_missing(target_base, suffix)
        if target.exists() and not force:
            raise FileExistsError(f"{target} already exists; pass --force to overwrite")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return target


def copy_input(source: Path, target_base: Path, force: bool) -> Path:
    if not source.exists():
        raise FileNotFoundError(source)
    target = with_suffix_if_missing(target_base, source.suffix.lower() or ".txt")
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires pypdf; run `.venv/bin/python -m pip install -r requirements.txt`.") from exc

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
        if index % 100 == 0:
            print(f"extracted_pages={index}", file=sys.stderr)
    return "\n\n".join(pages)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    raw_base = Path(args.out)
    text_out = Path(args.text_out)

    if args.url:
        raw_path = download(args.url, raw_base, args.force)
        source = {"url": args.url}
    else:
        raw_path = copy_input(Path(args.input), raw_base, args.force)
        source = {"input": str(Path(args.input).resolve())}

    text = normalize_text(extract_text(raw_path))
    if len(text.encode("utf-8")) < 1024:
        raise ValueError("Extracted text is unexpectedly small; check the source file.")

    if text_out.exists() and not args.force:
        raise FileExistsError(f"{text_out} already exists; pass --force to overwrite")
    text_out.parent.mkdir(parents=True, exist_ok=True)
    text_out.write_text(text, encoding="utf-8")

    metadata = {
        **source,
        "raw_path": str(raw_path),
        "text_path": str(text_out),
        "raw_bytes": raw_path.stat().st_size,
        "text_bytes": len(text.encode("utf-8")),
        "characters": len(text),
    }
    metadata_path = text_out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"raw={raw_path} text={text_out} text_bytes={metadata['text_bytes']} metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
