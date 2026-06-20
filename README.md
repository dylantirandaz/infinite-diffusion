# Masked Diffusion LM Lab

This repo contains my current Infinite Jest diffusion language model experiment.

The active model starts from RoBERTa-large and gets posttrained on `data/infinite_jest.txt` with a masked denoising objective. It is diffusion-only: no causal attention mask and no next-token prediction objective.

The Infinite Jest corpus is doing the style work. There is no DFW prompt template, style-transfer layer, or postprocessing pass. Every clean target in training comes from the book, and the model learns to reconstruct those targets from corrupted versions of the same text. That is how the run pushes the denoiser toward the book's local distribution: long clauses, bureaucratic flatness, tennis/AA/institutional vocabulary, strange compression, and the kind of sentence drift that shows up before the model falls apart.

The result should be read narrowly. This is not a general language model and it is not trying to recover exact paragraphs from the book. It is a small RoBERTa-style denoiser posttrained into a text diffusion generator whose samples are conditioned by the Infinite Jest training distribution.

Current checkpoint:

```text
outputs/roberta-large-infinite-jest-mdlm-subs-selfcond-step500-preserved
```

Current write-up:

[docs/current-dlm-writeup.md](docs/current-dlm-writeup.md)

Current GIFs:

![Office prompt diffusion trace](assets/token-diffusion-mdlm-subs-selfcond-step500.gif)

![Tennis prompt diffusion trace](assets/token-diffusion-mdlm-subs-selfcond-tennis.gif)

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Corpus

Fetch or extract the authorized Infinite Jest text into the training path:

```bash
make fetch
```

This downloads or extracts the authorized source into:

```text
data/infinite_jest.txt
```

## Train

The current training target posttrains the previous Infinite Jest diffusion checkpoint and writes a new HF masked-diffusion checkpoint.

```bash
make train
```

Default training configuration:

```text
objective: mdlm-subs
corruption: mixed
uniform corruption fraction: 0.75
mask distribution: high
full mask fraction: 0.35
loss weighting: mdlm
self-conditioning probability: 0.25
self-conditioning strength: 0.5
```

During posttraining, sampled continuation spans from `data/infinite_jest.txt` are corrupted with masks and random vocabulary tokens. The model sees the corrupted span and learns to predict the original book tokens at the corrupted positions. At sampling time, a prompt is held fixed and the continuation canvas is refined in parallel over repeated denoising steps.

## Sample

```bash
make sample
```

Override the prompt if needed:

```bash
make sample PROMPT="Serious juniors never pick up tennis balls with their hands."
```

The sampler uses uniform canvas initialization, uniform re-noising, entropy-based retention, cosine unmasking, and self-conditioning.

## GIF

```bash
make gif
```

The default output is:

```text
assets/token-diffusion-mdlm-subs-selfcond-step500.gif
```

## Ablations

```bash
make ablations
```

The ablation script compares:

```text
mask-refine
uniform-refine
uniform-selfcond
```

The latest report is:

```text
outputs/evals/hf-diffusion-ablations-mdlm-subs-step500.json
```
