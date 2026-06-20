# Masked diffusion language modeling on Infinite Jest

I wanted this to be a diffusion language model in the literal sense I cared about: not a chatbot wrapped in diffusion language, and not a next-token model with a different sampler. I treat generation as iterative denoising over a fixed token canvas. I hold a prompt fixed, initialize the continuation span as noise, and use a bidirectional masked language model to repeatedly predict cleaner token states for the whole span.

The checkpoint I used for the results below is:

```text
outputs/roberta-large-infinite-jest-mdlm-subs-selfcond-step500-preserved
```

The base architecture is RoBERTa-large. I posttrained it on `data/infinite_jest.txt`. The style is not added with a prompt, a hand-written template, or a postprocessing pass. It comes from the training distribution: the denoising targets are book tokens from Infinite Jest. Whatever DFW-like behavior shows up in the samples comes from pushing the masked denoiser toward that corpus.

The resulting model should be described narrowly: a RoBERTa-style masked denoiser posttrained into a small text diffusion model over one literary corpus. It is not a from-scratch foundation model.

## Basic formulation

Let `x0` be a clean token sequence from the corpus. I split the sequence into a visible prefix and a continuation region. The prefix is left unchanged. The continuation region is corrupted according to a sampled noise rate `t`.

The model receives the corrupted sequence `xt` and predicts the original clean book tokens at corrupted continuation positions:

```text
model input:      visible prefix + corrupted continuation
model target:     clean continuation tokens at corrupted positions
attention mask:   bidirectional
training loss:    denoising cross entropy, not causal LM loss
```

The denoiser is trained to estimate `p(x0 | xt, t)` for the corrupted positions. During generation, I reverse the process approximately: start from a noisy continuation and repeatedly apply the denoiser until the canvas stabilizes.

This is the part I care about most. A causal LM represents text as:

```text
p(x1, ..., xn) = product_i p(x_i | x_<i)
```

The diffusion sampler instead keeps a full continuation canvas in memory and revises positions in parallel.

## Corruption process

I used mixed discrete corruption. Selected continuation positions are corrupted in one of two ways:

```text
selected token -> [MASK]
selected token -> random vocabulary token
unselected token -> original token
```

The random-token branch is important because the sampler does not begin from pure `[MASK]` tokens. It begins from uniform vocabulary noise. A model trained only on mask replacement can learn a fill-in-the-blank task that is easier than the sampling problem. Mixed corruption moves training closer to the inference distribution I actually use.

The training configuration for this checkpoint is:

```text
objective: mdlm-subs
corruption: mixed
uniform corruption fraction: 0.75
mask distribution: high
full mask fraction: 0.35
loss weighting: mdlm
max loss weight: 8.0
prefix length: 64 tokens
sequence length: 256 tokens
self-conditioning probability: 0.25
self-conditioning strength: 0.5
```

The `mdlm-subs` setting removes special tokens and `<mask>` from the clean prediction space. The target is always a real Infinite Jest token. I reweight the loss by the sampled noise rate with a cap. This keeps high-noise and low-noise denoising cases in the objective without allowing very small corruption rates to dominate the update.

## Checkpoint selection

I continued the run to 1000 steps, but validation selected the earlier checkpoint.

```text
step 500   validation loss 4.8792
step 1000  validation loss 5.2819
```

The later checkpoint had worse validation loss under the same denoising objective. I use the step-500 checkpoint for the samples and animations in this note.

## Sampling procedure

The sampler I used here is full-canvas refinement. It initializes the continuation region with uniform token noise. Each reverse step predicts every generated position. Positions with lower uncertainty are retained. Positions with higher uncertainty are replaced with fresh noise and predicted again.

The sampler configuration is:

```text
sampler: refine
initial noise: uniform
re-noise: uniform
steps: 48
top-k: 64
temperature: 0.8 to 0.4
unmask schedule: cosine
remask strategy: entropy
self-conditioning strength: 0.5
```

I measure uncertainty from the predicted distribution at each generated position. In this run, entropy is used for remasking. Low-entropy positions are treated as more stable. High-entropy positions are treated as unresolved and are re-noised.

Self-conditioning adds a deterministic state signal between reverse steps. The previous step's predicted token distribution is projected through the embedding table to form an expected token embedding. That embedding is blended into the next denoising pass at generated positions. The model still predicts all positions in parallel. Self-conditioning does not introduce a left-to-right factorization.

## Trace: office prompt

The first animation uses a one-sentence prompt from the corpus. I keep the prompt fixed and sample the continuation by iterative denoising.

![Office prompt diffusion trace](../assets/token-diffusion-mdlm-subs-selfcond-step500.gif)

Final text from the trace:

```text
I am seated in an office, surrounded by heads and bodies. I have been there for a few short hours. My desk's right next to the door of the room. The door's white, and slightly deformed, and the back of a chair. It's clean, bright, and tidy.
```

I read this as conditional generation in the book's learned local style. It is not an attempt to recover the original paragraph.

## Trace: tennis prompt

The second animation uses a different sentence from the corpus. I included it because it shows the model honestly. It can stay near the topic and local style for a short span, then it degrades into brittle phrase structure and acronym-like fragments.

![Tennis prompt diffusion trace](../assets/token-diffusion-mdlm-subs-selfcond-tennis.gif)

This behavior is expected for the scale of the run. The model has a narrow corpus, a short posttraining schedule, and a heuristic reverse process. I still find the trace useful because it exposes the denoising dynamics directly. The failure happens inside the canvas refinement process, not after a hidden postprocessing step.

## Sampler ablation

I evaluated three samplers on the same checkpoint. The sweep used two prompts, three seeds, and 48 denoising steps.

| sampler | quality | unique word fraction | repeated bigrams |
| --- | ---: | ---: | ---: |
| mask-refine | -3.755 | 0.594 | 6.17 |
| uniform-refine | 26.132 | 0.722 | 2.17 |
| uniform-selfcond | 27.258 | 0.736 | 2.17 |

The quality score is a local diagnostic. It rewards generated length and lexical diversity and penalizes repetition and punctuation collapse. It is not a general language-model benchmark. In this setting I treat it as useful because the ranking matches sample inspection.

The main observation is that uniform-state refinement outperforms mask-only refinement. This is consistent with the training setup. The model was exposed to random-token corruption, and the sampler uses random-token initialization and re-noising. Self-conditioning gives a smaller gain on this sweep.

## Interpretation

I think the experiment supports a narrow claim:

```text
A masked language model can be posttrained into a small text diffusion generator by training it to denoise variable-rate mask and uniform-token corruption, then sampling with full-canvas refinement.
```

The strongest implementation details in this run were mixed corruption, high-noise continuation training, uniform re-noising, entropy-based token retention, and self-conditioning.

The result is not broad fluency. The model can produce short locally coherent continuations that have some of the Infinite Jest distribution in them, but it still repeats, loses syntax, and collapses into malformed fragments. The useful property is structural: I can represent generation as refinement of a noisy canvas rather than as next-token prediction.
