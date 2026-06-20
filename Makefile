PYTHON ?= python3
VENV ?= .venv
PY ?= $(VENV)/bin/python

TOKEN_CORPUS ?= data/infinite_jest.txt
RAW_OUT ?= data/raw/infinite_jest
URL ?= https://raisuman123.wordpress.com/wp-content/uploads/2013/05/david-foster-wallace-infinite-jest-v2-0.pdf

PROMPT ?= I am seated in an office, surrounded by heads and bodies.
NEW_TOKENS ?= 48
SAMPLE_STEPS ?= 48
TEMPERATURE ?= 0.8
FINAL_TEMPERATURE ?= 0.4
TOP_K ?= 64

HF_MASKED_MODEL ?= outputs/roberta-large-infinite-jest-mdlm-subs-selfcond-step500-preserved
HF_BASE_MODEL ?= outputs/roberta-large-infinite-jest-uniform-mixed-1k
HF_DIFFUSION_OUT ?= outputs/roberta-large-infinite-jest-mdlm-subs-selfcond
HF_DIFFUSION_STEPS ?= 1000
HF_DIFFUSION_BATCH_SIZE ?= 4
HF_DIFFUSION_SEQ_LEN ?= 256
HF_DIFFUSION_PREFIX_LEN ?= 64
HF_DIFFUSION_LR ?= 2e-5
HF_DIFFUSION_SEED ?= 61

HF_OBJECTIVE ?= mdlm-subs
HF_LOSS_WEIGHTING ?= mdlm
HF_MAX_LOSS_WEIGHT ?= 8.0
HF_SELF_CONDITIONING_PROB ?= 0.25
HF_SELF_CONDITIONING_STRENGTH ?= 0.5
HF_MASK_DISTRIBUTION ?= high
HF_FULL_MASK_FRACTION ?= 0.35
HF_HIGH_MASK_POWER ?= 2.0
HF_CORRUPTION ?= mixed
HF_UNIFORM_CORRUPTION_FRACTION ?= 0.75

HF_SAMPLER ?= refine
HF_INITIAL_NOISE ?= uniform
HF_RENOISE ?= uniform
HF_UNMASK_SCHEDULE ?= cosine
HF_REMASK_STRATEGY ?= entropy
HF_REPEAT_PENALTY ?= 3.5
HF_MAX_TOKEN_REPEAT_FRACTION ?= 0.12
HF_ENTROPY_BOUND ?= 0.1
HF_EARLY_STOP_ENTROPY ?= 0.005
HF_PIECE_LOGIT_PENALTY ?= 1.2
HF_CANDIDATE_COUNT ?= 1
HF_CANDIDATE_SEED_STRIDE ?= 9973

HF_GIF_OUT ?= assets/token-diffusion-mdlm-subs-selfcond-step500.gif
HF_GIF_FRAMES ?= 16
HF_GIF_SEED ?= 10074
HF_GIF_MIN_QUALITY_SCORE ?= 20
HF_GIF_MIN_GENERATED_WORDS ?= 30

HF_ABLATION_OUT ?= outputs/evals/hf-diffusion-ablations-mdlm-subs-step500.json
HF_ABLATION_SEEDS ?= 101,10074,20047
HF_ABLATION_SETTINGS ?= mask-refine,uniform-refine,uniform-selfcond

.PHONY: setup fetch train sample gif ablations

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

fetch:
	$(PY) scripts/fetch_corpus.py --url "$(URL)" --out "$(RAW_OUT)" --text-out "$(TOKEN_CORPUS)"

train:
	$(PY) scripts/train_hf_masked_diffusion.py --corpus "$(TOKEN_CORPUS)" --out "$(HF_DIFFUSION_OUT)" --model "$(HF_BASE_MODEL)" --steps $(HF_DIFFUSION_STEPS) --batch-size $(HF_DIFFUSION_BATCH_SIZE) --seq-len $(HF_DIFFUSION_SEQ_LEN) --prefix-len $(HF_DIFFUSION_PREFIX_LEN) --learning-rate $(HF_DIFFUSION_LR) --seed $(HF_DIFFUSION_SEED) --objective $(HF_OBJECTIVE) --loss-weighting $(HF_LOSS_WEIGHTING) --max-loss-weight $(HF_MAX_LOSS_WEIGHT) --self-conditioning-prob $(HF_SELF_CONDITIONING_PROB) --self-conditioning-strength $(HF_SELF_CONDITIONING_STRENGTH) --mask-distribution $(HF_MASK_DISTRIBUTION) --full-mask-fraction $(HF_FULL_MASK_FRACTION) --high-mask-power $(HF_HIGH_MASK_POWER) --corruption $(HF_CORRUPTION) --uniform-corruption-fraction $(HF_UNIFORM_CORRUPTION_FRACTION)

sample:
	$(PY) scripts/sample_hf_masked_diffusion.py --model "$(HF_MASKED_MODEL)" --prompt "$(PROMPT)" --new-tokens $(NEW_TOKENS) --steps $(SAMPLE_STEPS) --temperature $(TEMPERATURE) --final-temperature $(FINAL_TEMPERATURE) --top-k $(TOP_K) --sampler $(HF_SAMPLER) --initial-noise $(HF_INITIAL_NOISE) --renoise $(HF_RENOISE) --unmask-schedule $(HF_UNMASK_SCHEDULE) --remask-strategy $(HF_REMASK_STRATEGY) --blank-commit-penalty 6.0 --repeat-penalty $(HF_REPEAT_PENALTY) --max-token-repeat-fraction $(HF_MAX_TOKEN_REPEAT_FRACTION) --entropy-bound $(HF_ENTROPY_BOUND) --early-stop-entropy $(HF_EARLY_STOP_ENTROPY) --piece-logit-penalty $(HF_PIECE_LOGIT_PENALTY) --self-conditioning-strength $(HF_SELF_CONDITIONING_STRENGTH) --candidate-count $(HF_CANDIDATE_COUNT) --candidate-seed-stride $(HF_CANDIDATE_SEED_STRIDE)

gif:
	$(PY) scripts/make_hf_diffusion_gif.py --model "$(HF_MASKED_MODEL)" --out "$(HF_GIF_OUT)" --prompt "$(PROMPT)" --new-tokens $(NEW_TOKENS) --steps $(SAMPLE_STEPS) --frames $(HF_GIF_FRAMES) --temperature $(TEMPERATURE) --final-temperature $(FINAL_TEMPERATURE) --top-k $(TOP_K) --sampler $(HF_SAMPLER) --initial-noise $(HF_INITIAL_NOISE) --renoise $(HF_RENOISE) --unmask-schedule $(HF_UNMASK_SCHEDULE) --remask-strategy $(HF_REMASK_STRATEGY) --blank-commit-penalty 6.0 --repeat-penalty $(HF_REPEAT_PENALTY) --max-token-repeat-fraction $(HF_MAX_TOKEN_REPEAT_FRACTION) --entropy-bound $(HF_ENTROPY_BOUND) --early-stop-entropy $(HF_EARLY_STOP_ENTROPY) --piece-logit-penalty $(HF_PIECE_LOGIT_PENALTY) --self-conditioning-strength $(HF_SELF_CONDITIONING_STRENGTH) --seed $(HF_GIF_SEED) --min-quality-score $(HF_GIF_MIN_QUALITY_SCORE) --min-generated-words $(HF_GIF_MIN_GENERATED_WORDS)

ablations:
	$(PY) scripts/evaluate_hf_diffusion_ablations.py --model "$(HF_MASKED_MODEL)" --out "$(HF_ABLATION_OUT)" --seeds "$(HF_ABLATION_SEEDS)" --settings "$(HF_ABLATION_SETTINGS)" --new-tokens $(NEW_TOKENS) --steps $(SAMPLE_STEPS) --temperature $(TEMPERATURE) --final-temperature $(FINAL_TEMPERATURE) --top-k $(TOP_K) --unmask-schedule $(HF_UNMASK_SCHEDULE) --remask-strategy $(HF_REMASK_STRATEGY) --blank-commit-penalty 6.0 --repeat-penalty $(HF_REPEAT_PENALTY) --max-token-repeat-fraction $(HF_MAX_TOKEN_REPEAT_FRACTION) --entropy-bound $(HF_ENTROPY_BOUND) --early-stop-entropy $(HF_EARLY_STOP_ENTROPY) --piece-logit-penalty $(HF_PIECE_LOGIT_PENALTY) --self-conditioning-strength $(HF_SELF_CONDITIONING_STRENGTH)
