# GFlowNet Internship Project

Main project workspace.

## Recommended structure

- `main.tex` contains the literature review and working notes
- `Coding/` contains the dependency manifests and project experiments

Suggested layout:

```text
gflownet-internship-project/
├── main.tex
├── README.md
└── Coding/
    ├── envs/
    │   ├── gflownet-cpu-requirements.txt
    │   ├── gflownet-cpu-working-requirements.txt
    │   ├── chemgymrl-requirements.txt
    │   └── chemgymrl-working-requirements.txt
    ├── chemgymrl_gflownet/
    ├── recursionpharma-gflownet/
    └── chemgymrl/
```

`chemgymrl_gflownet` is the project-owned integration package. The
`recursionpharma-gflownet` and `chemgymrl` directories are local clones of the
upstream projects and are therefore excluded from this repository.

## Reproducing the GFlowNet dependency

The exact upstream revision and the small local compatibility patch are stored
in `Coding/upstream/`. From `Coding/`, prepare the dependency with:

```bash
git clone https://github.com/recursionpharma/gflownet.git recursionpharma-gflownet
cd recursionpharma-gflownet
git checkout da999404e997a302a81773eb183b1da6ec2a4449
git apply --unidiff-zero ../upstream/recursionpharma-gflownet-local-fixes.patch
```

The patch corrects Flow Matching initialisation and disables gradient tracking
during model-based trajectory sampling. It does not alter either loss equation.

## Python setup

Use a virtual environment, and do not install into the system Python.

For this project, Python `3.10` or `3.11` is the safer choice.

Your current system `python3` is `3.13`, which may be fine for simple work, but it is not the best match for the GFlowNet repo.

Basic setup:

```bash
cd Coding
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Then in VS Code, select the interpreter from `Coding/.venv`.

## Package lists

There are four package lists in `Coding/envs/`.

- `gflownet-cpu-requirements.txt`
  This is the direct package list based on the Recursion GFlowNet repo.
- `gflownet-cpu-working-requirements.txt`
  This is the pinned working CPU setup to try first on this laptop.
- `chemgymrl-requirements.txt`
  This is the direct package list based on the ChemGymRL repo.
- `chemgymrl-working-requirements.txt`
  This records the ChemGymRL package versions used in the working environment.

These files are reference lists. They are there so the setup is written down clearly in one place.

## GFlowNet install notes

The Recursion repo uses PyTorch Geometric packages such as `torch-scatter`, `torch-sparse`, and `torch-cluster`.

These often need prebuilt wheels matched to the exact PyTorch version, so the install command uses the PyG wheel index to find compatible CPU builds.

Recommended pattern:

```bash
cd Coding/recursionpharma-gflownet
python -m pip install -r ../envs/gflownet-cpu-working-requirements.txt --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html
python -m pip install -e . --no-deps
```

If the repo dependencies fight with the local project environment, use a separate venv for each repo.

## ChemGymRL install notes

ChemGymRL is lighter and should be easier to install in a standard CPU environment.

If you want to use the RL training scripts in that repo, you will also want:

```bash
pip install stable-baselines3
```

## Upper-tail experiment

The Wang-inspired proof of concept changes the positive terminal target without
changing the Trajectory Balance loss. The exact target sweep can be reproduced
from the saved exhaustive table with:

```bash
Coding/.venv/bin/python Coding/chemgymrl_gflownet/scripts/analyze_upper_tail_targets.py \
  --reference-csv Coding/chemgymrl_gflownet/results/exhaustive_reference_2026-07-28/procedures.csv \
  --output-dir Coding/chemgymrl_gflownet/results/upper_tail_target_sweep_threshold010 \
  --reward-scale 0.05 --tail-threshold 0.10 \
  --strengths 0 0.1 0.25 0.5 0.75 1.0 --selected-strength 0.25
```

A matched no-replay training run uses:

```bash
Coding/.venv/bin/python Coding/chemgymrl_gflownet/scripts/train_procedure_gflownet.py \
  --steps 1000 --batch-size 32 --seed 0 --device cpu \
  --reward-scale 0.05 --tail-threshold 0.10 --tail-strength 0.25 \
  --validate-every 100 --validation-batches 1 --final-batches 32 \
  --log-dir Coding/chemgymrl_gflownet/logs/upper_tail_seed0
```

The controlled follow-up keeps that target fixed and adds four discovered tail
and four discovered non-tail trajectories to each active training update:

```bash
Coding/.venv/bin/python Coding/chemgymrl_gflownet/scripts/train_procedure_gflownet.py \
  --steps 1000 --batch-size 32 --seed 0 --device cpu \
  --reward-scale 0.05 --tail-threshold 0.10 --tail-strength 0.25 \
  --fixed-tail-replay --replay-batch-size 8 --replay-capacity 512 \
  --fixed-tail-replay-fraction 0.5 \
  --validate-every 100 --validation-batches 1 --final-batches 32 \
  --log-dir Coding/chemgymrl_gflownet/logs/upper_tail_replay_seed0
```

Replay starts only after four examples have been discovered on each side of the
threshold. The replay trajectories do not require additional ChemGymRL calls.

To separate the effect of reward shaping from the effect of replay, the matched
control uses the original terminal target with the same replay settings. The
threshold classifies stored procedures for replay but does not alter their
terminal rewards because the tail strength remains zero:

```bash
Coding/.venv/bin/python Coding/chemgymrl_gflownet/scripts/train_procedure_gflownet.py \
  --steps 1000 --batch-size 32 --seed 0 --device cpu \
  --reward-scale 0.05 --tail-threshold 0.10 \
  --fixed-tail-replay --replay-batch-size 8 --replay-capacity 512 \
  --fixed-tail-replay-fraction 0.5 \
  --validate-every 100 --validation-batches 1 --final-batches 32 \
  --log-dir Coding/chemgymrl_gflownet/logs/baseline_target_replay_seed0
```

This is an adaptation of the upper-tail problem to a discrete GFlowNet terminal
reward. It is not a reproduction of Wang et al.'s continuous-model fine-tuning
algorithm.

## Current recommendation

- keep notes here
- keep code in `Coding/`
- keep the GFlowNet repo and ChemGymRL repo as separate folders inside `Coding/`
- use separate virtual environments if dependency conflicts appear
