# Replay and Upper-Tail Learning in GFlowNets for Simulated Chemical Experiments

This repository contains an internship research project on finite-training
behaviour in Generative Flow Networks (GFlowNets). It connects a discrete
Trajectory Balance GFlowNet to a ChemGymRL simulator, then studies two related
questions:

1. Can replay improve learning of a reward-proportional target when useful
   simulator outcomes are rare?
2. Once rare-state exposure is controlled, how does an upper-tail-shaped
   terminal reward affect tail frequency and diversity?

The main experiment uses ChemGymRL's fictitious `FictReactBandit-v1`
environment. A procedure has nine controls with three values each, giving an
exactly enumerable space of `3^9 = 19,683` terminal procedures. Enumeration
makes it possible to compare learned policies with their exact target
distributions instead of relying only on loss or mean reward.

This is a controlled proof of concept about simulator-guided GFlowNet training.
It is not a claim about arbitrary chemistry, and the upper-tail experiment is
not a reproduction of Wang et al.'s continuous-model fine-tuning method.

## Main findings

- Standard Trajectory Balance substantially underfit the known target within
  the fixed training budget.
- Positive replay increased exact target-mass coverage from `5.06%` to `21.52%`
  without additional simulator calls, but also increased repetition.
- With the same fixed-tail replay rule in both conditions, upper-tail shaping
  increased final tail frequency by `5.83` percentage points but did not
  increase the number of distinct tail procedures.
- In this benchmark, replay primarily addressed rare-state exposure, while
  shaping changed how much probability the learned policy assigned to the
  exposed tail.

The full methodology, limitations, and three-seed results are in
[`main.tex`](main.tex). Generated figures used by the report are stored in
[`figures/`](figures/).

## Repository layout

```text
.
├── main.tex                         # Research report
├── references.bib                   # Report bibliography
├── THIRD_PARTY_NOTICES.md           # Dependency licenses and attribution
├── figures/                         # Report figures
└── Coding/
    ├── chemgymrl_gflownet/
    │   ├── src/chemgymrl_gflownet/ # Bridge, task, replay, and reward code
    │   ├── scripts/                 # Training and analysis entry points
    │   └── tests/                   # Unit and integration tests
    ├── envs/                        # Direct and tested dependency lists
    └── upstream/                    # Pinned GFlowNet revision and local patch
```

The upstream `recursionpharma-gflownet` and `chemgymrl` repositories are cloned
locally under `Coding/` and are intentionally excluded from version control.

## Environment setup

The tested environment uses Python 3.10. Python 3.11 is supported by the local
package metadata, but the pinned dependency set was prepared with Python 3.10.

```bash
git clone https://github.com/Levey222/gflownet-internship-project.git
cd gflownet-internship-project

python3.10 -m venv Coding/.venv
source Coding/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Clone the tested upstream revisions:

```bash
git clone https://github.com/recursionpharma/gflownet.git Coding/recursionpharma-gflownet
git -C Coding/recursionpharma-gflownet checkout da999404e997a302a81773eb183b1da6ec2a4449
git -C Coding/recursionpharma-gflownet apply --unidiff-zero \
  ../upstream/recursionpharma-gflownet-local-fixes.patch

git clone https://github.com/chemgymrl/chemgymrl.git Coding/chemgymrl
git -C Coding/chemgymrl checkout ab8227b6b33f13617b7e551bdf6b894df7eec68d
```

Install the pinned CPU dependencies and the three local packages:

```bash
python -m pip install \
  -r Coding/envs/gflownet-cpu-working-requirements.txt \
  --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html
python -m pip install -r Coding/envs/chemgymrl-working-requirements.txt
python -m pip install pytest

python -m pip install -e Coding/recursionpharma-gflownet --no-deps
python -m pip install -e Coding/chemgymrl
python -m pip install -e Coding/chemgymrl_gflownet
```

The local GFlowNet patch corrects Flow Matching initialisation and disables
gradient tracking during model-based trajectory sampling. It does not change
the Flow Matching or Trajectory Balance loss equations.

## Verification

Run the project test suite from the repository root:

```bash
Coding/.venv/bin/python -m pytest Coding/chemgymrl_gflownet/tests
```

Run a small random ChemGymRL benchmark:

```bash
Coding/.venv/bin/python \
  Coding/chemgymrl_gflownet/scripts/benchmark_fict_react.py \
  --mode random --samples 100 --seed 0 \
  --output-dir Coding/chemgymrl_gflownet/results/smoke_benchmark
```

Generated logs and result tables are ignored by Git.

## Core experiments

An exhaustive reference over all 19,683 procedures can be generated with:

```bash
Coding/.venv/bin/python \
  Coding/chemgymrl_gflownet/scripts/benchmark_fict_react.py \
  --mode exhaustive --seed 0 --reward-scale 0.05 \
  --output-dir Coding/chemgymrl_gflownet/results/exhaustive_reference
```

The original-target baseline is:

```bash
Coding/.venv/bin/python \
  Coding/chemgymrl_gflownet/scripts/train_procedure_gflownet.py \
  --steps 1000 --batch-size 32 --seed 0 --device cpu \
  --reward-scale 0.05 \
  --validate-every 100 --validation-batches 1 --final-batches 32 \
  --log-dir Coding/chemgymrl_gflownet/logs/baseline_seed0
```

Add `--positive-replay --replay-batch-size 8 --replay-capacity 512` to run the
positive-replay condition.

The matched upper-tail condition uses the selected threshold and shaping
strength together with fixed-tail replay:

```bash
Coding/.venv/bin/python \
  Coding/chemgymrl_gflownet/scripts/train_procedure_gflownet.py \
  --steps 1000 --batch-size 32 --seed 0 --device cpu \
  --reward-scale 0.05 --tail-threshold 0.10 --tail-strength 0.25 \
  --fixed-tail-replay --replay-batch-size 8 --replay-capacity 512 \
  --fixed-tail-replay-fraction 0.5 \
  --validate-every 100 --validation-batches 1 --final-batches 32 \
  --log-dir Coding/chemgymrl_gflownet/logs/upper_tail_replay_seed0
```

For the matched original-target control, use the same command without
`--tail-strength 0.25`. The threshold then classifies replay samples but does
not reshape their terminal rewards.

The report uses seeds `0`, `17`, and `42`. Analysis and aggregation utilities
for these runs are available in
[`Coding/chemgymrl_gflownet/scripts/`](Coding/chemgymrl_gflownet/scripts/).

## Building the report

With a LaTeX distribution containing `latexmk`, `natbib`, and the standard
graphics packages installed:

```bash
latexmk -pdf main.tex
```

The generated `main.pdf` and LaTeX build files are ignored by Git.

## Licensing and attribution

This repository currently has no blanket license for its original code,
report, or figures. Making the repository public allows it to be viewed and
forked through GitHub, but does not otherwise grant permission to reproduce,
modify, or redistribute the original material.

The upstream GFlowNet and ChemGymRL dependencies retain their own licenses and
are not vendored in this repository. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for revisions, license
details, and attribution.
