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

## Current recommendation

- keep notes here
- keep code in `Coding/`
- keep the GFlowNet repo and ChemGymRL repo as separate folders inside `Coding/`
- use separate virtual environments if dependency conflicts appear
