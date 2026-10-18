# no-show-ml-demo

A 32-minute, pre-built notebook demo for a technical breakout talk on **developing AI/ML in a
health system**. One prediction problem everybody understands, predicting outpatient no-shows,
walked end to end: synthetic data, a time-based split, a baseline, a look under the hood of
`fit()` (logistic regression by hand in NumPy, the same loop in PyTorch, a small neural network
that loses to the tree), honest evaluation with hand-counted metrics, a planted target-leakage
trap, a bias check, and a drift batch that hands off to the governance half of the talk.

Everything is synthetic and generated in the first cell. No PHI, no downloads, nothing to break
on a managed laptop.

## Layout

```
notebooks/no_show_classic_ml.ipynb                 the whole talk: Part 1 intro cells, Part 2 the demo; clean, open this live
notebooks/backup/no_show_classic_ml_prerun.ipynb   same notebook, already executed; open if the kernel dies
notebooks/backup/no_show_classic_ml_prerun.html    every output with charts inline; source for backup slides
docs/intro-slides.md                               script for the Part 1 intro cells, with timings
docs/talk-track.md                                 per-block timing, talking points, and the two live changes
docs/figures/                                      word cloud and y = mx + b figures the intro cells display
tools/build_notebook.py                            the notebook as code; edit this, not the .ipynb
tools/run_backup.py                                executes the notebook headless and refreshes notebooks/backup/
tools/make_intro_figures.py                        draws docs/figures/ from the notebook's own generator
```

## Setup

Python 3.12 or newer (the pinned numpy needs 3.12). Everything else comes from `requirements.lock`,
the exact versions the backup notebook was executed with; `requirements.txt` lists the same
packages unpinned if you want the latest.

### Option A: a standard venv in the repo

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

`.venv/` is gitignored. On Linux the default PyPI wheel for torch is the CUDA build, about 2 GB;
the demo only needs CPU, so you can install the small build first and let the lock file skip it:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.lock
```

### Option B: the author's portable-SSD layout

The repo is kept on an exFAT SSD that moves between machines, so the venv lives on each machine's
local disk instead, built by the root-level `setup-venv.ps1` (not part of this repo):

```powershell
..\setup-venv.ps1 no-show-ml-demo
& "$env:LOCALAPPDATA\repo-venvs\no-show-ml-demo\Scripts\Activate.ps1"
```

### Pick the kernel

- **JupyterLab**: start it from the activated venv (`jupyter lab notebooks/no_show_classic_ml.ipynb`).
  The default "Python 3" kernel is the venv itself.
- **VS Code**: open this repo folder as the workspace, click **Select Kernel** in the notebook
  toolbar, then **Python Environments** and choose `.venv`. VS Code only auto-discovers venvs
  inside the workspace. If yours lives elsewhere, as in Option B, or you opened a parent folder,
  either add the venv's parent folder to the `python-envs.globalSearchPaths` user setting (the
  setup script does this), or register the venv as a named kernel and pick it under
  **Jupyter Kernel...**:

  ```powershell
  python -m ipykernel install --user --name no-show-ml-demo --display-name "Python (no-show-ml-demo)"
  ```

## Rehearse

```powershell
python tools/make_intro_figures.py  # redraw the intro figures (word cloud, the line) if they change
python tools/build_notebook.py      # regenerate the .ipynb after editing the cells
python tools/run_backup.py          # execute it end to end and refresh the backup copies
jupyter lab notebooks/no_show_classic_ml.ipynb
```

The whole notebook executes in about a minute on CPU; torch is the CPU build. Before presenting, run `run_backup.py` on
the machine you will present from so the backup matches the installed library versions.

## What the numbers land on (seed 42)

| Moment | Value |
|---|---|
| Base rate / "everyone shows up" accuracy | 18% / 82% |
| Honest model accuracy (logistic, boosted trees) | 83%, 83% |
| Honest ROC AUC | 0.76 |
| Logistic regression by hand vs scikit-learn, ROC AUC | 0.758 vs 0.758 |
| Neural network vs boosted trees, ROC AUC | 0.73 vs 0.76 |
| With the leaked `reminder_confirmed` column | 96% accuracy |
| Recall at threshold 0.30, portal vs no-portal patients | 52% vs 27% |
| Share flagged, validation period vs drifted batch | 17% vs 45% |

## Classic ML or LLM?

This is the classic ML version. It shows the full development loop, runs deterministically, and
raises no "what can I send to the model" question. An LLM contrast belongs at the end of the
talk, not in the demo, unless the event framing is explicitly generative AI. See
`docs/talk-track.md`.
