# no-show-ml-demo

A 25-minute, pre-built notebook demo for a technical breakout talk on **developing AI/ML in a
health system**. One prediction problem everybody understands, predicting outpatient no-shows,
walked end to end: synthetic data, a time-based split, a baseline, honest evaluation, a planted
target-leakage trap, a bias check, and a drift batch that hands off to the governance half of
the talk.

Everything is synthetic and generated in the first cell. No PHI, no downloads, nothing to break
on a managed laptop.

## Layout

```
notebooks/no_show_classic_ml.ipynb                 the demo, clean (no outputs); open this live
notebooks/backup/no_show_classic_ml_prerun.ipynb   same notebook, already executed; open if the kernel dies
notebooks/backup/no_show_classic_ml_prerun.html    every output with charts inline; source for backup slides
docs/talk-track.md                                 per-block timing, talking points, and the two live changes
tools/build_notebook.py                            the notebook as code; edit this, not the .ipynb
tools/run_backup.py                                executes the notebook headless and refreshes notebooks/backup/
```

## Setup

The venv lives on the local disk, not the SSD, built by the root `setup-venv.ps1`:

```powershell
..\setup-venv.ps1 no-show-ml-demo
& "$env:LOCALAPPDATA\repo-venvs\no-show-ml-demo\Scripts\Activate.ps1"
```

## Rehearse

```powershell
python tools/build_notebook.py      # regenerate the .ipynb after editing the cells
python tools/run_backup.py          # execute it end to end and refresh the backup copies
jupyter lab notebooks/no_show_classic_ml.ipynb
```

The whole notebook executes in well under a minute. Before presenting, run `run_backup.py` on
the machine you will present from so the backup matches the installed library versions.

## What the numbers land on (seed 42)

| Moment | Value |
|---|---|
| Base rate / "everyone shows up" accuracy | 18% / 82% |
| Honest model accuracy (logistic, boosted trees) | 83%, 83% |
| Honest ROC AUC | 0.76 |
| With the leaked `reminder_confirmed` column | 96% accuracy |
| Recall at threshold 0.30, portal vs no-portal patients | 52% vs 27% |
| Share flagged, validation period vs drifted batch | 17% vs 45% |

## Classic ML or LLM?

This is the classic ML version. It shows the full development loop, runs deterministically, and
raises no "what can I send to the model" question. An LLM contrast belongs at the end of the
talk, not in the demo, unless the event framing is explicitly generative AI. See
`docs/talk-track.md`.
