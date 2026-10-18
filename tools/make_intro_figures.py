"""Figures for the intro, drawn from the same synthetic data the notebook generates.

Run:  python tools/make_intro_figures.py
Writes docs/figures/00-wordcloud.png, 01-line.png, 02-line-fitted.png, 03-squashed.png
The notebook's intro cells display these files.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse the notebook's generator without executing the notebook: pull the cell source.
sys.path.insert(0, str(ROOT / "tools"))
import nbformat  # noqa: E402

nb = nbformat.read(ROOT / "notebooks" / "no_show_classic_ml.ipynb", as_version=4)
gen_src = next(c.source for c in nb.cells if c.cell_type == "code" and "def make_appointments" in c.source)
ns: dict = {"np": np}
gen_src = gen_src[: gen_src.index("df = make_appointments(")]   # drop the cell's trailing calls
exec("import pandas as pd\n" + gen_src, ns)
df = ns["make_appointments"](20_000, np.random.default_rng(42))

BLUE, ORANGE, GRAY, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e6e3"
plt.rcParams.update({"figure.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 12})

# ---------------------------------------------------------------------------------------
# 00: the word cloud. Every term is coloured by the ring it belongs to on the definitions
# slide, so the colours are the reveal: the mess on screen is really five groups.
from wordcloud import WordCloud  # noqa: E402

GROUPS = {
    # colour              terms (weight = how big)
    "#2a78d6": {  # AI, the outer ring
        "AI": 100, "artificial intelligence": 40, "generative AI": 45, "large language model": 40,
        "GPT": 30, "copilot": 25, "prompt engineering": 28, "hallucination": 22, "guardrails": 22,
        "responsible AI": 24, "tokens": 18, "RAG": 30, "vector database": 20, "embedding": 26,
        "computer vision": 22, "NLP": 24, "inference": 22, "GPU": 16,
    },
    "#eb6834": {  # machine learning
        "machine learning": 70, "supervised": 28, "unsupervised": 24, "classification": 30,
        "regression": 32, "feature": 26, "label": 22, "training": 36, "overfitting": 26,
        "gradient boosting": 22, "random forest": 20, "k-nearest neighbours": 16, "clustering": 20,
        "MLOps": 22, "drift": 24, "model": 30, "weights": 22,
    },
    "#1baf7a": {  # deep learning
        "deep learning": 55, "neural network": 45, "transformer": 34, "attention": 24,
        "backpropagation": 20, "fine-tuning": 26, "convolution": 16, "reinforcement learning": 22,
        "parameters": 20, "layers": 18,
    },
    "#eda100": {  # data science
        "data science": 50, "predictive analytics": 30, "data engineering": 24, "pipeline": 20,
        "dashboard": 18, "feature store": 14, "ETL": 16, "data lake": 18, "notebook": 16,
        "Python": 22, "SQL": 18, "pandas": 14,
    },
    "#e87ba4": {  # statistics
        "statistics": 48, "probability": 30, "base rate": 24, "bias": 30, "variance": 22,
        "confidence interval": 20, "p-value": 22, "sampling": 18, "correlation": 24,
        "precision": 22, "recall": 22, "ROC curve": 18, "logistic regression": 28,
    },
    "#6250d6": {  # agents
        "agent": 44, "agentic AI": 42, "multi-agent": 26, "MCP": 24, "tool calling": 22,
        "orchestration": 20, "autonomy": 18, "workflow": 18, "planning": 16,
    },
}
freq = {term: w for terms in GROUPS.values() for term, w in terms.items()}
colour_of = {term: c for c, terms in GROUPS.items() for term in terms}

_font = next((f for f in [Path("C:/Windows/Fonts/segoeuib.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")]
              if f.exists()), None)
wc = WordCloud(width=1600, height=800, background_color="white", prefer_horizontal=0.9,
               min_font_size=12, max_font_size=220, random_state=42, margin=6,
               font_path=str(_font) if _font else None,
               color_func=lambda word, **kw: colour_of[word]).generate_from_frequencies(freq)
wc.to_file(str(OUT / "00-wordcloud.png"))

# Binned no-show rate by lead time: the dots every slide uses.
bins = np.arange(0, 95, 5)
df["bin"] = np.digitize(df.lead_time_days, bins) - 1
agg = df[df.lead_time_days < 90].groupby("bin").agg(x=("lead_time_days", "mean"), y=("no_show", "mean"), n=("no_show", "size"))
agg = agg[agg.n > 50]
x, y = agg.x.to_numpy(), agg.y.to_numpy()

# Straight-line fit on the dots (what a slope and intercept look like here).
m, b = np.polyfit(x, y, 1)
xs = np.linspace(0, 90, 200)


def base(title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, y, s=60, color=BLUE, zorder=3)
    ax.set_xlabel("Days between scheduling and appointment (x)")
    ax.set_ylabel("Share who no-show (y)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 0.75)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    return fig, ax


# 01: just the dots
fig, ax = base("What the data says")
fig.tight_layout()
fig.savefig(OUT / "01-line.png")
plt.close(fig)

# 02: the fitted line with m and b called out
fig, ax = base("y = mx + b")
ax.plot(xs, m * xs + b, color=ORANGE, lw=2.5, zorder=2)
ax.annotate(f"b = {b:.0%}\nthe no-show rate at zero days", xy=(0, b), xytext=(6, 0.52),
            color=GRAY, arrowprops=dict(arrowstyle="->", color=GRAY))
ax.annotate(f"m = {m*10:.1%} per 10 days\nhow much each extra wait costs", xy=(55, m * 55 + b),
            xytext=(40, 0.12), color=GRAY, arrowprops=dict(arrowstyle="->", color=GRAY))
fig.tight_layout()
fig.savefig(OUT / "02-line-fitted.png")
plt.close(fig)

# 03: the squashed version (a logistic curve on the same data)
from sklearn.linear_model import LogisticRegression  # noqa: E402

lr = LogisticRegression().fit(df[["lead_time_days"]].to_numpy(), df.no_show)
xl = np.linspace(0, 160, 300)
ps = lr.predict_proba(xl.reshape(-1, 1))[:, 1]
fig, ax = base("Squash it so it stays between 0% and 100%")
ax.set_xlim(0, 160)
ax.set_ylim(0, 1.2)
ax.axhline(1.0, color=GRAY, lw=1, ls=":")
ax.text(158, 1.02, "100%: everyone no-shows", ha="right", color=GRAY, fontsize=10)
ax.plot(xl, m * xl + b, color=ORANGE, lw=1.5, ls="--", alpha=0.6, label="straight line (goes past 100%)")
ax.plot(xl, ps, color=ORANGE, lw=2.5, label="same idea, squashed (logistic regression)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "03-squashed.png")
plt.close(fig)

print(f"m = {m:.4f} per day ({m*10:.1%} per 10 days), b = {b:.1%}")
print(f"wrote 4 figures to {OUT.relative_to(ROOT)}/")
