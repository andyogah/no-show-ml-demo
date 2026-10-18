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
        "MLOps": 22, "drift": 24, "model": 30, "weights": 22, "logistic regression": 28,
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
        "precision": 22, "recall": 22, "ROC curve": 18,
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

# ---------------------------------------------------------------------------------------
# 04: more inputs, more m's. The equation as a picture: inputs, one weighted arrow each, a sum.
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

INPUTS = [("lead time", "45 days"), ("prior no-shows", "2"), ("reminder sent", "no"),
          ("distance", "30+ mi"), ("…", "")]
fig, ax = plt.subplots(figsize=(11, 6.0))
ax.set_xlim(0, 10)
ax.set_ylim(-1.0, 5.2)
ax.axis("off")
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.03)

x_in, x_sum, x_out = 1.6, 5.6, 8.6
ys = np.linspace(4.5, 0.9, len(INPUTS))
for (name, val), y_ in zip(INPUTS, ys):
    box = FancyBboxPatch((x_in - 1.4, y_ - 0.32), 2.8, 0.64, boxstyle="round,pad=0.02,rounding_size=0.12",
                         fc="white", ec=BLUE, lw=1.6)
    ax.add_patch(box)
    label = f"{name}   {val}" if val else name
    ax.text(x_in, y_, label, ha="center", va="center", fontsize=12, color="#0b0b0b")

for i, y_ in enumerate(ys):
    arrow = FancyArrowPatch((x_in + 1.45, y_), (x_sum - 0.62, 2.6 + (y_ - 2.6) * 0.12),
                            arrowstyle="-|>", mutation_scale=14, color=GRAY, lw=1.4,
                            shrinkA=0, shrinkB=0)
    ax.add_patch(arrow)
    if INPUTS[i][0] != "…":
        xm = (x_in + 1.45 + x_sum - 0.62) / 2
        ym = (y_ + 2.6 + (y_ - 2.6) * 0.12) / 2 + 0.17
        ax.text(xm, ym, f"m{chr(0x2080 + i + 1)}", ha="center", va="bottom", fontsize=14, color=ORANGE,
                fontweight="bold")

ax.add_patch(Circle((x_sum, 2.6), 0.62, fc="white", ec=ORANGE, lw=2.2))
ax.text(x_sum, 2.72, "add it up", ha="center", va="center", fontsize=12, color="#0b0b0b")
ax.text(x_sum, 2.38, "+ b", ha="center", va="center", fontsize=14, color=ORANGE, fontweight="bold")

ax.add_patch(FancyArrowPatch((x_sum + 0.64, 2.6), (x_out - 1.25, 2.6), arrowstyle="-|>",
                             mutation_scale=16, color=GRAY, lw=1.8, shrinkA=0, shrinkB=0))
ax.add_patch(FancyBboxPatch((x_out - 1.2, 2.2), 2.4, 0.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc=BLUE, ec=BLUE))
ax.text(x_out, 2.6, "y\nno-show rate", ha="center", va="center", fontsize=12, color="white", linespacing=1.3)

ax.text(x_in, 5.05, "inputs", ha="center", fontsize=11, color=GRAY)
ax.text((x_in + x_sum) / 2 + 0.1, 5.05, "one m per input: the weights", ha="center", fontsize=11, color=GRAY)
ax.text(x_out, 5.05, "output", ha="center", fontsize=11, color=GRAY)
# The equation, large, in a band across the bottom, with the m's and b in the arrow colour so
# the eye links each symbol to the diagram above it.
PIECES = [("y", "#0b0b0b"), ("  =  ", GRAY), ("m₁", ORANGE), (" · lead time", "#0b0b0b"), ("  +  ", GRAY),
          ("m₂", ORANGE), (" · prior no-shows", "#0b0b0b"), ("  +  ", GRAY), ("m₃", ORANGE),
          (" · reminder sent", "#0b0b0b"), ("  +  ", GRAY), ("m₄", ORANGE), (" · distance", "#0b0b0b"),
          ("  +  …  +  ", GRAY), ("b", ORANGE)]
EQ_SIZE = 14
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
# measure each piece in display units, lay them out centred, then draw the band around them
widths = []
for text, _ in PIECES:
    t = ax.text(0, 0, text, fontsize=EQ_SIZE, fontweight="bold", family="DejaVu Sans")
    widths.append(t.get_window_extent(renderer).width)
    t.remove()
total = sum(widths)
to_data = lambda xd: ax.transData.inverted().transform((xd, 0))[0]
x0_disp = ax.transData.transform((5.0, 0))[0] - total / 2
x_start, x_end = to_data(x0_disp), to_data(x0_disp + total)
y_eq = -0.42
ax.add_patch(FancyBboxPatch((x_start - 0.25, y_eq - 0.42), (x_end - x_start) + 0.5, 0.84,
                            boxstyle="round,pad=0.02,rounding_size=0.14", fc="#f6f6f4", ec=GRID, lw=1.0))
for (text, colour), w in zip(PIECES, widths):
    ax.text(to_data(x0_disp), y_eq, text, fontsize=EQ_SIZE, fontweight="bold", color=colour, family="DejaVu Sans",
            ha="left", va="center")
    x0_disp += w
fig.savefig(OUT / "04-weighted-sum.png")
plt.close(fig)

# ---------------------------------------------------------------------------------------
# 05: stack it. Four panels: one squashed line, a few feeding one, many stacked, and a tree.
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.15, 1.5, 1.35]})


def draw_network(ax, layers, title, subtitle, n_inputs=4):
    """layers: list of node counts per hidden/output layer, e.g. [1] or [3, 1] or [4, 4, 4, 1]."""
    ax.set_xlim(-0.2, len(layers) + 1.2)
    ax.set_ylim(-0.6, 5.3)
    ax.axis("off")
    cols = [np.linspace(3.4, 0.6, n_inputs)]                       # input column y positions
    for n in layers:
        cols.append(np.linspace(2.0 + 0.5 * (n - 1), 2.0 - 0.5 * (n - 1), n) if n > 1 else np.array([2.0]))
    xs_ = list(range(len(cols)))
    # inputs as small blue squares
    for y_ in cols[0]:
        ax.add_patch(FancyBboxPatch((xs_[0] - 0.14, y_ - 0.14), 0.28, 0.28, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc=BLUE, ec=BLUE))
    # edges then nodes
    for li in range(1, len(cols)):
        for y0 in cols[li - 1]:
            for y1 in cols[li]:
                ax.plot([xs_[li - 1] + (0.16 if li == 1 else 0.26), xs_[li] - 0.26], [y0, y1], color=GRAY, lw=0.9,
                        alpha=0.75, zorder=1)
        for y1 in cols[li]:
            ax.add_patch(Circle((xs_[li], y1), 0.26, fc="white", ec=ORANGE, lw=1.8, zorder=2))
            # a tiny sigmoid glyph inside each node
            t = np.linspace(-3, 3, 30)
            ax.plot(xs_[li] + t / 3 * 0.15, y1 + (1 / (1 + np.exp(-t)) - 0.5) * 0.26, color=ORANGE, lw=1.4, zorder=3)
    # output arrow
    last_x, last_y = xs_[-1], cols[-1][0]
    ax.add_patch(FancyArrowPatch((last_x + 0.28, last_y), (last_x + 0.95, last_y), arrowstyle="-|>",
                                 mutation_scale=12, color=GRAY, lw=1.4, shrinkA=0, shrinkB=0))
    ax.text(last_x + 1.0, last_y - 0.45, "y", ha="right", fontsize=11, color=GRAY)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#0b0b0b")
    ax.text(-0.15, 5.05, subtitle, fontsize=10.5, color=GRAY, va="top")


draw_network(axes[0], [1], "One squashed line", "logistic regression\nthe circle from two slides ago")
draw_network(axes[1], [3, 1], "A few feeding one", "a neural network")
draw_network(axes[2], [4, 4, 4, 1], "Many, stacked", "deep learning: images, text, speech, signals\n+ a trick for attending to words in order = an LLM")

# the tree panel
ax = axes[3]
ax.set_xlim(0, 10)
ax.set_ylim(-0.6, 5.3)
ax.axis("off")
ax.set_title("Stacked if-thens", loc="left", fontsize=13, fontweight="bold", color="#0b0b0b")
ax.text(0.1, 5.05, "trees: random forests, gradient boosting\nnot lines, but found by the same loop", fontsize=10.5,
        color=GRAY, va="top")


def node(x, y, text, leaf=False):
    w, h = (3.2, 0.62) if not leaf else (1.5, 0.55)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=(BLUE if leaf else "white"), ec=BLUE, lw=1.6, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color=("white" if leaf else "#0b0b0b"), zorder=3)


def edge(x0, y0, x1, y1, label):
    ax.plot([x0, x1], [y0 - 0.3, y1 + 0.3], color=GRAY, lw=1.1, zorder=1)
    ax.text((x0 + x1) / 2 + (0.25 if x1 > x0 else -0.25), (y0 + y1) / 2, label, fontsize=8.5, color=GRAY,
            ha="left" if x1 > x0 else "right", va="center")


node(5.0, 3.3, "lead time > 30 days?")
edge(5.0, 3.3, 2.5, 2.0, "no"); edge(5.0, 3.3, 7.5, 2.0, "yes")
node(2.5, 2.0, "prior no-shows > 0?")
node(7.5, 2.0, "reminder sent?")
edge(2.5, 2.0, 1.3, 0.7, "no"); edge(2.5, 2.0, 3.9, 0.7, "yes")
edge(7.5, 2.0, 6.1, 0.7, "yes"); edge(7.5, 2.0, 8.7, 0.7, "no")
node(1.3, 0.7, "9%", leaf=True); node(3.9, 0.7, "21%", leaf=True)
node(6.1, 0.7, "33%", leaf=True); node(8.7, 0.7, "58%", leaf=True)
ax.text(5.0, -0.35, "× hundreds of small trees, added up", ha="center", fontsize=10.5, color=ORANGE, fontweight="bold")

fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "05-stack-it.png")
plt.close(fig)

print(f"m = {m:.4f} per day ({m*10:.1%} per 10 days), b = {b:.1%}")
print(f"wrote 6 figures to {OUT.relative_to(ROOT)}/")
