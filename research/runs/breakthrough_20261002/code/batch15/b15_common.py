# -*- coding: utf-8 -*-
"""BATCH14 metrics-snapshot-20261003-r1 shared plotting helpers.
Read-only: loads existing metrics/predictions; writes only batch14 artifacts.
No GPU, no training, CPU-only matplotlib rendering.
"""
import os
import json
import hashlib
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

# ---- CJK font (readable, no missing glyphs) -------------------------------
def setup_fonts():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "fonts", "msyh_regular.ttf"),
              os.path.join(here, "fonts", "msyh_bold.ttf")):
        try:
            font_manager.fontManager.addfont(p)
        except Exception:
            pass
    matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["svg.fonttype"] = "none"   # keep editable <text> in SVG
    matplotlib.rcParams["svg.image_inline"] = True
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"

# ---- consistent colour scheme --------------------------------------------
TEAL   = "#00897B"   # full / augmented / candidate / best-baseline
GRAY   = "#9E9E9E"   # base / linear baseline / comparator
ORANGE = "#EF6C00"   # caution / candidate-that-did-not-win
DARK   = "#263238"

# ---- metrics helpers ------------------------------------------------------
def spearman(y, p):
    return float(stats.spearmanr(y, p).statistic)

def r2_score(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot

def mae_score(y, p):
    return float(np.mean(np.abs(y - p)))

# ---- io helpers -----------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_tsv(path):
    import csv
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    return rows

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def now_local():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S+08:00")

def save_png_svg(fig, stem, outdir, width_px=2400):
    """Save PNG (>=1800px wide) and editable SVG with preserved text."""
    png_path = os.path.join(outdir, stem + ".png")
    svg_path = os.path.join(outdir, stem + ".svg")
    dpi = 200
    w0, h0 = fig.get_size_inches()
    w_in = width_px / dpi
    h_in = h0 * (w_in / w0)
    fig.set_size_inches(w_in, h_in)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, svg_path
