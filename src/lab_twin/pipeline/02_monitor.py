from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from matplotlib.patches import Patch


width = 15.92 * 0.618 * 0.9

# =========================
# Global style / theme
# =========================
_BG_COLOR = "#002E54"  # deep navy background

# ---- Custom color palette (edit freely) ----
PALETTE = {
    "timeline_batch_colors": [
        "#4CC9F0",  # 0 - bright cyan
        "#3DB6F2",  # 1 - sky blue
        "#3491F0",  # 2 - medium blue
        "#4361EE",  # 3 - rich indigo
        "#3A0CA3",  # 4 - deep violet
        "#5A189A",  # 5 - royal purple
        "#7209B7",  # 6 - bright purple
        "#9D4EDD",  # 7 - orchid
        "#C77DFF",  # 8 - lavender
        "#E0AAFF",  # 9 - pale lilac
        "#FF6FAF",  # 10 - pink
        "#F72585",  # 11 - magenta
        "#D00070",  # 12 - crimson magenta
        "#B5179E",  # 13 - deep fuchsia
        "#8E1E8E",  # 14 - royal pink-violet
    ],
    "bottleneck_bars": [
        "#4361EE",
        "#3491F0",
        "#4CC9F0",
        "#ffa600",
        "#43EE7F",
    ],
    "hist_fill": "#4361EE",
    "hist_mean_line": "#B8282B",
    "sla_yes": "#3D7545",
    "sla_no": "#B8282B",
    "decision_yes": "#3D7545",
    "decision_no": "#B8282B",
    "white": "#FFFFFF",
}


def _init_matplotlib_style() -> None:
    """
    Register custom font weights and push global rcParams so that
    ALL plots share the same look:
    - Source Sans 3 typography
    - white text
    - dark navy background
    """
    # --- try to register Source Sans 3 family ---
    # If these font files aren't available on the runtime machine,
    # matplotlib will just fall back.
    fontdir = "/home/dorian/.local/share/fonts/source-sans-pro/source-sans-3.052R/TTF/"
    for f in [
        "SourceSans3-Light.ttf",
        "SourceSans3-Regular.ttf",
        "SourceSans3-Semibold.ttf",
        "SourceSans3-Bold.ttf",
    ]:
        fpath = Path(fontdir) / f
        if fpath.exists():
            try:
                fm.fontManager.addfont(str(fpath))
            except Exception:
                pass

    rcParams.update(
        {
            # --- Font setup ---
            "font.family": "Source Sans 3",
            "font.sans-serif": ["Source Sans 3"],
            "font.weight": "light",
            "axes.labelweight": "light",
            "axes.titleweight": "bold",

            # --- Text / tick colors ---
            "text.color": PALETTE["white"],
            "axes.labelcolor": PALETTE["white"],
            "axes.edgecolor": PALETTE["white"],
            "axes.titlecolor": PALETTE["white"],
            "xtick.color": PALETTE["white"],
            "ytick.color": PALETTE["white"],

            # --- Tick style ---
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,

            # --- Legend defaults ---
            "legend.frameon": False,
            "legend.fontsize": 11,

                # --- Figure / save ---
            "figure.dpi": 150,        # on-screen clarity
            "savefig.dpi": 300,       # high-quality raster export
            "savefig.bbox": "tight",  # trims whitespace
            "savefig.facecolor": _BG_COLOR,  # ensures dark bg on export
        }
    )


def _apply_ax_dark_theme(ax: plt.Axes) -> None:
    """
    Apply shared visual tweaks that aren't purely rcParams:
    - dark bg on both fig + axes
    - hide top/right/left spines, only bottom spine visible + white
    - consistent label font sizes
    """
    fig = ax.figure
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # Spines
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # bottom spine stays for reference, in white
    ax.spines["bottom"].set_color(PALETTE["white"])
    ax.spines["bottom"].set_linewidth(1.0)

    # default label sizes
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.title.set_fontsize(11)

    # ticks
    ax.tick_params(
        axis="both",
        which="major",
        length=8,
        width=1,
        color=PALETTE["white"],
        pad=6,
        labelsize=11,
    )


def _apply_legend_dark_theme(leg: plt.Legend | None) -> None:
    """
    Legend matches dark background (bg same as fig, white text).
    """
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor(_BG_COLOR)
    frame.set_edgecolor(_BG_COLOR)
    for text in leg.get_texts():
        text.set_color(PALETTE["white"])
    if leg.get_title() is not None:
        leg.get_title().set_color(PALETTE["white"])


# -------------------------
# Safe helpers
# -------------------------
def _safe_read_csv(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


# -------------------------
# Plots
# -------------------------
def _plot_timeline(events: pd.DataFrame, outpath: Path, max_legend: int = 20) -> None:
    """
    Workflow timeline (Gantt-style):
    - Y axis = process_id, ordered by when it first runs in the workflow.
    - Each bar = an execution window [sim_start, sim_end] for that process.
    - Color = batch_id.
    This shows where time is actually spent in the end-to-end workflow.
    """
    if events is None or events.empty:
        return

    ev = events[~events["process_id"].astype(str).str.startswith("decision:")].copy()

    needed_cols = {"sim_start", "sim_end", "process_id", "batch_id"}
    if not needed_cols.issubset(ev.columns):
        return

    # to numeric
    for c in ["sim_start", "sim_end"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev = ev.dropna(subset=["sim_start", "sim_end"])
    if ev.empty:
        return

    # Order processes by first time they appear in the sim
    first_seen = (
        ev.groupby("process_id")["sim_start"]
        .min()
        .sort_values()
    )
    process_order = list(first_seen.index)
    order_map = {proc: i for i, proc in enumerate(process_order)}
    ev["y_idx"] = ev["process_id"].map(order_map)

    # Assign colors per batch using your palette
    batches = ev["batch_id"].drop_duplicates().tolist()
    timeline_colors = PALETTE["timeline_batch_colors"]
    batch_color = {
        b: timeline_colors[i % len(timeline_colors)]
        for i, b in enumerate(batches)
    }

    # Dynamic fig height based on number of processes
    # fig_h = max(5, 0.5 * len(process_order) + 2)
    # fig, ax = plt.subplots(figsize=(12, fig_h))

    # fig, ax = plt.subplots(figsize=(15.92, 24.62))
    fig, ax = plt.subplots(figsize=(15.92, width))

    # Style bg/spines/etc
    _apply_ax_dark_theme(ax)

    bar_height = 0.6 if len(process_order) < 20 else 0.4

    for row in ev.itertuples(index=False):
        start = float(row.sim_start)
        end = float(row.sim_end)
        dur = end - start
        if dur <= 0:
            continue
        ax.barh(
            y=row.y_idx,
            width=dur,
            left=start,
            height=bar_height,
            color=batch_color.get(row.batch_id, "#888888"),
            edgecolor=PALETTE["white"],    # thin white outline reads better on dark bg
            linewidth=0.3,
        )

    # Axes, labels
    ax.set_yticks(range(len(process_order)))
    ax.set_yticklabels(process_order, color=PALETTE["white"])
    ax.set_ylabel("Process (execution order)", color=PALETTE["white"])
    ax.set_xlabel("Simulation time (min)", color=PALETTE["white"])
    ax.set_title(
        "Workflow timeline",
        loc="left",
        fontweight="bold",
        color=PALETTE["white"],
        pad=15,
    )

    # Light gridlines in x for readability (white w/ low alpha)
    ax.grid(
        True,
        axis="x",
        linestyle="--",
        linewidth=0.5,
        color=PALETTE["white"],
        alpha=0.2,
    )

    # Legend (first few batches only)
    shown_batches = batches[:max_legend]
    handles = [
        Patch(
            facecolor=batch_color[b],
            edgecolor=PALETTE["white"],
            linewidth=0.3,
            label=str(b),
        )
        for b in shown_batches
    ]
    legend_title = (
        f"Batches (showing {len(shown_batches)} of {len(batches)})"
        if len(batches) > max_legend
        else "Batches"
    )
    leg = ax.legend(
        handles=handles,
        title=legend_title,
        loc="upper right",
        bbox_to_anchor=(1.25, 1.0),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=1.5,
        handlelength=1.25,
        handletextpad=0.6,
        fontsize=11,
    )
    _apply_legend_dark_theme(leg)

    fig.tight_layout()
    # fig.savefig(outpath, dpi=150, facecolor=_BG_COLOR)
    fig.savefig(outpath.with_suffix(".png"))  # 300 dpi via rcParams
    fig.savefig(outpath.with_suffix(".pdf"))  # vector, infinite zoom
    plt.close(fig)


def _plot_avg_wait_bar(kpp: pd.DataFrame, outpath: Path, top_n: int = 3) -> None:
    """
    Plot the top-N processes by average waiting time.
    This is the bottleneck slide: 'these are the steps slowing us down'.
    Dark theme, same font, same palette.
    """
    if kpp is None or kpp.empty or not {"process_id", "avg_wait_min"}.issubset(kpp.columns):
        return

    kpp2 = kpp.copy()
    kpp2["avg_wait_min"] = pd.to_numeric(kpp2["avg_wait_min"], errors="coerce")
    kpp2 = kpp2.dropna(subset=["avg_wait_min"]).sort_values(
        "avg_wait_min", ascending=False
    )

    if kpp2.empty:
        return

    top = kpp2.head(top_n)

    fig, ax = plt.subplots(figsize=(width*0.618, width*0.618*0.618))
    _apply_ax_dark_theme(ax)

    custom_colors = PALETTE["bottleneck_bars"]
    bar_colors = [custom_colors[i % len(custom_colors)] for i in range(len(top))]
    bars = ax.bar(
        top["process_id"],
        top["avg_wait_min"],
        color=bar_colors,
        edgecolor=PALETTE["white"],
        linewidth=0.6,
    )

    ax.set_xticklabels(top["process_id"], rotation=60, ha="right", color=PALETTE["white"])
    ax.set_ylabel("Average wait (min)", color=PALETTE["white"])
    ax.set_xlabel("")  # cleaner
    ax.set_title(
        f"Top {len(top)} waiting steps (bottlenecks)",
        loc="left",
        fontweight="bold",
        color=PALETTE["white"],
        pad=15,
    )

    # Annotate bars
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["white"],
        )

    fig.tight_layout()
    # fig.savefig(outpath, dpi=160, facecolor=_BG_COLOR)
    fig.savefig(outpath.with_suffix(".png"))  # 300 dpi via rcParams
    fig.savefig(outpath.with_suffix(".pdf"))  # vector, infinite zoom
    plt.close(fig)


def _plot_tat_and_sla(batch_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Two figures:
    - Distribution of per-batch turnaround times (TAT)
    - SLA compliance split (within SLA / exceeded SLA)
    Assumes batch_df is kpi_batches.csv.
    Dark-theme versions.
    """
    if batch_df is None or batch_df.empty:
        return

    # --- TAT extraction ---
    if "tat_min_events" in batch_df.columns:
        tat = pd.to_numeric(batch_df["tat_min_events"], errors="coerce").dropna()
    else:
        if {"end_min", "start_min"}.issubset(batch_df.columns):
            tat = (
                pd.to_numeric(batch_df["end_min"], errors="coerce")
                - pd.to_numeric(batch_df["start_min"], errors="coerce")
            )
            tat = tat.dropna()
        else:
            return

    # --- SLA flag ---
    if "sla_hit" in batch_df.columns:
        sla_hits = batch_df["sla_hit"].astype(bool)
    else:
        if {"sla_pct"}.issubset(batch_df.columns):
            sla_hits = batch_df["sla_pct"].fillna(0).astype(float) >= 100.0
        else:
            sla_hits = pd.Series([False] * len(batch_df), index=batch_df.index)

    # ---------- Plot TAT histogram ----------
    fig1, ax1 = plt.subplots(figsize=(width*0.618, width*0.618*0.618))
    _apply_ax_dark_theme(ax1)

    n, bins, patches = ax1.hist(
        tat,
        bins=20,
        color=PALETTE["hist_fill"],
        edgecolor=PALETTE["white"],
        alpha=0.8,
    )

    if len(tat):
        mean_tat = tat.mean()
        ax1.axvline(
            mean_tat,
            color=PALETTE["hist_mean_line"],
            linestyle="--",
            linewidth=1.25,
        )
        # label text above the line (no rotation)
        ax1.text(
            mean_tat,
            max(n) * 1.02,        # a bit above the top of the bar area
            f"Mean = {mean_tat:.0f} min",
            ha="center",           # center align horizontally
            va="bottom",           # position text above line
            color=PALETTE["white"],
            fontsize=10,
            rotation=0,            # horizontal
            fontweight="bold",
        )

    ax1.set_xlabel("Turnaround Time (minutes)", color=PALETTE["white"])
    ax1.set_ylabel("Batch count", color=PALETTE["white"])
    ax1.set_title(
        "Distribution of Batch Turnaround Times",
        loc="left",
        fontweight="bold",
        color=PALETTE["white"],
        pad=15,
    )

    fig1.tight_layout()
    fig1.savefig(out_dir / "plot_tat_distribution.png", dpi=300, facecolor=_BG_COLOR)
    fig1.savefig(out_dir / "plot_tat_distribution.pdf")  # vector version
    plt.close(fig1)

    # ---------- Plot SLA compliance ----------
    sla_rate = float(sla_hits.mean() * 100.0) if len(sla_hits) else 0.0

    fig2, ax2 = plt.subplots(figsize=(width*0.618*0.618, width*0.618*0.618))
    _apply_ax_dark_theme(ax2)

    # two bars: within SLA / exceeded SLA
    vals = [sla_rate, 100.0 - sla_rate]
    labels = ["Within SLA", "Exceeded SLA"]

    bars = ax2.bar(
        labels,
        vals,
        color=[PALETTE["sla_yes"], PALETTE["sla_no"]],
        edgecolor=PALETTE["white"],
        linewidth=0.6,
    )

    # annotate %
    for bar, v in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=PALETTE["white"],
        )

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Share of batches (%)", color=PALETTE["white"])
    ax2.set_xlabel("")  # cleaner
    ax2.set_title(
        f"SLA Compliance: {sla_rate:.1f}% of batches within target",
        loc="left",
        fontweight="bold",
        color=PALETTE["white"],
        pad=15,
    )

    fig2.tight_layout()
    fig2.savefig(out_dir / "plot_sla_compliance.png", dpi=300, facecolor=_BG_COLOR)
    fig2.savefig(out_dir / "plot_sla_compliance.pdf")  # vector version
    plt.close(fig2)



def _plot_decisions(events: pd.DataFrame, outpath: Path):
    """
    Bar chart of routing decisions (yes/no) per decision node.
    Dark theme + same palette.
    """
    d = events[events["process_id"].astype(str).str.startswith("decision:")].copy()
    if d.empty:
        return

    d["route"] = (
        d["status"].str.extract(r"ROUTED:(yes|no)", expand=False).fillna("unknown")
    )
    d["decision"] = d["process_id"].str.replace("^decision:", "", regex=True)

    piv = d.pivot_table(
        index="decision",
        columns="route",
        values="batch_id",
        aggfunc="count",
        fill_value=0,
    )

    # Make sure yes/no exist
    for col in ["yes", "no"]:
        if col not in piv.columns:
            piv[col] = 0
    piv = piv[["yes", "no"]].sort_index()

    x = np.arange(len(piv))
    w = 0.4

    fig, ax = plt.subplots(figsize=(width*0.618, width*0.618*0.618))
    _apply_ax_dark_theme(ax)

    yes_color = PALETTE["decision_yes"]
    no_color = PALETTE["decision_no"]

    ax.bar(
        x - w / 2,
        piv["yes"],
        width=w,
        label="yes",
        color=yes_color,
        edgecolor=PALETTE["white"],
        linewidth=0.6,
    )
    ax.bar(
        x + w / 2,
        piv["no"],
        width=w,
        label="no",
        color=no_color,
        edgecolor=PALETTE["white"],
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=30, ha="right", color=PALETTE["white"])
    ax.set_ylabel("routed batches", color=PALETTE["white"])
    ax.set_xlabel("")  # minimal
    ax.set_title(
        "Decision Outcomes",
        loc="left",
        fontweight="bold",
        color=PALETTE["white"],
        pad=15,
    )

    leg = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.25, 1.0),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=1.5,
        handlelength=1.25,
        handletextpad=0.6,
        fontsize=11,
    )
    _apply_legend_dark_theme(leg)

    fig.tight_layout()
    # fig.savefig(outpath, dpi=160, facecolor=_BG_COLOR)
    fig.savefig(outpath.with_suffix(".png"))  # 300 dpi via rcParams
    fig.savefig(outpath.with_suffix(".pdf"))  # vector, infinite zoom
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main(run_dir: str):
    rd = Path(run_dir)
    if not rd.exists():
        raise FileNotFoundError(f"Run directory not found: {rd}")

    # >>> apply global style ONCE <<<
    _init_matplotlib_style()

    # Load artifacts
    ev = _safe_read_csv(rd / "events_report.csv")
    kpo = _safe_read_csv(rd / "kpi_overall.csv")
    kpp = _safe_read_csv(rd / "kpi_process.csv")
    kpb = _safe_read_csv(rd / "kpi_batches.csv")

    # ---- 1. Print executive KPIs ----
    if kpo is not None and len(kpo):
        o = kpo.iloc[0]
        cols = [
            "avg_tat_min",
            "p90_tat_min",
            "sla_pct",
            "makespan_min",
            "n_batches",
            "n_samples",
            "bottleneck_process",
            "bottleneck_avg_wait_min",
            "max_queue_len",
        ]
        cols = [c for c in cols if c in kpo.columns]
        print("\n== KPI overall ==")
        print(o[cols])
    else:
        print("kpi_overall.csv missing or empty.")

    # Prepare numeric event columns for plotting
    if ev is not None and len(ev):
        for c in ["sim_start", "sim_end", "service_min", "wait_min", "queue_len_on_arrival"]:
            if c in ev.columns:
                ev[c] = pd.to_numeric(ev[c], errors="coerce")

    # ---- 2. Plot the workflow timeline ----
    if ev is not None and len(ev):
        _plot_timeline(ev, rd / "plot_timeline.png")

    # ---- 3. Plot top bottlenecks ----
    if kpp is not None and len(kpp):
        _plot_avg_wait_bar(kpp, rd / "plot_bottlenecks.png")

        # also print top-5 bottlenecks in text
        try:
            kpp2 = kpp.copy()
            kpp2["avg_wait_min"] = pd.to_numeric(kpp2["avg_wait_min"], errors="coerce")
            kpp2 = kpp2.sort_values("avg_wait_min", ascending=False)
            cols_print = [
                "process_id",
                "n_events",
                "avg_wait_min",
                "p90_wait_min",
                "avg_service_min",
                "p90_service_min",
                "max_queue_len",
            ]
            cols_print = [c for c in cols_print if c in kpp2.columns]
            print("\n== Top bottlenecks by avg_wait_min ==")
            print(kpp2[cols_print].head(5).to_string(index=False))
        except Exception:
            pass

    # ---- 4. Plot TAT distribution + SLA compliance ----
    if kpb is not None and len(kpb):
        _plot_tat_and_sla(kpb, rd)

    # ---- 5. Plot decision outcomes ----
    if ev is not None and len(ev):
        _plot_decisions(ev, rd / "plot_decisions.png")

    print(f"\nMonitor artifacts written to: {rd.resolve()}")
    for fn in [
        "plot_timeline.png",
        "plot_bottlenecks.png",
        "plot_tat_distribution.png",
        "plot_sla_compliance.png",
        "plot_decisions.png",
    ]:
        p = rd / fn
        if p.exists():
            print(" -", p.name)


if __name__ == "__main__":
    # If a run dir is passed, use that. Otherwise grab the latest baseline_* under outputs_pipeline/.
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        root = Path("outputs_pipeline")
        cand = sorted(root.glob("baseline*"))
        if not cand:
            raise SystemExit(
                "No baseline_* run found under outputs_pipeline/. Run 01_simulate first."
            )
        main(str(cand[-1]))
