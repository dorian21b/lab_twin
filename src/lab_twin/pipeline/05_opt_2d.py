# # # lab_twin/pipeline/05_opt_2d.py
# # """
# # Performs a lightweight 2D optimization sweep over (min_gap, wip_cap)
# # and writes results inside the same run folder as the baseline simulation.

# # ✅ PoC version:
# #    - Reuses latest folder under outputs_pipeline/baseline_*
# #    - Writes 2D optimization artifacts there
# #    - Keeps runtime short (limited seeds)
# # """

# # from pathlib import Path
# # import yaml
# # from lab_twin.opt.runner_2d import grid_search_2d, OptConfig

# # MAX_SEEDS_FOR_POC = 3  # runtime cap


# # def get_latest_baseline_folder(root: str = "outputs_pipeline") -> Path:
# #     """Find the latest baseline_* folder under outputs_pipeline."""
# #     root_path = Path(root)
# #     runs = sorted(root_path.glob("baseline_stress*"))
# #     if not runs:
# #         raise FileNotFoundError(f"No baseline_* runs found in {root_path}. Run 01_simulate first.")
# #     return runs[-1]  # newest


# # def main():
# #     # --- Locate baseline run ---
# #     run_dir = get_latest_baseline_folder()
# #     print(f"\n=== Running 2D optimization sweep in {run_dir.name} ===")

# #     # --- Load config ---
# #     cfg_path = Path(__file__).with_name("config.yaml")
# #     if not cfg_path.exists():
# #         raise FileNotFoundError(f"Missing config file: {cfg_path}")
# #     cfg = yaml.safe_load(cfg_path.read_text())

# #     # --- Cap seeds ---
# #     seeds = tuple(cfg.get("seeds", [1, 2, 3]))[:MAX_SEEDS_FOR_POC]
# #     print(f"Seeds (capped): {seeds}")

# #     # --- Ensure grids exist ---
# #     if "min_gap_grid" not in cfg or "wip_caps" not in cfg:
# #         raise ValueError("config.yaml must define both 'min_gap_grid' and 'wip_caps'.")

# #     # --- Prepare subfolder for optimization ---
# #     out_dir = run_dir / "opt_2d"
# #     out_dir.mkdir(exist_ok=True)

# #     # --- Run 2D sweep ---
# #     grid_search_2d(
# #         out_root=out_dir,
# #         config=OptConfig(
# #             n_batches=cfg.get("n_batches", 6),
# #             samples_per_batch=cfg.get("samples_per_batch", 96),
# #             seeds=seeds,
# #             sla_min=cfg.get("sla_min", 43200),
# #             alpha_avg=cfg.get("alpha_avg", 1.0),
# #             beta_p90=cfg.get("beta_p90", 0.5),
# #             eta_sla=cfg.get("eta_sla", 2.0),
# #             zeta_wip=cfg.get("zeta_wip", 0.0),
# #         ),
# #         min_gap_grid=tuple(cfg["min_gap_grid"]),
# #         wip_caps=tuple(cfg["wip_caps"]),
# #     )

# #     # --- Summary printout ---
# #     print("\n✅ 2D optimization complete!")
# #     print(f"Results saved under: {out_dir.resolve()}")
# #     print("Artifacts generated:")
# #     print(" - agg_2d.csv              (aggregated metrics)")
# #     print(" - heatmap_J.png           (overall score surface)")
# #     print(" - heatmap_SLA.png         (SLA surface)")
# #     print(" - best_config_summary.csv (best-performing policy)\n")


# # if __name__ == "__main__":
# #     main()


# # lab_twin/pipeline/05_opt_2d.py
# """
# 2D optimization sweep over (min_gap, wip_cap) for a given scenario run.

# What it does:
# - Takes a run directory (e.g. outputs_pipeline/stress_20251101_102200)
# - Runs grid_search_2d() on that scenario's workload assumptions
# - Saves results to <run_dir>/opt_2d/

# If you don't pass a run_dir, it auto-picks the most recent folder
# under outputs_pipeline/ matching baseline_*, stress_*, or controlled_*.

# This is what you show in slides when you say:
# "Given this intake volume, this is the best release policy."
# """

# from pathlib import Path
# import sys
# import yaml
# from lab_twin.opt.runner_2d import grid_search_2d, OptConfig

# MAX_SEEDS_FOR_POC = 3  # keep runtime short


# def pick_latest_run(root: str = "outputs_pipeline") -> Path:
#     """
#     Return the most recent scenario folder under outputs_pipeline/.
#     We accept folders that start with baseline_, stress_, or controlled_.
#     """
#     root_path = Path(root)
#     candidates = sorted([
#         p for p in root_path.glob("*_*")  # e.g. baseline_2025..., stress_2025...
#         if p.is_dir() and (
#             p.name.startswith("baseline_")
#             or p.name.startswith("stress_")
#             or p.name.startswith("controlled_")
#         )
#     ])
#     if not candidates:
#         raise FileNotFoundError(
#             f"No scenario folders found in {root_path}. "
#             f"Run 01_simulate first."
#         )
#     return candidates[-1]  # newest by sort order


# def main(run_dir_arg: str | None = None):
#     # -------- 1. Choose which scenario folder to optimize --------
#     if run_dir_arg:
#         run_dir = Path(run_dir_arg)
#         if not run_dir.exists():
#             raise FileNotFoundError(f"Provided run_dir does not exist: {run_dir}")
#     else:
#         run_dir = pick_latest_run()

#     print(f"\n=== Running 2D optimization sweep in: {run_dir} ===")

#     # -------- 2. Load global optimization config --------
#     # Note: this config is not the same as pipeline/config.yaml.
#     # This one should define the sweep ranges.
#     cfg_path = Path(__file__).with_name("config.yaml")
#     if not cfg_path.exists():
#         raise FileNotFoundError(f"Missing config file: {cfg_path}")
#     cfg = yaml.safe_load(cfg_path.read_text())

#     # Cap seeds for speed
#     seeds = tuple(cfg.get("seeds", [1, 2, 3]))[:MAX_SEEDS_FOR_POC]
#     print(f"Seeds used: {seeds}")

#     # Validate sweep grids
#     if "min_gap_grid" not in cfg or "wip_caps" not in cfg:
#         raise ValueError("config.yaml must define both 'min_gap_grid' and 'wip_caps'.")

#     # -------- 3. Create output subfolder inside the scenario run --------
#     out_dir = run_dir / "opt_2d"
#     out_dir.mkdir(exist_ok=True)

#     # -------- 4. Actually run the sweep --------
#     # Important: n_batches etc. should ideally match that scenario.
#     # We'll try to infer them from the scenario folder name or
#     # reuse what's in this optimization config as a fallback.
#     grid_search_2d(
#         out_root=out_dir,
#         config=OptConfig(
#             n_batches=cfg.get("n_batches", 6),
#             samples_per_batch=cfg.get("samples_per_batch", 96),
#             seeds=seeds,
#             sla_min=cfg.get("sla_min", 43200),     # SLA target in minutes
#             alpha_avg=cfg.get("alpha_avg", 1.0),   # weight on mean turnaround time
#             beta_p90=cfg.get("beta_p90", 0.5),     # weight on tail turnaround
#             eta_sla=cfg.get("eta_sla", 2.0),       # weight on % within SLA
#             zeta_wip=cfg.get("zeta_wip", 0.0),     # optional penalty on WIP
#         ),
#         min_gap_grid=tuple(cfg["min_gap_grid"]),   # e.g. [30, 60, 90, 120]
#         wip_caps=tuple(cfg["wip_caps"]),           # e.g. [1, 2, 3, 4]
#     )

#     # -------- 5. Tell the user what to look at --------
#     print("\n✅ 2D optimization complete!")
#     print(f"Results saved under: {out_dir.resolve()}")
#     print("Artifacts generated:")
#     print(" - agg_2d.csv              (summary of all tested policies)")
#     print(" - heatmap_J.png           (score surface: lower is better)")
#     print(" - heatmap_SLA.png         (% SLA hit across policies)")
#     print(" - best_config_summary.csv (the chosen 'best' policy for this scenario)")
#     print("You will show these two heatmaps in slides to say:")
#     print("  'Here is how intake rules change SLA and turnaround under this load.'\n")


# if __name__ == "__main__":
#     # optional CLI arg: path to a scenario folder
#     # ex:
#     #   python -m lab_twin.pipeline.05_opt_2d outputs_pipeline/stress_20251101_102200
#     # or just:
#     #   python -m lab_twin.pipeline.05_opt_2d
#     arg = sys.argv[1] if len(sys.argv) > 1 else None
#     main(arg)


# lab_twin/pipeline/05_opt_2d.py
"""
2D optimization sweep over (min_gap, wip_cap) for a given scenario run.

What it does:
- Takes a run directory (e.g. outputs_pipeline/stress_20251101_102200)
- Runs grid_search_2d() on that scenario's workload assumptions
- Saves results to <run_dir>/opt_2d/

If you don't pass a run_dir, it auto-picks the most recent folder
under outputs_pipeline/ matching baseline_*, stress_*, or controlled_*.

This is what you show in slides when you say:
"Given this intake volume, this is the best release policy."
"""

from pathlib import Path
import sys
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib.font_manager as fm

from lab_twin.opt.runner_2d import grid_search_2d, OptConfig


# =========================
# Global dark theme settings
# =========================

_BG_COLOR = "#002E54"           # same deep navy
_ACCENT_CMAP = cm.get_cmap("Accent", 5)
MAX_SEEDS_FOR_POC = 3  # keep runtime short


def _init_matplotlib_style() -> None:
    """
    Global style so 2D sweep plots (heatmaps, etc.) look like the dashboard:
    - Source Sans 3 font (light body, bold titles)
    - white text
    - dark navy background
    - outward ticks
    """
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
            # Typography
            "font.family": "Source Sans 3",
            "font.sans-serif": ["Source Sans 3"],
            "font.weight": "light",
            "axes.labelweight": "light",
            "axes.titleweight": "bold",

            # Colors
            "text.color": "white",
            "axes.labelcolor": "white",
            "axes.edgecolor": "white",
            "axes.titlecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",

            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,

            # Legend defaults
            "legend.frameon": False,
            "legend.fontsize": 11,

            # Output
            "figure.dpi": 150,
        }
    )

    # --- Ensure exports are publication-ready ---
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.facecolor"] = _BG_COLOR
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["svg.fonttype"] = "none"



def apply_dark_heatmap_theme(ax, title_text: str | None = None, cbar=None) -> None:
    """
    Helper to be used inside runner_2d after pcolormesh/imshow/etc.
    - navy bg on fig + ax
    - white ticks/labels/gridlines
    - bold white title aligned left
    - tidy colorbar to also match dark bg
    """
    fig = ax.figure
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # Spines: only bottom spine visible and white
    for spine_name, spine in ax.spines.items():
        if spine_name == "bottom":
            spine.set_visible(True)
            spine.set_color("white")
            spine.set_linewidth(1.0)
        else:
            spine.set_visible(False)

    # labels/ticks sizing
    ax.tick_params(
        axis="both",
        which="major",
        length=8,
        width=1,
        color="white",
        pad=6,
        labelsize=11,
    )

    # grid (optional subtle grid for readability on top of heatmap cell boundaries is usually noisy,
    # so we skip gridlines here in the default)

    # axis labels
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    # tick labels already white via rcParams, but make sure:
    for ticklbl in ax.get_xticklabels() + ax.get_yticklabels():
        ticklbl.set_color("white")

    # title
    if title_text is not None:
        ax.set_title(
            title_text,
            loc="left",
            fontweight="bold",
            color="white",
            fontsize=11,
            pad=15,
        )

    # colorbar formatting
    if cbar is not None:
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(1.0)
        cbar.ax.tick_params(
            colors="white",
            length=6,
            width=1,
            labelsize=10,
        )
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.figure.patch.set_facecolor(_BG_COLOR)
        cbar.ax.set_facecolor(_BG_COLOR)


# =========================
# Core logic
# =========================

def pick_latest_run(root: str = "outputs_pipeline") -> Path:
    """
    Return the most recent scenario folder under outputs_pipeline/.
    We accept folders that start with baseline_, stress_, or controlled_.
    """
    root_path = Path(root)
    candidates = sorted([
        p for p in root_path.glob("*_*")  # e.g. baseline_2025..., stress_2025...
        if p.is_dir() and (
            p.name.startswith("baseline_")
            or p.name.startswith("stress_")
            or p.name.startswith("controlled_")
        )
    ])
    if not candidates:
        raise FileNotFoundError(
            f"No scenario folders found in {root_path}. "
            f"Run 01_simulate first."
        )
    return candidates[-1]  # newest by sort order


def main(run_dir_arg: str | None = None):
    # -------- 0. Style (must be first so everything downstream inherits it)
    _init_matplotlib_style()

    # -------- 1. Choose which scenario folder to optimize --------
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        if not run_dir.exists():
            raise FileNotFoundError(f"Provided run_dir does not exist: {run_dir}")
    else:
        run_dir = pick_latest_run()

    print(f"\n=== Running 2D optimization sweep in: {run_dir} ===")

    # -------- 2. Load global optimization config --------
    # Note: this config is not the same as pipeline/config.yaml.
    # This one should define the sweep ranges.
    cfg_path = Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    # Cap seeds for speed
    seeds = tuple(cfg.get("seeds", [1, 2, 3]))[:MAX_SEEDS_FOR_POC]
    print(f"Seeds used: {seeds}")

    # Validate sweep grids
    if "min_gap_grid" not in cfg or "wip_caps" not in cfg:
        raise ValueError("config.yaml must define both 'min_gap_grid' and 'wip_caps'.")

    # -------- 3. Create output subfolder inside the scenario run --------
    out_dir = run_dir / "opt_2d"
    out_dir.mkdir(exist_ok=True)

    # -------- 4. Actually run the sweep --------
    # IMPORTANT:
    # grid_search_2d should internally generate:
    #   - agg_2d.csv
    #   - heatmap_J.png
    #   - heatmap_SLA.png
    #
    # We assume grid_search_2d will call matplotlib. Since we've already set
    # rcParams globally via _init_matplotlib_style(), those plots will:
    #   - use Source Sans 3
    #   - use white text
    #   - etc.
    #
    # To *fully* match the dashboard look, modify grid_search_2d to:
    #   - set fig.patch.set_facecolor(_BG_COLOR)
    #   - ax.set_facecolor(_BG_COLOR)
    #   - call apply_dark_heatmap_theme(ax, "...title...", cbar)
    #
    # See below for a reference implementation.
    grid_search_2d(
        out_root=out_dir,
        config=OptConfig(
            n_batches=cfg.get("n_batches", 6),
            samples_per_batch=cfg.get("samples_per_batch", 96),
            seeds=seeds,
            sla_min=cfg.get("sla_min", 43200),     # SLA target in minutes
            alpha_avg=cfg.get("alpha_avg", 1.0),   # weight on mean turnaround time
            beta_p90=cfg.get("beta_p90", 0.5),     # weight on tail turnaround
            eta_sla=cfg.get("eta_sla", 2.0),       # weight on % within SLA
            zeta_wip=cfg.get("zeta_wip", 0.0),     # optional penalty on WIP
        ),
        min_gap_grid=tuple(cfg["min_gap_grid"]),   # e.g. [30, 60, 90, 120]
        wip_caps=tuple(cfg["wip_caps"]),           # e.g. [1, 2, 3, 4]
    )

    # -------- 5. Tell the user what to look at --------
    print("\n✅ 2D optimization complete!")
    print(f"Results saved under: {out_dir.resolve()}")
    print("Artifacts generated:")
    print(" - agg_2d.csv              (summary of all tested policies)")
    print(" - heatmap_J.png           (score surface: lower is better)")
    print(" - heatmap_SLA.png         (% SLA hit across policies)")
    print(" - best_config_summary.csv (the chosen 'best' policy for this scenario)")
    print("You will show these two heatmaps in slides to say:")
    print("  'Here is how intake rules change SLA and turnaround under this load.'\n")


if __name__ == "__main__":
    # optional CLI arg: path to a scenario folder
    # ex:
    #   python -m lab_twin.pipeline.05_opt_2d outputs_pipeline/stress_20251101_102200
    # or just:
    #   python -m lab_twin.pipeline.05_opt_2d
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
