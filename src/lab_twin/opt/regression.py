# lab_twin/opt/regression.py
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_agg_2d(root: Path) -> pd.DataFrame:
    """
    Expect runner_2d.py outputs at <root>/agg_2d.csv.
    Columns needed: min_gap, wip_cap, J_mean, SLA_mean
    """
    root = Path(root)
    cand = root / "agg_2d.csv"
    if not cand.exists():
        raise FileNotFoundError(
            f"Could not find {cand}. "
            "Run: python -m lab_twin.opt.runner_2d (it generates agg_2d.csv)"
        )
    df = pd.read_csv(cand)
    req = {"min_gap", "wip_cap", "J_mean", "SLA_mean"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"agg_2d.csv missing columns: {sorted(missing)}")
    return df


def fit_forest(df: pd.DataFrame, target: str, seed: int = 42) -> tuple[RandomForestRegressor, float]:
    X = df[["min_gap", "wip_cap"]].copy()
    y = df[target].astype(float).copy()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
        oob_score=False,
    )
    rf.fit(Xtr, ytr)
    r2 = r2_score(yte, rf.predict(Xte))
    return rf, r2


def build_dense_grid(df: pd.DataFrame, gap_points: int = 61) -> pd.DataFrame:
    """
    Create a dense what-if grid across the observed ranges.
    - min_gap: linspace from min..max (gap_points points)
    - wip_cap: all integer values between min..max inclusive
    """
    gmin, gmax = float(df["min_gap"].min()), float(df["min_gap"].max())
    wmin, wmax = int(df["wip_cap"].min()), int(df["wip_cap"].max())

    gaps = np.linspace(gmin, gmax, gap_points)
    wips = np.arange(wmin, wmax + 1, 1)

    grid = pd.DataFrame(
        [(float(g), int(w)) for w in wips for g in gaps],
        columns=["min_gap", "wip_cap"]
    )
    return grid


def plot_surface_3d(grid: pd.DataFrame, zcol: str, title: str, outpath: Path) -> None:
    """
    3D surface for a metric (zcol) defined on a grid of (min_gap, wip_cap).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

    # Reshape to matrix (rows=wip_cap, cols=min_gap)
    piv = grid.pivot(index="wip_cap", columns="min_gap", values=zcol)
    X = np.array(piv.columns.tolist(), dtype=float)
    Y = np.array(piv.index.tolist(), dtype=float)
    XX, YY = np.meshgrid(X, Y)
    ZZ = piv.values

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=True)
    ax.set_xlabel("min_gap (min)")
    ax.set_ylabel("WIP cap")
    ax.set_zlabel(zcol)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_contour(grid: pd.DataFrame, zcol: str, title: str, outpath: Path) -> None:
    piv = grid.pivot(index="wip_cap", columns="min_gap", values=zcol)
    X = np.array(piv.columns.tolist(), dtype=float)
    Y = np.array(piv.index.tolist(), dtype=float)
    XX, YY = np.meshgrid(X, Y)
    ZZ = piv.values

    fig = plt.figure(figsize=(8, 5))
    cs = plt.contourf(XX, YY, ZZ, levels=15)
    plt.xlabel("min_gap (min)")
    plt.ylabel("WIP cap")
    plt.title(title)
    plt.colorbar(cs, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)


def main():
    # CLI: python -m lab_twin.opt.regression [outputs_opt_2d_dir] [out_dir]
    in_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs_opt_2d")
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs_ml") / f"ml_{_stamp()}"
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_agg_2d(in_root)

    # --- Fit models
    model_J, r2_J = fit_forest(df, target="J_mean", seed=42)
    model_S, r2_S = fit_forest(df, target="SLA_mean", seed=42)

    print(f"R² (J_mean)  = {r2_J:.4f}")
    print(f"R² (SLA_mean)= {r2_S:.4f}")

    # Save models
    joblib.dump(model_J, out_root / "rf_J_mean.pkl")
    joblib.dump(model_S, out_root / "rf_SLA_mean.pkl")

    # Feature importance
    fi_J = pd.Series(model_J.feature_importances_, index=["min_gap", "wip_cap"]).sort_values(ascending=False)
    fi_S = pd.Series(model_S.feature_importances_, index=["min_gap", "wip_cap"]).sort_values(ascending=False)
    fi = pd.DataFrame({"J_mean": fi_J, "SLA_mean": fi_S})
    fi.to_csv(out_root / "feature_importances.csv")
    print("\nFeature importances:")
    print(fi)

    # --- Dense prediction grid
    grid = build_dense_grid(df, gap_points=101)  # smoother curves
    # IMPORTANT: pass a DataFrame (keeps feature names, avoids the sklearn warning)
    grid["J_pred"] = model_J.predict(grid[["min_gap", "wip_cap"]])
    grid["SLA_pred"] = model_S.predict(grid[["min_gap", "wip_cap"]])

    # Identify AI top policies (by J), include SLA as tie-breaker
    best = (
        grid.sort_values(["J_pred", "SLA_pred"], ascending=[True, False])
            .groupby("wip_cap", as_index=False)  # one per WIP cap (handy table)
            .first()
            .sort_values("J_pred")
    )
    # Also: global top 10
    best10 = grid.sort_values(["J_pred", "SLA_pred"], ascending=[True, False]).head(10)

    grid.to_csv(out_root / "predicted_grid.csv", index=False)
    best.to_csv(out_root / "ai_best_per_wip.csv", index=False)
    best10.to_csv(out_root / "ai_best_overall_top10.csv", index=False)

    # --- Plots
    plot_surface_3d(grid, "J_pred", "AI surface: Predicted J (lower=better)", out_root / "surface3d_J.png")
    plot_contour(grid, "J_pred", "AI contour: Predicted J (lower=better)", out_root / "contour_J.png")

    plot_surface_3d(grid, "SLA_pred", "AI surface: Predicted SLA% (higher=better)", out_root / "surface3d_SLA.png")
    plot_contour(grid, "SLA_pred", "AI contour: Predicted SLA% (higher=better)", out_root / "contour_SLA.png")

    # Quick stdout summary
    print("\nAI top policies (overall):")
    print(best10[["min_gap", "wip_cap", "J_pred", "SLA_pred"]].to_string(index=False))

    print(f"\nArtifacts saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
