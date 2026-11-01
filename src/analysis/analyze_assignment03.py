"""Validate raw Assignment 3 data and produce reproducible summaries and plots."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {
    "method", "algorithm", "parallel_model", "vectorisation", "n", "threads",
    "actual_threads", "oversubscribed", "numeric_type", "repetition", "time_s",
    "peak_memory_mib", "structural_memory_mib",
}
FULL_SIZES = [250, 500, 750, 1000, 1500, 2000]
FULL_THREADS = [1, 2, 4, 8, 10, 12, 16, 24]
EXPECTED_REPETITIONS = {1, 2, 3, 4, 5}
HARDWARE_CONTEXTS = 12
PHYSICAL_CORE_COUNT = 10
LOGICAL_CONTEXTS = 12

SIZE_COLORS = {
    500: "#1b9e77", 1000: "#386cb0", 1500: "#e7298a", 2000: "#7570b3"
}
VEC_COLORS = {"no_vec": "#e41a1c", "autovec_O3": "#377eb8", "avx2_intrinsics": "#4daf4a"}


def _numeric_columns(df: pd.DataFrame, filepath: str) -> None:
    columns = [
        "n", "threads", "actual_threads", "oversubscribed", "repetition", "time_s",
        "peak_memory_mib", "structural_memory_mib",
    ]
    for column in columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(f"Column {column!r} contains non-numeric or non-finite values in {filepath}.")
        df[column] = converted


def _validate_configuration_grid(df: pd.DataFrame, filepath: str, allow_partial: bool) -> None:
    sizes = sorted(df["n"].astype(int).unique().tolist())
    thread_values = sorted(
        df.loc[df["parallel_model"] == "openmp", "threads"].astype(int).unique().tolist()
    )
    observed = set(
        df[["method", "algorithm", "vectorisation", "parallel_model", "n", "threads"]]
        .itertuples(index=False, name=None)
    )

    algorithm_pairs = sorted(set(zip(df["algorithm"], df["vectorisation"])))
    missing = []
    for algorithm, vectorisation in algorithm_pairs:
        seq_method = f"seq_{algorithm}"
        omp_method = f"omp_{algorithm}"
        for n in sizes:
            if (seq_method, algorithm, vectorisation, "sequential", n, 1) not in observed:
                missing.append((seq_method, n, 1))
            for threads in thread_values:
                if (omp_method, algorithm, vectorisation, "openmp", n, threads) not in observed:
                    missing.append((omp_method, n, threads))

    if missing:
        message = f"Missing matched configurations in {filepath}: {missing[:20]}"
        if allow_partial:
            print(f"[WARNING] {message}", file=sys.stderr)
        else:
            raise ValueError(message)

    if not allow_partial:
        if sizes != FULL_SIZES:
            raise ValueError(f"Expected full size sequence {FULL_SIZES}; found {sizes}.")
        if thread_values != FULL_THREADS:
            raise ValueError(f"Expected full thread sequence {FULL_THREADS}; found {thread_values}.")


def validate_parallel_dataframe(
    df: pd.DataFrame, filepath: str, allow_partial: bool = False
) -> None:
    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {filepath}: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError(f"No measurement rows found in {filepath}.")

    df = df.copy()
    _numeric_columns(df, filepath)
    text_columns = ["method", "algorithm", "parallel_model", "vectorisation", "numeric_type"]
    if df[text_columns].isna().any().any():
        raise ValueError(f"Missing string metadata in {filepath}.")
    if df["numeric_type"].ne("double").any():
        raise ValueError("This analysis expects numeric_type='double' in every row.")
    if not df["parallel_model"].isin(["sequential", "openmp"]).all():
        raise ValueError("parallel_model must be 'sequential' or 'openmp'.")
    if (df["time_s"] <= 0).any():
        raise ValueError("Execution times must be positive.")
    if (df["peak_memory_mib"] <= 0).any() or (df["structural_memory_mib"] <= 0).any():
        raise ValueError("Memory measurements must be positive.")
    if ((df["n"] <= 0) | (df["threads"] <= 0) | (df["actual_threads"] <= 0)).any():
        raise ValueError("Matrix sizes and thread counts must be positive.")
    if not df["repetition"].isin(EXPECTED_REPETITIONS).all():
        raise ValueError(f"Repetition values must be from {sorted(EXPECTED_REPETITIONS)}.")

    sequential = df["parallel_model"].eq("sequential")
    if (df.loc[sequential, "threads"] != 1).any() or (df.loc[sequential, "actual_threads"] != 1).any():
        raise ValueError("Sequential rows must record one requested and actual thread.")
    openmp = ~sequential
    if (df.loc[openmp, "threads"] != df.loc[openmp, "actual_threads"]).any():
        raise ValueError("Requested and actual OpenMP team sizes differ.")

    expected_oversubscription = (df["threads"] > HARDWARE_CONTEXTS).astype(int)
    if (df["oversubscribed"].astype(int) != expected_oversubscription).any():
        raise ValueError(f"oversubscribed must be 1 only when requested threads exceed {HARDWARE_CONTEXTS}.")

    keys = ["method", "algorithm", "vectorisation", "parallel_model", "n", "threads"]
    duplicate_config = df.duplicated(keys + ["repetition"], keep=False)
    if duplicate_config.any():
        raise ValueError(f"Duplicate configuration/repetition rows found in {filepath}.")

    for config, group in df.groupby(keys, dropna=False):
        reps = set(group["repetition"].astype(int))
        if reps != EXPECTED_REPETITIONS or len(group) != len(EXPECTED_REPETITIONS):
            raise ValueError(
                f"Configuration {config} has repetitions {sorted(reps)} (rows={len(group)}); "
                f"expected exactly {sorted(EXPECTED_REPETITIONS)}."
            )

    _validate_configuration_grid(df, filepath, allow_partial)


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "method", "algorithm", "parallel_model", "vectorisation", "n", "threads",
        "actual_threads", "oversubscribed",
    ]
    summary = df.groupby(group_cols, as_index=False).agg(
        median_time=("time_s", "median"),
        q1_time=("time_s", lambda values: np.percentile(values, 25)),
        q3_time=("time_s", lambda values: np.percentile(values, 75)),
        reps=("repetition", "count"),
        median_process_peak_mib=("peak_memory_mib", "median"),
        structural_memory_mib=("structural_memory_mib", "first"),
    )
    summary["iqr_time"] = summary["q3_time"] - summary["q1_time"]

    baseline = summary[summary["parallel_model"] == "sequential"].set_index(
        ["algorithm", "vectorisation", "n"]
    )["median_time"]

    def baseline_for(row: pd.Series) -> float:
        key = (row["algorithm"], row["vectorisation"], row["n"])
        if key not in baseline.index:
            return np.nan
        return float(baseline.loc[key])

    summary["matched_sequential_median_time"] = summary.apply(baseline_for, axis=1)
    summary["speedup_vs_seq"] = (
        summary["matched_sequential_median_time"] / summary["median_time"]
    )
    summary["efficiency"] = summary["speedup_vs_seq"] / summary["threads"]
    summary["omp_p1_overhead_pct"] = np.where(
        (summary["parallel_model"] == "openmp") & (summary["threads"] == 1),
        (summary["median_time"] / summary["matched_sequential_median_time"] - 1.0) * 100.0,
        np.nan,
    )
    return summary


def plot_runtime_scaling(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    configs = [
        ("seq_ikj", 1, "Sequential ikj", "#252525", "o", "-"),
        ("omp_ikj", 1, "OpenMP ikj, p=1", "#08519c", "s", "--"),
        ("omp_ikj", 4, "OpenMP ikj, p=4", "#3182bd", "^", "--"),
        ("omp_ikj", 8, "OpenMP ikj, p=8", "#6baed6", "v", "--"),
        ("omp_ikj", 12, "OpenMP ikj, p=12", "#006d2c", "D", "-"),
        ("omp_ikj", 24, "OpenMP ikj, p=24", "#e6550d", "P", "--"),
        ("seq_tiled_b32", 1, "Sequential tiled, B=32", "#756bb1", "X", ":"),
        ("omp_tiled_b32", 12, "OpenMP tiled, B=32, p=12", "#e7298a", "*", "-"),
    ]
    for method, threads, label, color, marker, linestyle in configs:
        part = summary[
            (summary["method"] == method)
            & (summary["threads"] == threads)
            & (summary["vectorisation"] == "autovec_O3")
        ].sort_values("n")
        if part.empty:
            continue
        ax.errorbar(
            part["n"], part["median_time"],
            yerr=[part["median_time"] - part["q1_time"], part["q3_time"] - part["median_time"]],
            label=label, color=color, marker=marker, linestyle=linestyle,
            linewidth=1.8, markersize=6, capsize=3,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Matrix dimension, n (elements per side)")
    ax.set_ylabel("Kernel-call wall time (s, log scale)")
    ax.set_title("Matrix multiplication runtime by size and configuration")
    ax.set_xticks(sorted(summary["n"].unique()))
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output_dir / "parallel_runtime_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_scaling_metric(summary: pd.DataFrame, output_dir: Path, metric: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    target_sizes = [n for n in [500, 1000, 1500, 2000] if n in set(summary["n"])]
    for n in target_sizes:
        part = summary[
            (summary["method"] == "omp_ikj")
            & (summary["n"] == n)
            & (summary["vectorisation"] == "autovec_O3")
        ].sort_values("threads")
        if part.empty:
            continue
        values = part[metric] * (100.0 if metric == "efficiency" else 1.0)
        ax.plot(
            part["threads"], values, marker="o", linewidth=2, markersize=5,
            label=f"n={n}", color=SIZE_COLORS.get(n, "#444444"),
        )

    threads = sorted(summary.loc[summary["parallel_model"] == "openmp", "threads"].unique())
    max_threads = max(threads)
    if metric == "speedup_vs_seq":
        ax.plot([1, max_threads], [1, max_threads], "k--", linewidth=1.3, label="Ideal S(p)=p")
        ax.set_ylabel("Speedup, S(p) = Tseq(n) / Tp(n)")
        ax.set_title("Matched-baseline speedup across thread counts")
    else:
        ax.axhline(100, color="black", linestyle="--", linewidth=1.3, label="Ideal efficiency (100%)")
        ax.set_ylabel("Parallel efficiency, E(p) (%)")
        ax.set_ylim(bottom=0)
        ax.set_title("Parallel efficiency across thread counts")

    ax.axvline(PHYSICAL_CORE_COUNT, color="#e6550d", linestyle=":", linewidth=1.5, label="10 physical cores")
    ax.axvline(LOGICAL_CONTEXTS, color="#de2d26", linestyle="--", linewidth=1.4, label="12 logical contexts")
    if max_threads > LOGICAL_CONTEXTS:
        ax.axvspan(LOGICAL_CONTEXTS + 0.5, max_threads + 0.5, color="#fee0d2", alpha=0.35, label="Oversubscribed")
    ax.set_xlabel("Requested and observed OpenMP team size, p (threads)")
    ax.set_xticks(threads)
    ax.set_xlim(0.5, max_threads + 0.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vectorisation_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    modes = [mode for mode in ["no_vec", "autovec_O3", "avx2_intrinsics"] if mode in set(summary["vectorisation"])]
    if len(modes) < 2:
        return
    sizes = [n for n in [1000, 2000] if n in set(summary["n"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for ax, threads, title in zip(axes, [1, 12], ["Single-threaded (p=1)", "Multi-threaded (p=12)"]):
        x = np.arange(len(sizes))
        width = 0.8 / len(modes)
        for idx, mode in enumerate(modes):
            method = "seq_ikj" if threads == 1 and mode == "autovec_O3" else (
                "seq_ikj_novec" if threads == 1 else "omp_ikj_novec" if mode == "no_vec" else
                "omp_ikj" if mode == "autovec_O3" else "seq_ikj_avx2" if threads == 1 else "omp_ikj_avx2"
            )
            values, lower, upper = [], [], []
            for n in sizes:
                part = summary[(summary["method"] == method) & (summary["n"] == n) & (summary["threads"] == threads)]
                part = part[part["vectorisation"] == mode]
                if part.empty:
                    values.append(np.nan); lower.append(0.0); upper.append(0.0)
                else:
                    row = part.iloc[0]
                    values.append(row["median_time"])
                    lower.append(row["median_time"] - row["q1_time"])
                    upper.append(row["q3_time"] - row["median_time"])
            offset = (idx - (len(modes) - 1) / 2) * width
            ax.bar(x + offset, values, width, yerr=[lower, upper], capsize=3,
                   color=VEC_COLORS[mode], label=mode, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Matrix dimension, n")
        ax.set_xticks(x, [str(n) for n in sizes])
        ax.set_ylabel("Kernel-call wall time (s)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Optional vectorisation comparison with IQR")
    fig.tight_layout()
    fig.savefig(output_dir / "vectorisation_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="Raw CSV; defaults to results/parallel_results.csv")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow a declared pilot/subset while still checking that each observed size has matched baselines and threads.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    results_csv = Path(args.csv).resolve() if args.csv else base_dir / "results" / "parallel_results.csv"
    summary_csv = results_csv.with_name(f"{results_csv.stem}_summary.csv")
    plots_dir = results_csv.with_name(f"{results_csv.stem}_plots")
    if not results_csv.is_file():
        parser.error(f"Raw results file does not exist: {results_csv}")

    df = pd.read_csv(results_csv)
    validate_parallel_dataframe(df, str(results_csv), allow_partial=args.allow_partial)
    summary = compute_summary_statistics(df)
    summary.to_csv(summary_csv, index=False, float_format="%.9g")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_runtime_scaling(summary, plots_dir)
    _plot_scaling_metric(summary, plots_dir, "speedup_vs_seq", "parallel_speedup_curves.png")
    _plot_scaling_metric(summary, plots_dir, "efficiency", "parallel_efficiency_curves.png")
    plot_vectorisation_comparison(summary, plots_dir)
    print(f"Validated {len(df)} raw measurements across {len(summary)} configurations.")
    print(f"Summary: {summary_csv}")
    print(f"Plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
