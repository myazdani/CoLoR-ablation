#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_parent, load_config


POST_HOC_TEXT = (
    "This follow-up is post-hoc and does not overturn the pre-registered weak_or_ambiguous call. "
    "It only informs the forward decision about whether to spend GPU on a real conditional-seed ensemble."
)
CAUTION_TEXT = (
    "The crude ensemble is biased structural perturbation, not a posterior; a positive result licenses "
    "testing the real ensemble, not a claim that uncertainty-aware selection works."
)
ENRICHED_TEXT = (
    "The enriched set is paper-pipeline-selected, not an independent gold standard; retention is a calibration diagnostic."
)

DEFAULT_ENSEMBLES = ("mild", "mild_no_full", "mid", "wide")
POSITIVE_LAMBDAS = (0.25, 0.5, 1.0, 2.0)
CONTROL_LAMBDAS = (-1.0, -0.5)
DEFAULT_RATES = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)


def _load_pretest_module():
    path = ROOT / "scripts" / "07_uncertainty_pretest.py"
    spec = importlib.util.spec_from_file_location("uncertainty_pretest_07", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


U07 = _load_pretest_module()


@dataclass(frozen=True)
class Inputs:
    results_dir: Path
    pool_meta: Path
    pretest_csv: Path
    pretest_provenance: Path
    output_csv: Path
    output_provenance: Path
    report_md: Path
    figures_dir: Path


@dataclass(frozen=True)
class FollowupDecision:
    analysis_a: str
    analysis_b: str
    mild_no_full_sign_matches: bool
    recall_ok: bool

    @property
    def call(self) -> str:
        if self.analysis_b == "positive_but_recall_loss":
            return "not_supported_investigate"
        if self.analysis_a == "underpowered" and self.analysis_b == "positive" and self.mild_no_full_sign_matches:
            return "build_real_ensemble"
        if self.analysis_b in {"null", "negative"} and self.analysis_a == "informative_null":
            return "do_not_build_sigma_fails_here"
        if self.analysis_b in {"null", "negative"}:
            return "do_not_build_strict_call_stands"
        if self.analysis_a == "informative_null" and self.analysis_b == "positive" and self.mild_no_full_sign_matches:
            return "ambiguous_inspect_lean_do_not_build"
        return "ambiguous_inspect"


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_hash(payload: dict[str, Any]) -> str:
    clean = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def _resolve_inputs(args: argparse.Namespace, config: dict[str, Any]) -> Inputs:
    pretest_provenance = Path(args.pretest_provenance)
    if not pretest_provenance.exists():
        raise FileNotFoundError(f"Missing pre-test provenance: {pretest_provenance}")
    provenance = json.loads(pretest_provenance.read_text())

    artifact_root = Path(args.artifact_root).expanduser() if args.artifact_root else None
    results_dir = (
        Path(args.results_dir).expanduser()
        if args.results_dir
        else artifact_root / "results"
        if artifact_root
        else Path(provenance["inputs"]["results_dir"])
    )
    pool_meta = (
        Path(args.pool_meta).expanduser()
        if args.pool_meta
        else artifact_root / "data" / "books_pool_meta.parquet"
        if artifact_root
        else Path(provenance["inputs"]["pool_meta"])
    )
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else report_dir / "figures"
    return Inputs(
        results_dir=results_dir,
        pool_meta=pool_meta,
        pretest_csv=Path(args.pretest_csv),
        pretest_provenance=pretest_provenance,
        output_csv=output_dir / "pretest_followup.csv",
        output_provenance=output_dir / "pretest_followup_provenance.json",
        report_md=report_dir / "followup.md",
        figures_dir=figures_dir,
    )


def _pretest_best_positive_lambda(pretest: pd.DataFrame, ensemble: str, rate: float) -> float:
    rows = pretest[
        (pretest["ensemble"] == ensemble)
        & (pretest["test"] == "c_pessimistic_selection")
        & (pretest["metric"] == "enriched_retention")
        & np.isclose(pretest["selection_rate"].astype(float), rate)
        & (pretest["lambda"].astype(float) > 0)
    ].copy()
    if rows.empty:
        raise ValueError(f"No positive lambda rows for {ensemble} at {rate}")
    rows = rows.sort_values(["point", "lambda"], ascending=[False, True])
    return float(rows.iloc[0]["lambda"])


def _pretest_gain(pretest: pd.DataFrame, ensemble: str, rate: float, lambda_value: float) -> float:
    rows = pretest[
        (pretest["ensemble"] == ensemble)
        & (pretest["test"] == "c_pessimistic_selection")
        & (pretest["metric"] == "enriched_retention_diff_vs_lambda0")
        & np.isclose(pretest["selection_rate"].astype(float), rate)
        & np.isclose(pretest["lambda"].astype(float), lambda_value)
    ]
    if rows.empty:
        raise ValueError(f"Missing pre-test gain for {ensemble}, rate={rate}, lambda={lambda_value}")
    return float(rows.iloc[0]["point"])


def _all_scores(frame: pd.DataFrame, lambda_value: float) -> np.ndarray:
    return frame["mu"].to_numpy(dtype=np.float64) + lambda_value * frame["sigma"].to_numpy(dtype=np.float64)


def _selection_arrays(frame: pd.DataFrame, rate: float, lambda_value: float) -> tuple[np.ndarray, np.ndarray, float]:
    pure = frame[~frame["enriched"]]
    enriched = frame[frame["enriched"]]
    pure_score = _all_scores(pure, lambda_value)
    enriched_score = _all_scores(enriched, lambda_value)
    selected_pure, threshold, _ = U07._selected(pure_score, rate)
    selected_enriched = enriched_score <= threshold
    return selected_pure, selected_enriched, threshold


def _mcnemar_exact_ci(b: int, c: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    discordant = b + c
    if discordant == 0:
        return 0.0, 0.0
    result = stats.binomtest(c, discordant, p=0.5)
    ci = result.proportion_ci(confidence_level=confidence, method="exact")
    return float((2 * ci.low - 1) * discordant / n_total), float((2 * ci.high - 1) * discordant / n_total)


def _analysis_a(
    *,
    frame: pd.DataFrame,
    pretest: pd.DataFrame,
    lambda_best: float,
    rates: tuple[float, ...],
    discordance_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_enriched = int(frame["enriched"].sum())
    for rate in (1 / 64, 1 / 32):
        _, base_enriched, _ = _selection_arrays(frame, rate, 0.0)
        _, best_enriched, _ = _selection_arrays(frame, rate, lambda_best)
        b = int((base_enriched & ~best_enriched).sum())
        c = int((best_enriched & ~base_enriched).sum())
        diff = (c - b) / n_enriched
        ci_low, ci_high = _mcnemar_exact_ci(b, c, n_enriched)
        mde = (ci_high - ci_low) / 2
        rows.extend(
            [
                _row("A", "mild", rate, lambda_best, "discordant_lost_b", b, n=n_enriched),
                _row("A", "mild", rate, lambda_best, "discordant_gained_c", c, n=n_enriched),
                _row("A", "mild", rate, lambda_best, "discordant_total_b_plus_c", b + c, n=n_enriched),
                _row("A", "mild", rate, lambda_best, "retention_diff_c_minus_b_over_n", diff, ci_low, ci_high, n=n_enriched),
                _row("A", "mild", rate, lambda_best, "mde_ci_half_width", mde, n=n_enriched),
            ]
        )

    looser_rates = tuple(rate for rate in rates if rate > 1 / 64)
    gains = np.asarray([_pretest_gain(pretest, "mild", rate, lambda_best) for rate in looser_rates], dtype=np.float64)
    log_x = np.log2(np.asarray(looser_rates, dtype=np.float64))
    lin_x = np.asarray(looser_rates, dtype=np.float64)
    expected_log = float(np.polyval(np.polyfit(log_x, gains, deg=1), math.log2(1 / 64)))
    expected_linear = float(np.polyval(np.polyfit(lin_x, gains, deg=1), 1 / 64))
    wide_lambda = _pretest_best_positive_lambda(pretest, "wide", 1 / 64)
    mild_no_full_lambda = _pretest_best_positive_lambda(pretest, "mild_no_full", 1 / 64)
    wide_anchor = _pretest_gain(pretest, "wide", 1 / 64, wide_lambda)
    mild_no_full_anchor = _pretest_gain(pretest, "mild_no_full", 1 / 64, mild_no_full_lambda)
    rows.extend(
        [
            _row("A", "mild", 1 / 64, lambda_best, "expected_effect_log2_extrapolated_from_looser_rates", expected_log),
            _row("A", "mild", 1 / 64, lambda_best, "expected_effect_linear_rate_extrapolated_from_looser_rates", expected_linear),
            _row("A", "wide", 1 / 64, wide_lambda, "anchor_effect_pretest_best_positive_lambda", wide_anchor),
            _row("A", "mild_no_full", 1 / 64, mild_no_full_lambda, "anchor_effect_pretest_best_positive_lambda", mild_no_full_anchor),
        ]
    )

    a64 = {row["statistic"]: row["value"] for row in rows if row["analysis"] == "A" and row["ensemble"] == "mild" and np.isclose(row["rate"], 1 / 64)}
    discordant = int(a64["discordant_total_b_plus_c"])
    mde = float(a64["mde_ci_half_width"])
    diff = float(a64["retention_diff_c_minus_b_over_n"])
    expected_primary = float(expected_log)
    if discordant <= discordance_threshold and mde > expected_primary:
        status = "underpowered"
    elif discordant > discordance_threshold and mde < expected_primary and abs(diff) <= mde:
        status = "informative_null"
    else:
        status = "indeterminate"
    details = {
        "status": status,
        "discordance_threshold": discordance_threshold,
        "discordant": discordant,
        "mde": mde,
        "retention_diff": diff,
        "expected_effect_primary_log2": expected_primary,
        "expected_effect_linear_rate": expected_linear,
        "wide_anchor": wide_anchor,
        "mild_no_full_anchor": mild_no_full_anchor,
        "lambda_best": lambda_best,
    }
    rows.append(_row("A", "mild", 1 / 64, lambda_best, f"analysis_a_status_{status}", 1.0))
    return rows, details


def _weights(rates: tuple[float, ...]) -> dict[str, np.ndarray]:
    x = np.log2(np.asarray(rates, dtype=np.float64))
    raw = np.empty(len(rates), dtype=np.float64)
    raw[0] = (x[1] - x[0]) / 2
    raw[-1] = (x[-1] - x[-2]) / 2
    for i in range(1, len(rates) - 1):
        raw[i] = (x[i + 1] - x[i - 1]) / 2
    raw = np.abs(raw)
    equal_log = raw / raw.sum()
    tail = raw.copy()
    tail[:2] *= 2.0
    tail_weighted = tail / tail.sum()
    return {"equal_log": equal_log, "tail_weighted": tail_weighted}


def _point_gain_curve(
    *,
    frame: pd.DataFrame,
    lambdas: tuple[float, ...],
    rates: tuple[float, ...],
    full_color_pure: np.ndarray,
) -> pd.DataFrame:
    pure = frame[~frame["enriched"]]
    enriched = frame[frame["enriched"]]
    records: dict[tuple[float, float], dict[str, float]] = {}
    full_selected = {rate: U07._selected(full_color_pure, rate)[0] for rate in rates}
    for rate in rates:
        for lambda_value in (0.0, *lambdas):
            pure_score = _all_scores(pure, lambda_value)
            enriched_score = _all_scores(enriched, lambda_value)
            selected_pure, threshold, _ = U07._selected(pure_score, rate)
            selected_enriched = enriched_score <= threshold
            recall = float((selected_pure & full_selected[rate]).sum() / full_selected[rate].sum())
            retention = float(selected_enriched.mean())
            records[(rate, lambda_value)] = {"retention": retention, "recall": recall}

    rows = []
    for rate in rates:
        base = records[(rate, 0.0)]
        for lambda_value in lambdas:
            current = records[(rate, lambda_value)]
            rows.append(
                {
                    "rate": rate,
                    "lambda": lambda_value,
                    "retention_gain": current["retention"] - base["retention"],
                    "recall_gain": current["recall"] - base["recall"],
                    "retention": current["retention"],
                    "recall": current["recall"],
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_integrated_gains(
    *,
    frame: pd.DataFrame,
    lambdas: tuple[float, ...],
    rates: tuple[float, ...],
    weightings: dict[str, np.ndarray],
    reps: int,
    seed: int,
    chunk_size: int,
) -> dict[str, dict[str, dict[float, dict[str, np.ndarray]]]]:
    pure = frame[~frame["enriched"]]
    enriched = frame[frame["enriched"]]
    pure_mu = pure["mu"].to_numpy(dtype=np.float64)
    pure_sigma = pure["sigma"].to_numpy(dtype=np.float64)
    enriched_mu = enriched["mu"].to_numpy(dtype=np.float64)
    enriched_sigma = enriched["sigma"].to_numpy(dtype=np.float64)
    full_pure = pure["color_full"].to_numpy(dtype=np.float64)

    out: dict[str, dict[str, dict[float, dict[str, np.ndarray]]]] = {
        name: {
            "retention": {lambda_value: np.empty(reps, dtype=np.float64) for lambda_value in lambdas},
            "recall": {lambda_value: np.empty(reps, dtype=np.float64) for lambda_value in lambdas},
        }
        for name in weightings
    }
    rng = np.random.default_rng(seed)
    n_pure = len(pure)
    n_enriched = len(enriched)
    offset = 0
    while offset < reps:
        size = min(chunk_size, reps - offset)
        pure_idx = rng.integers(0, n_pure, size=(size, n_pure), dtype=np.int32)
        enriched_idx = rng.integers(0, n_enriched, size=(size, n_enriched), dtype=np.int32)
        base_by_rate: dict[float, dict[str, np.ndarray]] = {}
        full_selected_by_rate: dict[float, np.ndarray] = {}

        pure_score0 = pure_mu
        enriched_score0 = enriched_mu
        sampled_score0 = pure_score0[pure_idx]
        sampled_enriched0 = enriched_score0[enriched_idx]
        sampled_full = full_pure[pure_idx]
        for rate in rates:
            k = max(1, int(math.floor(n_pure * rate)))
            threshold0 = np.partition(sampled_score0, k - 1, axis=1)[:, k - 1]
            threshold_full = np.partition(sampled_full, k - 1, axis=1)[:, k - 1]
            selected_enriched0 = sampled_enriched0 <= threshold0[:, None]
            selected_pure0 = sampled_score0 <= threshold0[:, None]
            selected_full = sampled_full <= threshold_full[:, None]
            denom = selected_full.sum(axis=1)
            base_by_rate[rate] = {
                "retention": selected_enriched0.mean(axis=1),
                "recall": (selected_pure0 & selected_full).sum(axis=1) / denom,
            }
            full_selected_by_rate[rate] = selected_full

        for lambda_value in lambdas:
            pure_score = pure_mu + lambda_value * pure_sigma
            enriched_score = enriched_mu + lambda_value * enriched_sigma
            sampled_pure = pure_score[pure_idx]
            sampled_enriched = enriched_score[enriched_idx]
            gain_retention = []
            gain_recall = []
            for rate in rates:
                k = max(1, int(math.floor(n_pure * rate)))
                threshold = np.partition(sampled_pure, k - 1, axis=1)[:, k - 1]
                selected_enriched = sampled_enriched <= threshold[:, None]
                selected_pure = sampled_pure <= threshold[:, None]
                selected_full = full_selected_by_rate[rate]
                denom = selected_full.sum(axis=1)
                gain_retention.append(selected_enriched.mean(axis=1) - base_by_rate[rate]["retention"])
                gain_recall.append((selected_pure & selected_full).sum(axis=1) / denom - base_by_rate[rate]["recall"])
            gain_retention_arr = np.vstack(gain_retention)
            gain_recall_arr = np.vstack(gain_recall)
            for name, weights in weightings.items():
                out[name]["retention"][lambda_value][offset : offset + size] = weights @ gain_retention_arr
                out[name]["recall"][lambda_value][offset : offset + size] = weights @ gain_recall_arr
        offset += size
    return out


def _analysis_b(
    *,
    ensemble: str,
    frame: pd.DataFrame,
    lambdas: tuple[float, ...],
    rates: tuple[float, ...],
    full_color_pure: np.ndarray,
    weightings: dict[str, np.ndarray],
    reps: int,
    seed: int,
    chunk_size: int,
    bootstrap: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    curve = _point_gain_curve(frame=frame, lambdas=lambdas, rates=rates, full_color_pure=full_color_pure)
    boot = (
        _bootstrap_integrated_gains(
            frame=frame,
            lambdas=lambdas,
            rates=rates,
            weightings=weightings,
            reps=reps,
            seed=seed,
            chunk_size=chunk_size,
        )
        if bootstrap
        else None
    )
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"lambda_star": {}, "curve": curve}
    for weighting_name, weights in weightings.items():
        integrated = []
        for lambda_value in lambdas:
            sub = curve[np.isclose(curve["lambda"], lambda_value)].sort_values("rate")
            retention_ig = float(weights @ sub["retention_gain"].to_numpy(dtype=np.float64))
            recall_ig = float(weights @ sub["recall_gain"].to_numpy(dtype=np.float64))
            if boot is not None:
                retention_low, retention_high = U07._ci_from_samples(boot[weighting_name]["retention"][lambda_value])
                recall_low, recall_high = U07._ci_from_samples(boot[weighting_name]["recall"][lambda_value])
            else:
                retention_low = retention_high = recall_low = recall_high = float("nan")
            rows.extend(
                [
                    _row("B", ensemble, np.nan, lambda_value, f"integrated_retention_gain_{weighting_name}", retention_ig, retention_low, retention_high),
                    _row("B", ensemble, np.nan, lambda_value, f"integrated_recall_gain_{weighting_name}", recall_ig, recall_low, recall_high),
                ]
            )
            integrated.append((lambda_value, retention_ig, recall_ig, retention_low, retention_high, recall_low, recall_high))
        positives = [item for item in integrated if item[0] in POSITIVE_LAMBDAS]
        lambda_star, retention_star, recall_star, retention_low, retention_high, recall_low, recall_high = max(
            positives, key=lambda item: (item[1], -item[0])
        )
        details["lambda_star"][weighting_name] = {
            "lambda": lambda_star,
            "retention": retention_star,
            "retention_ci_low": retention_low,
            "retention_ci_high": retention_high,
            "recall": recall_star,
            "recall_ci_low": recall_low,
            "recall_ci_high": recall_high,
        }
    return rows, details


def _row(
    analysis: str,
    ensemble: str,
    rate: float,
    lambda_value: float,
    statistic: str,
    value: float,
    ci_low: float = np.nan,
    ci_high: float = np.nan,
    *,
    n: int = 0,
    run_hash: str = "",
) -> dict[str, Any]:
    return {
        "run_hash": run_hash,
        "analysis": analysis,
        "ensemble": ensemble,
        "rate": rate,
        "lambda": lambda_value,
        "statistic": statistic,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
    }


def _existing_run_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, nrows=1)
    if "run_hash" not in frame.columns or frame.empty:
        raise RuntimeError(f"Refusing to overwrite {path}: existing CSV has no run_hash column")
    return str(frame["run_hash"].iloc[0])


def _setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def _plot_discordance(results: pd.DataFrame, output: Path, details: dict[str, Any]) -> None:
    plt = _setup_matplotlib()
    rows = []
    for rate in (1 / 64, 1 / 32):
        b = results[(results["analysis"] == "A") & (results["statistic"] == "discordant_lost_b") & np.isclose(results["rate"], rate)]["value"].iloc[0]
        c = results[(results["analysis"] == "A") & (results["statistic"] == "discordant_gained_c") & np.isclose(results["rate"], rate)]["value"].iloc[0]
        rows.extend([{"rate": rate, "kind": "b: lost", "count": b}, {"rate": rate, "kind": "c: gained", "count": c}])
    frame = pd.DataFrame(rows)
    labels = ["1/64", "1/32"]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for offset, kind in [(-width / 2, "b: lost"), (width / 2, "c: gained")]:
        vals = [frame[(np.isclose(frame["rate"], rate)) & (frame["kind"] == kind)]["count"].iloc[0] for rate in (1 / 64, 1 / 32)]
        ax.bar(x + offset, vals, width=width, label=kind)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Discordant enriched rows")
    ax.set_title("Analysis A: paired discordance under lambda_best")
    ax.legend()
    note = (
        f"1/64 b+c={details['discordant']}; MDE={details['mde']:.4f}; "
        f"log2 expected={details['expected_effect_primary_log2']:.4f}"
    )
    ax.text(0.01, 0.98, note, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_ig_curve(results: pd.DataFrame, output: Path, *, ensembles: tuple[str, ...]) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, weighting in zip(axes, ("equal_log", "tail_weighted")):
        for ensemble in ensembles:
            sub = results[
                (results["analysis"] == "B")
                & (results["ensemble"] == ensemble)
                & (results["statistic"] == f"integrated_retention_gain_{weighting}")
            ].sort_values("lambda")
            x = sub["lambda"].to_numpy(dtype=np.float64)
            y = sub["value"].to_numpy(dtype=np.float64)
            low = sub["ci_low"].to_numpy(dtype=np.float64)
            high = sub["ci_high"].to_numpy(dtype=np.float64)
            ax.plot(x, y, marker="o", label=ensemble)
            if np.isfinite(low).all():
                ax.fill_between(x, low, high, alpha=0.14)
            pos = sub[sub["lambda"].isin(POSITIVE_LAMBDAS)].sort_values(["value", "lambda"], ascending=[False, True]).iloc[0]
            ax.axvline(float(pos["lambda"]), color="black", linestyle=":", linewidth=0.8, alpha=0.35)
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(weighting)
        ax.set_xlabel("lambda")
        ax.set_ylabel("Integrated retention gain")
        ax.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_ig_scatter(results: pd.DataFrame, output: Path) -> None:
    plt = _setup_matplotlib()
    rows = []
    for ensemble in DEFAULT_ENSEMBLES:
        sub = results[
            (results["analysis"] == "B")
            & (results["ensemble"] == ensemble)
            & (results["statistic"] == "integrated_retention_gain_equal_log")
            & (results["lambda"].isin(POSITIVE_LAMBDAS))
        ].sort_values(["value", "lambda"], ascending=[False, True])
        if sub.empty:
            continue
        lambda_star = float(sub.iloc[0]["lambda"])
        ret = float(sub.iloc[0]["value"])
        rec = float(
            results[
                (results["analysis"] == "B")
                & (results["ensemble"] == ensemble)
                & (results["statistic"] == "integrated_recall_gain_equal_log")
                & np.isclose(results["lambda"], lambda_star)
            ]["value"].iloc[0]
        )
        rows.append({"ensemble": ensemble, "lambda": lambda_star, "retention": ret, "recall": rec})
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(frame["recall"], frame["retention"], s=56)
    for _, row in frame.iterrows():
        ax.annotate(f"{row['ensemble']} (lambda={row['lambda']:g})", (row["recall"], row["retention"]), fontsize=8)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Integrated pure-C4 recall gain")
    ax.set_ylabel("Integrated enriched-retention gain")
    ax.set_title("Integrated gain vs recall guard")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _lookup(results: pd.DataFrame, analysis: str, ensemble: str, statistic: str, *, rate: float | None = None, lambda_value: float | None = None) -> pd.Series:
    mask = (results["analysis"] == analysis) & (results["ensemble"] == ensemble) & (results["statistic"] == statistic)
    if rate is not None:
        mask &= np.isclose(results["rate"], rate)
    if lambda_value is not None:
        mask &= np.isclose(results["lambda"], lambda_value)
    rows = results[mask]
    if rows.empty:
        raise KeyError((analysis, ensemble, statistic, rate, lambda_value))
    return rows.iloc[0]


def _write_report(
    *,
    output: Path,
    results: pd.DataFrame,
    figures_dir: Path,
    a_details: dict[str, Any],
    b_details: dict[str, Any],
    decision: FollowupDecision,
    lambda_best_mild: float,
    run_hash: str,
    provenance_path: Path,
) -> None:
    mild_equal = b_details["mild"]["lambda_star"]["equal_log"]
    mild_no_full_at_mild_lambda = _lookup(
        results,
        "B",
        "mild_no_full",
        "integrated_retention_gain_equal_log",
        lambda_value=mild_equal["lambda"],
    )
    lines = [
        "# Power + Integrated-Gain Follow-Up",
        "",
        "## Executive Summary",
        "",
        POST_HOC_TEXT,
        "",
        CAUTION_TEXT,
        "",
        ENRICHED_TEXT,
        "",
        f"Forward call: **{decision.call}**.",
        "",
        "This does not change the registered pre-test call. The registered gate remains **weak_or_ambiguous**.",
        "",
        "## Pre-Registered Rules for This Follow-Up",
        "",
        "Analysis A was declared underpowered if `b + c <= 30` and the exact-CI MDE exceeded the log2-rate extrapolated expected effect. It was declared an informative null if `b + c > 30`, MDE was smaller than the expected effect, and the point estimate was approximately zero.",
        "",
        "Analysis B used a single shared lambda across all rates. The primary statistic is `IG_retention(lambda*)` under `equal_log` weights, where `lambda*` maximizes integrated retention gain over positive lambdas only. A build call requires positive `mild` IG with CI excluding zero, recall not below zero beyond its bootstrap half-width, and `mild_no_full` matching the sign at the same lambda.",
        "",
        "## Analysis A: 1/64 Power and Discordance",
        "",
        f"`mild` best positive lambda from the pre-test at 1/64: `{lambda_best_mild:g}`.",
        "",
        "| rate | b lost | c gained | b+c | diff | exact 95% CI | MDE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rate, label in ((1 / 64, "1/64"), (1 / 32, "1/32")):
        b = _lookup(results, "A", "mild", "discordant_lost_b", rate=rate)["value"]
        c = _lookup(results, "A", "mild", "discordant_gained_c", rate=rate)["value"]
        d = _lookup(results, "A", "mild", "discordant_total_b_plus_c", rate=rate)["value"]
        diff = _lookup(results, "A", "mild", "retention_diff_c_minus_b_over_n", rate=rate)
        mde = _lookup(results, "A", "mild", "mde_ci_half_width", rate=rate)["value"]
        lines.append(
            f"| {label} | {int(b)} | {int(c)} | {int(d)} | {_fmt(diff['value'])} | [{_fmt(diff['ci_low'])}, {_fmt(diff['ci_high'])}] | {_fmt(mde)} |"
        )
    lines.extend(
        [
            "",
            f"At 1/64, `b+c={a_details['discordant']}`, MDE=`{a_details['mde']:.4f}`, and the primary log2-rate expected effect is `{a_details['expected_effect_primary_log2']:.4f}`.",
            f"Analysis A status: **{a_details['status']}**.",
            "",
            f"![Analysis A discordance]({(figures_dir / 'followup_discordance_mild.png').relative_to(output.parent).as_posix()})",
            "",
            "## Analysis B: Integrated Gain Across Rates",
            "",
            "| ensemble | weighting | lambda* | IG retention | 95% CI | IG recall | 95% CI |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ensemble in DEFAULT_ENSEMBLES:
        for weighting in ("equal_log", "tail_weighted"):
            star = b_details[ensemble]["lambda_star"][weighting]
            lines.append(
                f"| {ensemble} | {weighting} | {star['lambda']:g} | {_fmt(star['retention'])} | "
                f"[{_fmt(star['retention_ci_low'])}, {_fmt(star['retention_ci_high'])}] | "
                f"{_fmt(star['recall'])} | [{_fmt(star['recall_ci_low'])}, {_fmt(star['recall_ci_high'])}] |"
            )
    recall_half_width = (mild_equal["recall_ci_high"] - mild_equal["recall_ci_low"]) / 2
    lines.extend(
        [
            "",
            f"Primary `mild` equal-log lambda* is `{mild_equal['lambda']:g}` with retention IG `{mild_equal['retention']:.4f}` and recall IG `{mild_equal['recall']:.4f}`. Recall tolerance is the bootstrap half-width `{recall_half_width:.4f}`.",
            f"`mild_no_full` at the same lambda has equal-log retention IG `{mild_no_full_at_mild_lambda['value']:.4f}`.",
            "",
            f"![Integrated retention gain curves]({(figures_dir / 'followup_ig_curves.png').relative_to(output.parent).as_posix()})",
            "",
            f"![Integrated gain vs recall guard]({(figures_dir / 'followup_ig_recall_scatter.png').relative_to(output.parent).as_posix()})",
            "",
            "## Combined Decision Table",
            "",
            "| Analysis A (1/64) | Analysis B (`mild` IG) | mild_no_full IG sign | Forward call |",
            "|---|---|---|---|",
        ]
    )
    b_status = "positive, CI excl. 0, recall ok" if decision.analysis_b == "positive" else decision.analysis_b
    sign_status = "matches" if decision.mild_no_full_sign_matches else "does not match"
    lines.append(f"| {a_details['status']} | {b_status} | {sign_status} | **{decision.call}** |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if decision.call == "build_real_ensemble":
        lines.append(
            "The registered gate failed on an underpowered cell, and pooled evidence is suggestive enough to justify the next experiment. "
            + CAUTION_TEXT
        )
    elif "do_not_build" in decision.call:
        lines.append("The follow-up does not justify GPU spend for the real ensemble at this depth. The strict call stands.")
    else:
        lines.append("The follow-up is mixed. This should be inspected before committing GPU time.")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Run hash: `{run_hash}`",
            f"- Follow-up provenance: `{provenance_path.as_posix()}`",
            "- Bootstrap resamples pure-C4 and enriched rows separately and recomputes pure-C4 thresholds within each replicate.",
            "- Analysis B uses one lambda shared across all rates; there is no per-rate argmax.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision(
    *,
    a_details: dict[str, Any],
    b_details: dict[str, Any],
    results: pd.DataFrame,
) -> FollowupDecision:
    mild_equal = b_details["mild"]["lambda_star"]["equal_log"]
    retention_positive = mild_equal["retention"] > 0 and mild_equal["retention_ci_low"] > 0
    recall_half_width = (mild_equal["recall_ci_high"] - mild_equal["recall_ci_low"]) / 2
    recall_ok = mild_equal["recall"] >= -recall_half_width
    if retention_positive and recall_ok:
        analysis_b = "positive"
    elif retention_positive and not recall_ok:
        analysis_b = "positive_but_recall_loss"
    elif mild_equal["retention_ci_high"] < 0:
        analysis_b = "negative"
    else:
        analysis_b = "null"
    mild_no_full_same_lambda = _lookup(
        results,
        "B",
        "mild_no_full",
        "integrated_retention_gain_equal_log",
        lambda_value=mild_equal["lambda"],
    )
    sign_matches = np.sign(mild_no_full_same_lambda["value"]) == np.sign(mild_equal["retention"])
    return FollowupDecision(a_details["status"], analysis_b, bool(sign_matches), bool(recall_ok))


def main() -> None:
    parser = argparse.ArgumentParser(description="Power and integrated-gain follow-up for uncertainty pre-test.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--pool-meta", default=None)
    parser.add_argument("--pretest-csv", default="results/uncertainty_pretest.csv")
    parser.add_argument("--pretest-provenance", default="results/uncertainty_pretest_provenance.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--report-dir", default="reports/uncertainty-pretest")
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1017)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=20)
    parser.add_argument("--discordance-threshold", type=int, default=30)
    parser.add_argument("--rates", default=",".join(str(rate) for rate in DEFAULT_RATES))
    parser.add_argument("--lambdas", default=",".join(str(value) for value in (*CONTROL_LAMBDAS, *POSITIVE_LAMBDAS)))
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = _resolve_inputs(args, config)
    pretest_provenance = json.loads(inputs.pretest_provenance.read_text())
    pretest = pd.read_csv(inputs.pretest_csv)
    rates = _parse_float_list(args.rates)
    lambdas = _parse_float_list(args.lambdas)
    for required in (1 / 64, 1 / 32):
        if not any(np.isclose(rate, required) for rate in rates):
            raise ValueError(f"Rates must include {required}")
    if not all(value in lambdas for value in POSITIVE_LAMBDAS):
        raise ValueError(f"Lambdas must include {POSITIVE_LAMBDAS}")

    ensembles = pretest_provenance["ensembles"]
    variants = tuple(sorted({member for name in DEFAULT_ENSEMBLES for member in ensembles[name]}))
    pool_meta, scores, alignment = U07._load_aligned_scores(
        results_dir=inputs.results_dir,
        pool_meta_path=inputs.pool_meta,
        variants=variants,
    )
    full_color = scores["full"]["color"].to_numpy(dtype=np.float64)
    pure_mask = ~pool_meta["enriched"].to_numpy(dtype=bool)
    full_color_pure = full_color[pure_mask]
    frames = {
        name: U07._ensemble_frame(
            pool_meta=pool_meta,
            scores=scores,
            members=tuple(ensembles[name]),
            full_color=full_color,
        )
        for name in DEFAULT_ENSEMBLES
    }

    run_payload = {
        "pretest_provenance_hash": _sha256(inputs.pretest_provenance),
        "pretest_csv_hash": _sha256(inputs.pretest_csv),
        "rates": rates,
        "lambdas": lambdas,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "discordance_threshold": args.discordance_threshold,
        "weightings": "equal_log_trapezoid_normalized_tail_first_two_doubled",
    }
    run_hash = _run_hash(run_payload)
    existing = _existing_run_hash(inputs.output_csv)
    if existing is not None and existing != run_hash:
        raise RuntimeError(
            f"Refusing to overwrite {inputs.output_csv}: existing run_hash {existing} does not match current {run_hash}"
        )

    lambda_best_mild = _pretest_best_positive_lambda(pretest, "mild", 1 / 64)
    all_rows, a_details = _analysis_a(
        frame=frames["mild"],
        pretest=pretest,
        lambda_best=lambda_best_mild,
        rates=rates,
        discordance_threshold=args.discordance_threshold,
    )
    weightings = _weights(rates)
    b_details: dict[str, Any] = {}
    for idx, ensemble in enumerate(DEFAULT_ENSEMBLES):
        bootstrap = ensemble in ("mild", "mild_no_full")
        rows, details = _analysis_b(
            ensemble=ensemble,
            frame=frames[ensemble],
            lambdas=lambdas,
            rates=rates,
            full_color_pure=full_color_pure,
            weightings=weightings,
            reps=args.bootstrap_reps,
            seed=args.bootstrap_seed + 10000 * idx,
            chunk_size=args.bootstrap_chunk_size,
            bootstrap=bootstrap,
        )
        all_rows.extend(rows)
        b_details[ensemble] = details

    results = pd.DataFrame(all_rows)
    results["run_hash"] = run_hash
    decision = _decision(a_details=a_details, b_details=b_details, results=results)
    results = pd.concat(
        [
            results,
            pd.DataFrame(
                [
                    _row("decision", "global", np.nan, np.nan, f"forward_call_{decision.call}", 1.0, run_hash=run_hash),
                    _row("decision", "global", np.nan, np.nan, f"analysis_a_{decision.analysis_a}", 1.0, run_hash=run_hash),
                    _row("decision", "global", np.nan, np.nan, f"analysis_b_{decision.analysis_b}", 1.0, run_hash=run_hash),
                ]
            ),
        ],
        ignore_index=True,
    )
    ensure_parent(inputs.output_csv)
    results.to_csv(inputs.output_csv, index=False)

    inputs.figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_discordance(results, inputs.figures_dir / "followup_discordance_mild.png", a_details)
    _plot_ig_curve(results, inputs.figures_dir / "followup_ig_curves.png", ensembles=("mild", "mild_no_full"))
    _plot_ig_scatter(results, inputs.figures_dir / "followup_ig_recall_scatter.png")

    provenance = {
        "post_hoc_integrity": POST_HOC_TEXT,
        "caution": CAUTION_TEXT,
        "enriched_caveat": ENRICHED_TEXT,
        "run_hash": run_hash,
        "run_payload": run_payload,
        "inputs": {
            "results_dir": str(inputs.results_dir),
            "pool_meta": str(inputs.pool_meta),
            "pretest_csv": str(inputs.pretest_csv),
            "pretest_provenance": str(inputs.pretest_provenance),
        },
        "source_hashes": {
            "pool_meta": _sha256(inputs.pool_meta),
            "pretest_csv": _sha256(inputs.pretest_csv),
            "pretest_provenance": _sha256(inputs.pretest_provenance),
            **{f"scores_{variant}": _sha256(inputs.results_dir / f"scores_{variant}.parquet") for variant in variants},
        },
        "alignment": alignment,
        "ensembles": {name: ensembles[name] for name in DEFAULT_ENSEMBLES},
        "weights": {name: values.tolist() for name, values in weightings.items()},
        "analysis_a": a_details,
        "analysis_b": {name: {k: v for k, v in details.items() if k != "curve"} for name, details in b_details.items()},
        "decision": {
            "call": decision.call,
            "analysis_a": decision.analysis_a,
            "analysis_b": decision.analysis_b,
            "mild_no_full_sign_matches": decision.mild_no_full_sign_matches,
            "recall_ok": decision.recall_ok,
        },
    }
    ensure_parent(inputs.output_provenance)
    inputs.output_provenance.write_text(json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8")
    _write_report(
        output=inputs.report_md,
        results=results,
        figures_dir=inputs.figures_dir,
        a_details=a_details,
        b_details=b_details,
        decision=decision,
        lambda_best_mild=lambda_best_mild,
        run_hash=run_hash,
        provenance_path=inputs.output_provenance,
    )
    print(json.dumps({"decision": decision.call, "csv": str(inputs.output_csv), "report": str(inputs.report_md)}, indent=2))


if __name__ == "__main__":
    main()
