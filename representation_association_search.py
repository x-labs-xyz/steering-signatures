#!/usr/bin/env python3
"""
Compute a set of representation metrics from cached steering activations
and correlate them with steerability.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


EPS = 1e-12
FEATURE_COLUMNS = [
    "fisher_trace_ratio",
    "mean_cosine_alignment_diff",
    "glue_capacity",
    "twonn_intrinsic_dimension",
]


@dataclass(frozen=True)
class CacheEntry:
    model_id: str
    dataset_id: str
    layer_idx: int
    n_alloc: int
    n_valid: int
    hidden_dim: int
    dtype: torch.dtype
    pos_path: str
    neg_path: str
    meta_path: str
    mtime: float


def _infer_model_id_from_pickle(path: str) -> str:
    name = os.path.basename(path)
    marker = "_diffmeans"
    if marker in name:
        return name.split(marker, 1)[0]
    return os.path.splitext(name)[0]


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _torch_dtype_from_string(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).replace("torch.", "")
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported dtype string: {dtype_name}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid dtype string: {dtype_name}")
    return dtype


def _select_better_cache(a: CacheEntry, b: CacheEntry) -> CacheEntry:
    if a.n_valid != b.n_valid:
        return a if a.n_valid > b.n_valid else b
    if a.n_alloc != b.n_alloc:
        return a if a.n_alloc > b.n_alloc else b
    return a if a.mtime >= b.mtime else b


def load_steerability_rows(pickle_paths: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in pickle_paths:
        model_id = _infer_model_id_from_pickle(path)
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, dict):
            continue

        datasets = list(obj.get("dataset", []))
        layers = list(obj.get("layer", []))
        results = list(obj.get("results", []))
        n_rows = min(len(datasets), len(layers), len(results))
        for idx in range(n_rows):
            result = results[idx] if isinstance(results[idx], dict) else {}
            lambdas = result.get("lambda_values", [])
            mean_ld = result.get("mean_logit_diffs", [])
            rows.append(
                {
                    "model_id": str(model_id),
                    "dataset_id": str(datasets[idx]),
                    "layer_idx": int(layers[idx]),
                    "steerability": _safe_float(result.get("steerability")),
                    "steer_intercept": _safe_float(result.get("intercept")),
                    "lambda_min": _safe_float(min(lambdas)) if lambdas else float("nan"),
                    "lambda_max": _safe_float(max(lambdas)) if lambdas else float("nan"),
                    "lambda_count": int(len(lambdas)) if isinstance(lambdas, list) else 0,
                    "mld_span": (
                        _safe_float(max(mean_ld) - min(mean_ld))
                        if isinstance(mean_ld, list) and mean_ld
                        else float("nan")
                    ),
                    "source_pickle": path,
                }
            )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise ValueError("No steerability rows were loaded from the provided pickles.")
    return dataframe


def build_cache_index(cache_dir: str) -> Dict[Tuple[str, str, int], CacheEntry]:
    index: Dict[Tuple[str, str, int], CacheEntry] = {}
    meta_paths = glob.glob(os.path.join(cache_dir, "*__meta.json"))
    for meta_path in meta_paths:
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            entry = CacheEntry(
                model_id=str(meta["model_id"]),
                dataset_id=str(meta["dataset_id"]),
                layer_idx=int(meta["layer_idx"]),
                n_alloc=int(meta["N"]),
                n_valid=max(0, min(int(meta.get("count", meta["N"])), int(meta["N"]))),
                hidden_dim=int(meta["d"]),
                dtype=_torch_dtype_from_string(str(meta["dtype"])),
                pos_path=meta_path.replace("__meta.json", "__pos.dat"),
                neg_path=meta_path.replace("__meta.json", "__neg.dat"),
                meta_path=meta_path,
                mtime=os.path.getmtime(meta_path),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

        if not os.path.exists(entry.pos_path) or not os.path.exists(entry.neg_path):
            continue

        key = (entry.model_id, entry.dataset_id, entry.layer_idx)
        if key in index:
            index[key] = _select_better_cache(index[key], entry)
        else:
            index[key] = entry
    return index


def open_memmap_2d(path: str, n_rows: int, hidden_dim: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_file(
        path,
        shared=False,
        size=int(n_rows) * int(hidden_dim),
        dtype=dtype,
    ).view(int(n_rows), int(hidden_dim))


def _subsample_rows(
    x: torch.Tensor,
    max_rows: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    idx_np = rng.choice(x.shape[0], size=max_rows, replace=False)
    idx_np.sort()
    return x.index_select(0, torch.from_numpy(idx_np).long())


def _subsample_paired_rows(
    pos: torch.Tensor,
    neg: torch.Tensor,
    max_rows: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_rows <= 0:
        return pos, neg
    if pos.shape[0] == neg.shape[0] and pos.shape[0] > max_rows:
        idx_np = rng.choice(pos.shape[0], size=max_rows, replace=False)
        idx_np.sort()
        idx = torch.from_numpy(idx_np).long()
        return pos.index_select(0, idx), neg.index_select(0, idx)
    return _subsample_rows(pos, max_rows, rng), _subsample_rows(neg, max_rows, rng)


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    sorted_vals = values[order]
    i = 0
    n = values.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x.astype(float, copy=False) - float(np.mean(x))
    y = y.astype(float, copy=False) - float(np.mean(y))
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denom <= EPS:
        return float("nan")
    return float(np.dot(x, y) / denom)


def spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return pearsonr_np(_rankdata_average(x), _rankdata_average(y))


def fisher_pvalue_from_r(r: float, n: int, n_controls: int = 0) -> float:
    if not np.isfinite(r) or n <= (n_controls + 3):
        return float("nan")
    r_clip = float(np.clip(r, -0.999999999, 0.999999999))
    z = 0.5 * math.log((1.0 + r_clip) / (1.0 - r_clip))
    z *= math.sqrt(float(n - n_controls - 3))
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    if valid.sum() == 0:
        return qvals

    ranked = pvals[valid]
    m = ranked.size
    order = np.argsort(ranked, kind="mergesort")
    sorted_p = ranked[order]
    raw = sorted_p * m / np.arange(1, m + 1, dtype=float)
    raw = np.minimum.accumulate(raw[::-1])[::-1]
    raw = np.clip(raw, 0.0, 1.0)
    unsorted = np.empty_like(sorted_p)
    unsorted[order] = raw
    qvals[valid] = unsorted
    return qvals


def _design_matrix(df: pd.DataFrame, controls: Sequence[str]) -> np.ndarray:
    blocks: List[np.ndarray] = [np.ones((len(df), 1), dtype=float)]
    for col in controls:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            blocks.append(series.to_numpy(dtype=float).reshape(-1, 1))
        else:
            dummies = pd.get_dummies(series.astype(str), drop_first=True)
            if dummies.shape[1] > 0:
                blocks.append(dummies.to_numpy(dtype=float))
    return np.concatenate(blocks, axis=1)


def _partial_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    controls: Sequence[str],
    *,
    method: str,
) -> float:
    if len(df) < 3:
        return float("nan")
    available_controls = [col for col in controls if col in df.columns]
    cols = [x_col, y_col, *available_controls]
    sub = df.loc[:, cols].copy()

    mask = np.ones(len(sub), dtype=bool)
    for col in cols:
        series = sub[col]
        if pd.api.types.is_numeric_dtype(series):
            mask &= np.isfinite(series.to_numpy(dtype=float))
        else:
            mask &= series.notna().to_numpy(dtype=bool, copy=False)
    sub = sub.loc[mask].copy()
    if len(sub) < 3:
        return float("nan")

    if method == "spearman":
        sub[x_col] = _rankdata_average(sub[x_col].to_numpy(dtype=float))
        sub[y_col] = _rankdata_average(sub[y_col].to_numpy(dtype=float))
        for col in available_controls:
            if pd.api.types.is_numeric_dtype(sub[col]):
                sub[col] = _rankdata_average(sub[col].to_numpy(dtype=float))
    elif method != "pearson":
        raise ValueError(f"Unsupported method: {method}")

    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    if not available_controls:
        return spearmanr_np(x, y) if method == "spearman" else pearsonr_np(x, y)

    controls_mat = _design_matrix(sub, available_controls)
    beta_x, *_ = np.linalg.lstsq(controls_mat, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(controls_mat, y, rcond=None)
    residual_x = x - controls_mat @ beta_x
    residual_y = y - controls_mat @ beta_y
    return pearsonr_np(residual_x, residual_y)


def _group_weighted_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    min_group_size: int,
) -> Tuple[float, float, int]:
    rs: List[float] = []
    weights: List[int] = []
    for _, group in df.groupby(group_col):
        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        n_valid = int(mask.sum())
        if n_valid < min_group_size:
            continue
        corr = pearsonr_np(x[mask], y[mask])
        if np.isfinite(corr):
            rs.append(float(corr))
            weights.append(n_valid)
    if not rs:
        return float("nan"), float("nan"), 0
    return (
        float(np.average(np.asarray(rs), weights=np.asarray(weights))),
        float(np.median(np.asarray(rs))),
        len(rs),
    )


def _mean_cosine_to_reference(reference: torch.Tensor, vectors: torch.Tensor) -> float:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return float("nan")
    ref_norm = torch.linalg.vector_norm(reference).item()
    if ref_norm <= EPS:
        return float("nan")
    ref_unit = reference / ref_norm
    vec_norms = torch.linalg.vector_norm(vectors, dim=1)
    valid = vec_norms > EPS
    if not bool(valid.any().item()):
        return float("nan")
    cosine = (vectors[valid] @ ref_unit) / vec_norms[valid]
    return float(cosine.mean().item())


def _zscore_features(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    if x.shape[0] > 1:
        std = x.std(dim=0, unbiased=True, keepdim=True)
    else:
        std = torch.ones_like(mean)
    std = torch.where(torch.isfinite(std), std, torch.ones_like(std))
    return (x - mean) / std.clamp_min(EPS)


def _rank_gaussianize_features(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    n_rows, n_cols = x.shape
    if n_rows < 2 or n_cols < 1:
        return x.clone()
    order = torch.argsort(x, dim=0)
    ranks = torch.empty((n_rows, n_cols), dtype=torch.float32, device=x.device)
    base = torch.arange(n_rows, dtype=torch.float32, device=x.device).unsqueeze(1)
    ranks.scatter_(0, order, base.expand(n_rows, n_cols) + 0.5)
    u = (ranks / float(n_rows)).clamp(1e-6, 1.0 - 1e-6)
    gaussian = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return _zscore_features(gaussian)


def _gaussianize_pooled_manifolds(
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pooled = torch.cat([pos, neg], dim=0)
    gaussianized = _rank_gaussianize_features(pooled)
    return gaussianized[: pos.shape[0]], gaussianized[pos.shape[0] :]


def _projected_nnls(
    gram: torch.Tensor,
    target: torch.Tensor,
    *,
    max_steps: int,
    tol: float,
) -> torch.Tensor:
    gram = gram.double()
    target = target.double()
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Expected a square Gram matrix.")
    if target.ndim != 1 or target.shape[0] != gram.shape[0]:
        raise ValueError("Target shape must match Gram matrix.")
    if gram.shape[0] == 0:
        return torch.zeros(0, dtype=gram.dtype, device=gram.device)

    eigvals = torch.linalg.eigvalsh(gram)
    lipschitz = max(float(eigvals.max().item()), EPS)
    weights = torch.zeros(gram.shape[0], dtype=torch.double, device=gram.device)
    for _ in range(max_steps):
        grad = gram @ weights - target
        next_weights = torch.clamp(weights - grad / lipschitz, min=0.0)
        delta = torch.linalg.vector_norm(next_weights - weights).item()
        scale = 1.0 + torch.linalg.vector_norm(weights).item()
        weights = next_weights
        if delta <= tol * scale:
            break
    return weights


def _binary_glue_anchor_matrix(
    pos_signed: torch.Tensor,
    neg_signed: torch.Tensor,
    probe_t: torch.Tensor,
    *,
    max_steps: int,
    tol: float,
) -> torch.Tensor:
    n_pos = pos_signed.shape[0]
    n_neg = neg_signed.shape[0]
    if n_pos < 1 or n_neg < 1:
        raise ValueError("Binary GLUE requires at least one example per class.")

    combined = torch.cat([pos_signed, neg_signed], dim=0).double()
    probe_t = probe_t.double()
    gram = combined @ combined.T
    target = combined @ probe_t
    weights = _projected_nnls(gram, target, max_steps=max_steps, tol=tol)

    def weighted_anchor(rows: torch.Tensor, local_weights: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        total_weight = float(local_weights.sum().item())
        if total_weight > EPS:
            return ((local_weights[:, None] * rows.double()).sum(dim=0) / total_weight).float()
        best_idx = int(torch.argmax(scores).item())
        return rows[best_idx].float()

    pos_anchor = weighted_anchor(pos_signed, weights[:n_pos], target[:n_pos])
    neg_anchor = weighted_anchor(neg_signed, weights[n_pos : n_pos + n_neg], target[n_pos : n_pos + n_neg])
    return torch.stack([pos_anchor, neg_anchor], dim=0)


def _glue_quadratic_form(
    basis: torch.Tensor,
    probe_t: torch.Tensor,
    *,
    extra_gram: Optional[torch.Tensor] = None,
) -> float:
    basis = basis.double()
    probe_t = probe_t.double()
    rhs = basis @ probe_t
    gram = basis @ basis.T
    if extra_gram is not None:
        gram = gram + extra_gram.double()
    return float((rhs @ torch.linalg.pinv(gram) @ rhs).item())


def compute_glue_capacity(
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    rng: np.random.Generator,
    max_rows: int,
    t_samples: int,
    opt_steps: int,
    tol: float,
    gaussianize: bool,
) -> float:
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
        return float("nan")
    if pos.shape[0] < 1 or neg.shape[0] < 1 or t_samples <= 0:
        return float("nan")

    pos, neg = _subsample_paired_rows(pos, neg, max_rows, rng)
    pos = pos.float()
    neg = neg.float()
    if gaussianize:
        pos, neg = _gaussianize_pooled_manifolds(pos, neg)

    pos_signed = pos
    neg_signed = -neg
    dim = pos.shape[1]
    anchor_mats: List[torch.Tensor] = []
    probes: List[torch.Tensor] = []
    a_values: List[float] = []
    for _ in range(t_samples):
        probe_np = rng.standard_normal(dim).astype(np.float32, copy=False)
        probe_t = torch.from_numpy(probe_np)
        anchor_mat = _binary_glue_anchor_matrix(
            pos_signed,
            neg_signed,
            probe_t,
            max_steps=opt_steps,
            tol=tol,
        )
        anchor_mats.append(anchor_mat)
        probes.append(probe_t)
        a_values.append(_glue_quadratic_form(anchor_mat, probe_t))

    if not a_values:
        return float("nan")
    mean_a = float(np.mean(a_values))
    return float(2.0 / mean_a) if mean_a > EPS else float("nan")


@torch.no_grad()
def two_nn_intrinsic_dimension(
    x: torch.Tensor,
    *,
    eps: float = EPS,
) -> float:
    if x.ndim != 2 or x.shape[0] < 3:
        return float("nan")

    x = x.float()
    dists = torch.cdist(x, x, p=2)
    dists.fill_diagonal_(float("inf"))
    nearest, _ = torch.topk(dists, k=2, dim=1, largest=False)
    r1 = nearest[:, 0].clamp_min(eps)
    r2 = nearest[:, 1].clamp_min(eps)
    mu = (r2 / r1).clamp_min(1.0 + eps).sort().values
    empirical_cdf = torch.arange(1, x.shape[0] + 1, dtype=mu.dtype) / (x.shape[0] + 1.0)
    xx = torch.log(mu)
    yy = -torch.log((1.0 - empirical_cdf).clamp_min(eps))
    denom = torch.sum(xx * xx)
    if float(denom.item()) <= eps:
        return float("nan")
    return float((torch.sum(xx * yy) / denom).item())


def compute_representation_metrics(
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    rng: np.random.Generator,
    twonn_max_rows: int,
    twonn_max_dims: int,
    glue_max_rows: int,
    glue_t_samples: int,
    glue_opt_steps: int,
    glue_tol: float,
    glue_gaussianize: bool,
) -> Dict[str, float]:
    pos = pos.float()
    neg = neg.float()

    mu_pos = pos.mean(dim=0)
    mu_neg = neg.mean(dim=0)
    delta = mu_pos - mu_neg
    var_pos = pos.var(dim=0, unbiased=False)
    var_neg = neg.var(dim=0, unbiased=False)

    fisher_trace_ratio = float(
        delta.pow(2).sum().item() / (var_pos.sum().item() + var_neg.sum().item() + EPS)
    )
    m_pair = min(pos.shape[0], neg.shape[0])
    if m_pair > 0:
        mean_cosine_alignment_diff = _mean_cosine_to_reference(
            delta,
            pos[:m_pair] - neg[:m_pair],
        )
    else:
        mean_cosine_alignment_diff = float("nan")

    twonn_pos, twonn_neg = _subsample_paired_rows(pos, neg, twonn_max_rows, rng)
    if twonn_max_dims > 0 and twonn_pos.shape[1] > twonn_max_dims:
        dim_idx_np = rng.choice(twonn_pos.shape[1], size=twonn_max_dims, replace=False)
        dim_idx_np.sort()
        dim_idx = torch.from_numpy(dim_idx_np).long()
        twonn_pos = twonn_pos.index_select(1, dim_idx)
        twonn_neg = twonn_neg.index_select(1, dim_idx)
    twonn_intrinsic = two_nn_intrinsic_dimension(torch.cat([twonn_pos, twonn_neg], dim=0))

    glue_capacity = compute_glue_capacity(
        pos,
        neg,
        rng=rng,
        max_rows=glue_max_rows,
        t_samples=glue_t_samples,
        opt_steps=glue_opt_steps,
        tol=glue_tol,
        gaussianize=glue_gaussianize,
    )

    return {
        "fisher_trace_ratio": fisher_trace_ratio,
        "mean_cosine_alignment_diff": mean_cosine_alignment_diff,
        "glue_capacity": glue_capacity,
        "twonn_intrinsic_dimension": twonn_intrinsic,
    }


def sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return safe.strip("._") or "scope"


def compute_correlation_table(
    df: pd.DataFrame,
    *,
    target_col: str,
    min_group_size: int,
    alpha: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            continue
        x = df[feature].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        n_valid = int(mask.sum())
        if n_valid < 6:
            continue

        sub = df.loc[mask].copy()
        pearson = pearsonr_np(x[mask], y[mask])
        spearman = spearmanr_np(x[mask], y[mask])
        partial_layer = _partial_corr(sub, feature, target_col, ["layer_idx"], method="pearson")
        partial_dataset = _partial_corr(sub, feature, target_col, ["dataset_id"], method="pearson")
        partial_layer_dataset = _partial_corr(
            sub,
            feature,
            target_col,
            ["layer_idx", "dataset_id"],
            method="pearson",
        )
        partial_spearman_layer = _partial_corr(
            sub,
            feature,
            target_col,
            ["layer_idx"],
            method="spearman",
        )
        partial_spearman_dataset = _partial_corr(
            sub,
            feature,
            target_col,
            ["dataset_id"],
            method="spearman",
        )
        partial_spearman_layer_dataset = _partial_corr(
            sub,
            feature,
            target_col,
            ["layer_idx", "dataset_id"],
            method="spearman",
        )
        within_layer_weighted, within_layer_median, n_layers = _group_weighted_corr(
            sub,
            feature,
            target_col,
            "layer_idx",
            min_group_size,
        )
        within_dataset_weighted, within_dataset_median, n_datasets = _group_weighted_corr(
            sub,
            feature,
            target_col,
            "dataset_id",
            min_group_size,
        )

        rows.append(
            {
                "feature": feature,
                "n": n_valid,
                "pearson_r": pearson,
                "spearman_r": spearman,
                "pearson_p": fisher_pvalue_from_r(pearson, n=n_valid, n_controls=0),
                "spearman_p": fisher_pvalue_from_r(spearman, n=n_valid, n_controls=0),
                "partial_r_layer": partial_layer,
                "partial_r_dataset": partial_dataset,
                "partial_r_layer_dataset": partial_layer_dataset,
                "partial_spearman_r_layer": partial_spearman_layer,
                "partial_spearman_r_dataset": partial_spearman_dataset,
                "partial_spearman_r_layer_dataset": partial_spearman_layer_dataset,
                "partial_p_layer": fisher_pvalue_from_r(partial_layer, n=n_valid, n_controls=1),
                "partial_p_dataset": fisher_pvalue_from_r(partial_dataset, n=n_valid, n_controls=1),
                "partial_p_layer_dataset": fisher_pvalue_from_r(partial_layer_dataset, n=n_valid, n_controls=2),
                "partial_spearman_p_layer": fisher_pvalue_from_r(partial_spearman_layer, n=n_valid, n_controls=1),
                "partial_spearman_p_dataset": fisher_pvalue_from_r(partial_spearman_dataset, n=n_valid, n_controls=1),
                "partial_spearman_p_layer_dataset": fisher_pvalue_from_r(
                    partial_spearman_layer_dataset,
                    n=n_valid,
                    n_controls=2,
                ),
                "within_layer_weighted_r": within_layer_weighted,
                "within_layer_median_r": within_layer_median,
                "n_layers_used": n_layers,
                "within_dataset_weighted_r": within_dataset_weighted,
                "within_dataset_median_r": within_dataset_median,
                "n_datasets_used": n_datasets,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["pearson_q_fdr"] = bh_fdr(out["pearson_p"].to_numpy(dtype=float))
    out["spearman_q_fdr"] = bh_fdr(out["spearman_p"].to_numpy(dtype=float))
    out["partial_q_layer_fdr"] = bh_fdr(out["partial_p_layer"].to_numpy(dtype=float))
    out["partial_q_dataset_fdr"] = bh_fdr(out["partial_p_dataset"].to_numpy(dtype=float))
    out["partial_q_layer_dataset_fdr"] = bh_fdr(out["partial_p_layer_dataset"].to_numpy(dtype=float))
    out["partial_spearman_q_layer_fdr"] = bh_fdr(out["partial_spearman_p_layer"].to_numpy(dtype=float))
    out["partial_spearman_q_dataset_fdr"] = bh_fdr(out["partial_spearman_p_dataset"].to_numpy(dtype=float))
    out["partial_spearman_q_layer_dataset_fdr"] = bh_fdr(
        out["partial_spearman_p_layer_dataset"].to_numpy(dtype=float)
    )

    out["pearson_sig_fdr"] = out["pearson_q_fdr"] <= float(alpha)
    out["spearman_sig_fdr"] = out["spearman_q_fdr"] <= float(alpha)
    out["partial_sig_layer_fdr"] = out["partial_q_layer_fdr"] <= float(alpha)
    out["partial_sig_dataset_fdr"] = out["partial_q_dataset_fdr"] <= float(alpha)
    out["partial_sig_layer_dataset_fdr"] = out["partial_q_layer_dataset_fdr"] <= float(alpha)
    out["partial_spearman_sig_layer_fdr"] = out["partial_spearman_q_layer_fdr"] <= float(alpha)
    out["partial_spearman_sig_dataset_fdr"] = out["partial_spearman_q_dataset_fdr"] <= float(alpha)
    out["partial_spearman_sig_layer_dataset_fdr"] = (
        out["partial_spearman_q_layer_dataset_fdr"] <= float(alpha)
    )

    out["abs_spearman_r"] = out["spearman_r"].abs()
    out["abs_partial_r_layer_dataset"] = out["partial_r_layer_dataset"].abs()
    out["ranking_score"] = (
        out["abs_spearman_r"]
        + out["abs_partial_r_layer_dataset"]
        + out["within_layer_weighted_r"].abs().fillna(0.0)
    ) / 3.0
    out = out.sort_values(
        ["ranking_score", "abs_partial_r_layer_dataset", "abs_spearman_r"],
        ascending=False,
    ).reset_index(drop=True)
    return out


def maybe_filter_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.models:
        out = out[out["model_id"].isin(set(args.models))]
    if args.datasets:
        out = out[out["dataset_id"].isin(set(args.datasets))]
    if args.exclude_datasets:
        out = out[~out["dataset_id"].isin(set(args.exclude_datasets))]
    if args.layers:
        out = out[out["layer_idx"].isin(set(args.layers))]
    if args.limit and args.limit > 0:
        out = out.iloc[: args.limit].copy()
    return out.reset_index(drop=True)


def extract_feature_table(
    steer_df: pd.DataFrame,
    cache_index: Dict[Tuple[str, str, int], CacheEntry],
    *,
    rng: np.random.Generator,
    activation_max_rows: int,
    twonn_max_rows: int,
    twonn_max_dims: int,
    glue_max_rows: int,
    glue_t_samples: int,
    glue_opt_steps: int,
    glue_tol: float,
    glue_gaussianize: bool,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    missing = 0
    errors = 0
    total = len(steer_df)
    for idx, row in enumerate(steer_df.itertuples(index=False), start=1):
        key = (str(row.model_id), str(row.dataset_id), int(row.layer_idx))
        entry = cache_index.get(key)
        if entry is None:
            missing += 1
            continue
        try:
            pos = open_memmap_2d(entry.pos_path, entry.n_alloc, entry.hidden_dim, entry.dtype)[: entry.n_valid]
            neg = open_memmap_2d(entry.neg_path, entry.n_alloc, entry.hidden_dim, entry.dtype)[: entry.n_valid]
            pos, neg = _subsample_paired_rows(pos, neg, activation_max_rows, rng)
            metrics = compute_representation_metrics(
                pos,
                neg,
                rng=rng,
                twonn_max_rows=twonn_max_rows,
                twonn_max_dims=twonn_max_dims,
                glue_max_rows=glue_max_rows,
                glue_t_samples=glue_t_samples,
                glue_opt_steps=glue_opt_steps,
                glue_tol=glue_tol,
                glue_gaussianize=glue_gaussianize,
            )
            record = {
                "model_id": row.model_id,
                "dataset_id": row.dataset_id,
                "layer_idx": int(row.layer_idx),
                "steerability": float(row.steerability),
                "steer_intercept": float(row.steer_intercept),
                "lambda_min": float(row.lambda_min),
                "lambda_max": float(row.lambda_max),
                "lambda_count": int(row.lambda_count),
                "mld_span": float(row.mld_span),
                "source_pickle": row.source_pickle,
                "cache_meta_path": entry.meta_path,
                "cache_pos_path": entry.pos_path,
                "cache_neg_path": entry.neg_path,
                "cache_n_alloc": entry.n_alloc,
                "cache_n_valid": entry.n_valid,
                "cache_dim": entry.hidden_dim,
            }
            record.update(metrics)
            records.append(record)
        except Exception as exc:  # pylint: disable=broad-except
            errors += 1
            print(f"[warn] failed for {key}: {exc}")

        if idx % 25 == 0 or idx == total:
            print(
                f"[progress] {idx}/{total} rows processed | kept={len(records)} "
                f"missing={missing} errors={errors}"
            )

    print(
        f"[summary] extracted rows={len(records)} missing_cache={missing} failed_rows={errors}"
    )
    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("Feature extraction produced no rows.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate steerability with a small set of representation metrics.",
    )
    parser.add_argument(
        "--steering-pkls",
        nargs="+",
        required=True,
        help="Pickle files produced by steering runs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="activation_cache_torch",
        help="Directory containing activation cache memmaps.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="representational_correlations",
        help="Output directory for feature and correlation CSVs.",
    )
    parser.add_argument(
        "--activation-max-rows",
        type=int,
        default=256,
        help="Max rows per class loaded before metric computation. <=0 uses all rows.",
    )
    parser.add_argument(
        "--twonn-max-rows",
        type=int,
        default=192,
        help="Max rows per class used for TwoNN.",
    )
    parser.add_argument(
        "--twonn-max-dims",
        type=int,
        default=128,
        help="Max hidden dimensions used for TwoNN.",
    )
    parser.add_argument(
        "--glue-max-rows",
        type=int,
        default=50,
        help="Max rows per class used for GLUE capacity.",
    )
    parser.add_argument(
        "--glue-t-samples",
        type=int,
        default=200,
        help="Number of Gaussian probes used for GLUE capacity.",
    )
    parser.add_argument(
        "--glue-opt-steps",
        type=int,
        default=250,
        help="Projected-gradient steps for the GLUE anchor solver.",
    )
    parser.add_argument(
        "--glue-tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for the GLUE anchor solver.",
    )
    parser.add_argument(
        "--no-glue-gaussianize",
        action="store_true",
        help="Disable pooled featurewise Gaussianization before GLUE estimation.",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Optional model_id filter.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset_id filter.")
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=None,
        help="Optional dataset_id exclusion list.",
    )
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="Optional layer filter.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=8,
        help="Minimum samples per layer or dataset for grouped correlations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR significance threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many rows to print per output scope.",
    )
    parser.add_argument(
        "--load-features-csv",
        type=str,
        default="",
        help="Optional precomputed feature table.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.load_features_csv:
        feature_df = pd.read_csv(args.load_features_csv)
        feature_df = maybe_filter_rows(feature_df, args)
        print(f"[load] loaded feature table from {args.load_features_csv} ({len(feature_df)} rows)")
    else:
        steer_df = load_steerability_rows(args.steering_pkls)
        steer_df = maybe_filter_rows(steer_df, args)
        print(f"[load] steerability rows after filters: {len(steer_df)}")
        cache_index = build_cache_index(args.cache_dir)
        print(f"[load] cache index keys: {len(cache_index)}")
        feature_df = extract_feature_table(
            steer_df,
            cache_index,
            rng=rng,
            activation_max_rows=args.activation_max_rows,
            twonn_max_rows=args.twonn_max_rows,
            twonn_max_dims=args.twonn_max_dims,
            glue_max_rows=args.glue_max_rows,
            glue_t_samples=args.glue_t_samples,
            glue_opt_steps=args.glue_opt_steps,
            glue_tol=args.glue_tol,
            glue_gaussianize=not bool(args.no_glue_gaussianize),
        )

    feature_path = out_dir / "activation_representation_metrics.csv"
    feature_df.to_csv(feature_path, index=False)
    print(f"[write] {feature_path}")

    scopes: List[Tuple[str, pd.DataFrame]] = [("all_models", feature_df)]
    for model_id, subset in feature_df.groupby("model_id"):
        scopes.append((str(model_id), subset.copy()))

    for scope_name, scope_df in scopes:
        corr_df = compute_correlation_table(
            scope_df,
            target_col="steerability",
            min_group_size=int(args.min_group_size),
            alpha=float(args.alpha),
        )
        output_path = out_dir / f"correlations_{sanitize_name(scope_name)}.csv"
        corr_df.to_csv(output_path, index=False)
        print(f"[write] {output_path} ({len(corr_df)} features)")
        if not corr_df.empty:
            sig_pearson = int(corr_df["partial_sig_layer_dataset_fdr"].sum())
            sig_spearman = int(corr_df["partial_spearman_sig_layer_dataset_fdr"].sum())
            print(
                f"[sig] {scope_name}: {sig_pearson}/{len(corr_df)} features "
                f"significant at FDR<={args.alpha} for partial Pearson(layer+dataset)"
            )
            print(
                f"[sig] {scope_name}: {sig_spearman}/{len(corr_df)} features "
                f"significant at FDR<={args.alpha} for partial Spearman(layer+dataset)"
            )
            cols = [
                "feature",
                "pearson_r",
                "spearman_r",
                "partial_r_layer_dataset",
                "partial_spearman_r_layer_dataset",
                "partial_p_layer_dataset",
                "partial_q_layer_dataset_fdr",
                "partial_sig_layer_dataset_fdr",
                "partial_spearman_p_layer_dataset",
                "partial_spearman_q_layer_dataset_fdr",
                "partial_spearman_sig_layer_dataset_fdr",
                "within_layer_weighted_r",
                "ranking_score",
            ]
            print(f"\n[top {min(len(corr_df), int(args.top_k))} | {scope_name}]")
            print(corr_df.head(int(args.top_k))[cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
