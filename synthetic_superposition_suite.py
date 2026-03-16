import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HAS_TQDM = True
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    HAS_TQDM = False

    def tqdm(iterable=None, *args, **kwargs):
        return iterable

HAS_PLOTTING = True
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    HAS_PLOTTING = False
    plt = None
    sns = None

if HAS_PLOTTING:
    sns.set_theme(style='whitegrid', context='talk')

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_DTYPE = torch.float32
DEFAULT_DICTIONARY_CACHE_DIR = Path('synthetic_dictionary_cache')

np.set_printoptions(suppress=True, precision=4)

def resolve_device(device: Optional[torch.device] = None) -> torch.device:
    if device is None:
        return DEFAULT_DEVICE
    if isinstance(device, str):
        return torch.device(device)
    return device


def _dictionary_cache_path(cache_dir: Path, cache_key_payload: Dict) -> Path:
    payload = json.dumps(cache_key_payload, sort_keys=True).encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return cache_dir / f'dictionary_{digest}.pt'


def _load_dictionary_cache(cache_path: Path):
    if not cache_path.exists():
        return None
    return torch.load(cache_path, map_location='cpu')


def _save_dictionary_cache(cache_path: Path, dictionary: torch.Tensor, dict_info: Dict, cache_key_payload: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'cache_key_payload': cache_key_payload,
        'dictionary': dictionary.detach().cpu(),
        'dict_info': dict(dict_info),
    }
    torch.save(payload, cache_path)


def unit_normalize_rows_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(eps)


def welch_lower_bound(n_features: int, d: int) -> float:
    if n_features <= d:
        return 0.0
    return float(math.sqrt((n_features - d) / (d * (n_features - 1))))


def coherence(v: torch.Tensor) -> float:
    gram = v @ v.T
    gram.fill_diagonal_(0.0)
    return float(gram.abs().max().item())


def dictionary_superposition_stats(
    v: torch.Tensor,
    gram_batch_size: Optional[int] = None,
    memory_budget_mb: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute dictionary-level superposition statistics in a memory-safe way.

    We use the participation-ratio effective rank
        r_eff = tr(G)^2 / tr(G^2),  with G = U^T U,
    then define superposition_load = n_features / r_eff.
    """
    n_features, d = v.shape
    if n_features == 0:
        return {
            'effective_rank': 0.0,
            'effective_rank_method': 'participation_ratio',
            'superposition_load': float('nan'),
            'interference': float('nan'),
            'superposition_complexity': float('nan'),
        }

    if n_features == 1:
        return {
            'effective_rank': 1.0,
            'effective_rank_method': 'participation_ratio',
            'superposition_load': 1.0,
            'interference': 0.0,
            'superposition_complexity': 0.0,
        }

    acc_dtype = torch.float32
    acc_elem_size = torch.tensor([], dtype=acc_dtype, device=v.device).element_size()

    if memory_budget_mb is None:
        if v.device.type == 'cuda':
            try:
                free_bytes, _ = torch.cuda.mem_get_info(v.device)
            except TypeError:
                free_bytes, _ = torch.cuda.mem_get_info()
            budget_bytes = int(max(256 * 1024**2, min(0.25 * free_bytes, 4 * 1024**3)))
        else:
            budget_bytes = 1024 * 1024**2
    else:
        budget_bytes = int(memory_budget_mb * 1024**2)

    if gram_batch_size is None:
        fixed_bytes = (d * d + d) * acc_elem_size
        row_bytes = max(d * acc_elem_size, 1)
        available_row_bytes = max(row_bytes, budget_bytes - fixed_bytes)
        gram_batch_size = max(1, min(n_features, available_row_bytes // row_bytes))
    else:
        gram_batch_size = max(1, min(n_features, int(gram_batch_size)))

    gram = torch.zeros((d, d), device=v.device, dtype=acc_dtype)
    sum_vec = torch.zeros(d, device=v.device, dtype=acc_dtype)

    with torch.inference_mode():
        for start in range(0, n_features, gram_batch_size):
            xb = v[start:start + gram_batch_size].to(dtype=acc_dtype)
            gram.addmm_(xb.T, xb, beta=1.0, alpha=1.0)
            sum_vec += xb.sum(dim=0)

    trace_gram = float(torch.trace(gram).item())
    gram_fro_sq = float(gram.square().sum().item())
    effective_rank = trace_gram * trace_gram / max(gram_fro_sq, 1e-12)
    load = float(n_features) / max(effective_rank, 1e-12)

    off_diag_sum = float(torch.dot(sum_vec, sum_vec).item()) - trace_gram
    interference = off_diag_sum / float(n_features * (n_features - 1))

    return {
        'effective_rank': effective_rank,
        'effective_rank_method': 'participation_ratio',
        'superposition_load': load,
        'interference': interference,
        'superposition_complexity': interference * load,
    }


def _ensure_dictionary_stats(
    v: torch.Tensor,
    dict_info: Dict[str, float],
    gram_batch_size: Optional[int] = None,
    memory_budget_mb: Optional[int] = None,
) -> Dict[str, float]:
    out = dict(dict_info)
    needed = {
        'effective_rank',
        'effective_rank_method',
        'superposition_load',
        'interference',
        'superposition_complexity',
    }
    if not needed.issubset(out.keys()):
        out.update(dictionary_superposition_stats(v, gram_batch_size=gram_batch_size, memory_budget_mb=memory_budget_mb))
    return out


def _approximate_coherence(
    v: torch.Tensor,
    block_size: Optional[int] = None,
    memory_budget_mb: Optional[int] = None,
) -> float:
    """
    Exact max off-diagonal coherence using block scanning.

    Computes max_{i != j} |<v_i, v_j>| without materializing the full n x n Gram matrix.
    Extra memory is O(b*d + b^2), where b is block_size.
    """
    n_features, d = v.shape
    if n_features <= 1:
        return 0.0

    elem_size = torch.tensor([], dtype=v.dtype, device=v.device).element_size()

    # Auto memory budget from available device memory.
    if memory_budget_mb is None:
        if v.device.type == 'cuda':
            try:
                free_bytes, _ = torch.cuda.mem_get_info(v.device)
            except TypeError:
                free_bytes, _ = torch.cuda.mem_get_info()
            # Use at most half currently free memory, capped to 8 GB and floored at 256 MB.
            budget_bytes = int(max(256 * 1024**2, min(0.5 * free_bytes, 8 * 1024**3)))
        else:
            budget_bytes = 1024 * 1024**2  # 1 GB default on CPU
    else:
        budget_bytes = int(memory_budget_mb * 1024**2)

    budget_elems = max(1, budget_bytes // max(1, elem_size))

    if block_size is None:
        # Solve b^2 + 2*d*b <= budget_elems  => b <= -d + sqrt(d^2 + budget_elems)
        b = int(max(1.0, -float(d) + math.sqrt(float(d) * float(d) + float(budget_elems))))
        block_size = max(1, min(n_features, b))
    else:
        block_size = max(1, min(n_features, int(block_size)))

    max_abs = torch.tensor(0.0, device=v.device, dtype=torch.float32)

    with torch.inference_mode():
        for i in range(0, n_features, block_size):
            xi = v[i:i + block_size]
            for j in range(i, n_features, block_size):
                xj = v[j:j + block_size]
                g = xi @ xj.T

                if i == j:
                    g.fill_diagonal_(0.0)

                block_max = g.abs().max().float()
                if block_max > max_abs:
                    max_abs = block_max

                del g

    return float(max_abs.item())



def generate_dictionary_with_coherence(
    n_features: int,
    d: int,
    epsilon_target: float,
    seed: int = 0,
    n_restarts: int = 3,
    n_steps: int = 300,
    lr: float = 0.08,
    tol: float = 1e-4,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    max_gram_features: int = 8192,
    coherence_scan_block_size: Optional[int] = None,
    coherence_scan_memory_mb: Optional[int] = None,
    compute_superposition_features: bool = True,
    use_dictionary_cache: bool = True,
    dictionary_cache_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = resolve_device(device)
    cache_dir = DEFAULT_DICTIONARY_CACHE_DIR if dictionary_cache_dir is None else Path(dictionary_cache_dir)
    cache_key_payload = {
        'cache_version': 2,
        'n_features': int(n_features),
        'd': int(d),
        'epsilon_target': float(epsilon_target),
        'seed': int(seed),
        'n_restarts': int(n_restarts),
        'n_steps': int(n_steps),
        'lr': float(lr),
        'tol': float(tol),
        'dtype': str(dtype).replace('torch.', ''),
        'max_gram_features': int(max_gram_features),
    }
    cache_path = _dictionary_cache_path(cache_dir, cache_key_payload)
    if use_dictionary_cache:
        cached = _load_dictionary_cache(cache_path)
        if cached is not None:
            v_cached = cached['dictionary'].to(device=device, dtype=dtype)
            dict_info_cached = dict(cached.get('dict_info', {}))
            if compute_superposition_features:
                updated_info = _ensure_dictionary_stats(
                    v_cached,
                    dict_info_cached,
                    gram_batch_size=coherence_scan_block_size,
                    memory_budget_mb=coherence_scan_memory_mb,
                )
                if updated_info != dict_info_cached:
                    _save_dictionary_cache(cache_path, v_cached, updated_info, cache_key_payload)
                dict_info_cached = updated_info
            return v_cached, dict_info_cached

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    welch_lb = welch_lower_bound(n_features, d)
    eps_goal = max(float(epsilon_target), welch_lb + 1e-4)

    # For very large dictionaries, skip O(n^2) coherence optimization and use random normalized vectors.
    if n_features > max_gram_features:
        with torch.inference_mode():
            v = torch.randn((n_features, d), device=device, dtype=dtype, generator=generator)
            v = unit_normalize_rows_torch(v)
            coh_est = _approximate_coherence(
                v,
                block_size=coherence_scan_block_size,
                memory_budget_mb=coherence_scan_memory_mb,
            )
        dict_info = {
            'epsilon_target': float(epsilon_target),
            'epsilon_goal': eps_goal,
            'achieved_coherence': coh_est,
            'welch_lb': welch_lb,
            'feasible': bool(coh_est <= eps_goal + tol),
            'coherence_estimated': False,
            'coherence_method': 'exact_block_scan',
        }
        if compute_superposition_features:
            dict_info = _ensure_dictionary_stats(
                v,
                dict_info,
                gram_batch_size=coherence_scan_block_size,
                memory_budget_mb=coherence_scan_memory_mb,
            )
        if use_dictionary_cache:
            _save_dictionary_cache(cache_path, v, dict_info, cache_key_payload)
        return v, dict_info

    best_v = None
    best_coh = float('inf')

    with torch.inference_mode():
        for _ in range(n_restarts):
            v = torch.randn((n_features, d), device=device, dtype=dtype, generator=generator)
            v = unit_normalize_rows_torch(v)
            cur_lr = lr

            for _ in range(n_steps):
                gram = v @ v.T
                gram.fill_diagonal_(0.0)
                abs_gram = gram.abs()
                coh = float(abs_gram.max().item())

                if coh < best_coh:
                    best_coh = coh
                    best_v = v.clone()

                if coh <= eps_goal + tol:
                    dict_info = {
                        'epsilon_target': float(epsilon_target),
                        'epsilon_goal': eps_goal,
                        'achieved_coherence': coh,
                        'welch_lb': welch_lb,
                        'feasible': True,
                        'coherence_estimated': False,
                        'coherence_method': 'optimized_full_gram',
                    }
                    if compute_superposition_features:
                        dict_info = _ensure_dictionary_stats(
                            v,
                            dict_info,
                            gram_batch_size=coherence_scan_block_size,
                            memory_budget_mb=coherence_scan_memory_mb,
                        )
                    if use_dictionary_cache:
                        _save_dictionary_cache(cache_path, v, dict_info, cache_key_payload)
                    return v, dict_info

                excess = (abs_gram - eps_goal).clamp_min_(0.0)
                if float(excess.max().item()) <= tol:
                    break

                grad = (2.0 * excess * gram.sign()) @ v
                v = unit_normalize_rows_torch(v - cur_lr * grad / max(1, n_features))
                cur_lr *= 0.997

    dict_info = {
        'epsilon_target': float(epsilon_target),
        'epsilon_goal': eps_goal,
        'achieved_coherence': float(best_coh),
        'welch_lb': welch_lb,
        'feasible': bool(best_coh <= eps_goal + tol),
        'coherence_estimated': False,
        'coherence_method': 'optimized_full_gram',
    }
    if compute_superposition_features:
        dict_info = _ensure_dictionary_stats(
            best_v,
            dict_info,
            gram_batch_size=coherence_scan_block_size,
            memory_budget_mb=coherence_scan_memory_mb,
        )
    if use_dictionary_cache:
        _save_dictionary_cache(cache_path, best_v, dict_info, cache_key_payload)
    return best_v, dict_info


def sample_distribution(spec: Dict, size: int, rng: np.random.Generator) -> np.ndarray:
    # Numpy helper retained for compatibility with ad hoc analysis cells.
    name = spec.get('name', 'normal')

    if name == 'normal':
        out = rng.normal(spec.get('mean', 0.0), spec.get('std', 1.0), size)
    elif name == 'uniform':
        out = rng.uniform(spec.get('low', -1.0), spec.get('high', 1.0), size)
    elif name == 'laplace':
        out = rng.laplace(spec.get('loc', 0.0), spec.get('scale', 1.0), size)
    elif name == 'lognormal':
        out = rng.lognormal(spec.get('mean', 0.0), spec.get('sigma', 1.0), size)
    else:
        raise ValueError(f'Unsupported distribution: {name}')

    if spec.get('nonnegative', False):
        out = np.clip(out, 0.0, None)
    return out


def sample_distribution_torch(
    spec: Dict,
    shape,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    name = spec.get('name', 'normal')

    if name == 'normal':
        out = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        out = out * float(spec.get('std', 1.0)) + float(spec.get('mean', 0.0))
    elif name == 'uniform':
        low = float(spec.get('low', -1.0))
        high = float(spec.get('high', 1.0))
        out = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        out = out * (high - low) + low
    elif name == 'laplace':
        loc = float(spec.get('loc', 0.0))
        scale = float(spec.get('scale', 1.0))
        u = torch.rand(shape, device=device, dtype=dtype, generator=generator) - 0.5
        out = loc - scale * torch.sign(u) * torch.log1p(-2.0 * u.abs().clamp_max(0.499999))
    elif name == 'lognormal':
        mean = float(spec.get('mean', 0.0))
        sigma = float(spec.get('sigma', 1.0))
        out = torch.exp(torch.randn(shape, device=device, dtype=dtype, generator=generator) * sigma + mean)
    else:
        raise ValueError(f'Unsupported distribution: {name}')

    if spec.get('nonnegative', False):
        out = out.clamp_min_(0.0)
    return out


def _draw_support_indices(
    num: int,
    n_features: int,
    k_active: int,
    generator: torch.Generator,
    device: torch.device,
    key_batch_size: int = 64,
    key_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if k_active > n_features:
        raise ValueError(f'k_active={k_active} cannot exceed n_features={n_features}.')

    if key_dtype is None:
        key_dtype = torch.float16 if device.type == 'cuda' else torch.float32

    chunks = []
    for start in range(0, num, key_batch_size):
        bsz = min(key_batch_size, num - start)
        keys = torch.rand((bsz, n_features), device=device, generator=generator, dtype=key_dtype)
        idx = torch.topk(keys, k=k_active, dim=1, largest=False).indices
        chunks.append(idx)
    return torch.cat(chunks, dim=0)


def _to_torch_array(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def generate_superposition_dataset(
    n_features: int,
    d: int,
    n_pos: int,
    n_neg: int,
    k_active: int,
    epsilon_target: float,
    support_mode: str,
    d_pos: Dict,
    d_neg: Dict,
    d_bg: Dict,
    seed: int,
    target_idx: int = 0,
    target_shift_only: bool = True,
    ensure_target_active: bool = True,
    obs_noise_std: float = 0.0,
    dictionary_override=None,
    dictionary_info_override: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    return_coefficients: bool = False,
    max_gram_features: int = 8192,
    sample_batch_size: int = 64,
    key_batch_size: int = 64,
):
    device = resolve_device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if dictionary_override is None:
        v, dict_info = generate_dictionary_with_coherence(
            n_features=n_features,
            d=d,
            epsilon_target=epsilon_target,
            seed=seed + 17,
            device=device,
            dtype=dtype,
            max_gram_features=max_gram_features,
        )
    else:
        v = _to_torch_array(dictionary_override, device=device, dtype=dtype)
        dict_info = dictionary_info_override or {
            'epsilon_target': epsilon_target,
            'epsilon_goal': np.nan,
            'achieved_coherence': coherence(v),
            'welch_lb': welch_lower_bound(n_features, d),
            'feasible': np.nan,
            'coherence_estimated': np.nan,
        }
        dict_info = _ensure_dictionary_stats(v, dict_info)

    if support_mode not in ('same_support', 'different_support'):
        raise ValueError("support_mode must be 'same_support' or 'different_support'.")

    fixed_support = None
    if support_mode == 'same_support':
        fixed_support = _draw_support_indices(
            1,
            n_features,
            k_active,
            generator,
            device,
            key_batch_size=1,
        )
        if ensure_target_active and int(target_idx) not in fixed_support[0].tolist():
            fixed_support[0, 0] = int(target_idx)

    def sample_class(num: int, class_dist: Dict):
        x = torch.empty((num, d), device=device, dtype=dtype)
        a = torch.zeros((num, n_features), device=device, dtype=dtype) if return_coefficients else None

        for start in range(0, num, sample_batch_size):
            bsz = min(sample_batch_size, num - start)

            if fixed_support is not None:
                support = fixed_support.expand(bsz, -1).clone()
            else:
                support = _draw_support_indices(
                    bsz,
                    n_features,
                    k_active,
                    generator,
                    device,
                    key_batch_size=min(key_batch_size, bsz),
                )

            if ensure_target_active:
                has_target = (support == int(target_idx)).any(dim=1)
                if (~has_target).any():
                    support[~has_target, 0] = int(target_idx)

            if target_shift_only:
                coeffs = sample_distribution_torch(d_bg, (bsz, k_active), generator, device, dtype)
                target_mask = support == int(target_idx)
                n_target = int(target_mask.sum().item())
                if n_target > 0:
                    coeffs[target_mask] = sample_distribution_torch(class_dist, (n_target,), generator, device, dtype)
            else:
                coeffs = sample_distribution_torch(class_dist, (bsz, k_active), generator, device, dtype)

            x_batch = torch.bmm(coeffs.unsqueeze(1), v[support]).squeeze(1)

            if obs_noise_std > 0:
                x_batch = x_batch + torch.randn(x_batch.shape, device=device, dtype=dtype, generator=generator) * float(obs_noise_std)

            x[start:start + bsz] = x_batch
            if return_coefficients:
                a[start:start + bsz].scatter_(1, support, coeffs)

        return x, a

    with torch.inference_mode():
        x_pos, a_pos = sample_class(n_pos, d_pos)
        x_neg, a_neg = sample_class(n_neg, d_neg)

    return {
        'X_pos': x_pos,
        'X_neg': x_neg,
        'A_pos': a_pos,
        'A_neg': a_neg,
        'V': v,
        'dict_info': dict_info,
    }

def compute_steering_vector(x_pos: torch.Tensor, x_neg: torch.Tensor) -> torch.Tensor:
    return x_pos.mean(dim=0) - x_neg.mean(dim=0)


def fisher_trace_ratio(x_pos: torch.Tensor, x_neg: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    mu_p = x_pos.mean(dim=0)
    mu_n = x_neg.mean(dim=0)
    delta = mu_p - mu_n

    tr_sb = float(torch.dot(delta, delta).item())
    tr_sw = float((x_pos.var(dim=0, unbiased=False).sum() + x_neg.var(dim=0, unbiased=False).sum()).item())
    return {
        'fisher_trace_ratio': tr_sb / (tr_sw + eps),
        'trace_sb': tr_sb,
        'trace_sw': tr_sw,
    }


TWONN_METRIC_NAMES = tuple(
    f'twonn_{metric_name}_{group_name}'
    for metric_name in ('euclidean', 'cosine')
    for group_name in ('pos', 'neg', 'mix', 'diff')
)


@torch.no_grad()
def two_nn_intrinsic_dimension(
    x: torch.Tensor,
    *,
    metric: str = 'euclidean',
    eps: float = 1e-12,
) -> float:
    if x.ndim != 2 or x.shape[0] < 3:
        return float('nan')
    if metric not in {'euclidean', 'cosine'}:
        raise ValueError(f'Unsupported metric: {metric}')

    x = x.float()
    n = x.shape[0]
    if metric == 'euclidean':
        dists = torch.cdist(x, x, p=2)
    else:
        x_norm = x / (torch.linalg.vector_norm(x, dim=1, keepdim=True) + eps)
        dists = (1.0 - x_norm @ x_norm.T).clamp_min(0.0)

    dists.fill_diagonal_(float('inf'))
    vals, _ = torch.topk(dists, k=2, dim=1, largest=False)
    r1 = vals[:, 0].clamp_min(eps)
    r2 = vals[:, 1].clamp_min(eps)
    mu = (r2 / r1).clamp_min(1.0 + eps).sort().values

    f_emp = torch.arange(1, n + 1, dtype=mu.dtype, device=mu.device) / (n + 1.0)
    xx = torch.log(mu)
    yy = -torch.log((1.0 - f_emp).clamp_min(eps))
    denom = torch.sum(xx * xx)
    if float(denom.item()) <= eps:
        return float('nan')
    d_hat = torch.sum(xx * yy) / denom
    return float(d_hat.item())


def _nan_twonn_metrics() -> Dict[str, float]:
    return {metric_name: float('nan') for metric_name in TWONN_METRIC_NAMES}


def _subsample_shared_columns_torch(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    max_dims: Optional[int],
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_dims is None or max_dims <= 0 or x_pos.shape[1] <= max_dims:
        return x_pos, x_neg
    dim_count = min(x_pos.shape[1], int(max_dims))
    idx_np = rng.choice(x_pos.shape[1], size=dim_count, replace=False)
    idx_np.sort()
    idx = torch.from_numpy(idx_np).to(device=x_pos.device, dtype=torch.long)
    return x_pos.index_select(1, idx), x_neg.index_select(1, idx)


def twonn_feature_geometry_metrics(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    *,
    rng: np.random.Generator,
    max_rows: Optional[int],
    max_dims: Optional[int],
) -> Dict[str, float]:
    metrics = _nan_twonn_metrics()
    if x_pos.ndim != 2 or x_neg.ndim != 2 or x_pos.shape[1] != x_neg.shape[1]:
        return metrics

    pos_twonn, neg_twonn = _subsample_paired_rows_torch(x_pos, x_neg, max_rows, rng)
    pos_twonn, neg_twonn = _subsample_shared_columns_torch(pos_twonn, neg_twonn, max_dims, rng)
    pos_twonn = pos_twonn.float()
    neg_twonn = neg_twonn.float()
    mix_twonn = torch.cat([pos_twonn, neg_twonn], dim=0)
    m = min(pos_twonn.shape[0], neg_twonn.shape[0])
    diff_twonn = pos_twonn[:m] - neg_twonn[:m] if m >= 3 else None

    for metric_name in ('euclidean', 'cosine'):
        metrics[f'twonn_{metric_name}_pos'] = two_nn_intrinsic_dimension(pos_twonn, metric=metric_name)
        metrics[f'twonn_{metric_name}_neg'] = two_nn_intrinsic_dimension(neg_twonn, metric=metric_name)
        metrics[f'twonn_{metric_name}_mix'] = two_nn_intrinsic_dimension(mix_twonn, metric=metric_name)
        metrics[f'twonn_{metric_name}_diff'] = (
            two_nn_intrinsic_dimension(diff_twonn, metric=metric_name)
            if diff_twonn is not None
            else float('nan')
        )
    return metrics


def _auc_from_scores(pos_scores: torch.Tensor, neg_scores: torch.Tensor, eps: float = 1e-12) -> float:
    s = torch.cat([pos_scores, neg_scores], dim=0)
    y = torch.cat([
        torch.ones(pos_scores.shape[0], device=s.device, dtype=torch.bool),
        torch.zeros(neg_scores.shape[0], device=s.device, dtype=torch.bool),
    ], dim=0)

    try:
        order = torch.argsort(s, stable=True)
    except TypeError:
        order = torch.argsort(s)

    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.float32)

    n_pos = float(pos_scores.shape[0])
    n_neg = float(neg_scores.shape[0])
    rank_sum_pos = float(ranks[y].sum().item())
    u = rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0
    return u / max(n_pos * n_neg, eps)


def projection_metrics(x_pos: torch.Tensor, x_neg: torch.Tensor, w: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    norm_w = float(torch.linalg.norm(w).item())
    if norm_w < eps:
        return {
            'proj_gap': 0.0,
            'proj_dprime': 0.0,
            'proj_auc': 0.5,
            'steering_norm': norm_w,
        }

    u = w / norm_w
    z_pos = x_pos @ u
    z_neg = x_neg @ u

    gap = float((z_pos.mean() - z_neg.mean()).item())
    pooled = float(torch.sqrt(0.5 * (z_pos.var(unbiased=False) + z_neg.var(unbiased=False)) + eps).item())
    dprime = float(abs(gap) / (pooled + eps))
    auc = _auc_from_scores(z_pos, z_neg)

    return {
        'proj_gap': gap,
        'proj_dprime': dprime,
        'proj_auc': auc,
        'steering_norm': norm_w,
    }


def cosine_diagnostics(w: torch.Tensor, v: torch.Tensor, target_idx: int = 0, eps: float = 1e-12) -> Dict[str, float]:
    norm_w = float(torch.linalg.norm(w).item())
    if norm_w < eps:
        return {
            'target_cos': 0.0,
            'target_abs_cos': 0.0,
            'max_non_target_abs_cos': 0.0,
            'cosine_margin': 0.0,
            'target_rank': float(v.shape[0]),
        }

    w_hat = (w / norm_w).to(device=v.device, dtype=v.dtype)
    c = (v @ w_hat).float()
    abs_c = c.abs()

    target_abs = float(abs_c[target_idx].item())
    if v.shape[0] > 1:
        mask = torch.ones(v.shape[0], device=v.device, dtype=torch.bool)
        mask[target_idx] = False
        max_other = float(abs_c[mask].max().item())
    else:
        max_other = 0.0

    rank = int(1 + int((abs_c > target_abs).sum().item()))

    return {
        'target_cos': float(c[target_idx].item()),
        'target_abs_cos': target_abs,
        'max_non_target_abs_cos': max_other,
        'cosine_margin': target_abs - max_other,
        'target_rank': float(rank),
    }


def _zscore_features(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    if x.shape[0] > 1:
        std = x.std(dim=0, unbiased=True, keepdim=True)
    else:
        std = torch.ones_like(mean)
    std = torch.where(torch.isfinite(std), std, torch.ones_like(std))
    std = std.clamp_min(eps)
    return (x - mean) / std


def _mean_cosine_to_reference(reference: torch.Tensor, vectors: torch.Tensor, eps: float = 1e-12) -> float:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return 0.0

    ref_norm = float(torch.linalg.norm(reference).item())
    if ref_norm < eps:
        return 0.0

    ref_unit = reference / ref_norm
    vec_norms = torch.linalg.norm(vectors, dim=1).clamp_min(eps)
    cos = (vectors @ ref_unit) / vec_norms
    return float(cos.mean().item())


def mean_cosine_delta_example_diff(x_pos: torch.Tensor, x_neg: torch.Tensor, w: torch.Tensor, eps: float = 1e-12) -> float:
    m = min(x_pos.shape[0], x_neg.shape[0])
    if m == 0:
        return 0.0
    return _mean_cosine_to_reference(w, x_pos[:m] - x_neg[:m], eps=eps)


def twonn_intrinsic_dimension_metric(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    *,
    rng: np.random.Generator,
    max_rows: Optional[int],
    max_dims: Optional[int],
) -> float:
    if x_pos.ndim != 2 or x_neg.ndim != 2 or x_pos.shape[1] != x_neg.shape[1]:
        return float('nan')

    pos_twonn, neg_twonn = _subsample_paired_rows_torch(x_pos, x_neg, max_rows, rng)
    pos_twonn, neg_twonn = _subsample_shared_columns_torch(pos_twonn, neg_twonn, max_dims, rng)
    mix_twonn = torch.cat([pos_twonn.float(), neg_twonn.float()], dim=0)
    return two_nn_intrinsic_dimension(mix_twonn, metric='euclidean')


def selected_representation_metrics(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    *,
    seed: int,
    twonn_max_rows: Optional[int],
    twonn_max_dims: Optional[int],
    glue_max_rows: int,
    glue_t_samples: int,
    glue_opt_steps: int,
    glue_tol: float,
    glue_gaussianize: bool,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    steering_vector = compute_steering_vector(x_pos, x_neg)
    fisher_metrics = fisher_trace_ratio(x_pos, x_neg)
    glue_metrics = _binary_glue_metrics(
        x_pos,
        x_neg,
        rng=rng,
        max_rows=glue_max_rows,
        t_samples=glue_t_samples,
        opt_steps=glue_opt_steps,
        tol=glue_tol,
        gaussianize=glue_gaussianize,
        preserve_pairing=True,
        name_prefix='glue',
    )
    return {
        'fisher_trace_ratio': fisher_metrics['fisher_trace_ratio'],
        'mean_cosine_diff': mean_cosine_delta_example_diff(x_pos, x_neg, steering_vector),
        'glue_capacity': glue_metrics['glue_capacity'],
        'twonn_intrinsic_dimension': twonn_intrinsic_dimension_metric(
            x_pos,
            x_neg,
            rng=rng,
            max_rows=twonn_max_rows,
            max_dims=twonn_max_dims,
        ),
    }




GLUE_METRIC_SUFFIXES = (
    'capacity',
    'effective_dimension',
    'effective_radius',
    'effective_utility',
    'a_mean',
    'b_mean',
    'c_mean',
)


def _nan_prefixed_metrics(prefix: str) -> Dict[str, float]:
    return {f'{prefix}_{suffix}': float('nan') for suffix in GLUE_METRIC_SUFFIXES}


def _subsample_rows_torch(x: torch.Tensor, max_rows: Optional[int], rng: np.random.Generator) -> torch.Tensor:
    if max_rows is None or max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    idx_np = rng.choice(x.shape[0], size=max_rows, replace=False)
    idx_np.sort()
    idx = torch.from_numpy(idx_np).to(device=x.device, dtype=torch.long)
    return x.index_select(0, idx)


def _subsample_paired_rows_torch(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    max_rows: Optional[int],
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_rows is None or max_rows <= 0:
        return x_pos, x_neg
    if x_pos.shape[0] == x_neg.shape[0] and x_pos.shape[0] > max_rows:
        idx_np = rng.choice(x_pos.shape[0], size=max_rows, replace=False)
        idx_np.sort()
        idx = torch.from_numpy(idx_np).to(device=x_pos.device, dtype=torch.long)
        return x_pos.index_select(0, idx), x_neg.index_select(0, idx)
    return _subsample_rows_torch(x_pos, max_rows, rng), _subsample_rows_torch(x_neg, max_rows, rng)


def _rank_gaussianize_features(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError('Expected a 2D tensor for Gaussianization.')
    x = x.float()
    n, d = x.shape
    if n < 2 or d < 1:
        return x.clone()

    order = torch.argsort(x, dim=0)
    ranks = torch.empty((n, d), dtype=torch.float32, device=x.device)
    base = torch.arange(n, dtype=torch.float32, device=x.device).unsqueeze(1).expand(n, d)
    ranks.scatter_(0, order, base + 0.5)
    u = (ranks / float(n)).clamp(1e-6, 1.0 - 1e-6)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return _zscore_features(z)


def _gaussianize_pooled_manifolds(x_pos: torch.Tensor, x_neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_pos.ndim != 2 or x_neg.ndim != 2 or x_pos.shape[1] != x_neg.shape[1]:
        raise ValueError('Expected x_pos/x_neg 2D tensors with matching feature size.')
    all_x = torch.cat([x_pos, x_neg], dim=0)
    all_z = _rank_gaussianize_features(all_x)
    return all_z[:x_pos.shape[0]], all_z[x_pos.shape[0]:]


def _projected_nnls(
    gram: torch.Tensor,
    target: torch.Tensor,
    *,
    max_steps: int,
    tol: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError('Expected a square Gram matrix.')
    if target.ndim != 1 or target.shape[0] != gram.shape[0]:
        raise ValueError('Target shape must match Gram size.')
    if gram.shape[0] == 0:
        return torch.zeros(0, dtype=gram.dtype, device=gram.device)

    gram = gram.double()
    target = target.double()
    eigvals = torch.linalg.eigvalsh(gram)
    lipschitz = max(float(eigvals.max().item()), eps)

    lam = torch.zeros(gram.shape[0], dtype=torch.double, device=gram.device)
    for _ in range(max_steps):
        grad = gram @ lam - target
        lam_next = torch.clamp(lam - grad / lipschitz, min=0.0)
        delta = torch.linalg.vector_norm(lam_next - lam).item()
        scale = 1.0 + torch.linalg.vector_norm(lam).item()
        lam = lam_next
        if delta <= tol * scale:
            break
    return lam


def _binary_glue_anchor_matrix(
    pos_signed: torch.Tensor,
    neg_signed: torch.Tensor,
    probe_t: torch.Tensor,
    *,
    max_steps: int,
    tol: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    n_pos = pos_signed.shape[0]
    n_neg = neg_signed.shape[0]
    if n_pos < 1 or n_neg < 1:
        raise ValueError('Binary GLUE requires at least one point per class.')

    g = torch.cat([pos_signed, neg_signed], dim=0).double()
    probe_t = probe_t.double()
    gram = g @ g.T
    target = g @ probe_t
    weights = _projected_nnls(gram, target, max_steps=max_steps, tol=tol, eps=eps)

    def _weighted_anchor(rows: torch.Tensor, w: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        weight_sum = float(w.sum().item())
        if weight_sum > eps:
            return ((w[:, None] * rows.double()).sum(dim=0) / weight_sum).float()
        best_idx = int(torch.argmax(scores).item())
        return rows[best_idx].float()

    pos_anchor = _weighted_anchor(pos_signed, weights[:n_pos], target[:n_pos])
    neg_anchor = _weighted_anchor(neg_signed, weights[n_pos:n_pos + n_neg], target[n_pos:n_pos + n_neg])
    return torch.stack([pos_anchor, neg_anchor], dim=0)


def _glue_quadratic_form(
    basis: torch.Tensor,
    probe_t: torch.Tensor,
    *,
    extra_gram: Optional[torch.Tensor] = None,
) -> float:
    if basis.ndim != 2 or probe_t.ndim != 1 or basis.shape[1] != probe_t.shape[0]:
        raise ValueError('Invalid shapes for GLUE quadratic form.')
    basis = basis.double()
    probe_t = probe_t.double()
    rhs = basis @ probe_t
    gram = basis @ basis.T
    if extra_gram is not None:
        gram = gram + extra_gram.double()
    pinv = torch.linalg.pinv(gram)
    return float((rhs @ pinv @ rhs).item())


def _binary_glue_metrics(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    *,
    rng: np.random.Generator,
    max_rows: int,
    t_samples: int,
    opt_steps: int,
    tol: float,
    gaussianize: bool,
    preserve_pairing: bool,
    name_prefix: str = 'glue',
) -> Dict[str, float]:
    metrics = _nan_prefixed_metrics(name_prefix)
    if x_pos.ndim != 2 or x_neg.ndim != 2 or x_pos.shape[1] != x_neg.shape[1]:
        return metrics
    if x_pos.shape[0] < 1 or x_neg.shape[0] < 1 or t_samples <= 0:
        return metrics

    if preserve_pairing:
        x_pos, x_neg = _subsample_paired_rows_torch(x_pos, x_neg, max_rows, rng)
    else:
        x_pos = _subsample_rows_torch(x_pos, max_rows, rng)
        x_neg = _subsample_rows_torch(x_neg, max_rows, rng)

    x_pos = x_pos.float()
    x_neg = x_neg.float()
    if gaussianize:
        x_pos, x_neg = _gaussianize_pooled_manifolds(x_pos, x_neg)

    pos_signed = x_pos
    neg_signed = -x_neg
    d = x_pos.shape[1]
    anchor_mats: List[torch.Tensor] = []
    probes: List[torch.Tensor] = []
    a_vals: List[float] = []

    for _ in range(t_samples):
        probe_np = rng.standard_normal(d).astype(np.float32, copy=False)
        probe_t = torch.from_numpy(probe_np).to(device=x_pos.device)
        s_mat = _binary_glue_anchor_matrix(
            pos_signed,
            neg_signed,
            probe_t,
            max_steps=opt_steps,
            tol=tol,
        )
        anchor_mats.append(s_mat)
        probes.append(probe_t)
        a_vals.append(_glue_quadratic_form(s_mat, probe_t))

    if not anchor_mats:
        return metrics

    anchors = torch.stack(anchor_mats, dim=0)
    s0 = anchors.mean(dim=0)
    extra_gram = s0.double() @ s0.double().T
    b_vals: List[float] = []
    c_vals: List[float] = []
    for s_mat, probe_t in zip(anchor_mats, probes):
        s1 = s_mat - s0
        b_vals.append(_glue_quadratic_form(s1, probe_t))
        c_vals.append(_glue_quadratic_form(s1, probe_t, extra_gram=extra_gram))

    mean_a = float(np.mean(a_vals))
    mean_b = float(np.mean(b_vals))
    mean_c = float(np.mean(c_vals))
    denom_radius = mean_b - mean_c
    if denom_radius <= 1e-12:
        radius = float('inf') if mean_c > 1e-12 else 0.0
    else:
        radius = float(math.sqrt(max(mean_c, 0.0) / denom_radius))

    capacity = float(2.0 / mean_a) if mean_a > 1e-12 else float('nan')
    utility = float(mean_c / mean_a) if mean_a > 1e-12 else float('nan')
    utility = float(np.clip(utility, 0.0, 1.0)) if np.isfinite(utility) else utility

    metrics.update(
        {
            f'{name_prefix}_capacity': capacity,
            f'{name_prefix}_effective_dimension': float(0.5 * mean_b),
            f'{name_prefix}_effective_radius': radius,
            f'{name_prefix}_effective_utility': utility,
            f'{name_prefix}_a_mean': mean_a,
            f'{name_prefix}_b_mean': mean_b,
            f'{name_prefix}_c_mean': mean_c,
        }
    )
    return metrics


def _paired_difference_glue_metrics(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    *,
    rng: np.random.Generator,
    max_rows: int,
    t_samples: int,
    opt_steps: int,
    tol: float,
    gaussianize: bool,
) -> Dict[str, float]:
    m = min(x_pos.shape[0], x_neg.shape[0])
    if m < 1:
        return _nan_prefixed_metrics('glue_paired_diff')

    diff = (x_pos[:m] - x_neg[:m]).float()
    return _binary_glue_metrics(
        diff,
        -diff,
        rng=rng,
        max_rows=max_rows,
        t_samples=t_samples,
        opt_steps=opt_steps,
        tol=tol,
        gaussianize=gaussianize,
        preserve_pairing=True,
        name_prefix='glue_paired_diff',
    )


def contrastive_cloud_spectral_metrics(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    # The synthetic suite has no natural prompt IDs, so contrastive-cloud metrics
    # use the same index-aligned pairing convention as the other paired diagnostics.
    m = min(x_pos.shape[0], x_neg.shape[0])
    if m < 1:
        return {
            'contrastive_cloud_pair_count': 0.0,
            'contrastive_cloud_trace_second_moment': float('nan'),
            'contrastive_cloud_lambda_max': float('nan'),
            'contrastive_cloud_spike_ratio': float('nan'),
            'contrastive_spectral_concentration': float('nan'),
        }

    diff = (x_pos[:m] - x_neg[:m]).float()
    trace_second_moment = float(diff.square().sum().item() / float(m))
    if trace_second_moment <= eps:
        return {
            'contrastive_cloud_pair_count': float(m),
            'contrastive_cloud_trace_second_moment': 0.0,
            'contrastive_cloud_lambda_max': 0.0,
            'contrastive_cloud_spike_ratio': float('nan'),
            'contrastive_spectral_concentration': float('nan'),
        }

    gram = (diff.double() @ diff.double().T) / float(m)
    lambda_max = float(torch.linalg.eigvalsh(gram).max().item())
    spike_ratio = float((diff.shape[1] * lambda_max) / (trace_second_moment + eps))
    concentration = float((m / float(diff.shape[1])) * (spike_ratio - 1.0))
    return {
        'contrastive_cloud_pair_count': float(m),
        'contrastive_cloud_trace_second_moment': trace_second_moment,
        'contrastive_cloud_lambda_max': lambda_max,
        'contrastive_cloud_spike_ratio': spike_ratio,
        'contrastive_spectral_concentration': concentration,
    }


def feature_space_comparison_metrics(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    *,
    seed: int,
    compute_twonn_metrics: bool,
    twonn_max_rows: Optional[int],
    twonn_max_dims: Optional[int],
    compute_glue_metrics: bool,
    glue_max_rows: int,
    glue_t_samples: int,
    glue_opt_steps: int,
    glue_tol: float,
    glue_gaussianize: bool,
) -> Dict[str, float]:
    metrics = contrastive_cloud_spectral_metrics(x_pos, x_neg)
    rng = np.random.default_rng(seed)
    if compute_twonn_metrics:
        metrics.update(
            twonn_feature_geometry_metrics(
                x_pos,
                x_neg,
                rng=rng,
                max_rows=twonn_max_rows,
                max_dims=twonn_max_dims,
            )
        )
    else:
        metrics.update(_nan_twonn_metrics())
    if not compute_glue_metrics:
        metrics.update(_nan_prefixed_metrics('glue'))
        metrics.update(_nan_prefixed_metrics('glue_paired_diff'))
        return metrics

    metrics.update(
        _binary_glue_metrics(
            x_pos,
            x_neg,
            rng=rng,
            max_rows=glue_max_rows,
            t_samples=glue_t_samples,
            opt_steps=glue_opt_steps,
            tol=glue_tol,
            gaussianize=glue_gaussianize,
            preserve_pairing=True,
            name_prefix='glue',
        )
    )
    metrics.update(
        _paired_difference_glue_metrics(
            x_pos,
            x_neg,
            rng=rng,
            max_rows=glue_max_rows,
            t_samples=glue_t_samples,
            opt_steps=glue_opt_steps,
            tol=glue_tol,
            gaussianize=glue_gaussianize,
        )
    )
    return metrics


def dist_effect_size(
    d_pos: Dict,
    d_neg: Dict,
    n_samples: int = 20000,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> float:
    device = resolve_device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    s_pos = sample_distribution_torch(d_pos, (n_samples,), generator, device, dtype)
    s_neg = sample_distribution_torch(d_neg, (n_samples,), generator, device, dtype)
    num = float((s_pos.mean() - s_neg.mean()).abs().item())
    den = float(torch.sqrt(0.5 * (s_pos.var(unbiased=False) + s_neg.var(unbiased=False)) + 1e-12).item())
    return num / (den + 1e-12)


def maybe_tqdm(iterable, show_progress: bool, desc: str = '', total: Optional[int] = None):
    if show_progress and HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total)
    return iterable


def run_superposition_sweep(
    grid: Dict,
    n_pos: int = 256,
    n_neg: int = 256,
    reps: int = 4,
    base_seed: int = 0,
    target_shift_only: bool = True,
    ensure_target_active: bool = True,
    obs_noise_std: float = 0.0,
    show_progress: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    coherence_restarts: int = 3,
    coherence_steps: int = 300,
    max_gram_features: int = 8192,
    coherence_scan_block_size: Optional[int] = None,
    coherence_scan_memory_mb: Optional[int] = None,
    compute_superposition_features: bool = True,
    use_dictionary_cache: bool = True,
    dictionary_cache_dir: Optional[str] = None,
    max_dictionary_elements: int = 4_000_000_000,
    auto_dtype: bool = True,
    large_tensor_threshold: int = 350_000_000,
    sample_batch_size: int = 64,
    key_batch_size: int = 64,
    empty_cache_between_dicts: bool = False,
    compute_twonn_metrics: bool = False,
    twonn_max_rows: Optional[int] = 128,
    twonn_max_dims: Optional[int] = 1024,
    compute_glue_metrics: bool = False,
    glue_max_rows: int = 128,
    glue_t_samples: int = 8,
    glue_opt_steps: int = 250,
    glue_tol: float = 1e-4,
    glue_gaussianize: bool = False,
) -> pd.DataFrame:
    device = resolve_device(device)
    records = []

    total_runs = (
        len(grid['d'])
        * len(grid['n_over_d'])
        * len(grid['epsilon'])
        * len(grid['k_active'])
        * len(grid['dist_gap'])
        * len(grid['support_mode'])
        * reps
    )
    progress = tqdm(total=total_runs, desc='superposition sweep') if show_progress and HAS_TQDM else None

    try:
        with torch.inference_mode():
            for d in grid['d']:
                for n_over_d in grid['n_over_d']:
                    n_features = max(2, int(round(n_over_d * d)))
                    dict_elements = n_features * int(d)

                    if dict_elements > max_dictionary_elements:
                        raise ValueError(
                            f'n_features*d={dict_elements:,} exceeds max_dictionary_elements={max_dictionary_elements:,}. '
                            'Increase max_dictionary_elements, reduce d/n_over_d, or use a larger GPU.'
                        )

                    if device.type != 'cuda' and dict_elements > 300_000_000:
                        raise ValueError(
                            'This grid is too large for CPU memory/runtime. Use a CUDA device for large settings.'
                        )

                    effective_dtype = dtype
                    if auto_dtype and device.type == 'cuda' and dict_elements >= large_tensor_threshold:
                        if dtype == torch.float32:
                            effective_dtype = torch.bfloat16

                    for epsilon_target in grid['epsilon']:
                        v, dict_info = generate_dictionary_with_coherence(
                            n_features=n_features,
                            d=d,
                            epsilon_target=epsilon_target,
                            seed=base_seed + hash((n_features, d, epsilon_target)) % 1_000_000,
                            n_restarts=coherence_restarts,
                            n_steps=coherence_steps,
                            device=device,
                            dtype=effective_dtype,
                            max_gram_features=max_gram_features,
                            coherence_scan_block_size=coherence_scan_block_size,
                            coherence_scan_memory_mb=coherence_scan_memory_mb,
                            compute_superposition_features=compute_superposition_features,
                            use_dictionary_cache=use_dictionary_cache,
                            dictionary_cache_dir=dictionary_cache_dir,
                        )
                        for k_active in grid['k_active']:
                            for gap in grid['dist_gap']:
                                mu0 = grid.get('dist_base_mean', 0.0)
                                std = grid.get('dist_std', 1.0)
                                dist_name = grid.get('dist_name', 'normal')
                                nonneg = grid.get('dist_nonnegative', False)

                                d_pos = {'name': dist_name, 'mean': mu0 + 0.5 * gap, 'std': std, 'nonnegative': nonneg}
                                d_neg = {'name': dist_name, 'mean': mu0 - 0.5 * gap, 'std': std, 'nonnegative': nonneg}
                                d_bg = {
                                    'name': grid.get('bg_dist_name', 'normal'),
                                    'mean': grid.get('bg_mean', 0.0),
                                    'std': grid.get('bg_std', std),
                                    'nonnegative': grid.get('bg_nonnegative', nonneg),
                                }

                                dist_sep = dist_effect_size(d_pos, d_neg, n_samples=4000, seed=base_seed, device=device, dtype=effective_dtype)

                                for support_mode in grid['support_mode']:
                                    for rep in range(reps):
                                        seed = (
                                            base_seed
                                            + 13 * rep
                                            + 31 * int(d)
                                            + 47 * int(round(10 * n_over_d))
                                            + 59 * int(round(100 * epsilon_target))
                                            + 71 * int(k_active)
                                            + 89 * int(round(100 * gap))
                                            + (0 if support_mode == 'same_support' else 1)
                                        )

                                        data = generate_superposition_dataset(
                                            n_features=n_features,
                                            d=d,
                                            n_pos=n_pos,
                                            n_neg=n_neg,
                                            k_active=k_active,
                                            epsilon_target=epsilon_target,
                                            support_mode=support_mode,
                                            d_pos=d_pos,
                                            d_neg=d_neg,
                                            d_bg=d_bg,
                                            seed=seed,
                                            target_idx=grid.get('target_idx', 0),
                                            target_shift_only=target_shift_only,
                                            ensure_target_active=ensure_target_active,
                                            obs_noise_std=obs_noise_std,
                                            dictionary_override=v,
                                            dictionary_info_override=dict_info,
                                            device=device,
                                            dtype=effective_dtype,
                                            return_coefficients=False,
                                            max_gram_features=max_gram_features,
                                            sample_batch_size=sample_batch_size,
                                            key_batch_size=key_batch_size,
                                        )

                                        # Metrics in float32 for numeric stability.
                                        x_pos = data['X_pos'].float()
                                        x_neg = data['X_neg'].float()

                                        rec = {
                                            'd': d,
                                            'n_features': n_features,
                                            'n_over_d': n_features / d,
                                            'epsilon_target': epsilon_target,
                                            'achieved_coherence': data['dict_info']['achieved_coherence'],
                                            'welch_lb': data['dict_info']['welch_lb'],
                                            'effective_rank': data['dict_info'].get('effective_rank', np.nan) if compute_superposition_features else np.nan,
                                            'effective_rank_method': data['dict_info'].get('effective_rank_method') if compute_superposition_features else None,
                                            'superposition_load': data['dict_info'].get('superposition_load', np.nan) if compute_superposition_features else np.nan,
                                            'interference': data['dict_info'].get('interference', np.nan) if compute_superposition_features else np.nan,
                                            'superposition_complexity': data['dict_info'].get('superposition_complexity', np.nan) if compute_superposition_features else np.nan,
                                            'k_active': k_active,
                                            'dist_gap': gap,
                                            'dist_effect_size': dist_sep,
                                            'support_mode': support_mode,
                                            'rep': rep,
                                            'target_shift_only': target_shift_only,
                                            'device': str(device),
                                            'dtype': str(effective_dtype).replace('torch.', ''),
                                        }
                                        rec.update(
                                            selected_representation_metrics(
                                                x_pos,
                                                x_neg,
                                                seed=seed,
                                                twonn_max_rows=twonn_max_rows if compute_twonn_metrics else None,
                                                twonn_max_dims=twonn_max_dims if compute_twonn_metrics else None,
                                                glue_max_rows=glue_max_rows,
                                                glue_t_samples=glue_t_samples if compute_glue_metrics else 0,
                                                glue_opt_steps=glue_opt_steps,
                                                glue_tol=glue_tol,
                                                glue_gaussianize=glue_gaussianize,
                                            )
                                        )
                                        records.append(rec)

                                        if progress is not None:
                                            progress.update(1)

                        del v
                        if device.type == 'cuda' and empty_cache_between_dicts:
                            torch.cuda.empty_cache()
    finally:
        if progress is not None:
            progress.close()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    return pd.DataFrame.from_records(records)


def add_ease_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def z(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

    # Assumption used in this notebook:
    # easier superposition => lower n/d, lower coherence, lower k, higher D+/D- separation.
    out['ease_geometry'] = -0.5 * z(out['n_over_d']) - 0.5 * z(out['achieved_coherence'])
    out['ease_sparsity'] = -z(out['k_active'])
    out['ease_distribution'] = z(out['dist_effect_size'])
    if 'superposition_load' in out.columns:
        out['ease_superposition_load'] = -z(out['superposition_load'])
    out['ease_score'] = (out['ease_geometry'] + out['ease_sparsity'] + out['ease_distribution']) / 3.0
    return out


def _corr_from_arrays(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float('nan')

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom < 1e-12:
        return float('nan')

    return float(np.dot(x_centered, y_centered) / denom)


def _safe_corr(sub: pd.DataFrame, x_col: str, y_col: str, method: str) -> float:
    pair = sub[[x_col, y_col]].dropna()
    if len(pair) < 2:
        return float('nan')

    x = pair[x_col].to_numpy(dtype=float)
    y = pair[y_col].to_numpy(dtype=float)
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float('nan')

    if method == 'spearman':
        x = pair[x_col].rank(method='average').to_numpy(dtype=float)
        y = pair[y_col].rank(method='average').to_numpy(dtype=float)
    elif method != 'pearson':
        raise ValueError(f'Unsupported correlation method: {method}')

    return _corr_from_arrays(x, y)


def _residualize_linear(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), controls])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _partial_corr(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    control_cols,
    method: str,
) -> float:
    cols = [x_col, y_col] + list(control_cols)
    work = sub[cols].dropna()
    if len(work) < max(3, len(control_cols) + 2):
        return float('nan')

    if method == 'spearman':
        work = work.rank(method='average')

    x = work[x_col].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)

    if len(control_cols) == 0:
        return _safe_corr(work, x_col, y_col, method='pearson')

    controls = work[list(control_cols)].to_numpy(dtype=float)
    x_res = _residualize_linear(x, controls)
    y_res = _residualize_linear(y, controls)

    if np.nanstd(x_res) < 1e-12 or np.nanstd(y_res) < 1e-12:
        return float('nan')

    return float(pd.Series(x_res).corr(pd.Series(y_res), method='pearson'))


def _weighted_within_group_corr(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    control_cols,
    method: str,
    min_group_size: int = 3,
) -> Tuple[float, int]:
    if len(control_cols) == 0:
        return _safe_corr(sub, x_col, y_col, method=method), int(len(sub) >= min_group_size)

    fisher_z = []
    weights = []
    used_groups = 0

    for _, grp in sub.groupby(list(control_cols), dropna=False):
        pair = grp[[x_col, y_col]].dropna()
        if len(pair) < min_group_size:
            continue

        r = _safe_corr(pair, x_col, y_col, method=method)
        if not np.isfinite(r):
            continue

        r = float(np.clip(r, -0.999999, 0.999999))
        fisher_z.append(np.arctanh(r))
        weights.append(max(len(pair) - 3, 1))
        used_groups += 1

    if used_groups == 0:
        return float('nan'), 0

    return float(np.tanh(np.average(fisher_z, weights=weights))), used_groups


def summarize_metric_correlations(
    df: pd.DataFrame,
    metrics,
    group_col: str = 'support_mode',
    control_cols = ('n_over_d', 'achieved_coherence', 'k_active'),
    sub_ease_cols = ('ease_geometry', 'ease_sparsity', 'ease_distribution', 'ease_superposition_load'),
) -> pd.DataFrame:
    rows = []

    def add_group(gname: str, sub: pd.DataFrame):
        if len(sub) < 3:
            return

        available_controls = tuple(col for col in control_cols if col in sub.columns)

        for metric in metrics:
            if metric not in sub.columns:
                continue

            row = {
                'group': gname,
                'metric': metric,
                'n': len(sub),
                'spearman_rho': _safe_corr(sub, metric, 'ease_score', method='spearman'),
                'pearson_r': _safe_corr(sub, metric, 'ease_score', method='pearson'),
                'partial_spearman_rho': _partial_corr(sub, metric, 'ease_score', available_controls, method='spearman'),
                'partial_pearson_r': _partial_corr(sub, metric, 'ease_score', available_controls, method='pearson'),
            }

            weighted_spearman, n_weighted_groups = _weighted_within_group_corr(
                sub,
                metric,
                'ease_score',
                available_controls,
                method='spearman',
            )
            weighted_pearson, _ = _weighted_within_group_corr(
                sub,
                metric,
                'ease_score',
                available_controls,
                method='pearson',
            )
            row['weighted_within_group_spearman_rho'] = weighted_spearman
            row['weighted_within_group_pearson_r'] = weighted_pearson
            row['n_weighted_within_groups'] = n_weighted_groups

            for ease_col in sub_ease_cols:
                if ease_col not in sub.columns:
                    row[f'{ease_col}_spearman_rho'] = float('nan')
                    row[f'{ease_col}_pearson_r'] = float('nan')
                    continue

                row[f'{ease_col}_spearman_rho'] = _safe_corr(sub, metric, ease_col, method='spearman')
                row[f'{ease_col}_pearson_r'] = _safe_corr(sub, metric, ease_col, method='pearson')

            rows.append(row)

    add_group('all', df)
    if group_col in df.columns:
        for gname, sub in df.groupby(group_col):
            add_group(str(gname), sub)

    return pd.DataFrame(rows)


def bootstrap_spearman_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_boot: int = 400,
    seed: int = 0,
    show_progress: bool = False,
    progress_desc: str = 'bootstrap',
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    n = len(df)
    vals = np.empty(n_boot, dtype=float)

    iterator = range(n_boot)
    if show_progress and HAS_TQDM:
        iterator = tqdm(iterator, desc=progress_desc, leave=False)

    for i in iterator:
        idx = rng.integers(0, n, size=n)
        vals[i] = _safe_corr(pd.DataFrame({x_col: x[idx], y_col: y[idx]}), x_col, y_col, method='spearman')

    center = float(np.nanmedian(vals))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return center, float(lo), float(hi)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj).replace('torch.', '')
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_superposition_bundle(
    results_df: pd.DataFrame,
    *,
    output_root: Path,
    run_name: str,
    sweep_params: Dict,
    grid_details: Dict,
) -> Path:
    output_root = Path(output_root)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bundle_dir = output_root / f'{run_name}_{timestamp}'
    bundle_dir.mkdir(parents=True, exist_ok=False)

    results_pickle_path = bundle_dir / 'results_df.pkl'
    results_csv_path = bundle_dir / 'results_df.csv'
    sweep_params_path = bundle_dir / 'sweep_params.json'
    grid_details_path = bundle_dir / 'grid_details.json'
    manifest_path = bundle_dir / 'manifest.json'

    results_df.to_pickle(results_pickle_path)
    results_df.to_csv(results_csv_path, index=False)

    sweep_payload = _to_jsonable(sweep_params)
    grid_payload = _to_jsonable(grid_details)
    sweep_params_path.write_text(json.dumps(sweep_payload, indent=2, sort_keys=True))
    grid_details_path.write_text(json.dumps(grid_payload, indent=2, sort_keys=True))

    manifest = {
        'bundle_dir': str(bundle_dir),
        'created_at_local': timestamp,
        'artifacts': {
            'results_df_pickle': results_pickle_path.name,
            'results_df_csv': results_csv_path.name,
            'sweep_params_json': sweep_params_path.name,
            'grid_details_json': grid_details_path.name,
        },
        'n_rows': int(len(results_df)),
        'n_columns': int(results_df.shape[1]),
        'columns': list(results_df.columns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return bundle_dir


def load_superposition_bundle(bundle_dir: Path) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / 'manifest.json').read_text())
    sweep_params = json.loads((bundle_dir / 'sweep_params.json').read_text())
    grid_details = json.loads((bundle_dir / 'grid_details.json').read_text())
    results_df = pd.read_pickle(bundle_dir / manifest['artifacts']['results_df_pickle'])
    return results_df, sweep_params, grid_details, manifest


SELECTED_METRICS = [
    'fisher_trace_ratio',
    'mean_cosine_diff',
    'glue_capacity',
    'twonn_intrinsic_dimension',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the synthetic superposition sweep and save results/correlations.',
    )
    parser.add_argument('--d', nargs='+', type=int, default=[32, 64], help='Ambient dimensions for the grid.')
    parser.add_argument(
        '--n-over-d',
        nargs='+',
        type=float,
        default=[1.5, 2.5, 4.0],
        help='Values for n_features / d.',
    )
    parser.add_argument(
        '--epsilon',
        nargs='+',
        type=float,
        default=[0.06, 0.12, 0.20],
        help='Target coherence values.',
    )
    parser.add_argument(
        '--k-active',
        nargs='+',
        type=int,
        default=[2, 4, 8],
        help='Numbers of active features.',
    )
    parser.add_argument(
        '--dist-gap',
        nargs='+',
        type=float,
        default=[0.3, 0.7, 1.1],
        help='Class-conditional distribution gaps.',
    )
    parser.add_argument(
        '--support-mode',
        nargs='+',
        type=str,
        default=['same_support', 'different_support'],
        help='Support modes to evaluate.',
    )
    parser.add_argument('--dist-name', type=str, default='normal', help='Foreground coefficient distribution.')
    parser.add_argument('--dist-std', type=float, default=1.0, help='Foreground coefficient std.')
    parser.add_argument('--dist-base-mean', type=float, default=0.0, help='Foreground base mean before applying gap.')
    parser.add_argument('--dist-nonnegative', action='store_true', help='Clamp foreground draws to nonnegative values when supported.')
    parser.add_argument('--bg-dist-name', type=str, default='normal', help='Background coefficient distribution.')
    parser.add_argument('--bg-mean', type=float, default=0.0, help='Background coefficient mean.')
    parser.add_argument('--bg-std', type=float, default=1.0, help='Background coefficient std.')
    parser.add_argument('--bg-nonnegative', action='store_true', help='Clamp background draws to nonnegative values when supported.')
    parser.add_argument('--target-idx', type=int, default=0, help='Target feature index.')
    parser.add_argument('--n-pos', type=int, default=256, help='Positive samples per sweep cell.')
    parser.add_argument('--n-neg', type=int, default=256, help='Negative samples per sweep cell.')
    parser.add_argument('--reps', type=int, default=3, help='Replicates per grid cell.')
    parser.add_argument('--base-seed', type=int, default=7, help='Base random seed.')
    parser.add_argument('--obs-noise-std', type=float, default=0.0, help='Observation noise added to samples.')
    parser.add_argument('--device', type=str, default=str(DEFAULT_DEVICE), help='Torch device.')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'], help='Base torch dtype.')
    parser.add_argument('--no-auto-dtype', action='store_true', help='Disable automatic lower-precision switching for large dictionaries.')
    parser.add_argument('--no-target-shift-only', action='store_true', help='Allow all active features to differ across classes.')
    parser.add_argument('--no-ensure-target-active', action='store_true', help='Do not force the target feature to appear in supports.')
    parser.add_argument('--no-dictionary-cache', action='store_true', help='Disable dictionary caching.')
    parser.add_argument('--dictionary-cache-dir', type=str, default=str(DEFAULT_DICTIONARY_CACHE_DIR), help='Dictionary cache directory.')
    parser.add_argument('--max-dictionary-elements', type=int, default=4_000_000_000, help='Safety cap for n_features * d.')
    parser.add_argument('--max-gram-features', type=int, default=50_000, help='Max feature count for exact Gram scans.')
    parser.add_argument('--coherence-restarts', type=int, default=3, help='Dictionary search restarts.')
    parser.add_argument('--coherence-steps', type=int, default=300, help='Dictionary search steps per restart.')
    parser.add_argument('--coherence-scan-block-size', type=int, default=0, help='Optional block size for coherence scanning. 0 uses auto.')
    parser.add_argument('--coherence-scan-memory-mb', type=int, default=1024, help='Memory budget for coherence scans.')
    parser.add_argument('--sample-batch-size', type=int, default=64, help='Synthetic sample batch size.')
    parser.add_argument('--key-batch-size', type=int, default=64, help='Active-key batch size.')
    parser.add_argument('--empty-cache-between-dicts', action='store_true', help='Call torch.cuda.empty_cache() between dictionaries.')
    parser.add_argument('--twonn-max-rows', type=int, default=128, help='Max rows used for the TwoNN estimate.')
    parser.add_argument('--twonn-max-dims', type=int, default=-1, help='Max dimensions used for the TwoNN estimate. <=0 uses all dims.')
    parser.add_argument('--glue-max-rows', type=int, default=128, help='Max rows used for GLUE capacity.')
    parser.add_argument('--glue-t-samples', type=int, default=12, help='Gaussian probes used for GLUE capacity.')
    parser.add_argument('--glue-opt-steps', type=int, default=200, help='Projected-gradient steps for the GLUE solver.')
    parser.add_argument('--glue-tol', type=float, default=1e-4, help='Convergence tolerance for the GLUE solver.')
    parser.add_argument('--glue-gaussianize', action='store_true', help='Apply pooled featurewise Gaussianization before GLUE.')
    parser.add_argument('--out-dir', type=str, default='synthetic_superposition_outputs', help='Directory for outputs.')
    parser.add_argument('--results-name', type=str, default='synthetic_superposition_results.csv', help='CSV filename for the raw sweep rows.')
    parser.add_argument('--corr-name', type=str, default='synthetic_superposition_correlations.csv', help='CSV filename for the correlation summary.')
    parser.add_argument('--params-name', type=str, default='synthetic_superposition_params.json', help='JSON filename for the run parameters.')
    parser.add_argument('--show-progress', action='store_true', help='Enable tqdm progress bars.')
    return parser.parse_args()


def _dtype_from_arg(dtype_name: str) -> torch.dtype:
    mapping = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    return mapping[dtype_name]


def build_grid_from_args(args: argparse.Namespace) -> Dict:
    return {
        'd': list(args.d),
        'n_over_d': list(args.n_over_d),
        'epsilon': list(args.epsilon),
        'k_active': list(args.k_active),
        'dist_gap': list(args.dist_gap),
        'support_mode': list(args.support_mode),
        'dist_name': args.dist_name,
        'dist_std': float(args.dist_std),
        'dist_base_mean': float(args.dist_base_mean),
        'dist_nonnegative': bool(args.dist_nonnegative),
        'bg_dist_name': args.bg_dist_name,
        'bg_mean': float(args.bg_mean),
        'bg_std': float(args.bg_std),
        'bg_nonnegative': bool(args.bg_nonnegative),
        'target_idx': int(args.target_idx),
    }


def run_from_args(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    dtype = _dtype_from_arg(args.dtype)
    grid = build_grid_from_args(args)

    results_df = run_superposition_sweep(
        grid,
        n_pos=int(args.n_pos),
        n_neg=int(args.n_neg),
        reps=int(args.reps),
        base_seed=int(args.base_seed),
        target_shift_only=not bool(args.no_target_shift_only),
        ensure_target_active=not bool(args.no_ensure_target_active),
        obs_noise_std=float(args.obs_noise_std),
        show_progress=bool(args.show_progress),
        device=device,
        dtype=dtype,
        coherence_restarts=int(args.coherence_restarts),
        coherence_steps=int(args.coherence_steps),
        max_gram_features=int(args.max_gram_features),
        coherence_scan_block_size=(
            None if int(args.coherence_scan_block_size) <= 0 else int(args.coherence_scan_block_size)
        ),
        coherence_scan_memory_mb=int(args.coherence_scan_memory_mb),
        compute_superposition_features=True,
        use_dictionary_cache=not bool(args.no_dictionary_cache),
        dictionary_cache_dir=args.dictionary_cache_dir,
        max_dictionary_elements=int(args.max_dictionary_elements),
        auto_dtype=not bool(args.no_auto_dtype),
        sample_batch_size=int(args.sample_batch_size),
        key_batch_size=int(args.key_batch_size),
        empty_cache_between_dicts=bool(args.empty_cache_between_dicts),
        compute_twonn_metrics=True,
        twonn_max_rows=(None if int(args.twonn_max_rows) <= 0 else int(args.twonn_max_rows)),
        twonn_max_dims=(None if int(args.twonn_max_dims) <= 0 else int(args.twonn_max_dims)),
        compute_glue_metrics=True,
        glue_max_rows=int(args.glue_max_rows),
        glue_t_samples=int(args.glue_t_samples),
        glue_opt_steps=int(args.glue_opt_steps),
        glue_tol=float(args.glue_tol),
        glue_gaussianize=bool(args.glue_gaussianize),
    )
    results_df = add_ease_score(results_df)

    metrics_to_compare = [
        metric for metric in SELECTED_METRICS
        if metric in results_df.columns and results_df[metric].notna().any()
    ]
    corr_df = summarize_metric_correlations(results_df, metrics_to_compare)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_payload = {
        'grid': grid,
        'sweep': {
            'n_pos': int(args.n_pos),
            'n_neg': int(args.n_neg),
            'reps': int(args.reps),
            'base_seed': int(args.base_seed),
            'obs_noise_std': float(args.obs_noise_std),
            'device': str(device),
            'dtype': str(dtype).replace('torch.', ''),
            'target_shift_only': not bool(args.no_target_shift_only),
            'ensure_target_active': not bool(args.no_ensure_target_active),
            'compute_metrics': list(SELECTED_METRICS),
            'twonn_max_rows': None if int(args.twonn_max_rows) <= 0 else int(args.twonn_max_rows),
            'twonn_max_dims': None if int(args.twonn_max_dims) <= 0 else int(args.twonn_max_dims),
            'glue_max_rows': int(args.glue_max_rows),
            'glue_t_samples': int(args.glue_t_samples),
            'glue_opt_steps': int(args.glue_opt_steps),
            'glue_tol': float(args.glue_tol),
            'glue_gaussianize': bool(args.glue_gaussianize),
        },
    }

    (out_dir / args.params_name).write_text(
        json.dumps(_to_jsonable(params_payload), indent=2, sort_keys=True)
    )
    results_df.to_csv(out_dir / args.results_name, index=False)
    corr_df.to_csv(out_dir / args.corr_name, index=False)


def main() -> None:
    args = parse_args()
    run_from_args(args)


if __name__ == '__main__':
    main()
