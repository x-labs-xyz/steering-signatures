#!/usr/bin/env python3
"""
Steering-vector extraction and evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
else:
    PreTrainedModel = Any
    PreTrainedTokenizerBase = Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TRANSFORMERS_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc


EPS = 1e-12
MWE_BASE_URL = (
    "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main"
)
DEFAULT_PROMPT_TEMPLATE = "{question}\n(A) {option_a}\n(B) {option_b}\nAnswer: ("


def _require_transformers() -> None:
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "transformers could not be imported. The CLI still works, but model "
            "loading requires a compatible transformers/tokenizers installation."
        ) from _TRANSFORMERS_IMPORT_ERROR


@dataclass(frozen=True)
class PromptSpec:
    prompt_pos: str
    prompt_neg: str
    eval_prompt: str
    positive_token: str
    negative_token: str


@dataclass
class ActivationCacheTorch:
    pos: torch.Tensor
    neg: torch.Tensor
    meta: Dict[str, Union[int, str, bool]]

    @property
    def n_rows(self) -> int:
        return int(self.pos.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.pos.shape[1])

    def diff_means(self) -> torch.Tensor:
        return self.pos.mean(dim=0) - self.neg.mean(dim=0)

    def diffs(self) -> torch.Tensor:
        return self.pos - self.neg

    def compute_w_transformation(
        self,
        beta: float = 1e-3,
        lr: float = 5e-2,
        steps: int = 2000,
        restarts: int = 8,
        seed: int = 0,
        eps: float = EPS,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.n_rows == 0:
            raise ValueError("Activation cache is empty.")

        x = self.diffs()
        diag = self.pos.pow(2).sum(dim=1) + self.neg.pow(2).sum(dim=1)
        b = (
            self.n_rows * torch.diag(diag)
            - (self.pos @ self.pos.T)
            - (self.neg @ self.neg.T)
        )
        return optimize_w_gd(
            x,
            b,
            beta=beta,
            lr=lr,
            steps=steps,
            restarts=restarts,
            seed=seed,
            eps=eps,
        )

    def scaled_diff_means(self, w_transformation: torch.Tensor) -> torch.Tensor:
        if w_transformation.ndim == 1:
            scale = w_transformation.unsqueeze(1)
        elif w_transformation.ndim == 2 and w_transformation.shape[1] == 1:
            scale = w_transformation
        else:
            raise ValueError("Expected weights with shape (N,) or (N, 1).")

        if scale.shape[0] != self.n_rows:
            raise ValueError("Weight vector length does not match cache rows.")

        scaled = (self.diffs() * scale).mean(dim=0)
        baseline = self.diff_means()
        scaled_norm = torch.linalg.vector_norm(scaled).item()
        baseline_norm = torch.linalg.vector_norm(baseline).item()
        if scaled_norm <= EPS or baseline_norm <= EPS:
            return scaled
        return scaled * (baseline_norm / scaled_norm)

    def compute_scaled_diff_means(
        self,
        beta: float = 1e-3,
        lr: float = 5e-2,
        steps: int = 2000,
        restarts: int = 8,
        seed: int = 0,
        eps: float = EPS,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        weights, info = self.compute_w_transformation(
            beta=beta,
            lr=lr,
            steps=steps,
            restarts=restarts,
            seed=seed,
            eps=eps,
        )
        return self.scaled_diff_means(weights), weights, info

    def fisher_mean(self) -> torch.Tensor:
        x = self.diffs()
        m = x @ x.T
        diag = self.pos.pow(2).sum(dim=1) + self.neg.pow(2).sum(dim=1)
        b = (
            self.n_rows * torch.diag(diag)
            - (self.pos @ self.pos.T)
            - (self.neg @ self.neg.T)
        )
        _, weights = max_gen_rayleigh_psd(m, b)
        return self.scaled_diff_means(weights)


class SteeringHook:
    def __init__(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        vector: np.ndarray,
        multiplier: float,
    ) -> None:
        self.model = model
        self.layer_idx = int(layer_idx)
        self.vector = torch.tensor(vector, dtype=torch.float32)
        self.multiplier = float(multiplier)
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _apply(self, hidden: Tensor) -> Tensor:
        steer = self.vector.to(hidden.device, dtype=hidden.dtype) * self.multiplier
        hidden[:, -1, :] = hidden[:, -1, :] + steer
        return hidden

    def _hook_fn(
        self,
        module: torch.nn.Module,
        inputs: Tuple[Tensor, ...],
        output: object,
    ) -> object:
        if isinstance(output, torch.Tensor):
            return self._apply(output)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (self._apply(output[0]), *output[1:])
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            output[0] = self._apply(output[0])
            return output
        return output

    def __enter__(self) -> "SteeringHook":
        module = get_layer_module(self.model, self.layer_idx)
        self.hook_handle = module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def download_jsonl(url: str) -> List[Dict[str, str]]:
    with urllib.request.urlopen(url) as response:
        lines = response.read().decode("utf-8").splitlines()
    records: List[Dict[str, str]] = []
    for line in lines:
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def download_mwe_dataset(
    category: str,
    names: Sequence[str],
    cache_dir: str = "./mwe_cache",
) -> Dict[str, List[Dict[str, str]]]:
    os.makedirs(cache_dir, exist_ok=True)
    datasets: Dict[str, List[Dict[str, str]]] = {}
    for name in names:
        cache_path = os.path.join(cache_dir, f"{category}_{name}.jsonl")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as handle:
                datasets[name] = [json.loads(line) for line in handle if line.strip()]
            continue

        url = f"{MWE_BASE_URL}/{category}/{name}.jsonl"
        examples = download_jsonl(url)
        datasets[name] = examples
        with open(cache_path, "w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example) + "\n")
    return datasets


def split_dataset(
    dataset: Sequence[Dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    total = len(dataset)
    if total == 0:
        return [], [], []

    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    ratios /= ratios.sum()
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    indices = list(range(total))
    random.Random(seed).shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def subset(selected: Sequence[int]) -> List[Dict[str, str]]:
        return [dataset[idx] for idx in selected]

    return subset(train_idx), subset(val_idx), subset(test_idx)


def build_prompt_spec(
    example: Dict[str, str],
    *,
    positive_token: str,
    negative_token: str,
    template: str,
    randomise_options: bool,
    rng: random.Random,
) -> PromptSpec:
    question = (example.get("question") or "").strip()
    matching = (example.get("answer_matching_behavior") or "").strip()
    not_matching = (example.get("answer_not_matching_behavior") or "").strip()

    flipped = randomise_options and (rng.random() < 0.5)
    if flipped:
        option_a = not_matching
        option_b = matching
        pos_label = negative_token
        neg_label = positive_token
    else:
        option_a = matching
        option_b = not_matching
        pos_label = positive_token
        neg_label = negative_token

    base_prompt = template.format(
        question=question,
        option_a=option_a,
        option_b=option_b,
    )
    return PromptSpec(
        prompt_pos=f"{base_prompt} {pos_label}",
        prompt_neg=f"{base_prompt} {neg_label}",
        eval_prompt=base_prompt,
        positive_token=pos_label,
        negative_token=neg_label,
    )


def get_layer_module(model: PreTrainedModel, layer_idx: int) -> torch.nn.Module:
    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("base_model", "model", "layers"),
    ]
    for path in candidates:
        current = model
        try:
            for attr in path:
                current = getattr(current, attr)
            if isinstance(current, (list, torch.nn.ModuleList)):
                return current[layer_idx]
        except AttributeError:
            continue
    raise AttributeError("Could not locate transformer layers for this model.")


def get_num_layers(model: PreTrainedModel) -> int:
    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("base_model", "model", "layers"),
    ]
    for path in candidates:
        current = model
        try:
            for attr in path:
                current = getattr(current, attr)
            if isinstance(current, (list, torch.nn.ModuleList)):
                return len(current)
        except AttributeError:
            continue
    raise AttributeError("Could not locate transformer layers for this model.")


def _extract_hidden_states_from_outputs(outputs: object) -> Sequence[Tensor]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    if hidden_states is None:
        hidden_states = getattr(outputs, "decoder_hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("decoder_hidden_states")
    if hidden_states is None:
        raise ValueError("Model output did not expose hidden states.")
    return hidden_states


def max_gen_rayleigh_psd(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Expected matching square matrices.")

    a = 0.5 * (a + a.T)
    b = 0.5 * (b + b.T)
    evals_b, evecs_b = torch.linalg.eigh(b)

    keep = evals_b > eps
    if int(keep.sum().item()) == 0:
        raise ValueError("Denominator matrix is numerically zero.")

    if int((~keep).sum().item()) > 0:
        ker_basis = evecs_b[:, ~keep]
        a0 = ker_basis.T @ a @ ker_basis
        if float(torch.linalg.eigvalsh(0.5 * (a0 + a0.T)).max().item()) > 1e-10:
            raise ValueError("Generalized Rayleigh quotient is unbounded.")

    u = evecs_b[:, keep]
    lam = evals_b[keep]
    inv_sqrt_lam = lam.rsqrt()
    m = u.T @ a @ u
    m = 0.5 * (m + m.T)
    c = (inv_sqrt_lam[:, None] * m) * inv_sqrt_lam[None, :]
    evals_c, evecs_c = torch.linalg.eigh(c)
    eigval = evals_c[-1]
    w = u @ (inv_sqrt_lam * evecs_c[:, -1])
    denom = (w @ (b @ w)).clamp_min(1e-30)
    return eigval, w / torch.sqrt(denom)


def project_b_sphere(w: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    denom = torch.sqrt(torch.clamp(w @ (b @ w), min=eps))
    return w / denom


@torch.no_grad()
def top_generalized_eigvec_power(
    x: torch.Tensor,
    b: torch.Tensor,
    beta: float = 1e-3,
    iters: int = 200,
    tol: float = 1e-6,
) -> torch.Tensor:
    n = x.shape[0]
    breg = b + beta * torch.eye(n, device=x.device, dtype=x.dtype)
    chol = torch.linalg.cholesky(breg)

    def solve(rhs: torch.Tensor) -> torch.Tensor:
        return torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)

    w = torch.randn(n, device=x.device, dtype=x.dtype)
    w = project_b_sphere(w, breg)
    last_val: Optional[float] = None
    for _ in range(iters):
        y = x.T @ w
        z = x @ y
        w_new = project_b_sphere(solve(z), breg)
        y_new = x.T @ w_new
        value = float((y_new @ y_new).item() / (w_new @ (breg @ w_new)).item())
        if last_val is not None and abs(value - last_val) <= tol * max(1.0, abs(last_val)):
            w = w_new
            break
        last_val = value
        w = w_new
    return w


def objective(
    w: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    u: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    y = x.T @ w
    s = torch.clamp(y @ y, min=eps)
    t = torch.clamp(w @ (b @ w), min=eps)
    rayleigh = s / t
    u_norm = torch.clamp(torch.linalg.norm(u), min=eps)
    y_norm = torch.sqrt(s)
    cosine = (u @ y) / (u_norm * torch.clamp(y_norm, min=eps))
    return rayleigh + cosine


def optimize_w_gd(
    x: torch.Tensor,
    b: torch.Tensor,
    beta: float = 1e-3,
    lr: float = 5e-2,
    steps: int = 2000,
    restarts: int = 8,
    seed: int = 0,
    eps: float = EPS,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    torch.manual_seed(seed)
    n = x.shape[0]

    work_dtype = x.dtype
    if work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32

    x_work = x.to(dtype=work_dtype)
    b_work = b.to(dtype=work_dtype)
    breg = b_work + beta * torch.eye(n, device=x.device, dtype=work_dtype)
    ones = torch.ones(n, device=x.device, dtype=work_dtype)
    u = x_work.T @ ones

    init_vectors = [
        top_generalized_eigvec_power(x_work, b_work, beta=beta, iters=300),
        project_b_sphere(ones.clone(), breg),
    ]
    for _ in range(max(0, restarts - len(init_vectors))):
        init_vectors.append(
            project_b_sphere(
                torch.randn(n, device=x.device, dtype=work_dtype),
                breg,
            )
        )

    best_vector: Optional[torch.Tensor] = None
    best_value = -float("inf")
    for init in init_vectors:
        w = init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            w_proj = project_b_sphere(w, breg, eps=eps)
            loss = -objective(w_proj, x_work, breg, u, eps=eps)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            w_final = project_b_sphere(w, breg, eps=eps)
            value = float(objective(w_final, x_work, breg, u, eps=eps).item())
            if value > best_value:
                best_value = value
                best_vector = w_final.clone()

    if best_vector is None:
        raise RuntimeError("Failed to optimize row weights.")

    return best_vector, {
        "best_objective": best_value,
        "beta": beta,
        "lr": lr,
        "steps": steps,
        "restarts": len(init_vectors),
    }


def _safe_id(value: str, max_len: int = 140) -> str:
    value = str(value).strip().replace(os.sep, "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:max_len] if len(value) > max_len else value


def _model_id(model: Union[str, PreTrainedModel]) -> str:
    if isinstance(model, str):
        return _safe_id(model)
    name = (
        getattr(getattr(model, "config", None), "name_or_path", None)
        or getattr(model, "name_or_path", None)
        or model.__class__.__name__
    )
    return _safe_id(name)


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _str_to_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.replace("torch.", "")
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported dtype string: {dtype_name}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid dtype string: {dtype_name}")
    return dtype


def _cache_paths(
    memmap_dir: str,
    *,
    model_id: str,
    dataset_id: str,
    layer_idx: int,
    n_rows: int,
    randomise_options: bool,
    prompt_seed: int,
) -> Dict[str, str]:
    base = (
        f"{_safe_id(model_id)}__{_safe_id(dataset_id)}__L{layer_idx}"
        f"__N{n_rows}__rand{int(randomise_options)}__seed{prompt_seed}"
    )
    return {
        "pos": os.path.join(memmap_dir, f"{base}__pos.dat"),
        "neg": os.path.join(memmap_dir, f"{base}__neg.dat"),
        "meta": os.path.join(memmap_dir, f"{base}__meta.json"),
    }


def _expected_bytes(n_rows: int, hidden_dim: int, dtype: torch.dtype) -> int:
    return (
        int(n_rows)
        * int(hidden_dim)
        * int(torch.tensor([], dtype=dtype).element_size())
    )


def _ensure_file_size(path: str, num_bytes: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "ab") as handle:
        handle.truncate(num_bytes)


def _save_meta(path: str, meta: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _load_meta(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _open_memmap_2d(
    path: str,
    n_rows: int,
    hidden_dim: int,
    dtype: torch.dtype,
    *,
    shared: bool,
) -> torch.Tensor:
    return torch.from_file(
        path,
        shared=shared,
        size=int(n_rows) * int(hidden_dim),
        dtype=dtype,
    ).view(int(n_rows), int(hidden_dim))


def _is_full_cache(paths: Dict[str, str], n_rows: int, hidden_dim: int, dtype: torch.dtype) -> bool:
    meta = _load_meta(paths["meta"])
    if meta is None:
        return False
    if int(meta.get("N", -1)) != int(n_rows):
        return False
    if int(meta.get("d", -1)) != int(hidden_dim):
        return False
    if int(meta.get("count", -1)) != int(n_rows):
        return False
    if _str_to_dtype(str(meta.get("dtype", ""))) != dtype:
        return False
    expected = _expected_bytes(n_rows, hidden_dim, dtype)
    for key in ("pos", "neg"):
        if not os.path.exists(paths[key]) or os.path.getsize(paths[key]) < expected:
            return False
    return True


@torch.no_grad()
def cache_prompt_activations_torch_memmap(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Iterable[Dict[str, str]],
    layer_idx: int,
    *,
    dataset_id: str,
    positive_token: str,
    negative_token: str,
    randomise_options: bool,
    prompt_seed: int,
    max_examples: Optional[int],
    memmap_dir: str,
    template: str,
    show_progress: bool,
) -> ActivationCacheTorch:
    examples = list(dataset)
    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]
    if not examples:
        raise ValueError("Dataset split is empty.")

    os.makedirs(memmap_dir, exist_ok=True)
    model_id = _model_id(model)
    n_rows = len(examples)
    num_layers = get_num_layers(model)
    if layer_idx < 0 or layer_idx >= num_layers:
        raise IndexError(f"Layer {layer_idx} is out of range for this model.")

    all_paths = {
        idx: _cache_paths(
            memmap_dir,
            model_id=model_id,
            dataset_id=dataset_id,
            layer_idx=idx,
            n_rows=n_rows,
            randomise_options=randomise_options,
            prompt_seed=prompt_seed,
        )
        for idx in range(num_layers)
    }
    requested_paths = all_paths[layer_idx]
    existing_meta = _load_meta(requested_paths["meta"])
    if existing_meta is not None:
        dtype = _str_to_dtype(str(existing_meta["dtype"]))
        hidden_dim = int(existing_meta["d"])
        if _is_full_cache(requested_paths, n_rows, hidden_dim, dtype):
            pos = _open_memmap_2d(
                requested_paths["pos"],
                n_rows,
                hidden_dim,
                dtype,
                shared=False,
            )
            neg = _open_memmap_2d(
                requested_paths["neg"],
                n_rows,
                hidden_dim,
                dtype,
                shared=False,
            )
            return ActivationCacheTorch(pos=pos, neg=neg, meta=existing_meta)

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    prompt_rng = random.Random(prompt_seed)
    first_spec = build_prompt_spec(
        examples[0],
        positive_token=positive_token,
        negative_token=negative_token,
        template=template,
        randomise_options=randomise_options,
        rng=prompt_rng,
    )
    first_inputs_pos = tokenizer(first_spec.prompt_pos, return_tensors="pt")
    first_inputs_neg = tokenizer(first_spec.prompt_neg, return_tensors="pt")
    first_inputs_pos = {k: v.to(device) for k, v in first_inputs_pos.items()}
    first_inputs_neg = {k: v.to(device) for k, v in first_inputs_neg.items()}
    first_outputs_pos = model(
        **first_inputs_pos,
        output_hidden_states=True,
        return_dict=True,
    )
    first_outputs_neg = model(
        **first_inputs_neg,
        output_hidden_states=True,
        return_dict=True,
    )
    first_hidden_pos = _extract_hidden_states_from_outputs(first_outputs_pos)
    first_hidden_neg = _extract_hidden_states_from_outputs(first_outputs_neg)

    layer_buffers: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    layer_meta: Dict[int, Dict[str, Any]] = {}
    layer_specs: Dict[int, Tuple[int, torch.dtype]] = {}
    for idx in range(num_layers):
        act_pos = first_hidden_pos[idx + 1][0, -1].detach()
        act_neg = first_hidden_neg[idx + 1][0, -1].detach()
        if act_pos.ndim != 1 or act_neg.ndim != 1 or act_pos.shape != act_neg.shape:
            raise ValueError("Unexpected hidden-state shape while initializing cache.")

        hidden_dim = int(act_pos.shape[0])
        dtype = act_pos.dtype
        paths = all_paths[idx]
        _ensure_file_size(paths["pos"], _expected_bytes(n_rows, hidden_dim, dtype))
        _ensure_file_size(paths["neg"], _expected_bytes(n_rows, hidden_dim, dtype))
        pos_memmap = _open_memmap_2d(
            paths["pos"],
            n_rows,
            hidden_dim,
            dtype,
            shared=True,
        )
        neg_memmap = _open_memmap_2d(
            paths["neg"],
            n_rows,
            hidden_dim,
            dtype,
            shared=True,
        )
        pos_memmap[0].copy_(act_pos.to("cpu"))
        neg_memmap[0].copy_(act_neg.to("cpu"))

        meta = {
            "model_id": model_id,
            "dataset_id": str(dataset_id),
            "layer_idx": int(idx),
            "N": int(n_rows),
            "d": hidden_dim,
            "dtype": _dtype_to_str(dtype),
            "count": 1,
            "positive_token": positive_token,
            "negative_token": negative_token,
            "randomise_options": bool(randomise_options),
            "prompt_seed": int(prompt_seed),
            "store": "torch_memmap",
            "all_layers_cached": True,
        }
        _save_meta(paths["meta"], meta)
        layer_buffers[idx] = (pos_memmap, neg_memmap)
        layer_meta[idx] = meta
        layer_specs[idx] = (hidden_dim, dtype)

    for row_idx, example in enumerate(examples[1:], start=1):
        spec = build_prompt_spec(
            example,
            positive_token=positive_token,
            negative_token=negative_token,
            template=template,
            randomise_options=randomise_options,
            rng=prompt_rng,
        )
        inputs_pos = tokenizer(spec.prompt_pos, return_tensors="pt")
        inputs_neg = tokenizer(spec.prompt_neg, return_tensors="pt")
        inputs_pos = {k: v.to(device) for k, v in inputs_pos.items()}
        inputs_neg = {k: v.to(device) for k, v in inputs_neg.items()}
        outputs_pos = model(**inputs_pos, output_hidden_states=True, return_dict=True)
        outputs_neg = model(**inputs_neg, output_hidden_states=True, return_dict=True)
        hidden_pos = _extract_hidden_states_from_outputs(outputs_pos)
        hidden_neg = _extract_hidden_states_from_outputs(outputs_neg)

        for idx in range(num_layers):
            act_pos = hidden_pos[idx + 1][0, -1].detach()
            act_neg = hidden_neg[idx + 1][0, -1].detach()
            expected_dim, expected_dtype = layer_specs[idx]
            if (
                act_pos.ndim != 1
                or act_neg.ndim != 1
                or int(act_pos.shape[0]) != expected_dim
                or int(act_neg.shape[0]) != expected_dim
                or act_pos.dtype != expected_dtype
                or act_neg.dtype != expected_dtype
            ):
                raise ValueError("Hidden-state shape or dtype changed during caching.")
            pos_memmap, neg_memmap = layer_buffers[idx]
            pos_memmap[row_idx].copy_(act_pos.to("cpu"))
            neg_memmap[row_idx].copy_(act_neg.to("cpu"))

        if show_progress and ((row_idx + 1) % 50 == 0 or row_idx + 1 == n_rows):
            print(f"[cache] {dataset_id}: stored {row_idx + 1}/{n_rows} prompt pairs", flush=True)

    for idx in range(num_layers):
        layer_meta[idx]["count"] = int(n_rows)
        _save_meta(all_paths[idx]["meta"], layer_meta[idx])

    requested_meta = layer_meta[layer_idx]
    requested_pos, requested_neg = layer_buffers[layer_idx]
    return ActivationCacheTorch(pos=requested_pos, neg=requested_neg, meta=requested_meta)


def _resolve_single_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) == 1:
        return int(encoded[0])
    encoded_spaced = tokenizer.encode(f" {token}", add_special_tokens=False)
    if len(encoded_spaced) == 1:
        return int(encoded_spaced[0])
    raise ValueError(f"Token '{token}' does not map to a single vocabulary id.")


def _estimate_eval_batch_size(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    show_progress: bool,
) -> int:
    if not prompts:
        return 1
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_available():
        return max(1, min(16, len(prompts)))

    sample_prompts = list(prompts[: min(32, len(prompts))])
    sample_ids = tokenizer(sample_prompts, add_special_tokens=False)["input_ids"]
    sample_lengths = [len(ids) for ids in sample_ids]
    seq_len = max(1, int(np.percentile(sample_lengths, 90)))

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) if config is not None else None
    hidden_size = int(
        getattr(config, "hidden_size", 0)
        or getattr(text_config, "hidden_size", 0)
        or 4096
    )
    vocab_size = int(
        getattr(config, "vocab_size", 0)
        or getattr(text_config, "vocab_size", 0)
        or getattr(tokenizer, "vocab_size", 32000)
    )
    num_layers = get_num_layers(model)
    bytes_per_elem = torch.tensor([], dtype=next(model.parameters()).dtype).element_size()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    usable_bytes = int(free_bytes * 0.70)
    logits_bytes = seq_len * vocab_size * bytes_per_elem
    layer_bytes = seq_len * hidden_size * num_layers * bytes_per_elem * 4
    per_sample_bytes = max(1, logits_bytes + layer_bytes)
    batch_size = int(max(1, min(64, len(prompts), usable_bytes // per_sample_bytes)))

    if show_progress:
        print(
            f"[eval batching] free={free_bytes / (1024 ** 3):.2f} GiB, "
            f"seq_len~p90={seq_len}, batch_size={batch_size}",
            flush=True,
        )
    return batch_size


def evaluate_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Iterable[Dict[str, str]],
    *,
    vector: Optional[np.ndarray],
    layer_idx: int,
    multipliers: Sequence[float],
    positive_token: str,
    negative_token: str,
    template: str,
    randomise_options: bool,
    prompt_seed: int,
    max_examples: Optional[int],
    eval_batch_size: int,
    show_progress: bool,
) -> Dict[str, object]:
    examples = list(dataset)
    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]
    if not examples:
        raise ValueError("No evaluation examples were provided.")

    rng = random.Random(prompt_seed)
    prompt_specs = [
        build_prompt_spec(
            example,
            positive_token=positive_token,
            negative_token=negative_token,
            template=template,
            randomise_options=randomise_options,
            rng=rng,
        )
        for example in examples
    ]
    prompts = [spec.eval_prompt for spec in prompt_specs]

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    batch_size = eval_batch_size if eval_batch_size > 0 else _estimate_eval_batch_size(
        model,
        tokenizer,
        prompts,
        show_progress=show_progress,
    )

    prev_padding_side = getattr(tokenizer, "padding_side", "right")
    prev_pad_token = getattr(tokenizer, "pad_token", None)
    prev_pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if prev_pad_token_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is None or eos_token_id is None:
            raise ValueError("Tokenizer needs a pad token or eos token for batched evaluation.")
        tokenizer.pad_token = eos_token
    tokenizer.padding_side = "left"

    mean_scores: List[float] = []
    try:
        for lambda_idx, lam in enumerate(multipliers):
            if show_progress:
                print(
                    f"[eval] layer={layer_idx} lambda={lam} "
                    f"({lambda_idx + 1}/{len(multipliers)})",
                    flush=True,
                )

            scores: List[float] = []
            hook: Optional[SteeringHook] = None
            if vector is not None:
                hook = SteeringHook(model, layer_idx, vector, lam)
                hook.__enter__()

            try:
                processed = 0
                while processed < len(prompt_specs):
                    current_bs = min(batch_size, len(prompt_specs) - processed)
                    while True:
                        batch_specs = prompt_specs[processed : processed + current_bs]
                        batch_prompts = [spec.eval_prompt for spec in batch_specs]
                        pos_ids = torch.tensor(
                            [
                                _resolve_single_token_id(tokenizer, spec.positive_token)
                                for spec in batch_specs
                            ],
                            device=device,
                            dtype=torch.long,
                        )
                        neg_ids = torch.tensor(
                            [
                                _resolve_single_token_id(tokenizer, spec.negative_token)
                                for spec in batch_specs
                            ],
                            device=device,
                            dtype=torch.long,
                        )
                        try:
                            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            with torch.no_grad():
                                outputs = model(**inputs)
                            next_token_logits = outputs.logits[:, -1, :]
                            batch_indices = torch.arange(current_bs, device=device)
                            batch_scores = (
                                next_token_logits[batch_indices, pos_ids]
                                - next_token_logits[batch_indices, neg_ids]
                            )
                            scores.extend(batch_scores.detach().float().cpu().tolist())
                            processed += current_bs
                            break
                        except RuntimeError as exc:
                            if "out of memory" not in str(exc).lower() or current_bs <= 1:
                                raise
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            current_bs = max(1, current_bs // 2)
                            batch_size = current_bs
                            if show_progress:
                                print(
                                    f"[eval] OOM at lambda={lam}; reducing batch_size to {batch_size}",
                                    flush=True,
                                )

                mean_scores.append(float(np.mean(scores)))
            finally:
                if hook is not None:
                    hook.__exit__(None, None, None)
    finally:
        tokenizer.padding_side = prev_padding_side
        if prev_pad_token_id is None:
            tokenizer.pad_token = prev_pad_token

    x = np.asarray(list(multipliers), dtype=float)
    y = np.asarray(mean_scores, dtype=float)
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "lambda_values": list(multipliers),
        "mean_logit_diffs": mean_scores,
        "steerability": float(slope),
        "intercept": float(intercept),
    }


def extract_steering_vector(
    cache: ActivationCacheTorch,
    *,
    method: str,
    beta: float,
    lr: float,
    steps: int,
    restarts: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method == "diff_means":
        vector = cache.diff_means()
        info = {"method": method}
    elif method == "scaled_diff_means":
        vector, weights, info = cache.compute_scaled_diff_means(
            beta=beta,
            lr=lr,
            steps=steps,
            restarts=restarts,
            seed=seed,
        )
        info = dict(info)
        info["method"] = method
        info["weights_shape"] = list(weights.shape)
    elif method == "fisher_mean":
        vector = cache.fisher_mean()
        info = {"method": method}
    else:
        raise ValueError(f"Unsupported vector method: {method}")

    vector_np = vector.detach().float().cpu().numpy()
    norm = float(np.linalg.norm(vector_np))
    info["vector_norm"] = norm
    return vector_np, info


def _dtype_arg_to_torch(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    _require_transformers()
    model_kwargs: Dict[str, Any] = {"trust_remote_code": bool(args.trust_remote_code)}
    torch_dtype = _dtype_arg_to_torch(args.dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if args.device_map == "none":
        if args.device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = args.device
        model.to(target_device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _default_results_path(model_name: str, vector_method: str) -> str:
    return f"{_safe_id(model_name)}_{vector_method}_steering_results.pkl"


def _load_or_init_results(results_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    results = {
        "config": {
            "model": args.model,
            "vector_method": args.vector_method,
            "dataset_category": args.dataset_category,
            "dataset_names": list(args.dataset_names),
            "multipliers": list(args.multipliers),
        },
        "dataset": [],
        "layer": [],
        "vector_method": [],
        "steering_vector": [],
        "vector_info": [],
        "results": [],
    }
    if not os.path.exists(results_path):
        return results

    with open(results_path, "rb") as handle:
        loaded = pickle.load(handle)
    if isinstance(loaded, dict) and {"dataset", "layer", "results"}.issubset(loaded.keys()):
        loaded.setdefault("vector_method", [])
        loaded.setdefault("steering_vector", [])
        loaded.setdefault("vector_info", [])
        loaded.setdefault("config", results["config"])
        return loaded
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MWE datasets, extract steering vectors, and run steering sweeps.",
    )
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model id.")
    parser.add_argument(
        "--dataset-category",
        type=str,
        required=True,
        help="MWE category, for example persona or advanced-ai-risk.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        required=True,
        help="One or more MWE dataset filenames without the .jsonl suffix.",
    )
    parser.add_argument(
        "--vector-method",
        type=str,
        default="diff_means",
        choices=["diff_means", "scaled_diff_means", "fisher_mean"],
        help="How to extract the steering vector from cached activations.",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="Optional layer indices. By default the script runs every layer.",
    )
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        help="Steering strengths used during evaluation.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default="./mwe_cache",
        help="Local cache for downloaded MWE jsonl files.",
    )
    parser.add_argument(
        "--activation-cache-dir",
        type=str,
        default="./activation_cache_torch",
        help="Directory for activation memmaps.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="",
        help="Pickle file used for incremental saving and resume.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.19,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=0,
        help="Optional cap for train examples per dataset. 0 means all.",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=0,
        help="Optional cap for evaluation examples per dataset. 0 means all.",
    )
    parser.add_argument(
        "--positive-token",
        type=str,
        default="A",
        help="Token used for the positive answer label.",
    )
    parser.add_argument(
        "--negative-token",
        type=str,
        default="B",
        help="Token used for the negative answer label.",
    )
    parser.add_argument(
        "--no-randomise-options",
        action="store_true",
        help="Disable answer-order randomisation when building extraction prompts.",
    )
    parser.add_argument(
        "--eval-randomise-options",
        action="store_true",
        help="Also randomise answer order during steering evaluation.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template with {question}, {option_a}, and {option_b} fields.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=None,
        help="Prompt-order seed for activation extraction. Defaults to --seed.",
    )
    parser.add_argument(
        "--eval-prompt-seed",
        type=int,
        default=None,
        help="Prompt-order seed for evaluation. Defaults to --seed + 1.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-3,
        help="Regularization strength for scaled_diff_means.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-2,
        help="Learning rate for scaled_diff_means.",
    )
    parser.add_argument(
        "--opt-steps",
        type=int,
        default=1500,
        help="Optimization steps for scaled_diff_means.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=10,
        help="Restart count for scaled_diff_means.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Evaluation batch size. 0 enables automatic estimation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Value passed to transformers.from_pretrained(device_map=...). Use 'none' to disable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device when --device-map none is used.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompt_seed = args.seed if args.prompt_seed is None else int(args.prompt_seed)
    eval_prompt_seed = (
        args.seed + 1 if args.eval_prompt_seed is None else int(args.eval_prompt_seed)
    )
    show_progress = not bool(args.quiet)

    model, tokenizer = load_model_and_tokenizer(args)
    datasets = download_mwe_dataset(
        args.dataset_category,
        args.dataset_names,
        cache_dir=args.dataset_cache_dir,
    )

    results_path = args.results_path or _default_results_path(
        args.model,
        args.vector_method,
    )
    results = _load_or_init_results(results_path, args)
    completed = {
        (str(dataset_name), int(layer_idx))
        for dataset_name, layer_idx in zip(results.get("dataset", []), results.get("layer", []))
    }

    layer_indices = args.layers if args.layers else list(range(get_num_layers(model)))
    for dataset_name in args.dataset_names:
        dataset = datasets[dataset_name]
        train_split, _, test_split = split_dataset(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        if args.max_train_examples > 0:
            train_split = train_split[: args.max_train_examples]
        if args.max_eval_examples > 0:
            test_split = test_split[: args.max_eval_examples]

        if show_progress:
            print(
                f"[dataset] {dataset_name}: train={len(train_split)} test={len(test_split)}",
                flush=True,
            )

        for layer_idx in layer_indices:
            key = (dataset_name, int(layer_idx))
            if key in completed:
                if show_progress:
                    print(f"[skip] {dataset_name} layer={layer_idx} already exists in {results_path}", flush=True)
                continue

            if show_progress:
                print(
                    f"[run] dataset={dataset_name} layer={layer_idx} method={args.vector_method}",
                    flush=True,
                )

            cache = cache_prompt_activations_torch_memmap(
                model,
                tokenizer,
                train_split,
                layer_idx,
                dataset_id=dataset_name,
                positive_token=args.positive_token,
                negative_token=args.negative_token,
                randomise_options=not bool(args.no_randomise_options),
                prompt_seed=prompt_seed,
                max_examples=None,
                memmap_dir=args.activation_cache_dir,
                template=args.template,
                show_progress=show_progress,
            )
            vector, vector_info = extract_steering_vector(
                cache,
                method=args.vector_method,
                beta=args.beta,
                lr=args.lr,
                steps=args.opt_steps,
                restarts=args.restarts,
                seed=args.seed,
            )
            eval_result = evaluate_steering(
                model,
                tokenizer,
                test_split,
                vector=vector,
                layer_idx=layer_idx,
                multipliers=args.multipliers,
                positive_token=args.positive_token,
                negative_token=args.negative_token,
                template=args.template,
                randomise_options=bool(args.eval_randomise_options),
                prompt_seed=eval_prompt_seed,
                max_examples=None,
                eval_batch_size=args.eval_batch_size,
                show_progress=show_progress,
            )

            results["dataset"].append(dataset_name)
            results["layer"].append(int(layer_idx))
            results["vector_method"].append(args.vector_method)
            results["steering_vector"].append(vector)
            results["vector_info"].append(vector_info)
            results["results"].append(eval_result)
            completed.add(key)

            with open(results_path, "wb") as handle:
                pickle.dump(results, handle)
            if show_progress:
                print(f"[write] {results_path}", flush=True)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
