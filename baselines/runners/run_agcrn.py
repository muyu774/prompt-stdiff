"""Train/evaluate AGCRN with optional Prompt-STDiff semantic injection.

This runner keeps the official AGCRN implementation out of tree. It uses this
repo's canonical dataloader, scaler, split, metrics, and result writer, then
wraps the AGCRN model with ``AGCRNSemanticWrapper`` when ``baseline.use_semantic``
or ``--use_semantic`` is enabled.

Expected official repo layout is flexible. The runner searches ``--agcrn_repo``
for a Python file containing ``class AGCRN`` and instantiates it with common
constructor kwargs. If that fails, pass ``--model_file`` explicitly.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from baselines.deterministic_semantic import AGCRNSemanticWrapper
from dataio.traffic_dataset import build_dataloaders
from models.semantic_adapter import BatchSemanticComposer, build_semantic_adapter_config
from semantic.semantic_cache import load_semantic_embeddings
from utils.config import deep_merge, load_config
from utils.device import get_device
from utils.logger import get_logger
from utils.metrics import compute_all_metrics
from utils.result_writer import ExperimentResult, write_experiment_results
from utils.seed import set_seed


class FallbackAGCRN(nn.Module):
    """Small AGCRN-shaped fallback for smoke tests only.

    It consumes ``[B,T,N,F]`` and returns ``[B,H,N,F_out]``. This is not a paper
    baseline; it exists so the integration path can be validated before the
    official repo is present on the same machine.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        num_nodes: int,
        rnn_units: int = 64,
        num_layers: int = 1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.num_nodes = int(num_nodes)
        self.encoder = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(rnn_units),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.proj = nn.Linear(int(rnn_units), self.horizon * self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"FallbackAGCRN expects [B,T,N,F], got {tuple(x.shape)}")
        bsz, steps, nodes, feat = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(bsz * nodes, steps, feat)
        _, h_last = self.encoder(x_flat)
        out = self.proj(h_last[-1]).view(bsz, nodes, self.horizon, self.output_dim)
        return out.permute(0, 2, 1, 3).contiguous()


class OfficialAGCRNForwardAdapter(nn.Module):
    """Normalize official AGCRN forward signatures to ``forward(x)``.

    The common AGCRN release defines ``forward(source, targets,
    teacher_forcing_ratio)`` even though inference only needs ``source``. This
    adapter keeps the official model unchanged while allowing our semantic
    wrapper to call it with canonical history tensors only.
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.base_model(x)
        except TypeError as first_error:
            try:
                return self.base_model(x, None, 0.0)
            except TypeError:
                try:
                    return self.base_model(x, targets=None, teacher_forcing_ratio=0.0)
                except TypeError as second_error:
                    raise TypeError(
                        "Unable to call official AGCRN forward with source-only inference. "
                        f"First error: {first_error!r}; second error: {second_error!r}"
                    ) from second_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate AGCRN baseline with optional semantics")
    parser.add_argument("--config", type=str, required=True, help="Prompt-STDiff config path")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(10)))
    parser.add_argument("--mode", type=str, default="train", choices=("train", "eval"))
    parser.add_argument("--agcrn_repo", type=str, default="baselines/external_repos/AGCRN")
    parser.add_argument("--model_file", type=str, default="", help="Explicit official AGCRN Python file")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints/agcrn")
    parser.add_argument("--save_tag", type=str, default="")
    parser.add_argument("--use_semantic", action="store_true", help="Enable deterministic semantic adapter")
    parser.add_argument("--semantic_proj_dim", type=int, default=None)
    parser.add_argument("--fallback", action="store_true", help="Use fallback smoke-test model if official import fails")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--input_feature_index",
        type=int,
        default=None,
        help="Optional input channel for AGCRN. Defaults to train.metric_feature_index when set.",
    )
    parser.add_argument(
        "--skip_agcrn_init",
        action="store_true",
        help="Do not initialize official AGCRN parameters. Use only to reproduce an external checkpoint exactly.",
    )
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None, help="Debug only: cap train batches per epoch")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default="outputs/results.csv")
    parser.add_argument("--results_md", type=str, default="RESULTS.md")
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides without changing the shared config file."""
    override: Dict[str, Any] = {"baseline": {}}
    if args.use_semantic:
        override["baseline"]["use_semantic"] = True
    if args.semantic_proj_dim is not None:
        override["baseline"]["semantic_proj_dim"] = int(args.semantic_proj_dim)
    if args.epochs is not None:
        override.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.lr is not None:
        override.setdefault("train", {})["lr"] = float(args.lr)
    if args.weight_decay is not None:
        override.setdefault("train", {})["weight_decay"] = float(args.weight_decay)
    if args.eval_interval is not None:
        override.setdefault("train", {})["eval_interval"] = int(args.eval_interval)
    if args.max_eval_batches is not None:
        override.setdefault("train", {})["max_eval_batches"] = int(args.max_eval_batches)
    if args.patience is not None:
        override.setdefault("train", {})["patience"] = int(args.patience)
    if args.batch_size is not None:
        override.setdefault("dataset", {})["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        override.setdefault("dataset", {})["num_workers"] = int(args.num_workers)
    if args.input_feature_index is not None:
        override["baseline"]["input_feature_index"] = int(args.input_feature_index)
    if args.hidden_dim is not None:
        override["baseline"]["hidden_dim"] = int(args.hidden_dim)
    if args.num_layers is not None:
        override["baseline"]["num_layers"] = int(args.num_layers)
    return deep_merge(config, override)


def find_agcrn_model_file(repo: Path) -> Optional[Path]:
    """Find a likely official AGCRN implementation file."""
    if not repo.exists():
        return None
    preferred = [
        "model/AGCRN.py",
        "model/agcrn.py",
        "models/AGCRN.py",
        "models/agcrn.py",
        "AGCRN.py",
        "agcrn.py",
    ]
    for rel in preferred:
        candidate = repo / rel
        if candidate.exists():
            return candidate
    for candidate in repo.rglob("*.py"):
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "class AGCRN" in text:
            return candidate
    return None


def import_agcrn_class(model_file: Path) -> type[nn.Module]:
    """Import ``AGCRN`` class from an external Python file."""
    repo_root = str(model_file.parent.parent if model_file.parent.name in {"model", "models"} else model_file.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    module_name = f"external_agcrn_{abs(hash(str(model_file.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, str(model_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import AGCRN module from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "AGCRN", None)
    if cls is None:
        raise ImportError(f"No class AGCRN found in {model_file}")
    return cls


def _constructor_value(name: str, config: Mapping[str, Any], input_dim: int) -> Any:
    dcfg = config["dataset"]
    bcfg = dict(config.get("baseline", {}) or {})
    tcfg = dict(config.get("train", {}) or {})
    output_dim = int(bcfg.get("output_dim", 1 if tcfg.get("metric_feature_index") is not None else dcfg["input_dim"]))
    mapping = {
        "num_nodes": int(dcfg["num_nodes"]),
        "node_num": int(dcfg["num_nodes"]),
        "input_dim": int(input_dim),
        "input_size": int(input_dim),
        "in_dim": int(input_dim),
        "output_dim": output_dim,
        "out_dim": output_dim,
        "horizon": int(dcfg["horizon_steps"]),
        "horizon_steps": int(dcfg["horizon_steps"]),
        "seq_len": int(dcfg["history_steps"]),
        "input_window": int(dcfg["history_steps"]),
        "output_window": int(dcfg["horizon_steps"]),
        "rnn_units": int(bcfg.get("hidden_dim", bcfg.get("rnn_units", 64))),
        "hidden_dim": int(bcfg.get("hidden_dim", bcfg.get("rnn_units", 64))),
        "num_layers": int(bcfg.get("num_layers", 1)),
        "embed_dim": int(bcfg.get("embed_dim", 10)),
        "cheb_k": int(bcfg.get("cheb_k", 2)),
        "default_graph": bool(bcfg.get("default_graph", True)),
        "device": None,
        "seed": int(tcfg.get("seed", 42)),
    }
    return mapping.get(name)


def build_agcrn_args_namespace(
    config: Mapping[str, Any],
    input_dim: int,
    device: torch.device,
) -> SimpleNamespace:
    """Build the ``args`` object expected by the official AGCRN release."""
    dcfg = config["dataset"]
    bcfg = dict(config.get("baseline", {}) or {})
    tcfg = dict(config.get("train", {}) or {})
    output_dim = int(bcfg.get("output_dim", 1 if tcfg.get("metric_feature_index") is not None else dcfg["input_dim"]))
    return SimpleNamespace(
        num_nodes=int(dcfg["num_nodes"]),
        input_dim=int(input_dim),
        output_dim=output_dim,
        horizon=int(dcfg["horizon_steps"]),
        rnn_units=int(bcfg.get("hidden_dim", bcfg.get("rnn_units", 64))),
        num_layers=int(bcfg.get("num_layers", 1)),
        embed_dim=int(bcfg.get("embed_dim", 10)),
        cheb_k=int(bcfg.get("cheb_k", 2)),
        default_graph=bool(bcfg.get("default_graph", True)),
        device=device,
        seed=int(tcfg.get("seed", 42)),
    )


def instantiate_agcrn(
    cls: type[nn.Module],
    config: Mapping[str, Any],
    input_dim: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate common AGCRN constructor variants."""
    sig = inspect.signature(cls)
    kwargs: Dict[str, Any] = {}
    missing_required: List[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name == "args":
            value = build_agcrn_args_namespace(config=config, input_dim=input_dim, device=device)
        else:
            value = _constructor_value(name, config=config, input_dim=input_dim)
        if name == "device":
            value = device
        if value is None:
            if param.default is inspect._empty:
                missing_required.append(name)
            continue
        kwargs[name] = value
    if missing_required:
        raise TypeError(
            "Cannot instantiate official AGCRN automatically; missing required "
            f"constructor args {missing_required}. Add them to this runner or use a small adapter."
        )
    return cls(**kwargs)


def initialize_official_agcrn_parameters(model: nn.Module) -> None:
    """Initialize official AGCRN parameters that are raw FloatTensor buffers.

    The common LeiBAI/AGCRN implementation creates adaptive graph convolution
    pools with ``torch.FloatTensor(...)``. Those tensors are uninitialized until
    an external script applies an initializer. Since this runner instantiates the
    model directly, we initialize here to avoid all-NaN predictions at the first
    forward pass.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        with torch.no_grad():
            if "bias" in name or param.dim() == 1:
                nn.init.zeros_(param)
            elif "node_embedding" in name or "node_embeddings" in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.xavier_uniform_(param)
        if not torch.isfinite(param).all():
            raise FloatingPointError(f"Non-finite AGCRN parameter after init: {name}")


def build_model(config: Mapping[str, Any], args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, str, str]:
    """Build official AGCRN when available, otherwise optional fallback."""
    dcfg = config["dataset"]
    bcfg = dict(config.get("baseline", {}) or {})
    tcfg = dict(config.get("train", {}) or {})
    adapter_cfg = build_semantic_adapter_config(config)
    use_semantic = bool(adapter_cfg.use_semantic)
    base_input_dim = int(dcfg["input_dim"])
    input_feature_index = bcfg.get("input_feature_index", tcfg.get("metric_feature_index"))
    if input_feature_index is not None:
        base_input_dim = 1
    input_dim = base_input_dim + (int(adapter_cfg.d_proj) if use_semantic else 0)

    model_file = Path(args.model_file) if args.model_file else find_agcrn_model_file(Path(args.agcrn_repo))
    implementation = "official"
    notes = "official AGCRN dynamically imported"
    try:
        if model_file is None:
            raise FileNotFoundError(f"AGCRN model file not found under {args.agcrn_repo}")
        cls = import_agcrn_class(model_file)
        raw_model = instantiate_agcrn(cls, config=config, input_dim=input_dim, device=device)
        if not args.skip_agcrn_init:
            initialize_official_agcrn_parameters(raw_model)
        base_model = OfficialAGCRNForwardAdapter(
            raw_model
        )
        notes += f" from {model_file}"
    except Exception as exc:
        if not args.fallback:
            raise RuntimeError(
                "Failed to load official AGCRN. Pass --agcrn_repo/--model_file pointing to the official repo, "
                "or pass --fallback only for smoke testing. Original error: " + repr(exc)
            ) from exc
        base_model = FallbackAGCRN(
            input_dim=input_dim,
            output_dim=int(
                dict(config.get("baseline", {}) or {}).get(
                    "output_dim",
                    1 if config["train"].get("metric_feature_index") is not None else dcfg["input_dim"],
                )
            ),
            horizon=int(dcfg["horizon_steps"]),
            num_nodes=int(dcfg["num_nodes"]),
            rnn_units=int(bcfg.get("hidden_dim", 64)),
            num_layers=int(bcfg.get("num_layers", 1)),
        )
        implementation = "fallback-smoke"
        notes = f"fallback smoke model; official import failed: {exc!r}"

    z_sem = load_semantic_embeddings(adapter_cfg.static_path)
    wrapper = AGCRNSemanticWrapper(
        base_model=base_model,
        sem_dim=int(z_sem.shape[-1]),
        d_proj=int(adapter_cfg.d_proj),
        use_semantic=use_semantic,
        baseline_layout=str(dict(config.get("baseline", {}) or {}).get("layout", "BTNF")),
        dropout=float(dict(config.get("baseline", {}) or {}).get("semantic_dropout", 0.0)),
    ).to(device)
    return wrapper, implementation, notes


def inverse_with_scaler(
    x: torch.Tensor,
    scaler: Optional[object],
    metric_feature_index: Optional[int] = None,
) -> torch.Tensor:
    if scaler is None:
        return x
    mean = getattr(scaler, "mean", None)
    std = getattr(scaler, "std", None)
    if metric_feature_index is not None and mean is not None and std is not None and int(x.shape[-1]) == 1:
        idx = int(metric_feature_index)
        mean_t = torch.as_tensor(mean[..., idx : idx + 1], dtype=x.dtype, device=x.device)
        std_t = torch.as_tensor(std[..., idx : idx + 1], dtype=x.dtype, device=x.device)
        return x * std_t + mean_t
    x_np = x.detach().cpu().numpy()
    inv = scaler.inverse_transform(x_np)
    return torch.from_numpy(inv).to(x.device)


def select_metric_feature(x: torch.Tensor, idx: Optional[int]) -> torch.Tensor:
    if idx is None:
        return x
    return x[..., int(idx) : int(idx) + 1]


def select_input_feature(x_his: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    """Select baseline input channel when official AGCRN expects one feature."""
    bcfg = dict(config.get("baseline", {}) or {})
    tcfg = dict(config.get("train", {}) or {})
    idx = bcfg.get("input_feature_index", tcfg.get("metric_feature_index"))
    if idx is None:
        return x_his
    return x_his[..., int(idx) : int(idx) + 1]


def select_training_target(x_fut: torch.Tensor, metric_feature_index: Optional[int]) -> torch.Tensor:
    """Select the supervised target channel for deterministic baselines."""
    if metric_feature_index is None:
        return x_fut
    return x_fut[..., int(metric_feature_index) : int(metric_feature_index) + 1]


def assert_finite(name: str, x: torch.Tensor, batch_idx: int) -> None:
    """Fail fast on NaN/Inf with useful context."""
    if torch.isfinite(x).all():
        return
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        stats = "all values are non-finite"
    else:
        stats = f"finite_min={float(finite.min().item()):.6g} finite_max={float(finite.max().item()):.6g}"
    raise FloatingPointError(f"Non-finite {name} at batch {batch_idx}: shape={tuple(x.shape)} {stats}")


def normalize_output(pred: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Normalize common AGCRN output layouts to [B,H,N,F]."""
    if pred.dim() != 4:
        raise ValueError(f"Expected AGCRN output as 4D tensor, got {tuple(pred.shape)}")
    b, h, n, f = map(int, target_shape)
    if tuple(pred.shape) == (b, h, n, f):
        return pred
    if tuple(pred.shape) == (b, n, h, f):
        return pred.permute(0, 2, 1, 3).contiguous()
    if tuple(pred.shape) == (b, h, f, n):
        return pred.permute(0, 1, 3, 2).contiguous()
    if tuple(pred.shape) == (b, f, n, h):
        return pred.permute(0, 3, 2, 1).contiguous()
    raise ValueError(f"Cannot infer AGCRN output layout: pred={tuple(pred.shape)}, target={tuple(target_shape)}")


@torch.no_grad()
def evaluate_agcrn(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    scaler: Optional[object],
    composer: Optional[BatchSemanticComposer],
    eval_horizons: Iterable[int],
    max_batches: Optional[int],
    metric_feature_index: Optional[int],
    mape_eps: float,
    mape_mask_threshold: float,
) -> Dict[str, float]:
    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for idx, batch in enumerate(loader, start=1):
        x_his = batch["x_his"].to(device=device, dtype=torch.float32)
        x_in = select_input_feature(x_his, config=getattr(model, "_run_config", {}))
        x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
        x_tgt = select_training_target(x_fut, metric_feature_index)
        assert_finite("x_his", x_in, idx)
        assert_finite("x_tgt", x_tgt, idx)
        z_batch = composer.compose(batch, device=device, num_nodes=x_his.shape[2]) if composer is not None else None
        pred = normalize_output(model(x_in, z_batch=z_batch), x_tgt.shape)
        assert_finite("prediction", pred, idx)
        pred_inv = inverse_with_scaler(pred, scaler, metric_feature_index=metric_feature_index)
        target_inv = inverse_with_scaler(x_tgt, scaler, metric_feature_index=metric_feature_index)
        preds.append(pred_inv)
        targets.append(target_inv)
        if max_batches is not None and max_batches > 0 and idx >= max_batches:
            break
    if not preds:
        raise RuntimeError("No evaluation batches processed.")
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    metrics = compute_all_metrics(pred_all, target_all, mape_eps=mape_eps, mape_mask_threshold=mape_mask_threshold)
    h_total = int(target_all.shape[1])
    for hh in eval_horizons:
        h_idx = int(hh) - 1
        if 0 <= h_idx < h_total:
            mh = compute_all_metrics(
                pred_all[:, h_idx : h_idx + 1],
                target_all[:, h_idx : h_idx + 1],
                mape_eps=mape_eps,
                mape_mask_threshold=mape_mask_threshold,
            )
            metrics[f"mae@{hh}"] = mh["mae"]
            metrics[f"rmse@{hh}"] = mh["rmse"]
            metrics[f"mape@{hh}"] = mh["mape"]
    return metrics


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": dict(metrics)}, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> Mapping[str, Any]:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    return ckpt


def result_rows(
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
    config_path: str,
    setting: str,
    seed: int,
    implementation: str,
    checkpoint: str,
    notes: str,
) -> List[ExperimentResult]:
    dcfg = config["dataset"]
    horizons = list(dcfg.get("eval_horizons", [dcfg["horizon_steps"]]))
    rows: List[ExperimentResult] = []
    settings_json = json.dumps(dict(config.get("baseline", {}) or {}), sort_keys=True)
    for hh in horizons:
        rows.append(
            ExperimentResult(
                dataset=str(dcfg["name"]),
                method="AGCRN",
                setting=setting,
                horizon=int(hh),
                mae=float(metrics.get(f"mae@{hh}", metrics["mae"])),
                rmse=float(metrics.get(f"rmse@{hh}", metrics["rmse"])),
                crps=None,
                seed=int(seed),
                config=config_path,
                implementation=implementation,
                checkpoint=checkpoint,
                settings_json=settings_json,
                notes=notes,
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    logger = get_logger()
    set_seed(int(config["train"].get("seed", 42)))

    device_arg = args.device if args.gpu_id is None else f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)
    artifacts = build_dataloaders(config)
    model, implementation, notes = build_model(config, args=args, device=device)
    setattr(model, "_run_config", config)

    adapter_cfg = build_semantic_adapter_config(config, logger=logger)
    composer = BatchSemanticComposer.from_config(config, device=device, logger=logger) if adapter_cfg.use_semantic else None
    setting = "+semantic" if adapter_cfg.use_semantic else "original"

    train_cfg = config["train"]
    metric_feature_index = train_cfg.get("metric_feature_index")
    metric_feature_index = None if metric_feature_index is None else int(metric_feature_index)
    eval_horizons = list(config["dataset"].get("eval_horizons", [config["dataset"]["horizon_steps"]]))
    max_eval_batches = train_cfg.get("max_eval_batches")
    max_eval_batches = None if max_eval_batches is None else int(max_eval_batches)
    mape_eps = float(train_cfg.get("mape_eps", 1e-5))
    mape_mask_threshold = float(train_cfg.get("mape_mask_threshold", 1.0))

    tag = args.save_tag or setting.replace("+", "sem_")
    save_dir = Path(args.save_dir) / str(config["dataset"]["name"]) / tag
    best_path = save_dir / "best.pt"

    if args.ckpt:
        load_checkpoint(Path(args.ckpt), model=model, device=device)

    if args.mode == "train":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        epochs = int(train_cfg.get("epochs", 50))
        eval_interval = int(train_cfg.get("eval_interval", 1))
        patience = int(train_cfg.get("patience", 20))
        best_mae = float("inf")
        bad_epochs = 0
        loss_fn = nn.L1Loss()
        for epoch in range(1, epochs + 1):
            model.train()
            losses: List[float] = []
            start = time.time()
            for train_batch_idx, batch in enumerate(artifacts.train_loader, start=1):
                x_his = batch["x_his"].to(device=device, dtype=torch.float32)
                x_in = select_input_feature(x_his, config=config)
                x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
                x_tgt = select_training_target(x_fut, metric_feature_index)
                assert_finite("x_his", x_in, train_batch_idx)
                assert_finite("x_tgt", x_tgt, train_batch_idx)
                z_batch = composer.compose(batch, device=device, num_nodes=x_his.shape[2]) if composer is not None else None
                pred = normalize_output(model(x_in, z_batch=z_batch), x_tgt.shape)
                assert_finite("prediction", pred, train_batch_idx)
                loss = loss_fn(pred, x_tgt)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite AGCRN loss at batch {train_batch_idx}: {loss.item()}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 5.0)))
                optimizer.step()
                losses.append(float(loss.item()))
                if args.max_train_batches is not None and args.max_train_batches > 0 and train_batch_idx >= args.max_train_batches:
                    break
            logger.info(
                "AGCRN Epoch %d | train_loss=%.6f | time=%.2fs | setting=%s",
                epoch,
                sum(losses) / max(len(losses), 1),
                time.time() - start,
                setting,
            )
            if eval_interval > 0 and epoch % eval_interval == 0:
                metrics = evaluate_agcrn(
                    model=model,
                    loader=artifacts.val_loader,
                    device=device,
                    scaler=artifacts.scaler,
                    composer=composer,
                    eval_horizons=eval_horizons,
                    max_batches=max_eval_batches,
                    metric_feature_index=metric_feature_index,
                    mape_eps=mape_eps,
                    mape_mask_threshold=mape_mask_threshold,
                )
                logger.info("AGCRN Epoch %d | val_mae=%.6f val_rmse=%.6f", epoch, metrics["mae"], metrics["rmse"])
                if metrics["mae"] < best_mae:
                    best_mae = float(metrics["mae"])
                    bad_epochs = 0
                    save_checkpoint(best_path, model=model, optimizer=optimizer, epoch=epoch, metrics=metrics)
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        logger.info("AGCRN early stop at epoch %d", epoch)
                        break
        if best_path.exists():
            load_checkpoint(best_path, model=model, device=device)

    metrics = evaluate_agcrn(
        model=model,
        loader=artifacts.test_loader,
        device=device,
        scaler=artifacts.scaler,
        composer=composer,
        eval_horizons=eval_horizons,
        max_batches=max_eval_batches,
        metric_feature_index=metric_feature_index,
        mape_eps=mape_eps,
        mape_mask_threshold=mape_mask_threshold,
    )
    logger.info("AGCRN test | setting=%s mae=%.6f rmse=%.6f", setting, metrics["mae"], metrics["rmse"])
    rows = result_rows(
        metrics=metrics,
        config=config,
        config_path=args.config,
        setting=setting,
        seed=int(config["train"].get("seed", 42)),
        implementation=implementation,
        checkpoint=str(best_path if best_path.exists() else args.ckpt),
        notes=notes,
    )
    write_experiment_results(
        rows,
        csv_path=Path(args.output_csv),
        md_path=Path(args.results_md),
        title="AGCRN Baseline Results",
    )


if __name__ == "__main__":
    main()
