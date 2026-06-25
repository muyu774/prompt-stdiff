"""CPU tests for incident-gated residual mean-correction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.incident_correction import (
    GraphMeanPropagator,
    MeanCorrectionHead,
    RegimeShiftHead,
    correction_regression_loss,
    detection_loss,
    regime_shift_labels,
    sparsity_loss,
)
from models.prompt_stdiff import PromptSTDiff


def _make_model(use_correction: bool) -> PromptSTDiff:
    torch.manual_seed(0)
    return PromptSTDiff(
        input_dim=1,
        sem_dim=4,
        hidden_dim=8,
        horizon_steps=3,
        time_embed_dim=8,
        router_hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        semantic_dropout_p=0.0,
        use_semantic=True,
        use_mean_head=True,
        center_residual_samples=True,
        use_incident_mean_correction=use_correction,
        incident_correction_hidden_dim=8,
        incident_correction_graph_hops=2,
    )


def _dummy_inputs(b=2, t=4, n=5, f=1, d=4):
    x_his = torch.randn(b, t, n, f)
    z_sem = torch.randn(n, d)
    a_phy = torch.softmax(torch.randn(n, n), dim=-1)  # row-stochastic adjacency
    return x_his, z_sem, a_phy


def test_regime_gate_starts_near_zero():
    head = RegimeShiftHead(input_dim=1, sem_dim=4, hidden_dim=8, horizon_steps=3)
    x_his, z_sem, _ = _dummy_inputs()
    gate = head(x_his=x_his, z_sem=z_sem)
    assert gate.shape == (2, 3, 5, 1)
    assert torch.all(gate >= 0) and torch.all(gate <= 1)
    assert gate.max().item() < 0.05, "gate should be near zero by default"


def test_mean_correction_starts_at_zero():
    head = MeanCorrectionHead(input_dim=1, sem_dim=4, hidden_dim=8, horizon_steps=3, max_shift=4.0)
    x_his, z_sem, _ = _dummy_inputs()
    delta = head(x_his=x_his, z_sem=z_sem)
    assert delta.shape == (2, 3, 5, 1)
    assert torch.allclose(delta, torch.zeros_like(delta)), "delta should start at zero"


def test_propagator_is_identity_initialized():
    prop = GraphMeanPropagator(num_hops=2)
    x_his, _, a_phy = _dummy_inputs()
    shift = torch.randn(2, 3, 5, 1)
    out = prop(shift=shift, a_phy=a_phy)
    assert out.shape == shift.shape
    # Identity-favoring init: output should be close to the input shift.
    assert torch.allclose(out, shift, atol=0.1)


def test_propagator_can_spread_to_neighbors():
    prop = GraphMeanPropagator(num_hops=1)
    # Force all weight onto hop 1 to verify propagation actually uses the graph.
    with torch.no_grad():
        prop.hop_logits.copy_(torch.tensor([-10.0, 10.0]))
    n = 3
    a_phy = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    shift = torch.zeros(1, 1, n, 1)
    shift[0, 0, 0, 0] = 1.0  # only node 0 has a correction
    out = prop(shift=shift, a_phy=a_phy)
    # A @ shift moves node-0's value to the node that points at node 0 (node 2).
    assert out[0, 0, 2, 0].item() > 0.9
    assert out[0, 0, 0, 0].item() < 0.1


def test_disabled_correction_is_noop():
    model = _make_model(use_correction=False)
    x_his, z_sem, a_phy = _dummy_inputs()
    ensemble = torch.randn(6, 2, 3, 5, 1)
    out = model.apply_mean_correction(ensemble, x_his=x_his, z_sem=z_sem, a_phy=a_phy)
    assert torch.equal(out, ensemble)
    gate = model.predict_regime_gate(x_his=x_his, z_sem=z_sem)
    assert torch.equal(gate, torch.zeros_like(gate))


def test_correction_shifts_center_not_spread():
    model = _make_model(use_correction=True)
    model.eval()
    x_his, z_sem, a_phy = _dummy_inputs()
    # Force the gate fully open and inject a known shift so the effect is visible.
    with torch.no_grad():
        model.regime_shift_head.gate_head[-1].bias.fill_(10.0)
        model.mean_correction_head.shift_head[-1].bias.fill_(0.5)
    ensemble = torch.randn(8, 2, 3, 5, 1)
    out = model.apply_mean_correction(ensemble, x_his=x_his, z_sem=z_sem, a_phy=a_phy)
    # Spread (deviation from per-window ensemble mean) is preserved exactly.
    spread_in = ensemble - ensemble.mean(dim=0, keepdim=True)
    spread_out = out - out.mean(dim=0, keepdim=True)
    assert torch.allclose(spread_in, spread_out, atol=1e-5)
    # Center actually moved.
    assert not torch.allclose(out.mean(dim=0), ensemble.mean(dim=0), atol=1e-3)


def test_default_correction_is_nearly_identity():
    model = _make_model(use_correction=True)
    model.eval()
    x_his, z_sem, a_phy = _dummy_inputs()
    ensemble = torch.randn(8, 2, 3, 5, 1)
    out = model.apply_mean_correction(ensemble, x_his=x_his, z_sem=z_sem, a_phy=a_phy)
    # Sparse gate + zero delta => off-incident behavior is mean-preserving.
    assert torch.allclose(out, ensemble, atol=1e-3)


def test_losses_are_finite_and_trainable():
    model = _make_model(use_correction=True)
    x_his, z_sem, a_phy = _dummy_inputs()
    residual_target = torch.randn(2, 3, 5, 1) * 3.0  # some large mean-level misses
    losses = model.incident_correction_losses(
        residual_target=residual_target,
        x_his=x_his,
        z_sem=z_sem,
        a_phy=a_phy,
        regime_threshold=2.0,
    )
    total = losses["detection"] + losses["regression"] + losses["sparsity"]
    assert torch.isfinite(total)
    total.backward()
    # Only the correction heads should receive gradients.
    assert model.regime_shift_head.gate_head[-1].bias.grad is not None
    assert next(model.epsilon_theta.parameters()).grad is None


def test_label_and_loss_helpers():
    std_resid = torch.tensor([[0.0, 3.0], [1.0, 2.5]])
    labels = regime_shift_labels(std_resid, threshold=2.0)
    assert torch.equal(labels, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    gate = torch.full_like(labels, 0.5)
    assert torch.isfinite(detection_loss(gate, labels))
    gated_shift = torch.zeros_like(labels)
    target = torch.ones_like(labels)
    assert correction_regression_loss(gated_shift, target, labels).item() >= 0.0
    assert sparsity_loss(torch.ones_like(labels), labels).item() >= 0.0


def test_enable_switch_round_trip():
    model = _make_model(use_correction=True)
    model.set_incident_correction_enabled(False)
    x_his, z_sem, a_phy = _dummy_inputs()
    ensemble = torch.randn(4, 2, 3, 5, 1)
    out = model.apply_mean_correction(ensemble, x_his=x_his, z_sem=z_sem, a_phy=a_phy)
    assert torch.equal(out, ensemble), "disabled switch must be a strict no-op"
    model.set_incident_correction_enabled(True)
    assert model.apply_incident_correction is True


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
