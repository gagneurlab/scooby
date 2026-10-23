"""Tests for the median-based collapse guard and the batched step metrics."""

import torch

from scooby.scag.train import CollapseGuard, StepMetrics


def run(guard, peaks, check_every=1):
    """Record one peak per step; check every ``check_every`` steps."""
    for step, peak in enumerate(peaks, 1):
        guard.record(torch.tensor([peak]))
        if step % check_every == 0 and guard.check(step) is not None:
            return step
    return None


def test_trips_when_the_median_stays_low_despite_spikes():
    # One healthy spike every ten steps would reset a consecutive count.
    peaks = [10.0 if i % 10 == 0 else 1e-8 for i in range(200)]
    assert run(CollapseGuard(patience=50, threshold=1e-5), peaks) == 50


def test_checking_rarely_still_uses_every_recorded_step():
    # Collapse starts at step 100; checking every 100 steps catches it at the
    # first check with a full window of collapsed steps.
    peaks = [1.0] * 100 + [1e-9] * 300
    assert run(CollapseGuard(patience=100, threshold=1e-5), peaks, check_every=100) == 200


def test_healthy_run_never_trips():
    peaks = [0.5 + 0.1 * (i % 3) for i in range(500)]
    assert run(CollapseGuard(patience=50, threshold=1e-5), peaks) is None


def test_waits_for_a_full_window():
    assert run(CollapseGuard(patience=50, threshold=1e-5), [1e-9] * 49) is None


def test_zero_patience_disables():
    assert run(CollapseGuard(patience=0, threshold=1e-5), [1e-9] * 100) is None


def test_step_metrics_average_loss_and_median_peak():
    metrics = StepMetrics()
    for loss, peak in [(1.0, 0.1), (2.0, 0.5), (3.0, 9.0)]:
        metrics.update(torch.tensor(loss), torch.tensor([peak]))
    stats = metrics.flush()
    assert abs(stats["loss"] - 2.0) < 1e-6 and abs(stats["output_max"] - 0.5) < 1e-6
    assert metrics.flush() == {}
