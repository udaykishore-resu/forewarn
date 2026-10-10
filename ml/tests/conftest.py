"""Shared fixtures: one small simulated bank, built once per test session."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forewarn.features import assign_split, build_features, build_labels  # noqa: E402
from forewarn.simulate import SimConfig, simulate  # noqa: E402


@pytest.fixture(scope="session")
def bank():
    return simulate(SimConfig(n_customers=6000, seed=11))


@pytest.fixture(scope="session")
def snapshots(bank):
    snap, panel = build_features(bank["customers"], bank["dda_transactions"], bank["card_transactions"], bank["card_statements"])
    labels = build_labels(snap, panel, bank["card_statements"], bank["outcomes"], bank["interventions"])
    labels["split"] = assign_split(labels["snapshot_month"].to_numpy(), panel.n_months)
    return snap, labels, panel
