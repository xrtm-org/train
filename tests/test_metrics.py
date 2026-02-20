
import numpy as np

from xrtm.train.metrics import Evaluator


def test_missing_prior():
    evaluator = Evaluator()
    samples = [{
        "target_alpha": 10.0,
        "target_beta": 10.0,
        # Missing prior_alpha and prior_beta
    }]

    metrics = evaluator.compute_metrics(samples)

    assert np.isnan(metrics["kl_divergence"])
    assert metrics["mae"] == 0.0
    assert metrics["brier_score"] == 0.0
    assert metrics["samples"] == 1

def test_compute_metrics():
    evaluator = Evaluator()
    samples = [
        {
            "target_alpha": 10.0,
            "target_beta": 10.0,
            "prior_alpha": 10.0,
            "prior_beta": 10.0,
        },
        {
            "target_alpha": 10.0,
            "target_beta": 10.0,
            "prior_alpha": 20.0,
            "prior_beta": 20.0,
        }
    ]

    metrics = evaluator.compute_metrics(samples)

    # Expected:
    # Sample 1: t_mean = 0.5, p_mean = 0.5. MAE=0, Brier=0, KL=0
    # Sample 2: t_mean = 0.5, p_mean = 0.5. MAE=0, Brier=0, KL>0 (variance differs)

    assert metrics["mae"] == 0.0
    assert metrics["brier_score"] == 0.0
    assert metrics["kl_divergence"] > 0.0
    assert metrics["samples"] == 2
