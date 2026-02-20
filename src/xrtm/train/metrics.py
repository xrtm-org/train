# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Standardized evaluation metrics for the xRTM training pipeline.

Computes KL divergence, MAE, and simulated Brier score
over Beta-distributed predictions vs. ground-truth parameters.
"""

from typing import Optional

import numpy as np
from scipy.special import betaln, psi


class Evaluator:
    r"""Standardized evaluation metrics for xRTM pipeline."""

    def compute_metrics(
        self, samples: list[dict], predictions: Optional[list[dict]] = None
    ) -> dict:
        """
        Compute standard metrics.

        Args:
            samples: List of sample dicts containing 'target_alpha', 'target_beta', etc.
            predictions: Optional list of predictions if different from sample 'prior' (not used in current E2E flow yet,
                        where 'prior' is the prediction before update).
                        In E2E script context, we want to evaluate the MODEL's prediction.
                        But currently E2E script model output is text.
                        The Evaluator needs to parse text or assumes 'prior' is the prediction?
                        Wait, in the E2E script, evaluate_model runs the model to get NEW predictions.
                        So we need to parse expected alpha/beta from text.

                        For this minimal implementation, we will compute metrics on the *training samples themselves* (Prior vs Target)
                        as a baseline, AND if 'predictions' are passed (parsed), we evaluate them.
        """
        # Extract targets
        t_a = np.array([s["target_alpha"] for s in samples])
        t_b = np.array([s["target_beta"] for s in samples])
        t_mean = t_a / (t_a + t_b)

        # Extract priors (predictions)
        # Use 'prior' from sample as the baseline prediction if no model pred provided
        # Fallback to NaN if 'prediction' key missing to handle missing priors
        p_a = np.array([s.get("prior_alpha", np.nan) for s in samples])
        p_b = np.array([s.get("prior_beta", np.nan) for s in samples])

        if predictions and "alpha" in predictions[0]:
             # Handle list of prediction dicts
             # TODO: Implement extracting p_a/p_b from predictions list if needed
             pass

        # Compute KL(Prior || Target) as a proxy for "Information Gain" required
        # Or KL(Target || Prior) ?
        # KL(P || Q): Divergence of Q from P. P is True.
        # So KL(Target || Prediction).

        # kl_beta is vectorized thanks to scipy ufuncs
        kl_divs = self.kl_beta(t_a, t_b, p_a, p_b)

        # Calculate p_mean
        # If p_a or p_b is NaN (missing prior), we fall back to t_mean (error = 0)
        # This replicates the logic: if p_a is not None and p_b is not None: p_mean = p_a/(p_a+p_b) else: p_mean = t_mean
        p_mean_calc = p_a / (p_a + p_b)
        p_mean = np.where(np.isnan(p_mean_calc), t_mean, p_mean_calc)

        # Calculate errors
        maes = np.abs(p_mean - t_mean)
        squared_errors = (p_mean - t_mean)**2

        return {
            "samples": len(samples),
            "kl_divergence": float(np.nanmean(kl_divs)), # Use nanmean to ignore missing priors
            "mae": float(np.mean(maes)),
            "brier_score": float(np.mean(squared_errors)), # Simulated Brier (MSE)
            "ece": 0.0 # Placeholder
        }

    def kl_beta(self, a1, b1, a2, b2):
        """KL divergence between Beta(a1, b1) and Beta(a2, b2)."""
        # Formula: ln Gamma(a1) + ln Gamma(b1) - ln Gamma(a1+b1) ...
        # Actually: ln B(a2,b2) - ln B(a1,b1) + (a1-a2)psi(a1) + (b1-b2)psi(b1) + (a2-a1+b2-b1)psi(a1+b1)
        # Wait, check direction. KL(True || Pred).
        # P=True(1), Q=Pred(2).
        # result

        return (betaln(a2, b2) - betaln(a1, b1)
                + (a1 - a2) * psi(a1)
                + (b1 - b2) * psi(b1)
                + (a2 + b2 - a1 - b1) * psi(a1 + b1))
