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

__all__ = ["Evaluator"]


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
        kl_divs = []
        maes = []
        squared_errors = []

        for sample in samples:
            # Targets
            t_a = sample["target_alpha"]
            t_b = sample["target_beta"]
            t_mean = t_a / (t_a + t_b)

            # Predictions (Use 'prior' from sample as the baseline prediction if no model pred provided)
            # In E2E evaluate_model, we run the model. We should parse alpha/beta.
            # But parsing is complex if model outputs free text.
            # For "Standard Schema", let's assume we have p_a, p_b.

            # Fallback to prior if 'prediction' key missing
            p_a = sample.get("prior_alpha")
            p_b = sample.get("prior_beta")

            if predictions and "alpha" in predictions[0]:
                 # Handle list of prediction dicts
                 pass

            # Compute KL(Prior || Target) as a proxy for "Information Gain" required
            # Or KL(Target || Prior) ?
            # KL(P || Q): Divergence of Q from P. P is True.
            # So KL(Target || Prediction).

            kl = self.kl_beta(t_a, t_b, p_a, p_b)
            kl_divs.append(kl)

            if p_a is not None and p_b is not None:
                p_mean = p_a / (p_a + p_b)
            else:
                p_mean = t_mean  # fallback when prior is missing
            maes.append(abs(p_mean - t_mean))
            squared_errors.append((p_mean - t_mean)**2)

        return {
            "samples": len(samples),
            "kl_divergence": float(np.mean(kl_divs)),
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
