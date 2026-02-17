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
Beta prior projector for LLM injection.

This module provides neural network components for projecting Beta distribution
parameters and temporal context into vector embeddings suitable for LLM injection.
Implements Decision 1 from the training architecture.

Example:
    >>> import torch
    >>> from xrtm.train.kit.projectors import BetaPriorProjector
    >>> projector = BetaPriorProjector(embed_dim=64, output_dim=768)
    >>> embedding = projector(
    ...     alpha=torch.tensor([7.0]),
    ...     beta=torch.tensor([3.0]),
    ...     silence_delta=torch.tensor([0.1]),
    ...     deadline_delta=torch.tensor([0.5]),
    ... )
    >>> print(embedding.shape)  # (1, 768)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    r"""
    Positional-style encoding for scalar values.

    Uses sinusoidal functions at multiple frequencies to embed
    continuous scalar values into fixed-dimensional vectors.
    Inspired by transformer positional encodings.

    Attributes:
        dim: Output embedding dimension.
        max_period: Maximum period for sinusoidal frequencies.

    Example:
        >>> embed = SinusoidalEmbedding(dim=64)
        >>> x = torch.tensor([0.5, 0.7, 0.3])
        >>> output = embed(x)
        >>> print(output.shape)  # (3, 64)
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        r"""
        Initialize the sinusoidal embedding layer.

        Args:
            dim: Output embedding dimension (must be even).
            max_period: Maximum period for frequency computation.
        """
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Embed scalar values into sinusoidal vectors.

        Args:
            x: Scalar values of shape (batch,) or (batch, 1).

        Returns:
            Embeddings of shape (batch, dim).
        """
        if x.dim() == 2:
            x = x.squeeze(-1)
        angles = x.unsqueeze(-1) * self.freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class BetaPriorProjector(nn.Module):
    r"""
    Projects Beta(α,β) + time deltas to LLM-compatible vector.

    Decision 1 Implementation: Projects prior state into a dense vector
    for injection into the LLM's hidden state or cross-attention.

    The projection formula:
        v_inject = MLP(Concat[
            FreqEmbed(α),
            FreqEmbed(β),
            FreqEmbed(δ_silence),
            FreqEmbed(τ_remain)
        ])

    Attributes:
        embed_dim: Dimension of each scalar embedding.
        output_dim: Final output dimension (matches LLM hidden size).

    Example:
        >>> projector = BetaPriorProjector(embed_dim=64, output_dim=768)
        >>> embedding = projector(
        ...     alpha=torch.tensor([7.0, 8.0]),
        ...     beta=torch.tensor([3.0, 2.0]),
        ...     silence_delta=torch.tensor([0.1, 0.2]),
        ...     deadline_delta=torch.tensor([0.5, 0.3]),
        ... )
        >>> print(embedding.shape)  # (2, 768)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        output_dim: int = 768,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        r"""
        Initialize the Beta prior projector.

        Args:
            embed_dim: Dimension for each scalar embedding.
            output_dim: Final output dimension (LLM hidden size).
            hidden_dim: MLP hidden dimension. Defaults to output_dim.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Separate embedders for each input
        self.alpha_embed = SinusoidalEmbedding(embed_dim)
        self.beta_embed = SinusoidalEmbedding(embed_dim)
        self.silence_embed = SinusoidalEmbedding(embed_dim)
        self.deadline_embed = SinusoidalEmbedding(embed_dim)

        # MLP projection
        hidden = hidden_dim or output_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        silence_delta: torch.Tensor,
        deadline_delta: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Project prior state to LLM-compatible embedding.

        Args:
            alpha: Beta distribution α parameters, shape (batch,).
            beta: Beta distribution β parameters, shape (batch,).
            silence_delta: Normalized silence time, shape (batch,).
            deadline_delta: Normalized remaining time, shape (batch,).

        Returns:
            Dense embedding of shape (batch, output_dim).
        """
        # Embed each scalar
        alpha_emb = self.alpha_embed(alpha)
        beta_emb = self.beta_embed(beta)
        silence_emb = self.silence_embed(silence_delta)
        deadline_emb = self.deadline_embed(deadline_delta)

        # Concatenate and project
        combined = torch.cat([alpha_emb, beta_emb, silence_emb, deadline_emb], dim=-1)
        return self.mlp(combined)

    @classmethod
    def from_config(
        cls,
        llm_hidden_size: int = 768,
        embed_dim: int = 64,
    ) -> "BetaPriorProjector":
        r"""
        Create projector from LLM configuration.

        Args:
            llm_hidden_size: Hidden size of the target LLM.
            embed_dim: Dimension for scalar embeddings.

        Returns:
            Configured BetaPriorProjector.
        """
        return cls(embed_dim=embed_dim, output_dim=llm_hidden_size)


class DualHeadProjector(nn.Module):
    r"""
    Dual-head output projector for Beta + Text generation.

    Projects LLM hidden states to both:
    - Head A: Beta distribution parameters (α, β)
    - Head B: Text logits (standard LM head)

    Attributes:
        input_dim: LLM hidden size.

    Example:
        >>> dual_head = DualHeadProjector(input_dim=768)
        >>> hidden = torch.randn(2, 768)
        >>> alpha, beta = dual_head.beta_head(hidden)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: Optional[int] = None,
    ) -> None:
        r"""
        Initialize the dual-head projector.

        Args:
            input_dim: LLM hidden size.
            hidden_dim: Intermediate hidden dimension.
        """
        super().__init__()
        hidden = hidden_dim or input_dim // 2

        # Head A: Beta parameters
        self.beta_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # (α, β)
            nn.Softplus(),  # Ensure positive outputs
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Project hidden states to Beta parameters.

        Args:
            hidden_states: LLM hidden states of shape (batch, input_dim).

        Returns:
            Tuple of (alpha, beta) tensors, each shape (batch,).
        """
        beta_params = self.beta_head(hidden_states)
        alpha = beta_params[:, 0] + 0.1  # Minimum α = 0.1
        beta = beta_params[:, 1] + 0.1   # Minimum β = 0.1
        return alpha, beta


__all__ = [
    "SinusoidalEmbedding",
    "BetaPriorProjector",
    "DualHeadProjector",
]
