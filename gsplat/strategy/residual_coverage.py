# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .default import DefaultStrategy
from .ops import duplicate, remove, split


@dataclass
class ResidualCoverageStrategy(DefaultStrategy):
    """Densification strategy that augments gradient heuristics with residual and coverage.

    This strategy is designed as a drop-in replacement for :class:`DefaultStrategy`.
    It keeps the same training schedule and optimizer mutation flow, while extending
    the running state with:

    - ``residual_ema``: an EMA of reconstruction error for visible Gaussians.
    - ``coverage_ema``: an EMA of how often a Gaussian is visible.

    The trainer is expected to populate ``info["residual_value"]`` before
    :meth:`step_post_backward`. A scalar batch-level residual is sufficient for a first
    implementation and keeps the strategy decoupled from the rasterizer internals.
    """

    lambda_grad: float = 0.6
    lambda_residual: float = 0.4
    grow_score: float = 0.55
    coverage_min: float = 0.05
    residual_ema_decay: float = 0.9
    coverage_ema_decay: float = 0.99
    cap_max: int = -1
    prune_opacity_weight: float = 0.5
    prune_coverage_weight: float = 0.3
    prune_residual_weight: float = 0.2
    residual_key: str = "residual_value"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = super().initialize_state(scene_scale=scene_scale)
        state["residual_ema"] = None
        state["coverage_ema"] = None
        return state

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        super()._update_state(params, state, info, packed=packed)

        if self.residual_key not in info:
            return

        n_gaussian = len(list(params.values())[0])
        device = params["means"].device
        if state["residual_ema"] is None:
            state["residual_ema"] = torch.zeros(n_gaussian, device=device)
        if state["coverage_ema"] is None:
            state["coverage_ema"] = torch.zeros(n_gaussian, device=device)

        residual_value = info[self.residual_key]
        if not torch.is_tensor(residual_value):
            residual_value = torch.tensor(residual_value, device=device)
        residual_value = residual_value.detach().to(device=device).float().reshape(-1)
        if residual_value.numel() == 0:
            return
        residual_scalar = residual_value.mean()

        if packed:
            gs_ids = info["gaussian_ids"]
            visible_mask = torch.zeros(n_gaussian, dtype=torch.bool, device=device)
            visible_mask[gs_ids] = True
            visible_ids = torch.where(visible_mask)[0]
        else:
            visible_mask = (info["radii"] > 0.0).all(dim=-1).any(dim=0)
            visible_ids = torch.where(visible_mask)[0]

        if len(visible_ids) == 0:
            state["coverage_ema"].mul_(self.coverage_ema_decay)
            return

        state["coverage_ema"].mul_(self.coverage_ema_decay)
        state["coverage_ema"][visible_ids] += 1.0 - self.coverage_ema_decay
        state["residual_ema"][visible_ids] = (
            state["residual_ema"][visible_ids] * self.residual_ema_decay
            + residual_scalar * (1.0 - self.residual_ema_decay)
        )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        grad_score = self._normalize(grads)
        residual_score = self._normalize(state["residual_ema"])
        score = self.lambda_grad * grad_score + self.lambda_residual * residual_score

        device = grads.device
        passes_score = score > self.grow_score
        passes_coverage = state["coverage_ema"] > self.coverage_min

        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = passes_score & passes_coverage & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = passes_score & passes_coverage & is_large
        if step < self.refine_scale2d_stop_iter and state.get("radii") is not None:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)]
        )

        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            if step < self.refine_scale2d_stop_iter and state.get("radii") is not None:
                is_too_big |= state["radii"] > self.prune_scale2d
            is_prune = is_prune | is_too_big

        if self.cap_max > 0 and len(params["means"]) > self.cap_max:
            n_extra = len(params["means"]) - self.cap_max
            keep_score = self._compute_keep_score(params, state)
            candidate_ids = torch.where(~is_prune)[0]
            if n_extra >= len(candidate_ids):
                is_prune[:] = True
            elif n_extra > 0:
                extra_ids = candidate_ids[
                    torch.topk(keep_score[candidate_ids], k=n_extra, largest=False).indices
                ]
                is_prune[extra_ids] = True

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        return n_prune

    def _compute_keep_score(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
    ) -> torch.Tensor:
        opacity_score = self._normalize(torch.sigmoid(params["opacities"].flatten()))
        coverage_score = self._normalize(state["coverage_ema"])
        residual_score = self._normalize(state["residual_ema"])
        return (
            self.prune_opacity_weight * opacity_score
            + self.prune_coverage_weight * coverage_score
            + self.prune_residual_weight * residual_score
        )

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if x is None or x.numel() == 0:
            return x
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + eps)
