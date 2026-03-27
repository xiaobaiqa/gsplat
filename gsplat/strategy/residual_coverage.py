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

    The trainer is expected to populate ``info["residual_map"]`` with a per-view residual
    image of shape ``[C, H, W]`` or ``[H, W]`` before :meth:`step_post_backward`.
    The strategy samples that map at each visible Gaussian projection, so the residual
    signal stays local instead of collapsing to a single image-wide scalar.
    """

    lambda_grad: float = 1.0
    lambda_residual: float = 0.35
    lambda_coverage: float = 0.0
    grow_score: float = 0.35
    coverage_min: float = 0.05
    residual_ema_decay: float = 0.9
    coverage_ema_decay: float = 0.99
    relaxed_grad_factor: float = 0.5
    residual_threshold: float = 0.25
    coverage_warmup_iter: int = 2000
    base_coverage_min: float = 0.01
    target_coverage: float = 0.2
    coverage_peak: float = 0.6
    growth_topk_ratio: float = 0.12
    residual_budget_ratio: float = 0.25
    residual_warmup_iter: int = 7000
    min_residual_budget: int = 16
    max_residual_budget: int = 2048
    growth_pressure_start: float = 18.0
    growth_pressure_end: float = 28.0
    grad_threshold_gain: float = 0.5
    residual_threshold_gain: float = 0.15
    max_new_gs: int = 16000
    cap_max: int = -1
    prune_opacity_weight: float = 0.5
    prune_coverage_weight: float = 0.3
    prune_residual_weight: float = 0.2
    residual_key: str = "residual_map"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = super().initialize_state(scene_scale=scene_scale)
        state["residual_ema"] = None
        state["coverage_ema"] = None
        state["init_n_gaussians"] = None
        return state

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        super()._update_state(params, state, info, packed=packed)

        n_gaussian = len(list(params.values())[0])
        device = params["means"].device
        if state["residual_ema"] is None:
            state["residual_ema"] = torch.zeros(n_gaussian, device=device)
        if state["coverage_ema"] is None:
            state["coverage_ema"] = torch.zeros(n_gaussian, device=device)
        if state["init_n_gaussians"] is None:
            state["init_n_gaussians"] = n_gaussian

        gs_ids, camera_ids, coords = self._get_visible_projection_info(info, packed=packed)
        if gs_ids.numel() == 0:
            state["coverage_ema"].mul_(self.coverage_ema_decay)
            return

        state["coverage_ema"].mul_(self.coverage_ema_decay)
        visible_weight = torch.zeros(n_gaussian, device=device)
        visible_weight.index_add_(
            0, gs_ids, torch.ones_like(gs_ids, device=device, dtype=torch.float32)
        )
        visible_ids = torch.where(visible_weight > 0)[0]
        state["coverage_ema"][visible_ids] += 1.0 - self.coverage_ema_decay

        if self.residual_key not in info:
            return

        residual_map = info[self.residual_key]
        if not torch.is_tensor(residual_map):
            residual_map = torch.tensor(residual_map, device=device)
        residual_map = residual_map.detach().to(device=device).float()
        if residual_map.ndim == 2:
            residual_map = residual_map.unsqueeze(0)
        if residual_map.numel() == 0:
            return

        sampled_residuals = self._sample_residual_map(
            residual_map=residual_map,
            camera_ids=camera_ids,
            coords=coords,
            width=info["width"],
            height=info["height"],
        )
        if sampled_residuals.numel() == 0:
            return

        residual_sum = torch.zeros(n_gaussian, device=device)
        residual_count = torch.zeros(n_gaussian, device=device)
        residual_sum.index_add_(0, gs_ids, sampled_residuals)
        residual_count.index_add_(
            0, gs_ids, torch.ones_like(sampled_residuals, dtype=torch.float32)
        )
        visible_ids = torch.where(residual_count > 0)[0]
        mean_residual = residual_sum[visible_ids] / residual_count[visible_ids].clamp_min(1)
        state["residual_ema"][visible_ids] = (
            state["residual_ema"][visible_ids] * self.residual_ema_decay
            + mean_residual * (1.0 - self.residual_ema_decay)
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
        residual_ema = state["residual_ema"]
        coverage_ema = state["coverage_ema"]
        coverage_score = self._coverage_reliability(coverage_ema, step=step)
        pressure = self._growth_pressure(state, len(params["means"]))
        grad_threshold = self.grow_grad2d * (1.0 + self.grad_threshold_gain * pressure)
        residual_threshold = min(
            self.residual_threshold + self.residual_threshold_gain * pressure,
            0.95,
        )

        device = grads.device
        base_grad_mask = grads > grad_threshold
        if step >= self.coverage_warmup_iter:
            base_grad_mask &= coverage_ema > self.base_coverage_min
        n_base = int(base_grad_mask.sum().item())

        relaxed_grad_mask = grads > (grad_threshold * self.relaxed_grad_factor)
        coverage_gate = coverage_ema > self._coverage_floor(step)
        residual_gate = residual_ema > residual_threshold
        residual_growth_mask = relaxed_grad_mask & residual_gate & coverage_gate & ~base_grad_mask

        score_gate = torch.zeros_like(base_grad_mask)
        candidate_ids = torch.where(residual_growth_mask)[0]
        n_candidates = int(candidate_ids.numel())
        if n_candidates > 0:
            residual_budget = n_candidates
            if self.growth_topk_ratio > 0:
                residual_budget = max(1, int(n_candidates * self.growth_topk_ratio))
            residual_budget = min(
                residual_budget,
                self._residual_budget_limit(n_base=n_base, step=step, pressure=pressure),
            )
            if self.max_new_gs > 0:
                residual_budget = min(
                    residual_budget,
                    max(self.max_new_gs - n_base, 0),
                )
            if residual_budget > 0:
                candidate_grad_ratio = (
                    grads[candidate_ids] / max(grad_threshold, 1e-8)
                ).clamp(0.0, 2.0)
                candidate_residual_ratio = (
                    residual_ema[candidate_ids] / max(residual_threshold, 1e-8)
                ).clamp(0.0, 3.0)
                candidate_score = (
                    self.lambda_residual
                    * candidate_residual_ratio
                    * (2.0 - candidate_grad_ratio)
                    * coverage_score[candidate_ids]
                )
                topk_ids = candidate_ids[
                    torch.topk(
                        candidate_score,
                        k=min(residual_budget, n_candidates),
                        largest=True,
                    ).indices
                ]
                score_gate[topk_ids] = True

        grow_mask = base_grad_mask | score_gate

        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = grow_mask & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = grow_mask & is_large
        if step < self.refine_scale2d_stop_iter and state.get("radii") is not None:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        if self.max_new_gs > 0 and (n_dupli + n_split) > self.max_new_gs:
            grad_priority = (grads / max(grad_threshold, 1e-8)).clamp_min(0.0)
            residual_priority = (
                residual_ema / max(residual_threshold, 1e-8)
            ).clamp(0.0, 3.0) * coverage_score
            candidate_score = torch.where(
                base_grad_mask,
                grad_priority,
                torch.full_like(grad_priority, -1.0),
            )
            candidate_score = torch.where(score_gate, residual_priority, candidate_score)
            keep_k = min(self.max_new_gs, int(grow_mask.sum().item()))
            limited_mask = torch.zeros_like(grow_mask)
            if keep_k > 0:
                keep_ids = torch.topk(candidate_score, k=keep_k, largest=True).indices
                limited_mask[keep_ids] = True
            is_dupli = limited_mask & is_small
            is_split = limited_mask & is_large
            if step < self.refine_scale2d_stop_iter and state.get("radii") is not None:
                is_split |= (state["radii"] > self.grow_scale2d) & limited_mask
            n_dupli = is_dupli.sum().item()
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
        opacity_score = torch.sigmoid(params["opacities"].flatten())
        coverage_score = self._coverage_reliability(state["coverage_ema"])
        residual_score = (state["residual_ema"] / max(self.residual_threshold, 1e-8)).clamp(
            0.0, 1.0
        )
        hard_region_bonus = residual_score * coverage_score
        return (
            self.prune_opacity_weight * opacity_score
            + self.prune_coverage_weight * coverage_score
            + self.prune_residual_weight * hard_region_bonus
        )

    @staticmethod
    def _get_visible_projection_info(
        info: Dict[str, Any], packed: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if packed:
            gs_ids = info["gaussian_ids"]
            camera_ids = info["camera_ids"]
            coords = info["means2d"]
        else:
            visible_mask = (info["radii"] > 0.0).all(dim=-1)
            camera_ids, gs_ids = torch.where(visible_mask)
            coords = info["means2d"][visible_mask]
        return gs_ids.long(), camera_ids.long(), coords

    @staticmethod
    def _sample_residual_map(
        residual_map: torch.Tensor,
        camera_ids: torch.Tensor,
        coords: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        if coords.numel() == 0:
            return torch.empty(0, device=residual_map.device)

        camera_ids = camera_ids.clamp_(0, residual_map.shape[0] - 1)
        xs = coords[:, 0].float().clamp(0.0, max(width - 1, 0))
        ys = coords[:, 1].float().clamp(0.0, max(height - 1, 0))

        x0 = torch.floor(xs).long()
        y0 = torch.floor(ys).long()
        x1 = (x0 + 1).clamp(max=width - 1)
        y1 = (y0 + 1).clamp(max=height - 1)

        wx = xs - x0.float()
        wy = ys - y0.float()

        v00 = residual_map[camera_ids, y0, x0]
        v01 = residual_map[camera_ids, y0, x1]
        v10 = residual_map[camera_ids, y1, x0]
        v11 = residual_map[camera_ids, y1, x1]

        return (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v01
            + (1.0 - wx) * wy * v10
            + wx * wy * v11
        )

    def _coverage_floor(self, step: int) -> float:
        if self.coverage_warmup_iter <= 0:
            return self.coverage_min
        progress = min(max(step, 0) / float(self.coverage_warmup_iter), 1.0)
        return self.coverage_min * progress

    def _coverage_reliability(
        self,
        coverage_ema: torch.Tensor,
        step: int | None = None,
    ) -> torch.Tensor:
        if coverage_ema is None or coverage_ema.numel() == 0:
            return coverage_ema
        target = max(self.target_coverage, 1e-6)
        peak = max(self.coverage_peak, target)
        coverage = (coverage_ema / target).clamp(0.0, 1.0)
        if peak > target:
            decay = ((coverage_ema - target) / (peak - target)).clamp(0.0, 1.0)
            coverage = coverage * (1.0 - 0.25 * decay)
        if step is not None and self.coverage_warmup_iter > 0:
            progress = min(max(step, 0) / float(self.coverage_warmup_iter), 1.0)
            coverage = (1.0 - progress) + progress * coverage
        return coverage

    def _residual_budget_limit(self, n_base: int, step: int, pressure: float) -> int:
        if self.residual_budget_ratio <= 0.0:
            return self.min_residual_budget if step >= self.coverage_warmup_iter else 0

        warmup_progress = 1.0
        if self.residual_warmup_iter > 0:
            warmup_progress = min(max(step, 0) / float(self.residual_warmup_iter), 1.0)

        base_budget = max(int(n_base * self.residual_budget_ratio), self.min_residual_budget)
        base_budget = min(base_budget, self.max_residual_budget)
        base_budget = int(base_budget * warmup_progress * (1.0 - pressure))
        return max(base_budget, 0)

    def _growth_pressure(self, state: Dict[str, Any], n_current: int) -> float:
        init_n = state.get("init_n_gaussians")
        if init_n is None or init_n <= 0:
            return 0.0
        multiplier = float(n_current) / float(init_n)
        if self.growth_pressure_end <= self.growth_pressure_start:
            return float(multiplier >= self.growth_pressure_start)
        pressure = (multiplier - self.growth_pressure_start) / (
            self.growth_pressure_end - self.growth_pressure_start
        )
        return float(min(max(pressure, 0.0), 1.0))
