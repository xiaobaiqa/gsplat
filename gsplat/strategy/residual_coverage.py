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
    residual_quantile: float = 0.8
    coverage_gate_min_score: float = 0.15
    coverage_score_power: float = 0.75
    prune_warmup_iter: int = 3500
    prune_opa_warmup_scale: float = 0.6
    replace_start_iter: int = 4000
    replace_budget_ratio: float = 0.08
    min_replace_budget: int = 64
    max_replace_budget: int = 4096
    stage_transition_iter: int = 4000
    stage_end_iter: int = 8000
    gate_score_early: float = 0.08
    gate_score_late: float = 0.08
    coverage_min_early: float = 0.02
    coverage_min_late: float = 0.03
    adaptive_growth_budget: bool = False
    growth_budget_min_scale: float = 1.0
    growth_budget_max_scale: float = 1.0
    replace_guard_residual_scale: float = 0.85
    replace_guard_steps: int = 5
    use_default_refine: bool = False
    growth_spike_guard: bool = True
    growth_spike_ratio_limit: float = 1.5
    growth_spike_warmup_iter: int = 2500
    contribution_ema_decay: float = 0.96
    prune_contribution_weight: float = 0.35
    residual_threshold_early_scale: float = 0.92
    residual_threshold_late_scale: float = 1.10
    prune_opacity_weight: float = 0.5
    prune_coverage_weight: float = 0.3
    prune_residual_weight: float = 0.2
    residual_key: str = "residual_map"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = super().initialize_state(scene_scale=scene_scale)
        state["residual_ema"] = None
        state["coverage_ema"] = None
        state["init_n_gaussians"] = None
        state["last_growth_count"] = 0
        state["global_residual_ema"] = 0.0
        state["low_residual_streak"] = 0
        state["contribution_ema"] = None
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
        if state["contribution_ema"] is None:
            state["contribution_ema"] = torch.zeros(n_gaussian, device=device)
        if state["init_n_gaussians"] is None:
            state["init_n_gaussians"] = n_gaussian

        gs_ids, camera_ids, coords = self._get_visible_projection_info(info, packed=packed)
        if gs_ids.numel() == 0:
            state["coverage_ema"].mul_(self.coverage_ema_decay)
            return

        state["coverage_ema"].mul_(self.coverage_ema_decay)
        state["contribution_ema"].mul_(self.contribution_ema_decay)
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
        if visible_ids.numel() == 0:
            return
        mean_residual = residual_sum[visible_ids] / residual_count[visible_ids].clamp_min(1)
        state["residual_ema"][visible_ids] = (
            state["residual_ema"][visible_ids] * self.residual_ema_decay
            + mean_residual * (1.0 - self.residual_ema_decay)
        )

        # Lightweight contribution proxy: stable visibility + hard-region residual.
        local_obs = residual_count[visible_ids]
        local_obs = local_obs / local_obs.max().clamp_min(1.0)
        local_residual = (mean_residual / max(self.residual_threshold, 1e-8)).clamp(0.0, 2.0)
        local_contribution = 0.55 * local_obs + 0.45 * local_residual
        state["contribution_ema"][visible_ids] = (
            state["contribution_ema"][visible_ids] * self.contribution_ema_decay
            + local_contribution * (1.0 - self.contribution_ema_decay)
        )

        global_mean = float(mean_residual.mean().item())
        prev_global = float(state.get("global_residual_ema", global_mean))
        global_ema = prev_global * self.residual_ema_decay + global_mean * (1.0 - self.residual_ema_decay)
        state["global_residual_ema"] = global_ema
        if global_ema < (self.residual_threshold * self.replace_guard_residual_scale):
            state["low_residual_streak"] = int(state.get("low_residual_streak", 0)) + 1
        else:
            state["low_residual_streak"] = 0

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        if self.use_default_refine:
            n_dupli, n_split = super()._grow_gs(
                params=params,
                optimizers=optimizers,
                state=state,
                step=step,
            )
            state["last_growth_count"] = int(n_dupli + n_split)
            return n_dupli, n_split

        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        residual_ema = state["residual_ema"]
        coverage_ema = state["coverage_ema"]
        coverage_score = self._coverage_reliability(coverage_ema, step=step)
        if self.coverage_score_power != 1.0:
            coverage_score = coverage_score.clamp(0.0, 1.0).pow(self.coverage_score_power)
        pressure = self._growth_pressure(state, len(params["means"]))
        grad_threshold = self.grow_grad2d * (1.0 + self.grad_threshold_gain * pressure)
        residual_threshold_base = min(
            self.residual_threshold + self.residual_threshold_gain * pressure,
            0.95,
        )
        residual_threshold_base *= self._scheduled_residual_threshold_scale(step)
        residual_threshold = self._adaptive_residual_threshold(
            residual_ema=residual_ema,
            coverage_score=coverage_score,
            base_threshold=residual_threshold_base,
        )

        device = grads.device
        coverage_floor = self._scheduled_coverage_min(step)
        gate_min_score = self._scheduled_gate_score(step)
        base_grad_mask = grads > grad_threshold
        if step >= self.coverage_warmup_iter:
            base_grad_mask &= coverage_ema > max(self.base_coverage_min, coverage_floor)
        n_base = int(base_grad_mask.sum().item())

        relaxed_grad_mask = grads > (grad_threshold * self.relaxed_grad_factor)
        coverage_gate = (coverage_score > gate_min_score) & (coverage_ema > coverage_floor)
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

        effective_max_new_gs = self._effective_growth_budget(
            max_new_gs=self.max_new_gs,
            residual_ema=residual_ema,
            coverage_score=coverage_score,
            residual_threshold=residual_threshold,
            gate_min_score=gate_min_score,
            step=step,
        )
        effective_max_new_gs = self._apply_growth_spike_guard(
            state=state,
            effective_max_new_gs=effective_max_new_gs,
            step=step,
        )

        if effective_max_new_gs > 0 and (n_dupli + n_split) > effective_max_new_gs:
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
            keep_k = min(effective_max_new_gs, int(grow_mask.sum().item()))
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
        state["last_growth_count"] = int(n_dupli + n_split)
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        if self.use_default_refine:
            return super()._prune_gs(
                params=params,
                optimizers=optimizers,
                state=state,
                step=step,
            )

        effective_prune_opa = self._effective_prune_opa(step)
        is_prune = torch.sigmoid(params["opacities"].flatten()) < effective_prune_opa
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

        # Reallocate budget: remove lowest-value GSs after warmup so new GSs can
        # better target hard regions without increasing overall model size.
        replace_budget = self._replace_budget_limit(state=state, step=step)
        if replace_budget > 0:
            keep_score = self._compute_keep_score(params, state)
            candidate_ids = torch.where(~is_prune)[0]
            if candidate_ids.numel() > 0:
                k = min(replace_budget, int(candidate_ids.numel()))
                if k > 0:
                    replace_ids = candidate_ids[
                        torch.topk(keep_score[candidate_ids], k=k, largest=False).indices
                    ]
                    is_prune[replace_ids] = True

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
        contribution_score = state.get("contribution_ema")
        if contribution_score is None:
            contribution_score = torch.zeros_like(opacity_score)
        contribution_score = contribution_score.clamp(0.0, 1.5)
        hard_region_bonus = residual_score * coverage_score
        total_w = (
            self.prune_opacity_weight
            + self.prune_coverage_weight
            + self.prune_residual_weight
            + self.prune_contribution_weight
        )
        total_w = max(total_w, 1e-6)
        return (
            self.prune_opacity_weight * opacity_score
            + self.prune_coverage_weight * coverage_score
            + self.prune_residual_weight * hard_region_bonus
            + self.prune_contribution_weight * contribution_score
        ) / total_w

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

    def _adaptive_residual_threshold(
        self,
        residual_ema: torch.Tensor,
        coverage_score: torch.Tensor,
        base_threshold: float,
    ) -> float:
        if residual_ema is None or residual_ema.numel() == 0:
            return base_threshold
        if not (0.0 < self.residual_quantile < 1.0):
            return base_threshold

        valid = coverage_score > self.coverage_gate_min_score
        pool = residual_ema[valid]
        if pool.numel() < 64:
            pool = residual_ema
        if pool.numel() == 0:
            return base_threshold

        quantile_threshold = torch.quantile(pool, self.residual_quantile).item()
        return float(max(base_threshold, min(quantile_threshold, 0.98)))

    def _effective_prune_opa(self, step: int) -> float:
        if self.prune_warmup_iter <= 0:
            return self.prune_opa
        progress = min(max(step, 0) / float(self.prune_warmup_iter), 1.0)
        scale = self.prune_opa_warmup_scale + (1.0 - self.prune_opa_warmup_scale) * progress
        return self.prune_opa * max(scale, 1e-3)

    def _replace_budget_limit(self, state: Dict[str, Any], step: int) -> int:
        if step < self.replace_start_iter or self.replace_budget_ratio <= 0.0:
            return 0
        if int(state.get("low_residual_streak", 0)) < self.replace_guard_steps:
            return 0
        last_growth = int(state.get("last_growth_count", 0))
        if last_growth <= 0:
            return 0
        budget = int(last_growth * self.replace_budget_ratio)
        budget = max(budget, self.min_replace_budget)
        budget = min(budget, self.max_replace_budget)
        return max(budget, 0)

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

    def _scheduled_gate_score(self, step: int) -> float:
        return self._linear_schedule(
            step=step,
            start_val=self.gate_score_early,
            end_val=self.gate_score_late,
            start_step=self.stage_transition_iter,
            end_step=self.stage_end_iter,
        )

    def _scheduled_coverage_min(self, step: int) -> float:
        return self._linear_schedule(
            step=step,
            start_val=self.coverage_min_early,
            end_val=self.coverage_min_late,
            start_step=self.stage_transition_iter,
            end_step=self.stage_end_iter,
        )

    def _effective_growth_budget(
        self,
        max_new_gs: int,
        residual_ema: torch.Tensor,
        coverage_score: torch.Tensor,
        residual_threshold: float,
        gate_min_score: float,
        step: int,
    ) -> int:
        if max_new_gs <= 0 or not self.adaptive_growth_budget:
            return max_new_gs

        valid = coverage_score > gate_min_score
        if valid.numel() == 0 or int(valid.sum().item()) == 0:
            return max_new_gs

        hard_ratio = float((residual_ema[valid] > residual_threshold).float().mean().item())
        demand_scale = self.growth_budget_min_scale + (
            self.growth_budget_max_scale - self.growth_budget_min_scale
        ) * hard_ratio
        if step < self.stage_transition_iter:
            demand_scale = max(demand_scale, 0.95)

        scaled_budget = int(max_new_gs * demand_scale)
        return max(scaled_budget, self.min_residual_budget)

    def _apply_growth_spike_guard(
        self,
        state: Dict[str, Any],
        effective_max_new_gs: int,
        step: int,
    ) -> int:
        if effective_max_new_gs <= 0 or not self.growth_spike_guard:
            return effective_max_new_gs
        if step < self.growth_spike_warmup_iter:
            return effective_max_new_gs

        prev_growth = int(state.get("last_growth_count", 0))
        if prev_growth <= 0:
            return effective_max_new_gs

        spike_cap = int(prev_growth * self.growth_spike_ratio_limit)
        spike_cap = max(spike_cap, self.min_residual_budget * 4)
        return min(effective_max_new_gs, spike_cap)

    def _scheduled_residual_threshold_scale(self, step: int) -> float:
        return self._linear_schedule(
            step=step,
            start_val=self.residual_threshold_early_scale,
            end_val=self.residual_threshold_late_scale,
            start_step=self.stage_transition_iter,
            end_step=self.stage_end_iter,
        )

    @staticmethod
    def _linear_schedule(
        step: int,
        start_val: float,
        end_val: float,
        start_step: int,
        end_step: int,
    ) -> float:
        if end_step <= start_step:
            return end_val
        if step <= start_step:
            return start_val
        if step >= end_step:
            return end_val
        ratio = (step - start_step) / float(end_step - start_step)
        return start_val + (end_val - start_val) * ratio
