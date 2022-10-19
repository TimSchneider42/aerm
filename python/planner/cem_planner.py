import math
import time
from collections import deque
from typing import Tuple, Dict, Optional, List, Literal

import numpy as np
import torch
from torch.autograd import profiler

from util.normalizer import BaseNormalizer
from .knn_database import KNNDatabase
from .planner import Planner, CostFunction


class CEMPlanner(Planner):
    def __init__(self, cost_function: CostFunction, state_normalizer: BaseNormalizer, action_normalizer: BaseNormalizer,
                 state_dims: int, min_action: torch.Tensor, max_action: torch.Tensor, planning_horizon: int = 12,
                 max_optimization_iters: int = 10, candidates: int = 100, initial_candidate_factor: float = 1.0,
                 top_candidates_prop: float = 0.1, top_candidates_total: Optional[int] = None,
                 policy_cache_size: int = 10000, policy_proposal_method: Literal["none", "knn", "knn_ivf"] = "knn",
                 knn_neighbor_count: int = 50, return_mean: bool = True, max_planning_time: Optional[float] = None,
                 repeat_proposals: bool = True, proposal_random_cutoff: bool = False, proposal_min_std_dev: float = 0.0,
                 minimum_action_std_dev: float = 0.0, use_jit: bool = True):
        super().__init__()
        assert min_action.shape == max_action.shape
        self._policy_proposal_method = policy_proposal_method
        self._knn_neighbor_count = knn_neighbor_count
        self._ivf_nlist = 4096
        self._state_fields = [
            "policy_cache_states", "policy_cache_action_means",
            "policy_cache_action_std_devs", "policy_cache_index", "policy_cache_size"
        ]
        self._episode_plans = deque()

        self._planning_module = _CEMPlanningModule(
            cost_function, state_normalizer, action_normalizer, state_dims, min_action, max_action, planning_horizon,
            max_optimization_iters, candidates, initial_candidate_factor, top_candidates_prop, top_candidates_total,
            policy_cache_size, policy_proposal_method, return_mean,
            max_planning_time, repeat_proposals, proposal_random_cutoff, proposal_min_std_dev, minimum_action_std_dev)
        if use_jit:
            self._planning_module = torch.jit.script(self._planning_module)
        self._knn_db: Optional[KNNDatabase] = None

    def plan(self, collect_cost_function_info: bool = False) \
            -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        start_time = time.time()
        device = self._planning_module.device()
        with torch.no_grad():
            if self._policy_proposal_method.startswith("knn") and self._planning_module.policy_cache_size > 0:
                state = self._planning_module.cost_function.get_current_state_estimate().to("cpu")
                idx = self._knn_db.search(state[None], k=self._knn_neighbor_count)[0]
                neighbor_states = idx[idx != -1].to(device)
            else:
                neighbor_states = torch.empty(0)
        action_clamped, info, cost_function_info, action_mean_unnormalized, action_std_dev_unnormalized = \
            self._planning_module.forward(start_time, neighbor_states, collect_cost_function_info)

        if self._policy_proposal_method != "none" and not self.evaluation_mode:
            # Save current action sequence in the plan cache
            state = self._planning_module.cost_function.get_current_state_estimate()
            self._episode_plans.append((state, action_mean_unnormalized, action_std_dev_unnormalized))

        return action_clamped, info, cost_function_info

    def on_episode_start(self):
        self._planning_module.cost_function.on_episode_start()
        if self._policy_proposal_method.startswith("knn") and self._planning_module.policy_cache_size > 0:
            points = self._planning_module.policy_cache_states[:self._planning_module.policy_cache_size]
            self._knn_db = KNNDatabase(points, exact=self._policy_proposal_method == "knn", ivf_nlist=self._ivf_nlist)

    def on_episode_end(self):
        self._planning_module.cost_function.on_episode_end()
        # Grab the list of plans encountered this episode and add them to the replay buffer
        if len(self._episode_plans) > 0 and self._policy_proposal_method != "none":
            pm = self._planning_module
            states = torch.stack([s for s, a_mean, a_std in self._episode_plans])
            action_means = torch.stack([a_mean for s, a_mean, a_std in self._episode_plans])
            action_std_devs = torch.stack([a_std for s, a_mean, a_std in self._episode_plans])
            capacity = pm.policy_cache_states.shape[0]
            storage_locations = torch.arange(
                pm.policy_cache_index, pm.policy_cache_index + states.shape[0]) % capacity
            pm.policy_cache_states[storage_locations] = states.cpu()
            pm.policy_cache_action_means[storage_locations] = action_means
            pm.policy_cache_action_std_devs[storage_locations] = action_std_devs
            pm.policy_cache_index += states.shape[0]
            pm.policy_cache_size = torch.clamp_max(pm.policy_cache_size + states.shape[0], capacity)
        self._episode_plans.clear()

    def on_step_start(self):
        self._planning_module.cost_function.on_step_start()

    def on_step_end(self, collected_reward: float):
        self._planning_module.cost_function.on_step_end(collected_reward)

    def custom_state_dict(self):
        state_dict = {f: self._planning_module.__getattr__(f) for f in self._state_fields}
        state_dict["_cost_function"] = self._planning_module.cost_function.custom_state_dict()
        return state_dict

    def custom_load_state_dict(self, state_dict):
        self._planning_module.cost_function.custom_load_state_dict(state_dict["_cost_function"])
        for f in self._state_fields:
            if f in state_dict:
                self._planning_module.__setattr__(f, state_dict[f].to(self._planning_module.__getattr__(f).device))
            elif "_" + f in state_dict:  # Backward compatibility
                self._planning_module.__setattr__(
                    f, state_dict["_" + f].to(self._planning_module.__getattr__(f).device))

    def clear_policy_cache(self):
        self._planning_module.policy_cache_size *= 0
        self._planning_module.policy_cache_index *= 0

    @property
    def evaluation_mode(self):
        return self._planning_module.cost_function.evaluation_mode

    @evaluation_mode.setter
    def evaluation_mode(self, value: bool):
        self._planning_module.cost_function.evaluation_mode = value


# Unfortunately, this class has to be split up like that to make it JIT compilable
class _CEMPlanningModule(torch.nn.Module):
    __constants__ = ["_planning_horizon", "_max_optimization_iters", "_candidates", "_top_candidates",
                     "_initial_candidates", "_policy_proposal_method", "_max_planning_time", "_repeat_proposals",
                     "_proposal_random_cutoff", "_proposal_min_std_dev", "_minimum_action_std_dev"]

    def __init__(self, cost_function: CostFunction, state_normalizer: BaseNormalizer, action_normalizer: BaseNormalizer,
                 state_dims: int, min_action: torch.Tensor, max_action: torch.Tensor, planning_horizon: int = 12,
                 max_optimization_iters: int = 10, candidates: int = 100, initial_candidate_factor: float = 1.0,
                 top_candidates_prop: float = 0.1, top_candidates_total: Optional[int] = None,
                 policy_cache_size: int = 1000, policy_proposal_method: Optional[Literal["knn"]] = "knn",
                 return_mean: bool = True, max_planning_time: Optional[float] = None, repeat_proposals: bool = True,
                 proposal_random_cutoff: bool = False, proposal_min_std_dev: float = 0.0,
                 minimum_action_std_dev: float = 0.0):
        super().__init__()
        self.cost_function = cost_function
        device = min_action.device
        assert min_action.shape == max_action.shape
        self._planning_horizon = planning_horizon
        self._max_optimization_iters = max_optimization_iters
        self._candidates = candidates
        self._top_candidates = int(
            candidates * top_candidates_prop) if top_candidates_total is None else top_candidates_total
        self._act_center = (min_action + max_action) / 2
        self._act_range = max_action - min_action
        self._action_normalizer = action_normalizer
        self._state_normalizer = state_normalizer
        self._policy_proposal_method = policy_proposal_method
        self._proposal_count = candidates
        self._initial_candidates = int(candidates * initial_candidate_factor)
        self._max_planning_time = max_planning_time
        self._repeat_proposals = repeat_proposals
        self._proposal_random_cutoff = proposal_random_cutoff
        action_dims = min_action.shape[0]
        if self._policy_proposal_method == "none":
            policy_cache_size = 0
        # Has to be on the CPU, as the KNN algorithm works on CPU only currently
        self.register_buffer(
            "policy_cache_states", torch.empty((policy_cache_size, state_dims), device="cpu"))
        self.register_buffer(
            "policy_cache_action_means",
            torch.empty((policy_cache_size, planning_horizon, action_dims), device=device))
        self.register_buffer(
            "policy_cache_action_std_devs",
            torch.empty((policy_cache_size, planning_horizon, action_dims), device=device))
        self.register_buffer("policy_cache_index", torch.tensor(0, device=device))
        self.register_buffer("policy_cache_size", torch.tensor(0, device=device))
        self.policy_cache_index: torch.Tensor
        self.policy_cache_size: torch.Tensor
        self.policy_cache_states: torch.Tensor
        self.policy_cache_action_means: torch.Tensor
        self.policy_cache_action_std_devs: torch.Tensor
        self._return_mean = return_mean
        self._proposal_min_std_dev = proposal_min_std_dev
        self._minimum_action_std_dev = minimum_action_std_dev

    @staticmethod
    @torch.jit.ignore
    def time() -> float:
        return time.time()

    def forward(self, start_time: float, neighbor_states: torch.Tensor,
                collect_cost_function_info: bool = False) \
            -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]],
                     List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
        cost_function_info: List[Dict[str, torch.Tensor]] = []
        reward_contributions: Dict[str, torch.Tensor] = {}
        reward_components_topk_mean: Dict[str, torch.Tensor] = {}
        reward_components_topk_std: Dict[str, torch.Tensor] = {}
        reward_components_mean: Dict[str, torch.Tensor] = {}
        reward_components_std: Dict[str, torch.Tensor] = {}
        reward_components_topk_avg_placement: Dict[str, torch.Tensor] = {}
        device = self._act_center.device

        action_sample_lst = []
        best_action_std_dev_lst = []

        with torch.no_grad():
            # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
            action_size, = self._act_center.shape
            action_mean = torch.zeros(self._planning_horizon, 1, action_size, device=device)
            action_std_dev = torch.ones(self._planning_horizon, 1, action_size, device=device)

            proposal_success_fraction = torch.tensor(np.nan, device=device)
            average_successful_cutoff_index = torch.tensor(np.nan, device=device)

            ignored_time = 0.0
            first_itr_time = self.time()
            total_iterations = self._max_optimization_iters
            for i in range(self._max_optimization_iters):
                if self._max_planning_time is not None and i != 0:
                    time_spent = self.time() - start_time
                    remaining_time = self._max_planning_time - time_spent
                    average_iteration_time = (self.time() - first_itr_time - ignored_time) / i
                    if average_iteration_time > remaining_time - 0.005:  # Leave 5ms for the rest
                        total_iterations = i
                        break
                candidate_count = self._candidates if i > 0 else self._initial_candidates
                # Sample actions (time x candidates x actions)
                rand_act = torch.randn(
                    self._planning_horizon, candidate_count, action_size, device=device)
                actions_normalized = (action_mean + action_std_dev * rand_act)
                cutoff_indices: Optional[torch.Tensor] = None
                with profiler.record_function("PLANNER/POLICY_PROPOSALS"):
                    if i == 0 and self._policy_proposal_method != "none":
                        # Ignore time of the policy proposals for mean iteration time computation as they are only
                        # computed in the first iteration
                        ignored_time_start = self.time()
                        state = self.cost_function.get_current_state_estimate().to(device)
                        if self._policy_proposal_method.startswith("knn") and self.policy_cache_size > 0:
                            if self._repeat_proposals:
                                repeat_count = int(math.ceil(self._proposal_count / neighbor_states.shape[0]))
                                neighbor_states_repeated = neighbor_states.repeat(repeat_count)[:self._proposal_count]
                            else:
                                neighbor_states_repeated = neighbor_states
                            action_prop_means = self.policy_cache_action_means[neighbor_states_repeated]
                            action_prop_means_trans = action_prop_means.transpose(0, 1)
                            action_prop_std_devs = self.policy_cache_action_std_devs[neighbor_states_repeated]
                            action_prop_std_devs_clipped = torch.clip(
                                action_prop_std_devs, min=self._proposal_min_std_dev)
                            action_prop_std_devs_trans = action_prop_std_devs_clipped.transpose(0, 1)
                            if self._proposal_random_cutoff:
                                cutoff_indices = torch.randint(
                                    1, self._planning_horizon, (neighbor_states_repeated.shape[0],))
                                cutoff_mask = torch.arange(self._planning_horizon).unsqueeze(1) \
                                              >= cutoff_indices.unsqueeze(0)
                                action_prop_means_trans[cutoff_mask] = 0
                                action_prop_std_devs_trans[cutoff_mask] = 1

                            rand_act_prop = torch.randn(
                                self._planning_horizon, neighbor_states_repeated.shape[0], action_size,
                                device=state.device)
                            action_prop_samples = action_prop_means_trans + rand_act_prop * action_prop_std_devs_trans
                            action_prop_samples_norm = (action_prop_samples - self._act_center) / (self._act_range / 2)
                            actions_normalized = torch.cat([actions_normalized, action_prop_samples_norm], dim=1)
                        ignored_time += self.time() - ignored_time_start

                actions_normalized_clamped = actions_normalized.clamp(min=-1, max=1)
                actions_unnormalized = actions_normalized_clamped * self._act_range / 2 + self._act_center
                actions_mean_var_normalized = self._action_normalizer.normalize(actions_unnormalized)

                with profiler.record_function("PLANNER/CF_EVAL"):
                    returns, reward_components, cf_info = self.cost_function.forward(
                        actions_mean_var_normalized)

                if collect_cost_function_info:
                    cf_info["returns"] = returns
                    cost_function_info.append(cf_info)

                # Re-fit belief to the K best action sequences and record some metrics
                returns_sort_indices = torch.argsort(returns, descending=True)
                topk = returns_sort_indices[:self._top_candidates]
                best_actions = actions_normalized_clamped[:, topk]
                # Update belief with new means and standard deviations
                action_mean = best_actions.mean(dim=1, keepdim=True)
                action_std_dev = best_actions.std(dim=1, unbiased=False, keepdim=True)
                best_action_std_dev_lst.append(action_std_dev)

                with profiler.record_function("PLANNER/METRICS"):
                    if self._policy_proposal_method != "none" and i == 0:
                        topk_contributed_by_proposal_network = (topk >= self._candidates).sum()
                        proposal_success_fraction = topk_contributed_by_proposal_network / topk.shape[0]
                        if cutoff_indices is not None:
                            average_successful_cutoff_index = torch.mean(
                                cutoff_indices[topk - self._candidates].float())

                    action_sample_lst.append(actions_unnormalized)

                    for r, v in reward_components.items():
                        _, component_topk_positions = torch.topk(
                            v[returns_sort_indices], self._top_candidates, largest=True, sorted=False)
                        if i == 0:
                            reward_contributions[r] = torch.zeros(total_iterations)
                            reward_components_topk_mean[r] = torch.zeros(total_iterations)
                            reward_components_topk_std[r] = torch.zeros(total_iterations)
                            reward_components_mean[r] = torch.zeros(total_iterations)
                            reward_components_std[r] = torch.zeros(total_iterations)
                            reward_components_topk_avg_placement[r] = torch.zeros(total_iterations)
                        mean_comp_topk = v[topk].mean()
                        mean_comp = v.mean()
                        reward_contributions[r][i] = mean_comp_topk - mean_comp
                        reward_components_topk_mean[r][i] = mean_comp_topk
                        reward_components_topk_std[r][i] = v[topk].std()
                        reward_components_mean[r][i] = mean_comp
                        reward_components_std[r][i] = v.std()
                        reward_components_topk_avg_placement[r][i] = \
                            torch.mean(component_topk_positions.float()) / (v.shape[0] - 1)

            info = {
                "reward_components_mean": reward_components_mean,
                "reward_components_std": reward_components_std,
                "reward_contributions": reward_contributions,
                "reward_components_topk_mean": reward_components_topk_mean,
                "reward_components_topk_std": reward_components_topk_std,
                "reward_components_topk_avg_placement_rel": reward_components_topk_avg_placement,
                "general": {
                    "action_samples": torch.cat(action_sample_lst, dim=1),
                    "action_samples_iterations": torch.repeat_interleave(
                        torch.arange(total_iterations),
                        torch.tensor([a.shape[1] for a in action_sample_lst])).to(device),
                    "total_iterations": torch.tensor(total_iterations, device=device),
                    "best_action_std_dev": torch.stack(best_action_std_dev_lst)
                }
            }
            if self._policy_proposal_method != "none":
                info["general"]["proposal_success_fraction"] = proposal_success_fraction
                info["general"]["average_successful_cutoff_index"] = average_successful_cutoff_index

            action_mean_unnormalized = (action_mean[:, 0] + self._act_center[None, :]) * self._act_range[None, :] / 2
            action_std_dev_unnormalized = action_std_dev[:, 0] * self._act_range[None, :] / 2

            # Return action means
            action = action_mean_unnormalized
            if not self._return_mean:
                action += torch.randn_like(action_mean_unnormalized) * torch.clip(
                    action_std_dev_unnormalized, min=self._minimum_action_std_dev)
            action_clamped = action.clamp(min=-1, max=1)
            info["general"]["time_needed"] = torch.tensor(self.time() - start_time, device=device)
            return action_clamped, info, cost_function_info, action_mean_unnormalized, action_std_dev_unnormalized

    @torch.jit.ignore
    def device(self):
        return self._act_center.device
