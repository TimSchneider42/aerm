import argparse
import json
import re
import shutil
import subprocess
import traceback

import time
from datetime import datetime, timedelta
from logging import FileHandler, Formatter
from pathlib import Path
import signal
from typing import Optional, Dict, Any, Tuple, Union

import torch

import robot_gym
from logger import logger
from util.weight_schedule import parse_weight_schedule

from torch.utils.tensorboard import SummaryWriter

from env import CallbackWrapperEnv, WarmStartWrapperEnv, get_env_factory, WarmUpEnv
from models import create_ensemble_model_mdp, EnsembleModel
from filter import MDPFilter, BaseFilter

import numpy as np

from planner import CEMPlanner, SGDPlanner, AMBXCostFunction, Planner, PlannerOptimizer, StaticPlanner
from trainer import BaseTrainer, MDPTrainer

from sisyphus_env.base_env import BaseEnv


def load_state(run_directory: Path, model_ensemble: EnsembleModel, planner: Planner,
               planner_optimizer: PlannerOptimizer, trainer: BaseTrainer, iteration: Optional[int] = None,
               load_transition_model: bool = True, load_reward_model: bool = True, load_planner: bool = True,
               map_location: Union[torch.device, str]=None) -> int:
    state_dir = run_directory / "state"
    if iteration is None:
        state_path = state_dir / "state.pkl"
    else:
        all_states = [s for s in state_dir.iterdir() if re.match(r"state_[0-9]+\.pkl", s.name)]
        all_iterations = [int(s.stem.split("_")[1]) for s in all_states]
        episodes_below = [(it, s) for it, s in zip(all_iterations, all_states) if it <= iteration]
        if len(episodes_below) == 0:
            return 0
        state_path = max(episodes_below, key=lambda x: x[0])[1]
    state = torch.load(state_path, map_location=map_location)
    model_ensemble.load_state_dict(state["model"], load_transition_model=load_transition_model,
                                   load_reward_model=load_reward_model)
    if planner is not None and load_planner:
        planner.custom_load_state_dict(state["planner"])
        planner_optimizer.load_state_dict(state["planner_optimizer"])
    trainer.load_state_dict(state["trainer"])
    try:
        trainer.load_replay_buffer(run_directory / "replay_buffer.pkl")
    except:
        logger.warning("Failed to load replay buffer!")
    return state["episode"]


def save_replay_buffer(run_path: Path, trainer: BaseTrainer):
    logger.info("Saving replay buffer...")
    trainer.save_replay_buffer(run_path / "replay_buffer.pkl")
    logger.info("Replay buffer saved.")


def save_state(run_path: Path, model_ensemble: EnsembleModel, planner: Planner, planner_optimizer: PlannerOptimizer,
               trainer: BaseTrainer, episode: int):
    logger.info("Saving state...")
    state_path = run_path / "state"
    state_path.mkdir(exist_ok=True)
    state_output_path = state_path / "state_{:06d}.pkl".format(episode)
    state_link_path = state_path / "state.pkl"
    state = {
        "model": model_ensemble.state_dict(),
        "planner": planner.custom_state_dict(),
        "trainer": trainer.state_dict(),
        "planner_optimizer": planner_optimizer.state_dict(),
        "episode": episode
    }
    torch.save(state, state_output_path)
    state_link_path.unlink(missing_ok=True)
    state_link_path.symlink_to(state_output_path.name)
    logger.info("State saved.")


def configure(args, dev, result_path: Optional[Path] = None, warm_up_trainer: bool = False) \
        -> Tuple[BaseEnv, EnsembleModel, BaseFilter, BaseTrainer, Planner, PlannerOptimizer, AMBXCostFunction, int]:
    batch_size = 32
    ensemble_size = args.ensemble_size
    replay_buffer_max_size = args.total_episodes

    if result_path is not None:
        env_log_dir = result_path / "env_log"
    else:
        env_log_dir = None

    env = get_env_factory(args.env)(headless=args.headless, log_dir=env_log_dir)

    if args.special_model is not None:
        model_ensemble = env.make_special_model(args.ensemble_size, args.special_model, device=dev)
    else:
        create_std_dev_heads = args.start_learning_std_devs is not None
        model_ensemble = create_ensemble_model_mdp(
            symbolic=True, ensemble_size=ensemble_size, state_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0], transition_hidden_size=args.ths, reward_hidden_size=args.rhs,
            device=dev, activation_function="leaky_relu", transition_min_std_dev=args.min_std_transition,
            transition_max_std_dev=args.max_std_transition, reward_min_std_dev=args.min_std_reward,
            reward_max_std_dev=args.max_std_reward, obs_lower_bound=env.observation_space.low,
            obs_upper_bound=env.observation_space.high, reward_hidden_layers=args.rhl,
            transition_hidden_layers=args.thl, reward_constant_std_dev=args.crs,
            transition_constant_std_dev=args.cts, no_jit=args.no_jit, create_std_dev_heads=create_std_dev_heads)
    model_ensemble.disable_learned_std_devs()

    filter = MDPFilter(model_ensemble, device=dev)

    if not args.no_jit:
        filter = torch.jit.script(filter)

    # JIT does not work here
    if args.multi_step_prediction_length is not None:
        mspl = args.multi_step_prediction_length
    else:
        mspl = args.horizon
    storage_device = args.rp_storage_device
    if storage_device == "cuda" and not torch.cuda.is_available():
        storage_device = "cpu"
    trainer = MDPTrainer(
        filter, model_ensemble, args.time_steps, batch_size, use_stein_gradient=args.use_stein, training_device=dev,
        storage_device=storage_device, replay_buffer_max_size=replay_buffer_max_size,
        use_jit=not args.no_jit, multi_step_prediction_length=mspl, individual_episodes=args.individual_episodes,
        replan_interval_steps=args.replan_interval, use_multi_step_prediction_reward=True,
        fixed_transition_model=args.no_train_trans, fixed_reward_model=args.no_train_rew,
        train_input_state_noise=args.training_noise_ps)

    if args.planner != "random":
        assert len(env.observation_space.shape) == 1
        extrinsic_schedule = parse_weight_schedule(args.reward_scale)
        if args.intrinsic_clamp_bound is not None:
            # Add one for the reward which is one dimensional
            intrinsic_clamp_bound_per_step = args.intrinsic_clamp_bound * (env.observation_space.shape[0] + 1)
            intrinsic_clamp_bound = intrinsic_clamp_bound_per_step * args.horizon
            logger.info("Setting intrinsic clamp bound to {:0.2f}.".format(intrinsic_clamp_bound))
            intrinsic_clamp_bounds = (-intrinsic_clamp_bound, intrinsic_clamp_bound)
        else:
            intrinsic_clamp_bounds = None
        cost_function = AMBXCostFunction(
            filter, model_ensemble.reward_normalizer, imagined_observations_per_action=1,
            information_method=args.information_method, intrinsic_clamp_bounds=intrinsic_clamp_bounds,
            extrinsic_weight_schedule=extrinsic_schedule, use_intrinsic_reward=not args.no_intrinsic,
            use_extrinsic_reward=not args.no_reward, adaptive_weight_strategy=args.adaptive_weight_strategy, device=dev,
            min_ll_per_step_and_dim=args.mll, ignore_model_weights=True,
            reward_moving_average_gain=args.adaptive_weight_reward_gain,
            adaptive_weight_reward_scale=args.adaptive_weight_reward_scale,
            mi_exclude_outer_sample_from_inner=not args.no_mi_exclude_outer_sample,
            ig_include=args.ig_include)
        if not args.no_jit:
            cost_function = torch.jit.script(cost_function)
        min_action = torch.from_numpy(env.action_space.low).to(dev)
        max_action = torch.from_numpy(env.action_space.high).to(dev)
        if args.planner == "sgd":
            planner = SGDPlanner(
                cost_function, filter.parameters(), model_ensemble.action_normalizer, min_action, max_action,
                planning_horizon=args.horizon, optimization_iters=10)
        elif args.planner == "cem":
            planner = CEMPlanner(
                cost_function, model_ensemble.observation_normalizer, model_ensemble.action_normalizer,
                model_ensemble.observation_size, min_action, max_action, planning_horizon=args.horizon,
                max_optimization_iters=args.cem_iters, candidates=args.cem_candidates,
                top_candidates_total=args.cem_top_candidates, return_mean=not args.gaussian_action_noise,
                policy_cache_size=args.policy_cache_size, max_planning_time=args.cem_max_planning_time,
                proposal_random_cutoff=args.proposal_random_cutoff, proposal_min_std_dev=args.proposal_min_std_dev,
                minimum_action_std_dev=args.cem_min_action_std_dev, use_jit=not args.no_jit,
                policy_proposal_method=args.policy_proposal_method, knn_neighbor_count=args.proposal_knn_neighbor_count)
        else:
            raise ValueError("Unknown planner")
    else:
        planner = cost_function = None

    if planner is not None:
        planner_optimizer = planner.make_planner_optimizer()
    else:
        planner_optimizer = None

    if warm_up_trainer:
        logger.info("Warming up trainer...")
        warm_up_env = WarmUpEnv(env)
        trainer.collect_rollouts(warm_up_env, planner)
        trainer.collect_rollouts(warm_up_env, planner)
        trainer.perform_evaluation_rollout(warm_up_env, planner)
        planner.clear_policy_cache()
        trainer.clear_replay_buffer()
        logger.info("Trainer warmed up.")

    if args.cont is not None:
        cont_split = args.cont.split(":")
        if len(cont_split) == 1:
            iteration = None
            load_trans = load_rew = True
        elif 2 <= len(cont_split) <= 3:
            iteration = int(cont_split[1]) if len(cont_split[1]) > 0 else None
            model_load_mode = "both" if len(cont_split) == 2 else cont_split[2]
            assert model_load_mode in ["both", "rew", "trans"]
            load_trans = model_load_mode in ["both", "trans"]
            load_rew = model_load_mode in ["both", "rew"]
        else:
            raise ValueError("Too many colons in cont.")
        episode = load_state(Path(cont_split[0]), model_ensemble, planner, planner_optimizer, trainer,
                             iteration=iteration, load_transition_model=load_trans, load_reward_model=load_rew,
                             load_planner=not args.no_load_planner, map_location=dev)
    else:
        episode = 0
    return env, model_ensemble, filter, trainer, planner, planner_optimizer, cost_function, episode


def determine_git_commit():
    try:
        cwd = Path(__file__).resolve().parent
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
    except:
        logger.warning("Could not determine git commit")
        return None


def add_visited_state(state_repr: np.ndarray):
    if state_repr.shape[0] > 0:
        if state_repr.shape[0] == 1:
            state_repr = np.array([state_repr[0], 0])
        bin = np.minimum(
            ((state_repr - state_repr_lower) / bin_size).astype(np.int_),
            np.array(test_state_coverage_hist.shape) - 1)
        if current_sample_is_evaluation_sample:
            if 0 <= bin[0] < test_state_coverage_hist.shape[0] and 0 <= bin[1] < test_state_coverage_hist.shape[1]:
                test_state_coverage_hist[tuple(bin)] += 1
        else:
            if 0 <= bin[0] < test_state_coverage_hist.shape[0] and 0 <= bin[1] < test_state_coverage_hist.shape[1]:
                train_state_coverage_hist[tuple(bin)] += 1


def on_post_step_callback(obs: np.ndarray, reward: float, done: bool, info: Dict[str, Any], action: np.ndarray,
                          planner_info: Dict[str, Any]):
    global step
    step += 1
    if "state_repr" in info:
        add_visited_state(info["state_repr"])


def on_post_reset_callback(obs: np.ndarray):
    global step, current_video_file, rollout_no
    step = 0
    rollout_no += 1
    record_train = args.video_log_interval_train > 0 and episode % args.video_log_interval_train == 0
    record_eval = args.video_log_interval_eval > 0 and (episode // eval_interval) % args.video_log_interval_eval == 0
    if record_train and not current_sample_is_evaluation_sample or record_eval and current_sample_is_evaluation_sample:
        episode_type = "eval" if current_sample_is_evaluation_sample else "train"
        current_video_file = record_path / "ep_{:05d}_{}_{}.mp4".format(episode, episode_type, rollout_no)
        env.start_video_log(current_video_file)


def on_episode_done_callback():
    global current_video_file
    if current_video_file is not None:
        env.stop_video_log()
        current_video_file = None


def on_episode_invalid_callback():
    global current_video_file
    if current_video_file is not None:
        env.stop_video_log()
        target_filename = None
        i = 0
        while target_filename is None or target_filename.exists():
            target_filename = current_video_file.parent / "{}_invalid_{:02d}{}".format(
                current_video_file.stem, i, current_video_file.suffix)
            i += 1
        shutil.copy(current_video_file, target_filename)
        current_video_file = None


def _terminate_signal_handler(*args, **kwargs):
    global terminate
    terminate = True


def parse_time(time_str: str) -> timedelta:
    split = list(map(int, time_str.split(":")))
    assert len(split) in [2, 3]

    if len(split) == 2:
        return timedelta(hours=split[0], minutes=split[1])
    else:
        return timedelta(hours=split[0], minutes=split[1], seconds=split[2])


parser = argparse.ArgumentParser()
parser.add_argument("--no-jit", action="store_true", help="Disable JIT.")
parser.add_argument("--no-cuda", action="store_true", help="Do not use CUDA.")
parser.add_argument("--headless", action="store_true", help="Run this program in headless mode.")
parser.add_argument("--evaluation-interval", type=int, default=10, help="Evaluate every N episodes.")
parser.add_argument("-ve", "--video-log-interval-eval", type=int, default=0,
                    help="Log video of every Nth evaluation episode.")
parser.add_argument("-vt", "--video-log-interval-train", type=int, default=0,
                    help="Log video of every Nth training episode.")
parser.add_argument("-nw", "--no-warm-up", action="store_true",
                    help="Do not warm up the trainer before starting the training.")
parser.add_argument("--pause-starts", type=str, help="Time when the pause starts.")
parser.add_argument("--pause-lengths", type=str, default="1:00,1:00,1:00,1:00,1:00,1:00,1:00",
                    help="Lengths of the pauses.")
parser.add_argument("--rp-storage-device", type=str, default="cuda", help="Device on which to store the replay buffer.")
subparsers = parser.add_subparsers(dest="command", required=True)
parser_new = subparsers.add_parser("new")
parser_new.add_argument("--episodes", type=int, help="Number of episodes to run in this instance (default: all)")
parser_new.add_argument("-e", "--env", type=str, default="car", help="Environment to train on.")
parser_new.add_argument("-c", "--continue", dest="cont", type=str, help="Continue training the model at this path.")
parser_new.add_argument("-t", "--test-mode", action="store_true", help="Test only, do not train.")
parser_new.add_argument("-T", "--time-steps", type=int, default=30,
                        help="Number of time steps to run this environment for.")
parser_new.add_argument("-p", "--planner", choices=["sgd", "cem", "random"], default="cem",
                        help="Select the type of planner.")
parser_new.add_argument("-H", "--horizon", type=int, default=20, help="Planning horizon.")
parser_new.add_argument("-E", "--ensemble-size", type=int, default=5, help="Number of models in the ensemble.")
parser_new.add_argument("--total-episodes", type=int, default=10000, help="Total number of episodes to run.")
parser_new.add_argument("--use-stein", action="store_true", help="Use Stein Variational Gradient Descent")
parser_new.add_argument("-i", "--information-method", type=str, default="mutual_information",
                        choices=["mutual_information", "lautum_information"])
parser_new.add_argument("--individual-episodes", action="store_true",
                        help="Collect different episodes for each ensemble model.")
parser_new.add_argument("--no-reward", action="store_true", help="Train without using the reward.")
parser_new.add_argument("--no-intrinsic", action="store_true", help="Train without using intrinsic motivation.")
parser_new.add_argument("-a", "--adaptive-weight-strategy", choices=["none", "max", "avg"], default="none",
                        help="Strategy for determining the extrinsic weight adaptively.")
parser_new.add_argument("-awrs", "--adaptive-weight-reward-scale", type=float, default=1.0,
                        help="Scale the reward by this factor before adding it to the intrinsic weight.")
parser_new.add_argument("-awrg", "--adaptive-weight-reward-gain", type=float, default=1e-3,
                        help="Gain to use for the exponential moving average of the reward.")
parser_new.add_argument("-b", "--intrinsic-clamp-bound", type=float,
                        help="Value at which to clamp the intrinsic value per step and observation dimension.",
                        default=None)
parser_new.add_argument("-S", "--special-model", type=str,
                        help="Use the special model of the environment.")
parser_new.add_argument("--cem-candidates", type=int, default=500,
                        help="Number of candidates to compute in each CEM iteration.")
parser_new.add_argument("--cem-top-candidates", type=int, default=20,
                        help="Number of candidates that get selected in each CEM iteration.")
parser_new.add_argument("--cem-iters", type=int, default=12, help="Number of CEM iterations.")
parser_new.add_argument("--cem-max-planning-time", type=float, help="Maximum time to spend planning per step (in s).")
parser_new.add_argument("--cem-min-action-std-dev", type=float, default=0.0,
                        help="Minimum variance of the Gaussian noise added to the action.")
parser_new.add_argument("-ths", type=int, default=64, help="Number of hidden units for the transition model.")
parser_new.add_argument("-rhs", type=int, default=32, help="Number of hidden units for the reward model.")
parser_new.add_argument("-thl", type=int, default=2, help="Number of hidden layers for the transition model.")
parser_new.add_argument("-rhl", type=int, default=1, help="Number of hidden layers for the reward model.")
parser_new.add_argument("-mll", type=float, default=-1000.0,
                        help="Minimum log likelihood per step and dimension to use for information gain "
                             "computation. Values below this value will be clipped.")
parser_new.add_argument("-crs", type=float, default=0.001, help="Use this constant for the reward standard deviation.")
parser_new.add_argument("-cts", type=float, default=0.001,
                        help="Use this constant for the transition standard deviation.")
parser_new.add_argument("-s", "--start-learning-std-devs", type=int,
                        help="Start learning standard deviations at this episode.")
parser_new.add_argument("--std-dev-initial-training-steps", type=int, default=20000,
                        help="Number of steps to train when starting to learn standard deviations.")
parser_new.add_argument("-r", "--replan-interval", type=int, default=1, help="Replan every n steps (default 1).")
parser_new.add_argument("-m", "--multi-step-prediction-length", type=int,
                        help="Length of the multi-step predictions used in the cost function.")
parser_new.add_argument("--max-std-reward", type=float, default=None,
                        help="Maximum standard deviation for the reward model.")
parser_new.add_argument("--max-std-transition", type=float, default=None,
                        help="Maximum standard deviation for the transition model.")
parser_new.add_argument("--min-std-reward", type=float, default=0.001,
                        help="Minimum standard deviation for the reward model.")
parser_new.add_argument("--min-std-transition", type=float, default=0.001,
                        help="Minimum standard deviation for the transition model.")
parser_new.add_argument("-Pc", "--proposal-random-cutoff", action="store_true",
                        help="Cut proposals off at random points to encourage exploration during planning.")
parser_new.add_argument("-Pk", "--proposal-knn-neighbor-count", type=int, default=50,
                        help="Number of neighbors to draw for policy proposals.")
parser_new.add_argument("-Ps", "--proposal-min-std-dev", type=float, default=0.0,
                        help="Minimum standard deviation of policy proposals.")
parser_new.add_argument("-P", "--policy-proposal-method", choices=["none", "knn", "knn_ivf"], default="knn",
                        const="knn", nargs="?", help="Method to use for policy proposal generation in the CEM planner.")
parser_new.add_argument("-o", "--output-path", type=str, help="Path to store the results in.")
parser_new.add_argument("-rs", "--reward-scale", type=str, default="1e6",
                        help="Scale of the extrinsic reward. In case of adaptive weights, this is the initial "
                             "weight.")
parser_new.add_argument("-gan", "--gaussian-action-noise", action="store_true",
                        help="Sample actions from the Gaussian distribution generated by the CEM planner instead of"
                             " taking the mean.")
parser_new.add_argument("--no-link-latest", action="store_true", help="Do not create a link to the latest run.")
parser_new.add_argument("--no-mi-exclude-outer-sample", action="store_true",
                        help="Do not exclude the generating sample in the inner estimator when approximating mutual"
                             " information.")
parser_new.add_argument("--alternative-name", type=str,
                        help="Create a symlink with this name that points to the run directory.")
parser_new.add_argument("-pcs", "--policy-cache-size", type=int, default=50000,
                        help="Size of the policy cache of the CEM planner.")
parser_new.add_argument("-w", "--warm-start", type=str, help="Path to a replay buffer used for a warm start.")
parser_new.add_argument("--resample-trajectories", action="store_true",
                        help="When performing a warm start, sample all trajectories from the replay buffer again using "
                             "only their initial state and the actions.")
parser_new.add_argument("--no-train-trans", action="store_true", help="Do not train the transition model.")
parser_new.add_argument("--no-train-rew", action="store_true", help="Do not train the reward model.")
parser_new.add_argument("--ig-include", choices=["both", "states", "rewards"], default="both",
                        help="Which components to include in the information gain computation.")
parser_new.add_argument("--no-load-planner", action="store_true",
                        help="Do not load the planner state. Only relevant if --cont is given.")
parser_new.add_argument("--training-noise-ps", type=float, default=0.0,
                        help="Noise std dev to add to the input state during training.")

parser_resume = subparsers.add_parser("resume")
parser_resume.add_argument("output_dir", type=str, help="Output directory of the run to resume.")
parser_resume.add_argument("--episodes", type=int, help="Number of episodes to run in this instance (default: all)")
parser_resume.add_argument("--no-load-model", action="store_true",
                           help="Do not load the model but rather learn it from scratch.")
parser_resume.add_argument("--max-initial-training-steps", type=int,
                           help="Maximum number of steps to train the model before the actual training starts. Only "
                                "applies if --no-load-model is set.")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.pause_starts is not None:
        pause_starts = [parse_time(e) for e in args.pause_starts.split(",")]
        pause_lengths = [parse_time(e) for e in args.pause_lengths.split(",")]
        assert len(pause_lengths) == len(pause_starts) == 7
        logger.info("Pause schedule:")
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for d, s, l in zip(weekdays, pause_starts, pause_lengths):
            logger.info("{:>10s}: {} ({} long)".format(d, s, l))
    else:
        pause_starts = pause_lengths = None

    if torch.cuda.is_available() and not args.no_cuda:
        dev = "cuda"
    else:
        dev = "cpu"
    logger.info("Using device {}.".format(dev))

    if args.command == "new":
        if args.output_path is None:
            results_path = Path(__file__).resolve().parents[1] / "results"
        else:
            results_path = Path(args.output_path).resolve()
        result_path = run_name = None
        i = 0
        while result_path is None:
            try:
                time_formatted = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                if i == 0:
                    run_name = "{}_{}".format(time_formatted, args.env)
                else:
                    run_name = "{}_{}_{}".format(time_formatted, i, args.env)
                result_path = results_path / run_name
                result_path.mkdir(parents=True)
            except FileExistsError:
                i += 1
                result_path = None

        latest_path = results_path / "latest"
        if args.alternative_name is not None:
            alternative_name_link = results_path / args.alternative_name
            alternative_name_link.unlink(missing_ok=True)
            alternative_name_link.symlink_to(run_name)
        if not args.no_link_latest:
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(run_name)
        with (result_path / "config.json").open("w") as f:
            json.dump(args.__dict__, f, indent=True)

        meta_info = {
            "git_commit": determine_git_commit()
        }

        with (result_path / "meta.json").open("w") as f:
            json.dump(meta_info, f, indent=True)
    else:
        result_path = Path(args.output_dir)
        with (result_path / "config.json").open() as f:
            args_dict = json.load(f)
        new_args = parser_new.parse_known_args([])[0]
        new_args.__dict__.update(**args_dict)
        new_args.__dict__.update(args.__dict__)
        args = new_args

    log_path = result_path / "log.txt"
    file_handler = FileHandler(log_path)
    file_handler.setFormatter(Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    robot_gym.logger.addHandler(file_handler)

    env, model_ensemble, filter, trainer, planner, planner_optimizer, cost_function, _ = configure(
        args, dev, result_path, warm_up_trainer=not args.no_warm_up)

    if args.command == "resume":
        initial_episode = load_state(
            result_path, model_ensemble, planner, planner_optimizer, trainer,
            load_transition_model=not args.no_load_model, load_reward_model=not args.no_load_model) + 1
    else:
        initial_episode = 0

    max_steps = args.time_steps
    ensemble_size = args.ensemble_size
    train_interval = 1
    eval_interval = args.evaluation_interval
    replay_buffer_saving_interval = 50
    model_saving_interval = max(int(args.total_episodes / 100), 10)
    max_training_steps_per_episode = max(int(train_interval * max_steps), 1)
    training_steps = max_training_steps_per_episode * 20
    # Maximum number of training steps per episode present in the replay buffer

    wrapped_env = CallbackWrapperEnv(env)

    if args.warm_start is not None:
        rp_path = Path(args.warm_start)
        if not args.resample_trajectories:
            trainer.load_replay_buffer(rp_path)
        else:
            state = torch.load(rp_path)
            initial_states = state["obs"][:, :, 0]
            actions = state["act"]
            warm_start_planner = StaticPlanner()
            ws_env = WarmStartWrapperEnv(env)

            for i, (init_state, act) in enumerate(zip(initial_states, actions)):
                logger.info("Resampling episode {}/{}...".format(i, actions.shape[0]))
                warm_start_planner.reset(act.reshape((-1, env.action_space.shape[0])))
                ws_env.set_initial_state_buffer(init_state.cpu().numpy())
                trainer.collect_rollouts(ws_env, warm_start_planner)
        save_replay_buffer(result_path, trainer)
        trainer.train_model(max_steps * trainer.replay_buffer_size)

    if args.command == "resume" and args.no_load_model:
        steps = max_steps * trainer.replay_buffer_size
        if args.max_initial_training_steps is not None:
            steps = min(steps, args.max_initial_training_steps)
        trainer.train_model(steps)

    trainer.post_step_event.handlers.append(on_post_step_callback)
    trainer.post_reset_event.handlers.append(on_post_reset_callback)
    trainer.episode_done_event.handlers.append(on_episode_done_callback)

    log_dir = result_path / "tensorboard"
    log_dir.mkdir(exist_ok=True)
    summary_writer = SummaryWriter(str(log_dir))

    state_repr_lower = env.state_repr_lower_bounds
    repr_upper = env.state_repr_upper_bounds
    if state_repr_lower.shape[0] < 2:
        assert state_repr_lower.shape[0] == 1
        state_repr_lower = np.array([state_repr_lower[0], -1])
    if repr_upper.shape[0] < 2:
        assert repr_upper.shape[0] == 1
        repr_upper = np.array([repr_upper[0], 1])

    train_state_coverage_hist = np.zeros((50, 50), dtype=np.int_)
    test_state_coverage_hist = np.zeros_like(train_state_coverage_hist)
    bin_size = (repr_upper - state_repr_lower) / np.array(train_state_coverage_hist.shape)

    current_sample_is_evaluation_sample = False

    init_episodes = 1

    step = 0
    current_video_file: Optional[Path] = None

    eval_rewards = []
    train_rewards = []

    record_path = result_path / "videos"
    if args.video_log_interval_train > 0 or args.video_log_interval_eval > 0:
        record_path.mkdir(exist_ok=True)

    last_episode = initial_episode - 1

    if args.episodes is None:
        episodes = args.total_episodes
    else:
        episodes = args.episodes

    terminate = False
    signal.signal(signal.SIGINT, _terminate_signal_handler)
    signal.signal(signal.SIGTERM, _terminate_signal_handler)
    signal.signal(signal.SIGHUP, _terminate_signal_handler)
    signal.signal(signal.SIGABRT, _terminate_signal_handler)

    rollout_no = None

    try:
        logger.info("Beginning training.")
        for episode in range(initial_episode, min(args.total_episodes, initial_episode + episodes)):
            if terminate:
                break
            if args.start_learning_std_devs is not None and episode >= args.start_learning_std_devs and \
                    not model_ensemble.std_dev_learning_enabled:
                logger.info("Enabling standard deviation learning.")
                model_ensemble.enable_learned_std_devs()
                if episode == args.start_learning_std_devs:
                    model_ensemble.reset_parameters()
                    trainer.train_model(min(
                        args.std_dev_initial_training_steps,
                        max_training_steps_per_episode * trainer.replay_buffer_size))

            if pause_starts is not None:
                today = datetime.today()
                weekday = today.weekday()
                start_time = pause_starts[weekday] + today.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = pause_lengths[weekday] + start_time
                now = datetime.now()
                if start_time <= now < end_time:
                    hours, remainder = divmod(int(round((end_time - now).total_seconds())), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    logger.info("Going into standby for {:02d}:{:02d}:{:02d}.".format(hours, minutes, seconds))
                    env.standby()
                    while (end_time - datetime.now()).total_seconds() > 0.0 and not terminate:
                        time.sleep(min(1.0, max(0.0, (end_time - datetime.now()).total_seconds())))
                    if terminate:
                        continue
                    logger.info("Waking up environment...")
                    env.wakeup()
                    logger.info("Environment woke up. Resuming training.")

            last_episode = episode
            logger.info("Episode {}/{} ({}/{})".format(
                episode, args.total_episodes, episode - initial_episode, episodes))
            progress = episode / (args.total_episodes - 1)
            cost_function.update_progress(progress)
            epoch_time = time.time()
            is_init_episode = episode < init_episodes and not args.test_mode and args.cont is None
            if not is_init_episode and (episode + 1) % eval_interval == 0 or args.test_mode:
                logger.info("Performing evaluation rollout...")
                current_sample_is_evaluation_sample = True
                rollout_no = -1
                start_time = time.time()
                meta_info = trainer.perform_evaluation_rollout(wrapped_env, planner)
                end_time = time.time()
                total_step_count = sum(map(len, meta_info)) - len(meta_info)
                steps_per_second = total_step_count / (end_time - start_time)
                test_reward = sum([mi.reward for mi in meta_info[1:]])
                summary_writer.add_scalar("test/reward", test_reward, global_step=episode, walltime=epoch_time)
                state_cov_log = np.log(test_state_coverage_hist + 1)
                summary_writer.add_image(
                    "test/coverage_log_hist",
                    img_tensor=torch.from_numpy(np.flip(state_cov_log, axis=1) / np.max(state_cov_log)),
                    global_step=episode, walltime=epoch_time, dataformats="WH")
                eval_rewards.append(test_reward)
                current_sample_is_evaluation_sample = False
                logger.info("Evaluation rollout done. ({:0.2f} steps/s).".format(steps_per_second))
                summary_writer.add_scalar(
                    "test/env_steps_per_second", steps_per_second, global_step=episode, walltime=epoch_time)
            if not args.test_mode:
                logger.info("Collecting {}episodes for training...".format("initial " if is_init_episode else ""))

                rollout_no = -1
                start_time = time.time()
                meta_info = trainer.collect_rollouts(wrapped_env, planner if not is_init_episode else None)
                end_time = time.time()
                total_step_count = sum(map(len, meta_info)) - len(meta_info)
                steps_per_second = total_step_count / (end_time - start_time)
                logger.info("Done collecting episodes ({:0.2f} steps/s).".format(steps_per_second))
                summary_writer.add_scalar(
                    "train/env_steps_per_second", steps_per_second, global_step=episode, walltime=epoch_time)

                if cost_function is not None and not is_init_episode:
                    summary_writer.add_scalar("cost/extrinsic_weight", cost_function.extrinsic_weight(),
                                              global_step=episode, walltime=epoch_time)
                    summary_writer.add_scalar("cost/reward_moving_average", cost_function.reward_moving_average(),
                                              global_step=episode, walltime=epoch_time)
                    cost_function.update_weights()
                train_reward = np.mean([sum(si.reward for si in mi[1:]) for mi in meta_info])
                summary_writer.add_scalar("train/reward", train_reward, global_step=episode, walltime=epoch_time)
                state_cov_log = np.log(train_state_coverage_hist + 1)
                summary_writer.add_image(
                    "train/coverage_log_hist",
                    img_tensor=torch.from_numpy(np.flip(state_cov_log, axis=1) / np.max(state_cov_log)),
                    global_step=episode, walltime=epoch_time, dataformats="WH")
                train_rewards.append(train_reward)
                if not is_init_episode:
                    fpi = meta_info[0][1].planner_info
                    if fpi is not None:
                        if "reward_contributions" in fpi:
                            reward_component_names = fpi["reward_contributions"].keys()
                            reward_contributions_structured = {
                                n: np.array([[si.planner_info["reward_contributions"][n].numpy()
                                              for si in mi[1:] if si.planner_info is not None] for mi in meta_info])
                                for n in reward_component_names
                            }
                            reward_contributions_mean = {
                                k: np.mean(v) for k, v in reward_contributions_structured.items()}
                            total_reward_contribution = sum(reward_contributions_mean.values())
                            reward_contributions_mean_rel = {
                                k: v / total_reward_contribution for k, v in reward_contributions_mean.items()
                            }
                            for n, v in reward_contributions_mean.items():
                                summary_writer.add_scalar(
                                    "planner/reward_contribution/{}".format(n), v, global_step=episode,
                                    walltime=epoch_time)
                            for n, v in reward_contributions_mean_rel.items():
                                summary_writer.add_scalar(
                                    "planner/reward_contribution_rel/{}".format(n), v, global_step=episode,
                                    walltime=epoch_time)

                        for metric_name, v_dict in fpi.items():
                            if metric_name.startswith("reward_components_"):
                                metric_short = metric_name[len("reward_components_"):]
                                reward_component_names = v_dict.keys()
                                final_values = {
                                    n: np.array(
                                        [
                                            [
                                                si.planner_info[metric_name][n][-1].numpy()
                                                for si in mi[1:] if si.planner_info is not None]
                                            for mi in meta_info
                                        ])
                                    for n in reward_component_names
                                }
                                mean_values = {
                                    n: np.array(
                                        [
                                            [
                                                si.planner_info[metric_name][n].mean().numpy()
                                                for si in mi[1:] if si.planner_info is not None]
                                            for mi in meta_info
                                        ])
                                    for n in reward_component_names
                                }
                                final_value_means = {k: np.mean(v) for k, v in final_values.items()}
                                mean_value_means = {k: np.mean(v) for k, v in mean_values.items()}
                                for n, v in final_value_means.items():
                                    summary_writer.add_scalar(
                                        "reward_components_final/{}/{}".format(metric_short, n), v, global_step=episode,
                                        walltime=epoch_time)
                                for n, v in mean_value_means.items():
                                    summary_writer.add_scalar(
                                        "reward_components_mean/{}/{}".format(metric_short, n), v, global_step=episode,
                                        walltime=epoch_time)
                        if "general" in fpi:
                            for metric_name in ["proposal_success_fraction", "total_iterations", "time_needed",
                                                "average_successful_cutoff_index"]:
                                if metric_name in fpi["general"]:
                                    average_value = np.mean(
                                        [si.planner_info["general"][metric_name].item()
                                         for mi in meta_info for si in mi[1:] if si.planner_info is not None])
                                    summary_writer.add_scalar(
                                        "planner/" + metric_name, average_value, global_step=episode,
                                        walltime=epoch_time)
                            if "best_action_std_dev" in fpi["general"]:
                                for t in set(map(int, np.linspace(1, args.time_steps, 5))):
                                    best_action_std_dev_lst = [
                                        mi[t].planner_info["general"]["best_action_std_dev"]
                                        for mi in meta_info if mi[t].planner_info is not None]
                                    if len(best_action_std_dev_lst) > 0:
                                        max_iterations = max([e.shape[0] for e in best_action_std_dev_lst])
                                        for i in range(0, max_iterations, 3):
                                            average_value = np.mean(
                                                [e[i, 0].mean().item() for e in best_action_std_dev_lst
                                                 if e.shape[0] > i])
                                            summary_writer.add_scalar(
                                                "planner/best_action_std_dev_s{:02d}_a0_itr{:02d}".format(t, i),
                                                average_value, global_step=episode, walltime=epoch_time)

                if (episode + 1 - init_episodes) % train_interval == 0 and \
                        not args.test_mode and episode + 1 >= init_episodes:
                    steps = min(training_steps, max_training_steps_per_episode * trainer.replay_buffer_size)
                    logger.info("Training for {} steps...".format(steps))
                    train_info = trainer.train_model(steps)
                    summary_writer.add_scalar(
                        "train/model_loss_mean_pre", train_info["model_losses"][0].mean().item(), global_step=episode,
                        walltime=epoch_time)
                    summary_writer.add_scalar(
                        "train/model_loss_mean", train_info["model_losses"].mean().item(), global_step=episode,
                        walltime=epoch_time)
                    summary_writer.add_scalar(
                        "train/model_loss_mean_post", train_info["model_losses"][-1].mean().item(), global_step=episode,
                        walltime=epoch_time)
                    summary_writer.add_scalar(
                        "train/model_steps_per_second", train_info["steps_per_s"].item(), global_step=episode,
                        walltime=epoch_time)
                    logger.info("Done training.")
                    if not is_init_episode:
                        planner_optimizer.optimize()

                    if (episode + 1) % model_saving_interval == 0:
                        save_state(result_path, model_ensemble, planner, planner_optimizer, trainer, episode)
                if (episode + 1) % replay_buffer_saving_interval == 0:
                    if not args.test_mode:
                        save_replay_buffer(result_path, trainer)
    except Exception as ex:
        logger.error(traceback.format_exc())
        raise
    finally:
        file_handler.flush()
        if not args.test_mode:
            save_replay_buffer(result_path, trainer)
            save_state(result_path, model_ensemble, planner, planner_optimizer, trainer, last_episode)
        wrapped_env.close()
