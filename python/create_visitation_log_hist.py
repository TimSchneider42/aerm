import argparse
import glob
import pickle
import shutil
from pathlib import Path
from pickle import Unpickler
from typing import Optional, Union, Sequence, Any

import numpy as np
import torch
from PIL import Image
from stable_baselines3 import HerReplayBuffer

from util.load_config import load_config


def create_incremental_histograms(states2d: torch.Tensor, lower_lims: torch.Tensor, upper_lims: torch.Tensor,
                                  bin_counts: torch.Tensor, final_only: bool = False):
    states_normalized = (states2d - lower_lims) / (upper_lims - lower_lims)
    bin_indices = (states_normalized * bin_counts).long()
    bin_indices.clip_(torch.zeros((2,), device=states_normalized.device, dtype=torch.int), bin_counts - 1)
    bin_indices_flat = bin_indices[:, :, 0] * bin_counts[1] + bin_indices[:, :, 1]
    step_histograms = torch.zeros((states_normalized.shape[0], bin_counts[0], bin_counts[1])).to(
        states_normalized.device)
    ones = torch.ones_like(bin_indices_flat, dtype=torch.float)
    step_histograms.view((states_normalized.shape[0], -1)).scatter_add_(1, bin_indices_flat, ones)
    if final_only:
        return torch.sum(step_histograms, dim=0)
    else:
        return torch.cumsum(step_histograms, dim=0)


def save_histogram_sequence(histograms: torch.Tensor, path: Path, save_episodes: Optional[Sequence[int]] = None,
                            scale_factor: int = 1, format: str = "png"):
    path.mkdir()
    hist_perm = histograms.permute((0, 2, 1)).flip(1)
    hist_log = torch.log(hist_perm + 1)
    normalized = hist_log / torch.max(hist_log)
    imgs_numpy = (normalized * 255).byte().cpu().numpy()
    img_size = (imgs_numpy.shape[1] * scale_factor, imgs_numpy.shape[2] * scale_factor)
    if save_episodes is None:
        save_episodes = range(len(imgs_numpy))
    for i in save_episodes:
        img = Image.fromarray(imgs_numpy[i]).resize(img_size, Image.NEAREST)
        with (path / "{:06d}.{}".format(i, format)).open("wb") as f:
            img.save(f)


def process_dir(run_dir: Path, x: int = 0, y: int = 1, bx: int = 50, by: int = 50, save_every: int = 50,
                scale_factor: int = 10, format: str = "png", output_path: Optional[Path] = None,
                dev: Union[str, torch.device] = "cpu", histogram_count: Optional[int] = None,
                max_episode_count: Optional[int] = None, save: bool = True, save_episodes: Sequence[int] = None):
    if output_path is None:
        output_path = run_dir / "histograms"
    if save:
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)

    if "checkpoints" in [e.name for e in run_dir.iterdir()]:
        # Load baselines replay buffer
        with (run_dir / "replay_buffer.pkl").open("rb") as f:
            class RobustUnpickler(Unpickler):
                def find_class(self, __module_name: str, __global_name: str) -> Any:
                    try:
                        return super(RobustUnpickler, self).find_class(__module_name, __global_name)
                    except AttributeError:
                        return None
            replay_buffer = RobustUnpickler(f).load()
        if isinstance(replay_buffer, HerReplayBuffer):
            obs_space = replay_buffer.observation_space["observation"]
            obs = replay_buffer.observations["observation"]
        else:
            obs_space = replay_buffer.observation_space
            obs = replay_buffer.observations
        lower_lims = torch.from_numpy(obs_space.low[[x, y]]).to(dev)
        higher_lims = torch.from_numpy(obs_space.high[[x, y]]).to(dev)
        done = np.where(replay_buffer.dones)[0]
        done = np.concatenate([[-1], done])
        episode_lengths = done[1:] - done[:-1]
        episode_length = episode_lengths[0]
        episode_count = replay_buffer.pos // episode_length
        assert np.all(episode_lengths == episode_length)
        state2d_np = obs[:episode_count * episode_length, :, [x, y]].reshape((episode_count, 1, episode_length, -1))
        state2d = torch.from_numpy(state2d_np).to(dev)
        env = None
    else:
        (env, _, _, trainer, _, _, _, _), run_args = load_config(str(run_dir), headless=True)

        episode_count = trainer.replay_buffer_size

        if max_episode_count is not None:
            episode_count = min(max_episode_count, episode_count)

        state2d = trainer.replay_buffer_obs[:episode_count, :, :, [x, y]].to(dev)

        lower_lims = torch.from_numpy(env.observation_space.low[[x, y]]).to(dev)
        higher_lims = torch.from_numpy(env.observation_space.high[[x, y]]).to(dev)
    rs, es, ts, _ = state2d.shape

    if save_episodes is None:
        if histogram_count is not None:
            save_every = episode_count // histogram_count
        save_episodes = range(0, episode_count, save_every)
    else:
        save_episodes = set(save_episodes)

    bin_counts = torch.tensor([bx, by]).to(dev)

    state2d_by_episodes = state2d.reshape((rs, es * ts, 2))
    histograms_by_episodes = create_incremental_histograms(state2d_by_episodes, lower_lims, higher_lims, bin_counts)
    if save:
        save_histogram_sequence(histograms_by_episodes, output_path / "by_episodes", save_episodes=save_episodes,
                                scale_factor=scale_factor, format=format)

    state2d_by_time_steps = state2d.reshape((rs * es, ts, 2)).permute((1, 0, 2))
    histograms_by_time_steps = create_incremental_histograms(state2d_by_time_steps, lower_lims, higher_lims, bin_counts)
    if save:
        save_histogram_sequence(histograms_by_time_steps, output_path / "by_time_steps", scale_factor=scale_factor,
                                format=format)
    if env is not None:
        env.close()
    return histograms_by_episodes, histograms_by_time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str)
    parser.add_argument("-x", type=int, default=0, help="State component to visualize on x-axis.")
    parser.add_argument("-y", type=int, default=1, help="State component to visualize on y-axis.")
    parser.add_argument("-bx", type=int, default=50, help="Bin count for the x-axis.")
    parser.add_argument("-by", type=int, default=50, help="Bin count for the y-axis.")
    parser.add_argument("--save-every", type=int, default=50, help="Save every X episodes.")
    parser.add_argument("-s", "--scale-factor", type=int, default=10, help="Scale image by this factor.")
    parser.add_argument("-f", "--format", type=str, default="png", help="Target image format.")
    args = parser.parse_args()

    if torch.cuda.is_available() and False:
        dev = "cuda"
    else:
        dev = "cpu"

    print("Using device {}.".format(dev))

    root_path = Path(args.root_dir).resolve()
    run_paths = [
        Path(p).resolve().parent for p in glob.glob(str(root_path / "**" / "replay_buffer.pkl"), recursive=True)]

    run_paths_unique = sorted(set(run_paths))

    for p in run_paths_unique:
        print("Processing \"{}\"...".format(p.relative_to(root_path)))
        process_dir(p, args.x, args.y, args.bx, args.by, args.save_every, args.scale_factor, args.format, dev=dev)
