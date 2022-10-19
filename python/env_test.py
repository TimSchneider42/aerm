import argparse
from pathlib import Path

import numpy as np
import PIL.Image as Image

from robot_gym.core.base_task import EpisodeInvalidException
from env.base_env import BaseEnv
from env import get_env_factory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="Environment to present.")
    parser.add_argument("-s", "--snapshot-path", type=str, help="Where to store the snapshot.")
    parser.add_argument("-k", action="store_true")
    parser.add_argument("-i", "--invert-controls", action="store_true",
                        help="Invert left/right and up/down in keyboard control.")
    parser.add_argument("--headless", action="store_true", help="Run environment in headless mode.")

    args = parser.parse_args()

    if args.k:
        import keyboard

    env: BaseEnv = get_env_factory(args.env)(args.headless)

    snapshot_taken = False

    try:
        while True:
            obs = env.reset()
            env.render()
            if not snapshot_taken and args.snapshot_path is not None:
                img_np = env.render(mode="rgb_array")
                img = Image.fromarray(img_np)
                with Path(args.snapshot_path).open("wb") as f:
                    img.save(f)
                snapshot_taken = True
            first = True
            done = False
            step = 0
            try:
                while not done:
                    step += 1
                    if args.env.startswith("ball_") and args.k:
                        inv = -1.0 if args.invert_controls else 1.0
                        ang_act = 0.0
                        if keyboard.is_pressed("a"):
                            ang_act = 0.5 * inv
                        elif keyboard.is_pressed("d"):
                            ang_act = -0.5 * inv
                        lin_act = np.zeros(2)
                        if keyboard.is_pressed("left"):
                            lin_act[0] = -inv
                        elif keyboard.is_pressed("right"):
                            lin_act[0] = inv
                        if keyboard.is_pressed("up"):
                            lin_act[1] = inv
                        elif keyboard.is_pressed("down"):
                            lin_act[1] = -inv
                        action = np.concatenate([lin_act, [ang_act]])
                    else:
                        action = env.action_space.sample()
                    obs, rew, done, info = env.step(action)
                    # time.sleep(5 * 0.005)
                    if rew > env.max_reward - 0.5 and first:
                        first = False
                        print(step)
                    prev_rew = rew
                    env.render()
            except EpisodeInvalidException as ex:
                print("Caught episode invalid exception: {}".format(ex))
    finally:
        env.close()
