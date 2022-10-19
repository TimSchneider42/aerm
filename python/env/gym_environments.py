import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sisyphus_env import SisyphusEnv

from util.rectangle import Rectangle
from sisyphus_env.base_env import BaseEnv


def _mk_ball_placing(
        headless: bool = False, log_dir: Optional[Path] = None, r: int = 0, w: int = 2, g: int = 20, f: int = 10,
        h: int = 0, ca: int = 0, t: int = 30, i: int = 20, b: int = 1) -> SisyphusEnv:
    hole_configs = [
        [],
        [
            Rectangle(np.array([-0.25, 0.3]), np.array([0.25, 0.5])),
            Rectangle(np.array([-1.0, -0.2]), np.array([-0.2, 0.0])),
            Rectangle(np.array([-1.0, 0.0]), np.array([-0.6, 1.0])),
            Rectangle(np.array([0.2, -0.2]), np.array([1.0, 0.0])),
            Rectangle(np.array([0.6, 0.0]), np.array([1.0, 1.0]))
        ]
    ]

    return SisyphusEnv(
        headless=headless, allow_rotation=bool(r), action_weight=0.001, gripper_width=g / 1000, ball_friction=f / 10,
        holes=hole_configs[h], control_lin_accel=bool(ca), ball_mass=w / 100, log_dir=log_dir, time_steps=t,
        sim_table_inclination_rad=i / 100, use_ball=bool(b))


def _mk_ball_placing_real(
        headless: bool = False, log_dir: Optional[Path] = None, a: int = 10, b: int = 1, r: int = 1, d: int = 0,
        ca: int = 1, tg: int = 0, db: int = 0, s: int = 0, t: int = 30, oa: int = 20) -> SisyphusEnv:
    return SisyphusEnv(
        action_weight=a / 1000, platform="real", use_ball=bool(b), allow_rotation=bool(r),
        headless=headless, dense_reward=bool(d), control_lin_accel=bool(ca), use_telegram_bot=bool(tg),
        done_on_border_contact=bool(db), shutdown_robot_on_close=bool(s), time_steps=t, log_dir=log_dir,
        obs_act_offset=oa / 100)


def _mk_ball_placing_sim_rt(
        headless: bool = False, log_dir: Optional[Path] = None, a: int = 10, b: int = 1, r: int = 1, d: int = 0,
        ca: int = 1, db: int = 0, t: int = 30, oa: int = 20, tx: int = 500, ty: int = 500) -> SisyphusEnv:
    return SisyphusEnv(
        action_weight=a / 1000, platform="sim_rt", use_ball=bool(b), allow_rotation=bool(r), headless=headless,
        dense_reward=bool(d), control_lin_accel=bool(ca), done_on_border_contact=bool(db), time_steps=t,
        log_dir=log_dir, obs_act_offset=oa / 100, table_extents=(tx / 1000, ty / 1000), ball_friction=0.1)


_ENV_FACTORIES = {
    "ball": _mk_ball_placing,
    "ball-real": _mk_ball_placing_real,
    "ball-simrt": _mk_ball_placing_sim_rt,
}


class InvalidEnvironmentDescException(Exception):
    pass


def get_env_factory(env_desc: str) -> Callable:
    env_name_stops = env_desc.rfind("_")
    env_name = env_desc[:env_name_stops] if env_name_stops != -1 else env_desc
    if env_name not in _ENV_FACTORIES:
        raise InvalidEnvironmentDescException("Unknown environment \"{}\".".format(env_name))
    params = env_desc[len(env_name) + 1:]
    params_split = re.split("([a-zA-z]+)", params)[1:]
    if len(params_split) % 2 != 0:
        raise InvalidEnvironmentDescException("Malformed parameter expression \"{}\"".format(params))
    try:
        kwargs = {n: int(v) for n, v in zip(params_split[::2], params_split[1::2])}
    except ValueError:
        raise InvalidEnvironmentDescException("Malformed parameter expression \"{}\"".format(params))

    def output(headless: bool = False, log_dir: Optional[Path] = None) -> BaseEnv:
        return _ENV_FACTORIES[env_name](headless=headless, log_dir=log_dir, **kwargs)

    return output
