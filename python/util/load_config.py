import json
from pathlib import Path
from typing import Union, Optional

import torch

from main import configure, parser_new


def load_config(run_dir: str, no_jit: bool = False, dev: Union[str, torch.device] = "cpu", headless: bool = False,
                override_env: Optional[str] = None, warm_up_trainer: bool = False):
    run_path = Path(run_dir.split(":")[0])
    with (run_path / "config.json").open() as f:
        run_args_dict = json.load(f)
    run_args = parser_new.parse_known_args([])[0]
    run_args.__dict__.update(**run_args_dict)
    run_args.cont = run_dir
    run_args.no_jit = no_jit
    run_args.headless = headless
    if run_args.special_model == False:
        run_args.special_model = None
    elif run_args.special_model == True:
        run_args.special_model = "default"
    if override_env is not None:
        run_args.env = override_env
    return configure(run_args, dev, warm_up_trainer=warm_up_trainer), run_args
