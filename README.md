# Active Exploration for Robotic Manipulation
This repository contains the code for our IROS 2022 paper "[Active Exploration for Robotic Manipulation](https://sites.google.com/view/aerm/)".

## Installation
This code requires a `python 3.7` or higher installation to run.
All requirements can be installed via 
```bash
pip install -r requirements.txt
```

## Running the experiments presented in the paper
Use the following commands to run the experiments presented in the paper.
Each run will create a log dir in `results/` which contains checkpoints and the tensorboard log.

#### Experiments on flat table (*Tilted Pushing*)
```bash
# Mutual Information
python main.py new -e ball_f02h0 --total-episodes 10000 -i mutual_information

# Lautum Information
python main.py new -e ball_f02h0 --total-episodes 10000 -i lautum_information -rs 2e5

# No intrinsic reward
python main.py new -e ball_f02h0 --total-episodes 10000 --no-intrinsic
```

#### Experiments on table with holes (*Tilted Pushing Maze*)
```bash
# Mutual Information
python main.py new -e ball_f05h1 --total-episodes 50000 -i mutual_information

# Lautum Information (here we use an adaptive weighting scheme)
python main.py new -e ball_f05h1 --total-episodes 50000 -i lautum_information -rs 2e5 -a max -awrs 1e8

# No intrinsic reward
python main.py new -e ball_f05h1 --total-episodes 50000 --no-intrinsic
```
If you want to run a different algorithm on these environments, check out https://github.com/TimSchneider42/sisyphus-env, which contains the environments and an example on how to use them.

For a documentation of the options of this software, run 
```bash
python main.py -h
```