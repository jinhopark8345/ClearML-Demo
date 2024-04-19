

import argparse
import os

import torch
import yaml
from clearml import Task
from munch import Munch
from ray import rllib
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


def run_demo(config):

    task = Task.init(project_name='ray', task_name='ray_demo')
    params = task.connect(config)


    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=config.a)
        .resources(num_gpus=config.b)
        .environment(env="CartPole-v1")
        .build()
    )

    for i in range(10):
        result = algo.train()
        # print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save().checkpoint.path
            print(f"Checkpoint saved in directory {checkpoint_dir}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='')

    return parser.parse_args()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default='config/default.yaml', required=False)

    args, left_argv = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config = Munch.fromDict(yaml.safe_load(f))

    run_demo(config.train)
