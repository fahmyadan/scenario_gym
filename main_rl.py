import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import argparse
import yaml
from pathlib import Path
import os
from scenario_gym.action import Action
from scenario_gym.observation import Observation
from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.integrations.openaigym import ScenarioGym as openai_gym
from scenario_gym.metrics import Metric
from scenario_gym.sensor import Sensor
from scenario_gym.state import State
from scenario_gym.scenario import Scenario
from scenario_gym.agent import Agent
import sys
from examples.ppo_agent import EpisodicReward, MapSensor
import random
from scenario_gym.entity import Entity
from scenario_gym.controller import Controller, VehicleController
from scenario_gym.manager import ScenarioManager
from sb3.stable_baselines3.ppo import PPO
from sb3.stable_baselines3.common.env_checker import check_env
from typing import Optional
import gymnasium as gym 


class SBPPO(Agent):

    def __init__(self, entity: Entity, controller: Controller, sensor: Sensor):
        super().__init__(entity, controller, sensor)
        self.model = PPO(cfg.policy, env, tensorboard_log= cfg.tb_logs).policy
    

    def _step(self, observation: Observation) -> Action:

        actions = self.model(observation)


        return actions



class SBPPO_Config(ScenarioManager):
    """
    Stores the config details of the MPPI agent
    """

    def __init__(self, config_path: str = None):

        pass



    def create_agent(self, scenario: Optional[Scenario], entity: Entity, *args) -> Agent:
        if entity.ref == "ego":
            controller = VehicleController(entity, max_steer=cfg.max_steer)
            self.sensor = MapSensor(entity)
            self.agent = SBPPO(
                entity,
                controller,
                self.sensor,
            )
            return self.agent
    
def scn_select():

    return cfg.scenario_path

def scn_call(gym: ScenarioGym) -> Scenario:

    return gym.state.scenario

def seed_all(seed: int) -> None:
    """Make deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run(args: argparse.Namespace) -> None:
    terminal_conditions = ["ego_off_road", "ego_collision", "max_length"]
    # scn_gym = ScenarioGym(
    #     terminal_conditions=terminal_conditions,
    #     timestep=0.1,
    # )
    metrics = EpisodicReward()
    # scn_gym.metrics.append(metrics)
    # scn_gym.load_scenario(args.scenario_path)
    global env, cfg 
    cfg = args
    ppo_cfg = SBPPO_Config()
    env = openai_gym(observation_space=config.obs_space,action_space=config.action_space, create_agent=ppo_cfg.create_agent)
    metrics = EpisodicReward()
    env.metrics.append(metrics)
    env.load_scenario(args.scenario_path, create_agent=ppo_cfg.create_agent)
    # model = PPO(args.policy, env, tensorboard_log= config.tb_logs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    for episode in range(args.n_episodes):
        obs = env.reset()
        obs_th = torch.tensor(obs, dtype=torch.bool).to(device=device)
        done = False
        reward = 0

        while not done:

            policy = ppo_cfg.agent.model.policy

            action = policy(obs_th)



        















    pass



if __name__ == "__main__":

    args_path = os.path.join(str(Path(__file__).parents[0]), 'rl_cfg', 'alg_cfg.yaml')

    with open(args_path, 'r') as f:
        try:

            cfg_dict = yaml.safe_load(f)
            config = argparse.Namespace(**cfg_dict)

        except:
            raise yaml.YAMLError
    
    if config.deterministic:
        seed_all(config.seed)
    
    if config.scenario_path is None:
        config.scenario_path = os.path.join(
            os.path.dirname(__file__),
            "tests",
            "input_files",
            "Scenarios",
            "d9726503-e04a-4e8b-b487-8805ef790c93.xosc",
        )
    if config.sensor == 'MapSensor':

        config.obs_space = gym.spaces.Box(low=0, high=1, shape=(20, 20, 2), dtype=np.bool_)
        n_actions = 2 #acc, steering 
        mean_low = -10 * np.ones(n_actions)
        mean_high = 10 * np.ones(n_actions)
        std_dev_low = np.zeros(n_actions)  # Standard deviations must be non-negative
        std_dev_high = 10 * np.ones(n_actions)
        combined_low = np.concatenate([mean_low, std_dev_low])
        combined_high = np.concatenate([mean_high, std_dev_high])
        config.action_space = gym.spaces.Box(low=combined_low, high=combined_high, dtype=np.float32)



    if config.tb_logs:
        from datetime import datetime
        current_date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        log_path = os.path.join(os.path.dirname(__file__), 'rl_results', 'run-'+ current_date_time)

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
        
        config.tb_logs = log_path
        


    run(config)

    






    print('check')


