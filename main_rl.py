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
import random
from scenario_gym.entity import Entity
from scenario_gym.controller import Controller, VehicleController
from scenario_gym.manager import ScenarioManager
from scenario_gym.sensor import RasterizedMapSensor
from sb3.stable_baselines3.ppo import PPO
from sb3.stable_baselines3.common.policies import ActorCriticPolicy 
from sb3.stable_baselines3.common.env_checker import check_env
from sb3.stable_baselines3.common.env_util import make_vec_env
from typing import Optional
import gymnasium as gym 

class MapSensor(RasterizedMapSensor):
    def _step(self, state):
        obs = super()._step(state)
        return obs.map[...,1]

class EgoSpeedMetric(Metric):
    """Compute the average speed of the ego."""
    
    def _reset(self, state):
        self.ds = 0.
        self.t = 0.
    
    def _step(self, state):
        ego = state.scenario.entities[0]
        self.ds += np.linalg.norm(state.velocities[ego][:2]) * state.dt
        self.t = state.t
        
    def get_state(self):
        return self.ds / self.t if self.t > 0 else 0.
        

class SBPPO(Agent):

    def __init__(self, entity: Entity, controller: Controller, sensor: Sensor):
        super().__init__(entity, controller, sensor)
        self.model = PPO(cfg.policy, env, tensorboard_log= cfg.tb_logs, n_steps= cfg.n_steps, batch_size = cfg.batch_size)
        self.entity
    
    def _reset(self):
        self.st_prev = None
        self.vel_prev = np.array([0.0, 0.0])
        self.r_prev = None
        self.pi_prev = None
        self.ds = 0.0
        self.ep_length = 0
        self.ep_reward = 0.0
    
    def _step(self, act):
        pass


    def _reward(self, state: State) -> float:
        ego = state.scenario.entities[0]
        current_v = state.velocities[ego][:2]
        dv = current_v - self.vel_prev 


        acc = np.linalg.norm(dv) / state.dt

        acc_penalty = -1 * acc**2

        speed = np.linalg.norm(dv)
        max_speed = state.agents[self.entity].controller.max_speed

        if max_speed is not None:

            norm_speed = speed / max_speed
        else:
            norm_speed = speed / 30


        return acc_penalty + norm_speed

    
    def _step(self, observation: Observation) -> Action:

        actions = self.model(observation)
        self.ep_reward += self.r_prev

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
    terminal_conditions = ["ego_off_road", "ego_collision"]
    metrics = EgoSpeedMetric()
    global env, cfg 
    cfg = args
    ppo_cfg = SBPPO_Config()
    env = openai_gym(observation_space=config.obs_space,action_space=config.action_space, create_agent=ppo_cfg.create_agent)
    env.metrics.append(metrics)
    env.load_scenario(args.scenario_path, create_agent=ppo_cfg.create_agent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vec_env = make_vec_env(lambda: env, n_envs=1)
    model = ppo_cfg.agent.model
    model.learn(total_timesteps=1e6, log_interval=10)
    model.save(config.model_path)



    
    # for episode in range(args.n_episodes):
    #     obs = env.reset().transpose(2,0,1)[1:,:,:]
    #     
    #     obs_th = torch.tensor(obs, dtype=torch.bool).to(device=device)
    #     done = False
    #     rewards = []

    #     ep_reward = 0
    #     count = 0 
    #     while not done:
    #         count+=1

    #         policy = ppo_cfg.agent.model.policy

    #         action, vf, log_probs = policy.forward(obs_th)
    #         action_np= np.array(action.detach().cpu().numpy())
    #         gym_action = (action_np[0][0], action_np[0][1])
    #         print(f'acc: {gym_action[0]}, steering: {gym_action[1]}')
    #         obs, reward, done, info = env.step(gym_action)
    #         rewards.append(reward)
    #         ep_reward += reward

    #         if done:
    #             print('CHECK')
    #     model.train() 
    #     print('done check')
    # pass



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

        config.obs_space = gym.spaces.Box(low=0, high=1, shape=(20, 20), dtype=np.bool_)
        n_actions = 2 #acc, steering 
        mean_low = -10 * np.ones(n_actions)
        mean_high = 10 * np.ones(n_actions)
        config.action_space = gym.spaces.Box(low=mean_low, high=mean_high, dtype=np.float32)



    if config.tb_logs:
        from datetime import datetime
        current_date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        log_path = os.path.join(os.path.dirname(__file__), 'rl_results', 'run-'+ current_date_time)

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
        
        config.tb_logs = log_path
    if config.render:


        vid_path = './rl_episodes/' + current_date_time 
        render_path = os.path.join(os.path.dirname(__file__), vid_path)
        if not os.path.exists(render_path):
            os.makedirs(render_path, exist_ok=True)
        config.vid_path = render_path
    
    if config.save_model:


        save_path = './saved_models/' + current_date_time 
        model_path = os.path.join(os.path.dirname(__file__), save_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        config.model_path = model_path




    run(config)

    






    print('check')


