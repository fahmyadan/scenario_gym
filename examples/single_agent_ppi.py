import numpy as np
from scenario_gym import ScenarioGym
from scenario_gym.controller import Controller, VehicleController
from scenario_gym.entity import Entity
from scenario_gym.scenario import Scenario
from scenario_gym.state import State
from scenario_gym.sensor import Sensor, RasterizedMapSensor
from scenario_gym.manager import ScenarioManager
from scenario_gym.xosc_interface import import_scenario
from scenario_gym.agent import Agent
from mppi import MPPI_Base
from typing import Any, Dict, List, Optional, Tuple
from scenario_gym.metrics import Metric
from copy import deepcopy
import copy
import argparse
import os


class MPPI_Agent(Agent, MPPI_Base):
    """
    MPPI_Agent processes the observations from the dynamic system and selects an action. 

    Uses  a physical model that describes the movement of the vehicle.

    Args:
    entity: Entity
        The entity that the agent will control.
    start_pose: np.ndarray
    goal_pose: np.ndarray
    time_horizon: int
        How far into the future the controller should simulate actions for.
     
    """

    def __init__(self, entity: Entity, controller: VehicleController, sensor: Sensor):
        super().__init__(entity, controller, sensor)
        self.entity = entity
        self.vehicle_model = controller
        self.start_pose = None 
        self.goal_pose = None 
        self.velocity = None
        self.time_horizon = flags.time_horizon
        self.k = flags.k
        self.lamda = flags.lambd
        self.sigma_acc = flags.sigma_acc
        self.sigma_steer = flags.sigma_steer
        self.sensor = sensor

        self.control_vector = np.zeros((2,self.time_horizon))  #2 dimensions for acc + steer

    def init_car_model(self):

        self.vehicle_model_copy =copy.copy(self.vehicle_model) 
        return  self.vehicle_model_copy
    
    def compute_cost(self, obs: np.ndarray, step):


        return 1.0

    
    def get_dynamic_obs(self, obs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        ego_pose, vel = obs[0], obs[1][0]

        x,y, yaw = ego_pose[0], ego_pose[1], ego_pose[3]
        goal_x, goal_y, goal_yaw = self.goal_pose[0], self.goal_pose[1], self.goal_pose[3]

        start = np.array([x,y, yaw, vel])
        goal = np.array([goal_x, goal_y, goal_yaw, 0.0])

        return start, goal 
    
    def compute_input(self, obs) -> np.ndarray: 

        """
        For VehicleController, the permitted actions (inputs to dynamic system) are acc and steer
        pose: [x, y, z, h, r, p]
        velocity: [longitudinal, lateral...] -> TODO: check this
        S: score array
        W: weight array
        Q: trajectory array, first row is x, second row is y, third row is yaw
        U: input array, first row is acceleration, second row is delta
        Returns: Tuple[acceleration:float, steer:float]
        """

        start, goal = self.get_dynamic_obs(obs)

        S = np.zeros(self.k)
        w = np.zeros(self.k)
        Q = np.zeros((self.k, 3, self.time_horizon))
        U = np.zeros((self.k, 2, self.time_horizon))

        for i in range(self.k):
            # noise array
            e_acc = np.random.normal(0, self.sigma_acc, self.time_horizon)
            e_steer = np.random.normal(0, self.sigma_steer, self.time_horizon)

            acc = self.control_vector[0, :] + e_acc
            steer = self.control_vector[1, :] + e_steer

            #Keep input in the range
            U[i, 0, :] = acc
            U[i, 1, :] = steer

            self.init_car_model()

            for t in range(self.time_horizon):
                # state for trajectory k at time t
                x, y, yaw  = start[0], start[1], start[2]

                Q[i, 0, t] = x  # x
                Q[i, 1, t] = y  # y
                Q[i, 2, t] = yaw  # yaw

                updated_poses = self.vehicle_model_copy._step(self.dynamic_state, U[i, :, t])
                start[0] = updated_poses[0]
                start[1] = updated_poses[1]
                start[2] = updated_poses[3]
                print('check')
            print('check')
        

        return np.array([1.0, 0.1])
    
    def reset(self, state: State) -> None:
        """Reset the agent state at the start of the scenario."""
        self.dynamic_state = state
        self.start_pose = state.poses[self.entity]
        self.current_vel = state.velocities[self.entity]
        self.goal_pose = state.scenario.ego.trajectory.data[-1]
        self.last_action = None
        self.last_reward = None
        self.sensor.reset(state)
        self.controller.reset(state)
        self._reset()
    
    def _reset(self) -> None:
        
        return None 
    
    def _step(self, state:State) :

        """
        Select an action from the current observation.

        state: global state as defined by the map sensor (e.g. Rasterized Map)
        dynamic_state: current state of the simulation (position, velocities etc of the entities) 
        """

        obs = self.dynamic_state.poses[self.entity]
        vel = self.dynamic_state.velocities[self.entity]

        actions = self.compute_input((obs, vel))

        new_poses = self.move_car(self.dynamic_state, actions)

        

        return actions
    

class MapSensor(RasterizedMapSensor):
    def _step(self, state):
        obs = super()._step(state)
        return obs.map   

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
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="MPPI Agent to steer a car."
    )
    parser.add_argument(
        "--seed",
        default=11,
        type=int,
        help="Set the random seed.",
    )
    parser.add_argument(
        "--max_steer",
        default=0.3,
        type=float,
        help="Maximum steering angle.",
    )
    parser.add_argument(
        "--sigma_acc",
        default=1.0,
        type=float,
        help="Variance of gaussian noise.",
    )
    parser.add_argument(
        "--sigma_steer",
        default=1.0,
        type=float,
        help="Variance of gaussian noise.",
    )
    
    parser.add_argument(
        "--k",
        default=100,
        type=int,
        help="Total number of sampled trajectories",
    )
    parser.add_argument(
        "--lambd",
        default=0.2,
        type=float,
        help="Hyperparameter for calculating weights for MPPI cost function.",
    )

    parser.add_argument(
        "--episodes",
        default=375,
        type=int,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--time_horizon",
        default=100,
        type=int,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--scenario_path",
        default=None,
        type=str,
        help="Path to the scenario file.",
    )
    parser.add_argument(
        "--verbose",
        default=-1,
        type=int,
        help="Print frequency per episode of loss and reward. -1 is no printing.",
    )
    return parser.parse_args()


class MPPI_Config(ScenarioManager):
    """
    Stores the config details of the MPPI agent
    """

    def __init__(self, config_path: str = None):

        pass



    def create_agent(self, scenario: Optional[Scenario], entity: Entity, *args) -> Agent:
        if entity.ref == "ego":
            controller = VehicleController(entity, max_steer=flags.max_steer)
            sensor = MapSensor(entity)
            return MPPI_Agent(
                entity,
                controller,
                sensor,
            )
        

def run(FLAGS):

    terminal_conditions = ["ego_off_road", "ego_collision"]
    gym = ScenarioGym(
        terminal_conditions=terminal_conditions,
    )
    metric = EgoSpeedMetric()
    gym.metrics.append(metric)

    mppi_config = MPPI_Config()
    gym.load_scenario(FLAGS.scenario_path, create_agent=mppi_config.create_agent, relabel=True)
    

    rewards, loss = 0.0, 0.0
    for episode in range(FLAGS.episodes):
        
        agent = gym.state.agents[gym.state.scenario.entity_by_name('ego')]
        gym.rollout(render=True)
        if FLAGS.verbose > 0:
            rewards += metric.get_state() / FLAGS.verbose
            if episode % FLAGS.verbose == 0:
                print(
                    "Episode {} Reward {:.4} Loss {:.4}".format(
                        episode, rewards, loss
                    )
                )
                rewards = 0.0

    # record a video of the agent
    gym.reset_scenario()
    gym.rollout(render=True, video_path='./mppi_test.mp4')


if __name__ == "__main__":
    global flags
    flags = parse_args()

    if flags.scenario_path is None:
        flags.scenario_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "tests",
            "input_files",
            "Scenarios",
            "d9726503-e04a-4e8b-b487-8805ef790c93.xosc",
        )
    run(flags)

    




    

