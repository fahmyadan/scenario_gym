from abc import ABC, abstractmethod
from scenario_gym.state import State
import numpy as np
from scenario_gym.controller import VehicleController
from typing import Tuple



class MPPI_Base(ABC):

    def __init__(self, vehicle_model: VehicleController, start_pose:np.ndarray, goal_pose: np.ndarray, time_horizon:int =50, k=100) -> None:

        self.vehicle_model = vehicle_model
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        self.time_horizon = time_horizon
        self.time_horizon = time_horizon

    @abstractmethod
    def init_car_model(self) -> VehicleController:
        pass
    
    @abstractmethod
    def compute_cost(self, obs: np.ndarray, step) -> float:
        pass
    
    @abstractmethod
    def compute_input(self, obs):
        pass
    

    def move_car(self, state: State, action: np.ndarray) -> np.ndarray: 
        
        return self.vehicle_model._step(state, action)
    


    def _step(
        self, state: np.ndarray, action: np.ndarray
    ):
        """
        Return the agent's next pose from the action.

        Updates the heading based on the steering angle. Then calculates
        the new speed to return the new velocity.
        """
        if isinstance(action, np.ndarray):
            accel, steer = action.acceleration, action.steering
        else:
            raise ValueError("Action must be a np.ndarra")

        accel = np.clip(accel, -self.max_accel, self.max_accel)
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        pose = state.poses[self.entity].copy()
        dt = state.next_t - state.t
        h = pose[3]

        dx = self.speed * np.cos(h)
        dy = self.speed * np.sin(h)
        dh = self.speed * np.tan(steer) / self.l

        pose[[0, 1]] += np.array([dx, dy]) * dt
        pose[3] += dh * dt

        speed = self.speed + accel * dt
        if not self.allow_reverse:
            speed = np.maximum(0.0, speed)
        if self.max_speed is not None:
            speed = np.minimum(self.max_speed, speed)
        self.speed = speed

        return pose