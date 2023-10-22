from abc import ABC, abstractmethod
from scenario_gym.state import State
import numpy as np
from scenario_gym.controller import VehicleController
from typing import Tuple

class SimulateBicycleModel:
    def __init__(self, dt, axle_length):
        self.dt = dt
        self.axle_length = axle_length

    def move(self, current_pose, current_speed, control_actions):
        """
        Simulate motion of a vehicle using a kinematic bicycle model.

        Args:
            current_pose (np.ndarray): Current pose of the vehicle [x, y, theta].
            current_speed (float): Current linear speed of the vehicle.
            control_actions (np.ndarray): Control actions [acceleration, steering_angle].

        Returns:
            new_pose (np.ndarray): New pose of the vehicle after time step [x, y, theta].
            new_speed (float): New linear speed of the vehicle.
        """
        x, y, theta = current_pose
        acceleration, steering_angle = control_actions

        # Update linear speed
        new_speed = current_speed + (acceleration * self.dt)

        # Update pose using kinematic bicycle model
        new_x = x + (new_speed * np.cos(theta) * self.dt)
        new_y = y + (new_speed * np.sin(theta) * self.dt)
        new_theta = theta + ((new_speed / self.axle_length) * np.tan(steering_angle) * self.dt)

        new_pose = np.array([new_x, new_y, new_theta])

        return new_pose, new_speed

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
    

    

