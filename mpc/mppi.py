from abc import ABC, abstractmethod
from scenario_gym.state import State
import numpy as np
from scenario_gym.controller import VehicleController
from typing import Tuple

class SimulateBicycleModel:
    def __init__(self, dt, axle_length, max_steering_angle,max_acc, pose, speed):
        self.dt = dt
        self.axle_length = axle_length
        self.max_steer = max_steering_angle
        self.max_accel = max_acc
        self.instant_speed = speed
        self.pose = pose
        self.allow_reverse = False
        self.max_speed = None

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
        accel, steer= control_actions
        acceleration = np.clip(accel, -self.max_accel, self.max_accel)
        steering_angle = np.clip(steer, -self.max_steer, self.max_steer)

        x, y, theta = current_pose

        

        # Update pose using kinematic bicycle model
        new_x = x + (self.instant_speed * np.cos(theta) * self.dt)
        new_y = y + (self.instant_speed * np.sin(theta) * self.dt)
        new_theta = theta + ((self.instant_speed / self.axle_length) * np.tan(steering_angle) * self.dt)

        

        new_pose = np.array([new_x, new_y, new_theta])
        #Update speed and pose attribute

        # Update linear speed
        new_speed = self.instant_speed + (acceleration * self.dt)

        if not self.allow_reverse:
            new_speed = np.maximum(0.0, new_speed)
        if self.max_speed is not None:
            new_speed = np.minimum(self.max_speed, new_speed)
  

        self.instant_speed = new_speed
        self.pose = new_pose

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
    

    

