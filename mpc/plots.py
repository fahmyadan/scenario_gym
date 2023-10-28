import matplotlib.pyplot as plt
import numpy as np
from scenario_gym.road_network import RoadNetwork
from scenario_gym.road_network.objects import Road, Lane, Intersection
from scenario_gym.entity import Entity
from typing import List, Tuple, Optional
from collections import namedtuple
from scenario_gym.state import State
import os 
from pathlib import Path
import matplotlib.cm as cm


class Plotter: 

    def __init__(self, network: Optional[Tuple[List[Road],List[Lane], List[Intersection]]], initial_pose, global_path) -> None:
        
        self._network = network
        self.initial_pose = initial_pose
        self.global_path = global_path 
        self.counter = 0 

        pass
    
    @property
    def network(self):

        network = namedtuple('Network', ['roads', 'lanes', 'intersections'])

        for net_item in self._network:
            if isinstance(net_item[0], Road): 
                network.roads = net_item
            elif isinstance(net_item[1], Lane):
                network.lanes = net_item
            elif isinstance(net_item[2], Intersection):
                network.intersections = net_item
            else: 
                raise Exception("Unknown network item")
            
        return network
    
    def get_all_lane_boundaries(self, lanes: List[Lane]):

        lane_list = []

        for lane in lanes:
            x,y = lane.boundary.exterior.xy
            lane_list.append((x,y))

        
        return lane_list
    
    def get_center_line(self, lane):
        return 
    
    def get_current_vehicle_pose(self, state: State, entity: Entity):

        full_pose = state.poses[entity]  #x,y,z,yaw,pitch,roll
        self.current_pose = np.array([full_pose[0], full_pose[1], full_pose[3]]) #x,y,yaw

        return self.current_pose
    

    def get_current_speed(self, state: State, entity: Entity):
        
        self.current_speed = np.linalg.norm(state.velocities[entity][:2])

        return self.current_speed

    
    def get_current_path_samples(self, state: State, trajectory: np.ndarray, weights: np.ndarray):

        self.current_path_samples = trajectory
        self.weights = weights
        
        return None 
    
    
    
    def transpose_reference_path(self, global_path: np.ndarray):

        reference_path = np.zeros((len(global_path), 3)) #x,y,yaw

        for i, traj in enumerate(global_path):
            reference_path[i] = np.array([traj[1], traj[2], traj[4]])


        
        return reference_path
    
    @property
    def reference_path(self):
        return self.transpose_reference_path(self.global_path)
    
    def get_goal_pose(self):

        final_pose = self.global_path[-1][1:]
        
        return final_pose 
    
    @property
    def goal_pose(self):
        return self.get_goal_pose()
    

    def plot(self):
        

        assert self.global_path is not None, "Global path is None; Please check"

        plot_dir = str(Path(__file__).parents[0]) + '/plots/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)


        plt.figure(figsize=(10, 10))
        plt.axis("equal")

        
        #Plot the goal pose
        goal_x, goal_y, goal_yaw = self.goal_pose[0], self.goal_pose[1], self.goal_pose[3]
        plt.plot(goal_x, goal_y, 'b*', markersize=10, label='Goal Pose')
        #plt.arrow(goal_x, goal_y, 0.1 * np.cos(goal_yaw), 0.1 * np.sin(goal_yaw), head_width=0.1, head_length=0.1, fc='b', ec='b')
        
    
        #Plot the initial pose
        init_x, init_y, init_yaw = self.initial_pose[0], self.initial_pose[1], self.initial_pose[2]
        plt.plot(init_x, init_y, 'go', markersize=7, label='Initial Pose')
        plt.arrow(init_x, init_y, 10 * np.cos(init_yaw), 10* np.sin(init_yaw), head_width=0.1, head_length=0.1, fc='g', ec='g')

        #Plot reference trajectory
        assert self.reference_path is not None, "Transposed reference path is None; Please check"
        
        ref_x, ref_y, ref_yaw = self.reference_path[:,0], self.reference_path[:,1], self.reference_path[:,2]
        plt.plot(ref_x, ref_y, 'r--', label='Reference Path')


        #Plot the road network
         
        lane_boundaries = self.get_all_lane_boundaries(self.network.lanes)

        for lane in lane_boundaries:
            plt.plot(lane[0], lane[1], 'k', label='__nolegend__')
        

        #Plot the current pose for step t
        plt.plot(self.current_pose[0], self.current_pose[1], 'ro', markersize=10, label=f'Current Pose {self.counter}')
        plt.arrow(self.current_pose[0], self.current_pose[1], 10 * np.cos(self.current_pose[2]), 10* np.sin(self.current_pose[2]), head_width=5, head_length=5, fc='r', ec='r')
        
        #Plot latest path samples
        normalized_weights = self.weights / np.max(self.weights)
        min_alpha = 0.3
        max_alpha = 1.0
        col_norm = plt.Normalize(0,0.01)
        cmap = cm.jet

        for k in range(self.current_path_samples.shape[0]):
           
            alpha = min_alpha + normalized_weights[k] * (max_alpha - min_alpha)

            col = cmap(col_norm(self.weights[k]))
            plt.plot(self.current_path_samples[k,0,:], self.current_path_samples[k,1,:], color=col, 
                   
                    alpha=alpha)


        plt.xlim(550, 350)
        plt.ylim(380, 150)
        plt.legend()
        
        plt.savefig(plot_dir+'plot_mppi_'+str(self.counter)+'.png')
        plt.close()
        self.counter += 1




