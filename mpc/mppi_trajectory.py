import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class MPPI_Trajectory:
    """
    An object that stores the global trajectory of the MPPI controller.
    
    At each time step, the controller will generate a new trajectory for a time 
    horizon of length T. 

    This object will take a new trajectory (length T), compare it to the global 
    trajectory (length N) and slice the reference trajectory considering the start
    and end of the predicted trajectory.

    The class will take the sliced global trajectory and ensure it is alligned 
    in time with the predicted trajectory. 

    
    """

    def __init__(self, global_path, state, time_horizon) -> None:

        self.global_path = global_path
        self.sim_state = state
        self.time_horizon = time_horizon
        self.dt = self.sim_state.dt
        self.global_section = None # The section of the global path that is being compared with a sample k
        pass

    def compare_trajectories(self, global_path, predicted_path):
        """
        Takes the global trajectory and slices it to match the predicted trajectory
        global_path: array (t,x,y,z,h,p,r)
        """

        predicted_path = predicted_path.T
        global_slice = np.zeros((global_path.shape[0], 4)) 

        for i in range(global_path.shape[0]):
            global_slice[i,0] = global_path[i,0]
            global_slice[i,1] = global_path[i,1]
            global_slice[i,2] = global_path[i,2]
            global_slice[i,3] = global_path[i,4]

        predicted_path = self.add_time(predicted_path)
        downsample_prediction, global_section = self.match_trajectory(global_slice, predicted_path)



        return downsample_prediction, global_section
    
    def match_trajectory(self, sliced_global_path, predicted_path):
        """
        Takes a timestamped predicted path and time stamped global path and matches them according to timesteps
        """

        # Find the start and end of the predicted path

        start = predicted_path[0,0]
        end = predicted_path[-1,0]

        #for each t in global path, find closest t and index to start and end of predicted path
        start_difference = np.zeros((sliced_global_path.shape[0], 1))
        end_difference = deepcopy(start_difference)
        for idx, t in enumerate(sliced_global_path[:,0]):
            start_difference[idx] = abs(t - start)
            end_difference[idx] = abs(t - end)
        
        start_idx = np.argmin(start_difference)
        end_idx = np.argmin(end_difference)
        self.global_section = sliced_global_path[start_idx:end_idx+1,:]

        target_length = abs(end_idx - start_idx) + 1

        # Downsample the longer trajectory to match the shorter one, when len(pred_path) > len(global_path)
        if target_length < self.time_horizon:
            # Downsample the predicted path
            factor = len(predicted_path)/target_length
            indices = [int(np.floor(i*factor)) for i in range(target_length)]
            adjusted_predicted_path = predicted_path[indices,:]
        else:
            raise ValueError("Path length is too long for the time horizon")
    
        assert adjusted_predicted_path.shape[0] == self.global_section.shape[0], "The length of the predicted path and global path do not match"

        return adjusted_predicted_path, self.global_section 
    
    
    def add_time(self, predicted_path):
        """
        Takes the predicted path and adds estimated timestamps 
        """
        timestamps = np.zeros((self.time_horizon,1))
        t = self.sim_state.t
        for i in range(self.time_horizon):
            timestamps[i] = t + ((i+1) *self.dt)


        predicted_path = np.concatenate((timestamps, predicted_path), axis=1)

        return predicted_path

    def align_trajectory(self, sliced_global_path, predicted_path):
        """
        Takes the sliced global trajectory and aligns it in time with the predicted trajectory
        """

        pass

    def interpolate_trajectory(self, aligned_global_path, predicted_path):
        """
        Takes the aligned global trajectory and interpolates it to match the predicted trajectory
        """

        pass