import numpy as np
import matplotlib.pyplot as plt


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
        pass

    def slice_trajectory(self, global_path, predicted_path):
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

        pass
    
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