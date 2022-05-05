import numpy as np

'''
This class implements the Goal Tolerance applied during training. The tolerance reduces through training to make it
easier initially to achieve goals and more difficult as the agent learns a policy.
'''

class GoalTolerance(object):
    def __init__(self, goal_tolerance_parameters):
        self.goal_tolerance_parameters = goal_tolerance_parameters
        self.inc_tol_obs = self.goal_tolerance_parameters['inc_tol_obs']
        self.init_tol = self.goal_tolerance_parameters['initial_tol']
        self.final_tol = self.goal_tolerance_parameters['final_tol']
        self.N_ts = self.goal_tolerance_parameters['N_ts']
        self.function = self.goal_tolerance_parameters['function']
        valid_functions = ['constant', 'linear', 'decay']
        assert self.function in valid_functions, 'Not a valid function. Choose constant, linear or decay.'

        if self.function == 'linear':
            self.a = (self.final_tol - self.init_tol) / self.N_ts
            self.b = self.init_tol

        if self.function == 'decay':
            self.a = self.init_tol
            self.r = 1 - np.power((self.final_tol / self.init_tol), 1 / self.N_ts)

        self.set_tol_value = self.goal_tolerance_parameters['set_tol']
        if self.set_tol_value == 0:
            self.current_tol = self.init_tol
        else:
            self.current_tol = self.set_tol_value
        self.training_step = 0

    def update(self, timestep):
        """
        Update current goal tolerance based on timestep of training.
        :param timestep: Current timestep of training to update goal tolerance.
        """
        # If set_tol is set to zero, update tolerance else use set tolerance.
        if self.set_tol_value == 0:
            if (self.function == 'linear') and (timestep <= self.N_ts):
                self.current_tol = self.a * timestep + self.b
            elif (self.function == 'decay') and (timestep <= self.N_ts):
                self.current_tol = self.a * np.power(1 - self.r, timestep)
            else:
                self.current_tol = self.final_tol
        else:
            self.current_tol = self.set_tol_value

    def get_tol(self):
        """
        Get the current updated goal tolerance.
        :return: The current tolerance.
        """
        return self.current_tol
