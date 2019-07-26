import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym_quantum_pong.envs.quantum_pong import QuantumPong
N_DISCRETE_ACTIONS = 3
PI = np.pi
HEIGHT, WIDTH, N_CHANNELS = 84, 84, 1


class QuantumPongEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        self.action_dictionary = {
                0: [0,0,0],
                1: [3,0,0],
                2: [-3,0,0]}
        self.QP = QuantumPong()
        self.done = False
        self.reward = 0
        
    def get_reward(self):
        reward = 0.0
        if self.QP.ball_pos[1] > self.QP.bat_pos_B[1]:
            reward = 0
        if self.QP.ball_pos[1] < self.QP.bat_pos_A[1]:
            reward = -1   
        if self.QP.ball_pos[0] >= self.QP.bat_pos_A[0] - 2 and self.QP.ball_pos[0] <= self.QP.bat_pos_A[0] + 2 and self.QP.ball_pos[1] <= self.QP.bat_pos_A[1] :
            reward = 0
        return reward
        

    def reset(self):
        self.QP = QuantumPong()
        self.QP._update_board()
        self.done = False
        self.reward = 0.0
        return np.expand_dims(self.QP.board, axis=2).astype(np.uint8)
        
    def step(self, action):
        if self.done == True:
            self.__init__()
        reward = 0
        score, observation, self.done, hit, win = self.QP.step(self.action_dictionary[action], n_steps = 1)
        observation = np.expand_dims(observation, axis=2).astype(np.uint8)
        if hit == True:
            reward = 0
        if win == 1:
            reward = 1
        if win == -1:
            reward = -1
        self.reward += reward    
        return observation, np.float32(reward), self.done, {}
    

         
    
       
       
        