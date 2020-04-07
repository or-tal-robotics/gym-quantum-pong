import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym_quantum_pong.envs.quantum_pong import QuantumPong
N_DISCRETE_ACTIONS = 6
PI = np.pi
HEIGHT, WIDTH, N_CHANNELS = 100, 100, 1


class QuantumPongEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, mode):
        self.action_space = spaces.MultiDiscrete([N_DISCRETE_ACTIONS,N_DISCRETE_ACTIONS])
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        
        self.done = False
        self.mode = mode
        self.QP = QuantumPong(mode = self.mode)
        self.MAX_STEPS = 20000
        self.step_count = 0
        self.win = 0
        
    def get_reward(self):
        reward = [0,0]
        if self.win == 1:
            reward = [1,1]
        elif self.win == -1:
            reward = [-1,-1]
        
        return reward
        

    def reset(self):
        self.QP = QuantumPong(mode = self.mode)
        self.QP._update_board()
        self.done = False
        self.reward = [0,0]
        self.step_count = 0
        return  [np.expand_dims(self.QP.board_right, axis=2).astype(np.uint8), np.expand_dims(self.QP.board, axis=2).astype(np.uint8)]
        
    def step(self, action):
        if self.done == True:
            self.__init__()
        reward = [0,0]
        score, observation, self.done, hit, self.win = self.QP.step(action[0], action[1])
        observation_right = np.expand_dims(observation[0], axis=2).astype(np.uint8)
        observation_left = np.expand_dims(observation[1], axis=2).astype(np.uint8)
        if self.win == 1:
            reward = [1,1]
        elif self.win == -1:
            reward = [-1,-1]
            
        self.step_count += 1
        if self.step_count > self.MAX_STEPS:
            self.done = True
            print("Game over!, too many steps!")
        return [observation_right, observation_left], np.float32(reward), self.done, hit
    
    

         
    
       
       
        