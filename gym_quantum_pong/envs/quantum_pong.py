import numpy as np
import matplotlib.pyplot as plt
import time


class QuantumPong():
    def __init__(self, n_players = 1, board_size = (80,80), V = [1,2], n_rounds = 21):
        self.board_size = board_size
        self.board = np.zeros(board_size)
        self.ball_pos = np.array([board_size[0]/2, board_size[1]/2], dtype=np.uint8)
        self.bat_pos_A = np.array([board_size[0]/2, 3], dtype=np.uint8)
        self.bat_pos_B = np.array([board_size[0]/2, board_size[1] - 3], dtype=np.uint8)
        self.score = np.zeros(2, dtype = np.int8)
        self.ball_vel = np.array(V, dtype = np.int8)
        self.n_players = n_players
        self.done = False
        self.n_rounds = n_rounds
        self.round = 0
        
        
        
    def _computer_action(self):
        alpha = np.arctan2(self.ball_vel[0], self.ball_vel[1])
        v = np.array([np.sin(alpha), np.cos(alpha)])
        d = np.round(self.ball_pos + v*np.linalg.norm(self.ball_pos-self.bat_pos_B))
        if self.bat_pos_B[0] != d[0] and self.ball_pos[1] > -self.board_size[1]/4 :
            action = 2*np.sign(d[0]-self.bat_pos_B[0])
        else:
            action = 0
        return action
    
    def _update_board(self):
        self.board = np.zeros(self.board_size)
        self.board[self.ball_pos[0], self.ball_pos[1]] = 255
        for ii in range(-2,3):
            self.board[self.bat_pos_A[0]+ii, self.bat_pos_A[1]] = 255
            self.board[self.bat_pos_B[0]+ii, self.bat_pos_B[1]] = 255
            
    def _quantum_action(self, alpha):
        psi = np.array([np.cos(alpha), 0, 0, np.sin(alpha)])
        A0 = np.array([[1,0],[0,-1]])
        A1 = np.array([[0,1],[1,0]])
        B0 = (A0+A1)/np.sqrt(2)
        B1 = (A0-A1)/np.sqrt(2)
        C = np.array([psi.dot(np.kron(A0,B0)).dot(psi.T),
             psi.dot(np.kron(A0,B1)).dot(psi.T),
             psi.dot(np.kron(A1,B0)).dot(psi.T),
             psi.dot(np.kron(A1,B1)).dot(psi.T)])
        i = np.random.binomial(1,0.5,2)
        j = i[0] + 2*i[1]
        i = np.random.binomial(1,(1+C[j])/2)
        if j <= 1:
            a = np.random.binomial(1,(1+ psi.dot(np.kron(A0,np.eye(2))).dot(psi.T))/2)
        else:
            a = np.random.binomial(1,(1+ psi.dot(np.kron(A1,np.eye(2))).dot(psi.T))/2)
        if i == 1:
            if a == 1:
                self.ball_pos[0] = self.board_size[0] - 1
                self.bat_pos_A[0] = self.board_size[0] - 3
            else:
                self.ball_pos[0] = 3
                self.bat_pos_A[0] = 3
        else:
            if a == 1:
                self.ball_pos[0] = self.board_size[0] - 1
                self.bat_pos_A[0] = 3
            else:
                self.ball_pos[0] = 3
                self.bat_pos_A[0] = self.board_size[0] - 3
                
        if self.bat_pos_A[0] < 3:
            self.bat_pos_A[0]  = 3
        if self.bat_pos_A[0] > self.board_size[0]-3:
            self.bat_pos_A[0]  = self.board_size[0]-3
                
                
    def step(self, Action_A, Action_B = [0]):
        hit = False
        win = 0
        # --- Single player game --- #
        if self.n_players == 1:
            Action_B[0] = self._computer_action()
        # --- Player A --- #
        self.bat_pos_A[0] += Action_A[0]
        if self.bat_pos_A[0] < 3:
            self.bat_pos_A[0]  = 3
        if self.bat_pos_A[0] >= self.board_size[0]-3:
            self.bat_pos_A[0]  = self.board_size[0]-3
        
        if Action_A[1] == 1 and np.sign(self.ball_vel[1]) == -1 and self.ball_pos[1] < 6:
            self._quantum_action(Action_A[2])
        
        # --- Player B --- #
        self.bat_pos_B[0] += Action_B[0]
        if self.bat_pos_B[0] < 3:
            self.bat_pos_B[0]  = 3
        if self.bat_pos_B[0] >= self.board_size[0]-3:
            self.bat_pos_B[0]  = self.board_size[0]-3
            
        # --- Ball step --- #
        self.ball_pos =  self.ball_pos + self.ball_vel
            
        # --- Ball --- #
        if self.ball_pos[0] > self.board_size[0]-1:
            self.ball_pos[0] = self.board_size[0]-1
            self.ball_vel[0] *= -1
            
            
        if self.ball_pos[0] < 1:
            self.ball_pos[0] = 1
            self.ball_vel[0] *= -1
        
        if self.ball_pos[1] > self.bat_pos_B[1]:
            self.score[0] += 1
            self.ball_pos[1] = self.bat_pos_B[1] -2
            self.ball_pos[0] = self.bat_pos_B[0]
            self.ball_vel[1] = -2
            self.round += 1
            win = 1
            
        if self.ball_pos[1] < self.bat_pos_A[1]:
            self.score[1] += 1
            self.ball_pos[1] = self.bat_pos_A[1] +2
            self.ball_pos[0] = self.bat_pos_A[0]
            self.ball_vel[1] = 2
            self.round += 1
            win = -1
            
        if self.ball_pos[0] >= self.bat_pos_A[0] - 2 and self.ball_pos[0] <= self.bat_pos_A[0] + 2 and self.ball_pos[1] <= self.bat_pos_A[1] :
            self.ball_vel[1] *= -1
            self.ball_vel[0] *= -1
            hit = True
            if np.random.choice(2):
                self.ball_vel[1] = np.sign(self.ball_vel[1])*2
            else:
                self.ball_vel[1] = np.sign(self.ball_vel[1])
                
        if self.ball_pos[0] >= self.bat_pos_B[0] - 2 and self.ball_pos[0] <= self.bat_pos_B[0] + 2 and self.ball_pos[1] >= self.bat_pos_B[1] :
            self.ball_vel[1] *= -1
            self.ball_vel[0] *= -1
            if np.random.choice(2):
                self.ball_vel[1] = np.sign(self.ball_vel[1])*2
            else:
                self.ball_vel[1] = np.sign(self.ball_vel[1])
        
        if self.round == 21:
            self.done = True
                
        
        self._update_board()
        return self.score, self.board, self.done, hit, win
    
    
if __name__ == '__main__':
    #plt.ion()
    #fig = plt.figure()
    QP = QuantumPong()
    done = False
    steps = 0
    while not done:
        action = [0,0,np.pi/4]
        a = QP.step(Action_A=action)
        print(a[0])
        done = a[2]
        steps +=1
        #plt.imshow(a[1])
        #plt.imshow(a[1])
        #fig.canvas.draw()
        #time.sleep(0.01)
        #fig.canvas.flush_events()
        
        
        
                
                
                
                
        