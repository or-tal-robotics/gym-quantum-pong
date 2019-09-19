import numpy as np
import matplotlib.pyplot as plt
import time


class QuantumPong():
    def __init__(self, n_players = 1, board_size = (84,84,70), V = [2,2], n_rounds = 21):
        self.bat_size = 5
        self.board_size = board_size
        self.board = np.zeros((board_size[0],board_size[1]))
        self.ball_pos = np.array([board_size[0]/2, board_size[1]/2], dtype=np.uint8)
        bpA = np.random.randint(self.bat_size+1, board_size[0] - self.bat_size-2)
        bpB = np.random.randint(self.bat_size+1, board_size[0] - self.bat_size-2)
        self.bat_pos_A = np.array([bpA, 3], dtype=np.uint8)
        self.bat_pos_B = np.array([bpB, board_size[1] - 3], dtype=np.uint8)
        self.score = np.zeros(2, dtype = np.int8)
        self.ball_vel = np.array(V, dtype = np.int8)
        self.n_players = n_players
        self.done = False
        self.n_rounds = n_rounds
        self.round = 0
        self.n_steps = 0
        self.last_action = 0
        self.quantum_A = 1
        self.quantum_i = 1
        self.quantum_j = 1
        self.ball_memory = 0
        self.QuantumState = 0
    
    def _height(self,x):
        m = (self.board_size[2] - self.board_size[0])/self.board_size[1]
        b = self.board_size[0]
        h = int(m*x+b)
        return h
    
    def _draw_A(self,p):
        x = np.random.binomial(1,p)
        if x == 0:
            return 1
        else:
            return -1
        
    def _hidden_action(self,Action_A, Action_B, side):
        if side == 'A':
            if Action_A[1]==1:
                return 0
            elif Action_A[1]==-1:
                return 1
        if side == 'B':
            if Action_B[1]==1:
                return 0
            elif Action_B[1]==-1:
                return 1
        
        
    
    def _update_board(self):
        self.board = np.zeros((self.board_size[0],self.board_size[1]))
        self.board[self.ball_pos[0], self.ball_pos[1]] = 255
        if self.ball_pos[0]+1 < self.board_size[0]:
            self.board[self.ball_pos[0]+1, self.ball_pos[1]] = 255
        if self.ball_pos[0]-1 > 0:
            self.board[self.ball_pos[0]-1, self.ball_pos[1]] = 255
        if self.ball_pos[1]+1 < self.board_size[1]:
            self.board[self.ball_pos[0], self.ball_pos[1]+1] = 255
        if self.ball_pos[1]-1 > 0:
            self.board[self.ball_pos[0], self.ball_pos[1]-1] = 255
        for ii in range(-self.bat_size,self.bat_size+1):
            for jj in range(2):
                self.board[self.bat_pos_A[0]+ii-1, self.bat_pos_A[1]-jj] = 127
                self.board[self.bat_pos_B[0]+ii-1, self.bat_pos_B[1]+jj] = 127
        
        for ii in range(self.board_size[1]):
            self.board[self._height(ii)-1, ii] = 200
            self.board[0, ii] = 200
        
                
                
    def step(self, Action_A, Action_B ):
        hit = 0
        win = 0
        # --- Player A --- #
        self.bat_pos_A[0] += Action_A[0]
        if self.bat_pos_A[0] <= self.bat_size:
            self.bat_pos_A[0]  = self.bat_size
        if self.bat_pos_A[0] >= self.board_size[0]-self.bat_size-1:
            self.bat_pos_A[0]  = self.board_size[0]-self.bat_size-1
        
        
        # --- Player B --- #
        self.bat_pos_B[0] += Action_B[0]
        if self.bat_pos_B[0] <= self.bat_size:
            self.bat_pos_B[0]  = self.bat_size
        if self.bat_pos_B[0] >= self.board_size[2]-self.bat_size-1:
            self.bat_pos_B[0]  = self.board_size[2]-self.bat_size-1
            
        # --- Ball step --- #
        self.ball_pos =  self.ball_pos + self.ball_vel
            
        # --- Ball --- #
        if self.ball_pos[0] > self._height(self.ball_pos[1]):
            self.ball_pos[0] = self._height(self.ball_pos[1])
            self.ball_vel[0] *= -1
            
            
        if self.ball_pos[0] < 1:
            self.ball_pos[0] = 1
            self.ball_vel[0] *= -1
            
        if (Action_A[1] != 0 or Action_B[1] != 0):
            self.QuantumState = 1
            if self.ball_vel[1] < 0:
                self.td = -1
            else:
                self.td = 1
                        

        
        if self.QuantumState == 0:
            if self.ball_pos[0] >= self.bat_pos_A[0] - self.bat_size and self.ball_pos[0] <= self.bat_pos_A[0] + self.bat_size and self.ball_pos[1] <= self.bat_pos_A[1] :
                self.ball_pos[1] = self.bat_pos_A[1]
                self.ball_vel[1] *= -1
                hit = 1
                if np.random.choice(2):
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[0] *= 1
                    
            elif self.ball_pos[1] <= self.bat_pos_A[1]:
                self.score[1] += 1
                self.ball_pos = np.array([self.board_size[0]/2, self.board_size[1]/2], dtype=np.uint8)
                #self.bat_pos_A[0] = self.ball_pos[0] 
                self.ball_vel[1] = 2*(-1)**np.random.randint(0,2)
                if self.ball_vel[1] == 0:
                   self.ball_vel[1] = 1
                self.ball_vel[0] = np.random.randint(-2,3)
                self.round += 1
                win = -1
                    
            if self.ball_pos[0] >= self.bat_pos_B[0] - self.bat_size and self.ball_pos[0] <= self.bat_pos_B[0] + self.bat_size and self.ball_pos[1] >= self.bat_pos_B[1] :
                self.ball_pos[1] = self.bat_pos_B[1]
                self.ball_vel[1] *= -1
                hit = -1
                if np.random.choice(2):
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[0] *= 1
               
            elif self.ball_pos[1] >= self.bat_pos_B[1]:
                self.score[0] += 1
                self.ball_pos = np.array([self.board_size[0]/2, self.board_size[1]/2], dtype=np.uint8)
                #self.bat_pos_B[0] = self.ball_pos[0]
                self.ball_vel[1] = 2*(-1)**np.random.randint(0,2)
                if self.ball_vel[1] == 0:
                    self.ball_vel[1] = 1
                self.ball_vel[0] = np.random.randint(-2,3)
                self.round += 1
                win = 1
                
        elif self.QuantumState  == 1:
            C = np.zeros((2,2))
            for i in range(2):
                for j in range(2):
                    C[i,j] = ((-1)**((1-i)*(1-j)))/np.sqrt(2)
            
            if self.ball_pos[0] >= self.bat_pos_A[0] - self.bat_size and self.ball_pos[0] <= self.bat_pos_A[0] + self.bat_size and self.ball_pos[1] <= self.bat_pos_A[1] :
                self.ball_pos[1] = self.bat_pos_A[1]
                if self.td == -1:
                    self.quantum_i = Action_A[1]
                    self.quantum_A = self._draw_A(0.5)
                    self.ball_vel[0] = self.quantum_A
                elif self.td == 1:
                    self.quantum_j = Action_A[1]
                    p = (1+C[self.quantum_i,self.quantum_j])/2
                    x = self._draw_A(p)
                    self.ball_vel[0] = x*self.quantum_A
                hit = 1
                self.ball_memory += 1
                self.ball_vel[1] *= -1
                    
            elif self.ball_pos[1] <= self.bat_pos_A[1]:
                self.score[1] += 1
                self.ball_pos = np.array([self.board_size[0]/2, self.board_size[1]/2], dtype=np.uint8)
                #self.bat_pos_A[0] = self.ball_pos[0] 
                self.ball_vel[1] = 2*(-1)**np.random.randint(0,2)
                if self.ball_vel[1] == 0:
                   self.ball_vel[1] = 1
                self.ball_vel[0] = np.random.randint(-2,3)
                self.round += 1
                win = -1
                self.ball_memory = 0
                self.QuantumState = 0

            
            if self.ball_pos[0] >= self.bat_pos_B[0] - self.bat_size and self.ball_pos[0] <= self.bat_pos_B[0] + self.bat_size and self.ball_pos[1] >= self.bat_pos_B[1] :
                self.ball_pos[1] = self.bat_pos_B[1]
                self.ball_vel[1] *= -1
                if self.td == 1:
                    self.quantum_i = Action_B[1]
                    self.quantum_A = self._draw_A(0.5)
                    self.ball_vel[0] = self.quantum_A
                elif self.td == -1:
                    self.quantum_j = Action_B[1]
                    p = (1+C[self.quantum_i,self.quantum_j])/2
                    x = self._draw_A(p)
                    self.ball_vel[0] = x*self.quantum_A
                hit = -1
                self.ball_memory += 1
               
            elif self.ball_pos[1] >= self.bat_pos_B[1]:
                self.score[0] += 1
                self.ball_pos = np.array([self.board_size[0]/2, self.board_size[1]/2], dtype=np.uint8)
                #self.bat_pos_B[0] = self.ball_pos[0]
                self.ball_vel[1] = 2*(-1)**np.random.randint(0,2)
                if self.ball_vel[1] == 0:
                    self.ball_vel[1] = 1
                self.ball_vel[0] = np.random.randint(-2,3)
                self.round += 1
                win = 1
                self.ball_memory = 0
                self.QuantumState = 0
                
            if self.ball_memory >= 2:
                self.ball_memory = 0
                self.QuantumState = 0
        
        if self.round == 21:
            self.done = True
            self.n_steps = 0
                
        self.n_steps += 1
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
        
        
        
                
                
                
                
        