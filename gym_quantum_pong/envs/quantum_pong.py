import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

class Player():
    def __init__(self, x, board_size, bat_size, dtheta = 15, dy = 3):
        self.x = x
        self.y = np.random.randint(bat_size+1, board_size[0] - bat_size-2)
        self.theta = np.random.uniform(0,2*np.pi)
        self.dy = dy
        self.dtheta = dtheta
        self.y_max = board_size[0]-bat_size-1
        self.y_min = bat_size
        self.score = 0
        
    def update(self, action):
        if action == 0: # Move up
            self.y += self.dy
            if self.y > self.y_max:
                self.y = self.y_max
        elif action == 1: # Move down
            self.y -= self.dy
            if self.y < self.y_min:
                self.y = self.y_min
        elif action == 3: # Torret up
            self.theta += self.dtheta
        elif action == 4: # Torret up
            self.theta -= self.dtheta
            
class Ball():
    def __init__(self, x, y, vx, vy, board_size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.quantum_hits = 0
    



class QuantumPong():
    def __init__(self, n_players = 1, board_size = (84,84,70), V = [2,2], n_rounds = 21):
        self.bat_size = 6
        self.board_size = board_size
        self.board = np.zeros((board_size[0],board_size[1]))
        self.board_p1 = np.zeros((board_size[0],board_size[1]))
        self.board_p2 = np.zeros((board_size[0],board_size[1]))
        self.ball_pos = np.array([board_size[0]/2, board_size[1]/2], dtype=np.uint8)
        self.left_player = Player(6, board_size, self.bat_size)
        self.right_player = Player(board_size[1] - 6, board_size, self.bat_size)
        self.ball_vel = np.array(V, dtype = np.int8)
        self.done = False
        self.n_rounds = n_rounds
        self.round = 0
        self.n_steps = 0
        

    
    def _height(self,x):
        m = (self.board_size[2] - self.board_size[0])/self.board_size[1]
        b = self.board_size[0]
        h = int(m*x+b)
        return h

        

        
    
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
                self.board[self.left_player.y+ii-1, self.left_player.x-jj] = 127
                self.board[self.right_player.y+ii-1, self.right_player.x+jj] = 127
        
        for ii in range(self.board_size[1]):
            self.board[self._height(ii)-1, ii] = 200
            self.board[0, ii] = 200
         
        self.board_temp = np.zeros((self.board_size[0],self.board_size[1]))    
        self.board_temp = self.board.copy()
        
        x_tor = np.array([self.left_player.x, self.left_player.x + 5*np.sin(np.deg2rad(self.left_player.theta))]).astype(np.int)
        y_tor = np.array([self.left_player.y, self.left_player.y + 5*np.cos(np.deg2rad(self.left_player.theta))]) .astype(np.int)
        self.board_p1 = cv2.line(self.board_temp,(x_tor[0],y_tor[0] ),(x_tor[1],y_tor[1] ),255,1)
        
        self.board_temp = np.zeros((self.board_size[0],self.board_size[1]))    
        self.board_temp = self.board.copy()
        self.board = self.board_p1.copy()
        
        x_tor = np.array([self.right_player.x, self.right_player.x + 5*np.sin(np.deg2rad(self.right_player.theta))]).astype(np.int)
        y_tor = np.array([self.right_player.y, self.right_player.y + 5*np.cos(np.deg2rad(self.right_player.theta))]) .astype(np.int)
        self.board_p2 = cv2.line(self.board_temp,(x_tor[0],y_tor[0] ),(x_tor[1],y_tor[1] ),255,1)
        self.board = cv2.line(self.board,(x_tor[0],y_tor[0] ),(x_tor[1],y_tor[1] ),255,1)
        
        
                
                
    def step(self, Action_A, Action_B ):
        hit = 0
        win = 0
        # --- Player A --- #
        self.left_player.update(Action_A)
        
        
        # --- Player B --- #
        self.right_player.update(Action_B)
            
        # --- Ball step --- #
        self.ball_pos =  self.ball_pos + self.ball_vel
            
        # --- Ball --- #
        if self.ball_pos[0] > self._height(self.ball_pos[1]):
            self.ball_pos[0] = self._height(self.ball_pos[1])
            self.ball_vel[0] *= -1
            
            
        if self.ball_pos[0] < 1:
            self.ball_pos[0] = 1
            self.ball_vel[0] *= -1
            
       
                        

        
        if self.ball_pos[0] >= self.left_player.y - self.bat_size and self.ball_pos[0] <= self.left_player.y + self.bat_size and self.ball_pos[1] <= self.left_player.x :
            self.ball_pos[1] = self.left_player.x
            self.ball_vel[1] *= -1
            hit = 1
            if np.random.choice(2):
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[0] *= 1
                
        elif self.ball_pos[1] <= self.left_player.x:
            self.right_player.score += 1
            self.ball_pos = np.array([self.board_size[0]/2, self.board_size[1]/2], dtype=np.uint8)
            #self.bat_pos_A[0] = self.ball_pos[0] 
            self.ball_vel[1] = 2*(-1)**np.random.randint(0,2)
            if self.ball_vel[1] == 0:
               self.ball_vel[1] = 1
            self.ball_vel[0] = np.random.randint(-2,3)
            self.round += 1
            win = -1
                
        if self.ball_pos[0] >= self.right_player.y - self.bat_size and self.ball_pos[0] <= self.right_player.y + self.bat_size and self.ball_pos[1] >= self.right_player.x :
            self.ball_pos[1] = self.right_player.x
            self.ball_vel[1] *= -1
            hit = -1
            if np.random.choice(2):
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[0] *= 1
           
        elif self.ball_pos[1] >= self.right_player.x:
            self.left_player.score += 1
            self.ball_pos = np.array([self.board_size[0]/2, self.board_size[1]/2], dtype=np.uint8)
            #self.bat_pos_B[0] = self.ball_pos[0]
            self.ball_vel[1] = 2*(-1)**np.random.randint(0,2)
            if self.ball_vel[1] == 0:
                self.ball_vel[1] = 1
            self.ball_vel[0] = np.random.randint(-2,3)
            self.round += 1
            win = 1
                
        
        if self.round == 21:
            self.done = True
            self.n_steps = 0
                
        self.n_steps += 1
        self._update_board()
        return [self.left_player.score, self.right_player.score], [self.board_p1, self.board_p2, self.board], self.done, hit, win
    
    
if __name__ == '__main__':
    #plt.ion()
    #fig = plt.figure()
    QP = QuantumPong()
    done = False
    steps = 0
    while not done:
        a = QP.step(3,4)
        print(a[0])
        done = a[2]
        steps +=1
        plt.imshow(a[1][0])
        #plt.imshow(a[1])
        #plt.imshow(a[1])
        #fig.canvas.draw()
        #time.sleep(0.01)
        #fig.canvas.flush_events()
        
        
        
                
                
                
                
        