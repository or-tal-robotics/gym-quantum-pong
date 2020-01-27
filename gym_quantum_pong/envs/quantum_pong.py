import numpy as np
import time
import cv2
import pandas as pd
import os
import csv
# ---- Globals ---- #
Pauli_x = np.array([[0,1],[1,0]], dtype = np.float32)
Pauli_z = np.array([[1,0],[0,-1]], dtype = np.float32)

STARTIN_ANGLES = (np.array([0.0,0.25,0.5,0.75,1.0], dtype = np.float32) + 0.01)*np.pi

class Stats():
    def __init__(self):
        self.left_player_theta_mes1 = []
        self.left_player_theta_mes2 = []
        self.right_player_theta_ent = []
        self.right_player_theta_mes1 = []
        self.right_player_theta_mes2 = []
        self.ball_psi_left = []
        self.ball_psi_right = []
        self.ball_polarization = []
        self.left_player_crystal_state = []
        self.right_player_crystal_state = []
        self.win = []
        self.ball_thata_left = []
        self.ball_thata_right = []
        self.left_player_crystal_state_p = []
        self.right_player_crystal_state_p = []
        self.u1 = []
        self.u2 = []
        self.u3 = []
        self.u4 = []
        self.ball_polarization_left = []
        
        self.a_left = []
        
def save_stat(stat, file_path, ex, f = None):
    data = {'left_player_theta_mes1': stat.left_player_theta_mes1,
            'left_player_theta_mes2': stat.left_player_theta_mes2,
            'right_player_theta_ent': stat.right_player_theta_ent,
            'right_player_theta_mes1': stat.right_player_theta_mes1,
            'right_player_theta_mes2': stat.right_player_theta_mes2,
            'ball_psi_left': stat.ball_psi_left,
            'ball_psi_right': stat.ball_psi_right,
            'ball_polarization': stat.ball_polarization,
            'left_player_crystal_state': stat.left_player_crystal_state,
            'right_player_crystal_state': stat.right_player_crystal_state,
            'win': stat.win,
            'ball_thata_left': stat.ball_thata_left,
            'ball_thata_right': stat.ball_thata_right,
            'left_player_crystal_state_p': stat.left_player_crystal_state_p,
            'right_player_crystal_state_p': stat.right_player_crystal_state_p,
            'a_left': stat.a_left,
            'u1': stat.u1,
            'u2': stat.u2,
            'u3': stat.u3,
            'u4': stat.u4,
            'ball_polarization_left': stat.ball_polarization_left
            }
    df = pd.DataFrame(data, columns= ['left_player_theta_mes1',
                                      'left_player_theta_mes2',
                                      'right_player_theta_ent',
                                      'right_player_theta_mes1',
                                      'right_player_theta_mes2',
                                      'ball_psi_left', 
                                      'ball_psi_right',
                                      'left_player_crystal_state',
                                      'right_player_crystal_state',
                                      'win', 
                                      'ball_thata_left', 
                                      'ball_thata_right',
                                      'left_player_crystal_state_p',
                                      'right_player_crystal_state_p',
                                      'a_left',
                                      'u1',
                                      'u2',
                                      'u3',
                                      'u4',
                                      'ball_polarization_left'])
    if ex==False:
        df.to_csv(file_path, sep="\t", quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")
    else:
        df.to_csv(f, sep="\t", quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\", header=False)
        
        
        

def sample_angle():
    d = np.random.binomial(1,0.5)
    if d==1:
        theta = 5*np.pi/4  
    else:
        theta = 3*np.pi/4  
    return theta
    

class Right_Player():
    def __init__(self, board_size,  dtheta_ent = np.pi/2.0, dtheta_mes = np.pi/12.0):
        self.x = board_size[1] - 7
        self.y = board_size[0]/2
        self.theta_mes1 = np.random.choice(STARTIN_ANGLES)
        self.theta_mes2 = np.random.choice(STARTIN_ANGLES)
        #self.theta_ent = np.random.choice(STARTIN_ANGLES)
        self.theta_ent = 0.0
        self.dtheta_mes = dtheta_mes
        self.dtheta_ent = dtheta_ent
        self.score = 0
        self.theta_mesured = 0
        self.crystal_state = np.random.binomial(1,0.5)*np.pi/2.0
        
    def update(self, action):
        if action == 0: # Torret up
            self.theta_mes1 += self.dtheta_mes
        elif action == 1: # Torret up
            self.theta_mes1 -= self.dtheta_mes
        elif action == 2: # Torret up
            self.theta_mes2 += self.dtheta_mes
        elif action == 3: # Torret up
            self.theta_mes2 -= self.dtheta_mes
        elif action == 4: # Torret up
            self.theta_ent += self.dtheta_ent
        elif action == 5: # Torret up
            self.theta_ent -= self.dtheta_ent
        
#        if self.theta_mes>2*np.pi: self.theta_mes = 2*np.pi
#        if self.theta_mes<0: self.theta_mes = 0
    
            

class Left_Player():
    def __init__(self, board_size, dtheta_mes = np.pi/12.0):
        self.x = 6
        self.y = board_size[0]/2
        self.theta_mes1 = np.random.choice(STARTIN_ANGLES)
        self.theta_mes2 = np.random.choice(STARTIN_ANGLES)
        self.dtheta_mes = dtheta_mes
        self.score = 0
        self.theta_mesured = 0
        self.crystal_state = np.random.binomial(1,0.5)*np.pi/2
        
    def update(self, action):
        if action == 0: # Torret up
            self.theta_mes1 += self.dtheta_mes
        elif action == 1: # Torret up
            self.theta_mes1 -= self.dtheta_mes
        elif action == 2: # Torret up
            self.theta_mes2 += self.dtheta_mes
        elif action == 3: # Torret up
            self.theta_mes2 -= self.dtheta_mes
            
       # if self.theta_ent>np.pi/4: self.theta_ent = np.pi/4
       # if self.theta_ent<0: self.theta_ent = 0
#        if self.theta_mes>2*np.pi: self.theta_mes = 2*np.pi
#        if self.theta_mes<0: self.theta_mes = 0
            
class Ball():
    def __init__(self,x, y, V, theta):
        self.V = V
        self.x = x
        self.y = y
        self.theta = theta
        self.quantum_hits = 0
        self.polarization = np.random.uniform(0.0, np.pi)
        self.psi = np.array([1,1,1,1], dtype = np.float32)/2.0
        self.visible = 255
    



class QuantumPong():
    def __init__(self, n_players = 1, board_size = (60,72,60), V = 4, n_rounds = 21, res = 0.2,max_rounds = 200, mode="quantum"):
        self.bat_size = 25
        self.board_size = board_size
        self.board = np.zeros((int(board_size[0]/res),int(board_size[1]/res)))
        self.res = res
        self.ball = Ball(x = board_size[1] - 6, y = board_size[0]/2, V = V, theta = sample_angle())
        self.mode = mode
        self.left_player = Left_Player(board_size)
        self.right_player = Right_Player(board_size)
        self.done = False
        self.n_rounds = n_rounds
        self.round = 0
        self.n_steps = 0
        self.board_angle = np.arctan((board_size[0]-board_size[2])/board_size[1])
        self.quantum_hits = 0
        self.V = V
        self._reset_ball()
        self.stat = Stats()
        self.wall_mem = 0
        self.max_rounds = max_rounds
        self.wall_polarization = True
        
        
    def _reset_ball(self):
        self.ball = Ball(x = self.board_size[1] - 6, y = self.board_size[0]/2, V = self.V, theta = sample_angle())
        
    def _left_player_mesurment(self):
        if np.sin(self.ball.theta) > 0:
            a = np.sin(self.left_player.theta_mes1)*Pauli_z + np.cos(self.left_player.theta_mes1)*Pauli_x
        else:
            a = np.sin(self.left_player.theta_mes2)*Pauli_z + np.cos(self.left_player.theta_mes2)*Pauli_x
        self.stat.a_left.append(a.reshape((-1)))
        self.stat.left_player_theta_mes1.append(self.left_player.theta_mes1)
        self.stat.left_player_theta_mes2.append(self.left_player.theta_mes2)
        self.stat.right_player_theta_ent.append(self.right_player.theta_ent)
        self.stat.ball_psi_left.append(self.ball.psi)
        #print("a:"+str(a))
        b = np.eye(2).astype(np.float32)
        #print("ball.psi:"+str(self.ball.psi))
        g = np.trace((np.outer(self.ball.psi,self.ball.psi)).dot(np.kron(a,b)))
        p = (1 + g)/2.0
        self.stat.left_player_crystal_state_p.append(p)
        crystal_state = np.random.binomial(1,p)*np.pi/2
        self.stat.left_player_crystal_state.append(crystal_state)
#        lamda, u = np.linalg.eig(np.kron(a,b))
        
        
        lamda, u = np.linalg.eig(a)
        idx = lamda.argsort()[::-1]
        
        idx = np.flip(idx)
        lamda = lamda[idx]
        u = u[:,idx]
        u1 = np.kron(u[:,0],np.array([1,0]))
        u2 = np.kron(u[:,0],np.array([0,1]))
        u3 = np.kron(u[:,1],np.array([1,0]))
        u4 = np.kron(u[:,1],np.array([0,1]))
        
        
        
        #print(np.linalg.det(u))
        self.stat.u1.append(u1)
        self.stat.u2.append(u2)
        self.stat.u3.append(u3)
        self.stat.u4.append(u4)
        
        if crystal_state == 0:
            psi = (np.outer(u1,u1) + np.outer(u2,u2)).dot(self.ball.psi)
            #print("psi:"+str(psi))
            #psi = psi + np.random.normal(0,0.0001,4)
            psi = psi/np.linalg.norm(psi)
            #print("psi_n:"+str(psi))
        else:
            psi = (np.outer(u3,u3) + np.outer(u4,u4)).dot(self.ball.psi)
            #print("psi:"+str(psi))
            #psi = psi + np.random.normal(0,0.0001,4)
            psi = psi/np.linalg.norm(psi)
            #print("psi_n:"+str(psi))
        self.stat.ball_psi_right.append(psi)
        
        return  crystal_state , psi
    
    def _right_player_mesurment(self):
        if np.sin(self.ball.theta) > 0:
            a = np.sin(self.right_player.theta_mes1)*Pauli_z + np.cos(self.right_player.theta_mes1)*Pauli_x
        else:
            a = np.sin(self.right_player.theta_mes2)*Pauli_z + np.cos(self.right_player.theta_mes2)*Pauli_x
        b = np.eye(2).astype(np.float32)
        g = np.trace((np.outer(self.ball.psi,self.ball.psi)).dot(np.kron(b,a)))
        
        self.stat.right_player_theta_mes1.append(self.right_player.theta_mes1)
        self.stat.right_player_theta_mes2.append(self.right_player.theta_mes2)
        #print("g:"+str(g))
        p = (1 + g)/2.0
        self.stat.right_player_crystal_state_p.append(p)
        if p<0: p=0
        if p>1: p=1
        crystal_state = np.random.binomial(1,p)*np.pi/2
        return  crystal_state 
    
    def _update_psi(self):
        return np.array([np.cos(self.right_player.theta_ent),0.0,0.0,np.sin(self.right_player.theta_ent)])
    
   
        

    
    

    
        

    
    def _height(self,x):
        m = (self.board_size[2]-self.board_size[0])/self.board_size[1]
        b = self.board_size[0]
        h = x*m+b
        return h

        

        
    
    def _update_board(self):
        self.board = np.zeros((round(self.board_size[1]/self.res),round(self.board_size[0]/self.res)))
        Bx = int(round(self.ball.x/self.res))
        By = int(round(self.ball.y/self.res))
        
        
        #self.board[50:-50,50:-50] = 0

        
        for ii in range(-round(self.bat_size/self.res),round(self.bat_size/self.res)+1):
            for jj in range(4):
                self.board[round(self.left_player.x/self.res)-jj, round(self.left_player.y/self.res)+ii-1] = 127
                self.board[round(self.right_player.x/self.res)+jj, round(self.right_player.y/self.res)+ii-1] = 127
#        
        for ii in range(round(self.board_size[1]/self.res)):
            self.board[ii, round(self._height(ii*self.res)/self.res)-3:round(self._height(ii*self.res)/self.res)-1] = 200
            self.board[ii,0:3] = 200
         
        
        
        
        
       
        
        x_tor = np.array([round(self.left_player.x/self.res), round(self.left_player.x/self.res) + (5/self.res)*np.sin((self.left_player.theta_mes1))]).astype(np.int)
        y_tor = np.array([round(self.left_player.y/self.res)+50, round(self.left_player.y/self.res)+50 + (5/self.res)*np.cos((self.left_player.theta_mes1))]) .astype(np.int)
        self.board = cv2.line(self.board,(y_tor[0],x_tor[0] ),(y_tor[1],x_tor[1] ),255,3)
        x_tor = np.array([round(self.left_player.x/self.res), round(self.left_player.x/self.res) + (8/self.res)*np.sin((self.left_player.theta_mes2))]).astype(np.int)
        y_tor = np.array([round(self.left_player.y/self.res)-50, round(self.left_player.y/self.res)-50 + (8/self.res)*np.cos((self.left_player.theta_mes2))]) .astype(np.int)
        self.board = cv2.line(self.board,(y_tor[0],x_tor[0] ),(y_tor[1],x_tor[1] ),255,3)
        
        
        x_tor = np.array([round(self.right_player.x/self.res), round(self.right_player.x/self.res)  + (5/self.res)*np.sin((self.right_player.theta_mes1))]).astype(np.int)
        y_tor = np.array([round(self.right_player.y/self.res)+100, round(self.right_player.y/self.res)+100 + (5/self.res)*np.cos((self.right_player.theta_mes1))]) .astype(np.int)
        self.board = cv2.line(self.board,(y_tor[0],x_tor[0] ),(y_tor[1],x_tor[1] ),255,3)
        x_tor = np.array([round(self.right_player.x/self.res), round(self.right_player.x/self.res) + (8/self.res)*np.sin((self.right_player.theta_ent))]).astype(np.int)
        y_tor = np.array([round(self.right_player.y/self.res)-100, round(self.right_player.y/self.res)-100 + (8/self.res)*np.cos((self.right_player.theta_ent))]) .astype(np.int)
        self.board = cv2.line(self.board,(y_tor[0],x_tor[0] ),(y_tor[1],x_tor[1] ),255,3)
        x_tor = np.array([round(self.right_player.x/self.res), round(self.right_player.x/self.res) + (8/self.res)*np.sin((self.right_player.theta_mes2))]).astype(np.int)
        y_tor = np.array([round(self.right_player.y/self.res), round(self.right_player.y/self.res) + (8/self.res)*np.cos((self.right_player.theta_mes2))]) .astype(np.int)
        self.board = cv2.line(self.board,(y_tor[0],x_tor[0] ),(y_tor[1],x_tor[1] ),255,3)
        
        #cv2.circle(self.board,(By, Bx), 10, (self.ball.visible,self.ball.visible,self.ball.visible), -1)
        #cv2.circle(self.board,(self.board.shape[1] - By + 1, Bx), 10, (self.ball.visible,self.ball.visible,self.ball.visible), -1)
        cv2.circle(self.board,(round(self.board_size[0]/self.res)//2, Bx), 10, (self.ball.visible,self.ball.visible,self.ball.visible), -1)
        
        
        
        
#        
        
                
                
    def step(self, Action_A, Action_B ):
        hit = 0
        win = 0
        # --- Player A --- #
        self.right_player.update(Action_A)
        
        
        # --- Player B --- #
        self.left_player.update(Action_B)
            
        # --- Ball step --- #
        self.ball.x +=  self.ball.V * np.cos(self.ball.theta)
        self.ball.y +=  self.ball.V * np.sin(self.ball.theta)
        #self.ball.visible -= 20 
            
        # --- Ball --- #
        if self.ball.visible < 0:
            self.ball.visible = 0
            
        if self.ball.y > self._height(self.ball.x):
            self.ball.y = self._height(self.ball.x)
            self.ball.theta = - self.ball.theta - 2*self.board_angle
            
            
        if self.ball.y < 1:
            self.ball.y = 1
            self.ball.theta = - self.ball.theta
            if self.wall_mem == 0:
                self.wall_mem = 1
            elif self.wall_mem == 1:
                self.wall_mem = 0
                if self.wall_polarization == True:
                    self.ball.polarization = self.ball.polarization + np.pi/2 
                    #print("wall!")
            
            
       
                        

        
        if self.ball.x <= self.left_player.x :
            self.stat.ball_thata_left.append(self.ball.theta)
            self.stat.ball_polarization_left.append(self.ball.polarization)
            self.ball.x = self.left_player.x
            hit = 1
            p = 0.5
            self.ball.psi = self._update_psi()
            self.left_player.crystal_state, self.ball.psi = self._left_player_mesurment()
            
            
            if np.random.binomial(1,p) == 1:
                self.ball.theta = np.pi - self.ball.theta
            else:
                self.ball.theta = np.pi + self.ball.theta
            self.ball.polarization = self.left_player.crystal_state
            self.stat.ball_polarization.append(self.ball.polarization)
            self.stat.ball_thata_right.append(self.ball.theta)
#        elif self.ball.x <= self.left_player.x:
#            self.right_player.score += 1
#            self.round += 1
#            win = -1
#            self._reset_ball()
                

            
        
        
    
                
        if self.ball.x >= self.right_player.x :
            self.wall_mem = 0
            self.ball.x = self.right_player.x
            hit = -1
            p = 0.5
            self.right_player.crystal_state = self._right_player_mesurment()
            self.stat.right_player_crystal_state.append(self.right_player.crystal_state)
            g = np.abs(np.cos(self.ball.polarization - self.right_player.crystal_state))
            if np.random.binomial(1,g) == 1:
                win = 1
                self.round += 1            
                self._reset_ball()
            else:
                self.round += 1
                win = -1
                self._reset_ball()
                #print("Lost!")
            
            self.stat.win.append(win)
                

            
       
                
        
        if self.round == self.max_rounds:
            self.done = True
            self.n_steps = 0
            self.round = 0
            if os.path.isfile('stat.csv'):
                with open('stat.csv', 'a') as f:
                    save_stat(self.stat, 'stat.csv', ex = True, f=f)
            else:
                save_stat(self.stat, 'stat.csv', ex = False)
                
        self.n_steps += 1
        self._update_board()
        return [self.left_player.score, self.right_player.score], self.board, self.done, hit, win
    
    
if __name__ == '__main__':

    QP = QuantumPong()
    done = False
    steps = 0
    while not done:
        a = QP.step(np.random.choice(5),np.random.choice(5))
        done = a[2]
        cv2.imshow("board",a[1]/255)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        
                
                
                
                
        