import numpy as np
import time
import cv2

# ---- Globals ---- #
Pauli_x = np.array([[0,1],[1,0]])
Pauli_z = np.array([[1,0],[0,-1]])

def sample_angle():
    d = np.random.binomial(1,0.5)
    if d==1:
        theta = 5*np.pi/4  
    else:
        theta = 3*np.pi/4  
    return theta
    

class Left_Player():
    def __init__(self, board_size, dtheta = np.pi/12):
        self.x = 6
        self.y = board_size[0]/2
        self.theta_mes = np.random.uniform(0,2*np.pi)
        self.dtheta = dtheta
        self.score = 0
        self.theta_mesured = 0
        self.crystal_state = np.random.binomial(1,0.5)*np.pi/2
        
    def update(self, action):
        if action == 0: # Torret up
            self.theta_mes += self.dtheta
        elif action == 1: # Torret up
            self.theta_mes -= self.dtheta
        
        if self.theta_mes>2*np.pi: self.theta_mes = 2*np.pi
        if self.theta_mes<0: self.theta_mes = 0
            

class Right_Player():
    def __init__(self, board_size, dtheta_ent = np.pi/12, dtheta_mes = np.pi/12):
        self.x = board_size[1] - 6
        self.y = board_size[0]/2
        self.theta_ent = np.random.uniform(0,2*np.pi)
        self.theta_mes = np.random.uniform(0,2*np.pi)
        self.dtheta_ent = dtheta_ent
        self.dtheta_mes = dtheta_mes
        self.score = 0
        self.theta_mesured = 0
        self.crystal_state = np.random.binomial(1,0.5)*np.pi/2
        
    def update(self, action):
        if action == 0: # Torret up
            self.theta_ent += self.dtheta_ent
        elif action == 1: # Torret up
            self.theta_ent -= self.dtheta_ent
        if action == 2: # Torret up
            self.theta_mes += self.dtheta_mes
        elif action == 3: # Torret up
            self.theta_mes -= self.dtheta_mes
            
        if self.theta_ent>np.pi/4: self.theta_ent = np.pi/4
        if self.theta_ent<0: self.theta_ent = 0
        if self.theta_mes>2*np.pi: self.theta_mes = 2*np.pi
        if self.theta_mes<0: self.theta_mes = 0
            
class Ball():
    def __init__(self,x, y, V, theta):
        self.V = V
        self.x = x
        self.y = y
        self.theta = theta
        self.quantum_hits = 0
        self.polarization = np.random.uniform(0.0, np.pi)
        self.psi = np.array([1,0,0,0])
        self.visible = 255
    



class QuantumPong():
    def __init__(self, n_players = 1, board_size = (60,72,60), V = 2, n_rounds = 21, res = 0.2, mode="quantum"):
        self.bat_size = 25
        self.board_size = board_size
        self.board = np.zeros((int(board_size[0]/res),int(board_size[1]/res)))
        self.res = res
        self.ball = Ball(x = board_size[0] - 6, y = board_size[1]/2, V = V, theta = sample_angle())
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
        
    def _reset_ball(self):
        self.ball = Ball(x = self.board_size[1] - 6, y = self.board_size[0]/2, V = self.V, theta = sample_angle())
        
    def _left_player_mesurment(self):
        a = np.sin(self.left_player.theta_mes)*Pauli_z + np.cos(self.left_player.theta_mes)*Pauli_x
        b = np.eye(2)
        g = np.trace((np.outer(self.ball.psi,self.ball.psi)).dot(np.kron(a,b)))
        p = (1 + g)/2.0
        return  np.random.binomial(1,p)*np.pi/2
    
    def _right_player_mesurment(self):
        a = np.sin(self.right_player.theta_mes)*Pauli_z + np.cos(self.right_player.theta_mes)*Pauli_x
        b = np.eye(2)
        g = np.trace((np.outer(self.ball.psi,self.ball.psi)).dot(np.kron(a,b)))
        p = (1 + g)/2.0
        crystal_state = np.random.binomial(1,p)*np.pi/2
        lamda, u = np.linalg.eig(np.kron(a,b))
        if crystal_state == 1:
            psi = (np.outer(u[0],u[0]) + np.outer(u[1],u[1])).dot(self.ball.psi)
            psi = psi/np.linalg.norm(psi)
        else:
            psi = (np.outer(u[2],u[2]) + np.outer(u[3],u[3])).dot(self.ball.psi)
            psi = psi/np.linalg.norm(psi)
        
        return  crystal_state , psi
    
    def _update_psi(self):
        return np.array([np.cos(self.ball.polarization),0,0,np.sin(self.ball.polarization)])
    
   
        

    
    

    
        

    
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
         
        
        x_tor = np.array([round(self.left_player.x/self.res), round(self.left_player.x/self.res) + (5/self.res)*np.sin((self.left_player.theta_mes))]).astype(np.int)
        y_tor = np.array([round(self.left_player.y/self.res), round(self.left_player.y/self.res) + (5/self.res)*np.cos((self.left_player.theta_mes))]) .astype(np.int)
        self.board = cv2.line(self.board,(y_tor[0],x_tor[0] ),(y_tor[1],x_tor[1] ),255,3)
        
       
        x_tor = np.array([round(self.right_player.x/self.res), round(self.right_player.x/self.res)  + (5/self.res)*np.sin((self.right_player.theta_mes))]).astype(np.int)
        y_tor = np.array([round(self.right_player.y/self.res)+50, round(self.right_player.y/self.res) +50+ (5/self.res)*np.cos((self.right_player.theta_mes))]) .astype(np.int)
        self.board = cv2.line(self.board,(y_tor[0],x_tor[0] ),(y_tor[1],x_tor[1] ),255,3)
        
        x_tor = np.array([round(self.right_player.x/self.res), round(self.right_player.x/self.res)  + (5/self.res)*np.sin((self.right_player.theta_ent))]).astype(np.int)
        y_tor = np.array([round(self.right_player.y/self.res)-50, round(self.right_player.y/self.res)-50 + (5/self.res)*np.cos((self.right_player.theta_ent))]) .astype(np.int)
        self.board = cv2.line(self.board,(y_tor[0],x_tor[0] ),(y_tor[1],x_tor[1] ),255,3)
        
        
        if np.cos(self.ball.theta)<0:
            self.right_output_board = self.board.copy()
        
        
        cv2.circle(self.board,(By, Bx), 10, (self.ball.visible,self.ball.visible,self.ball.visible), -1)
        self.left_output_board = self.board.copy()
        if np.cos(self.ball.theta)>=0:
            self.right_output_board = self.board.copy()
        
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
            
       
                        

        
        if self.ball.x <= self.left_player.x :
            self.ball.x = self.left_player.x
            hit = 1
            p = 0.5
            self.ball.psi = self._update_psi()
            self.left_player.crystal_state = self._left_player_mesurment()
            if np.random.binomial(1,p) == 1:
                self.ball.theta = np.pi - self.ball.theta
            else:
                self.ball.theta = np.pi + self.ball.theta
            self.ball.polarization = self.left_player.crystal_state
            
#        elif self.ball.x <= self.left_player.x:
#            self.right_player.score += 1
#            self.round += 1
#            win = -1
#            self._reset_ball()
                

            
        
        
    
                
        if self.ball.x >= self.right_player.x :
            self.ball.x = self.right_player.x
            hit = -1
            p = 0.5
            self.right_player.crystal_state, self.ball.psi = self._right_player_mesurment()
            g = np.cos(self.ball.polarization - self.right_player.crystal_state)
            if np.random.binomial(1,g) == 1:
                win = 1
                self.round += 1
                #print("won!")
                if np.random.binomial(1,p) == 1:
                    self.ball.theta = np.pi - self.ball.theta
                else:
                    self.ball.theta = np.pi + self.ball.theta
            else:
                self.round += 1
                win = -1
                self._reset_ball()
                #print("Lost!")
                

            
       
                
        
        if self.round == 10:
            self.done = True
            self.n_steps = 0
            self.round = 0
                
        self.n_steps += 1
        self._update_board()
        return [self.left_player.score, self.right_player.score], [self.right_output_board, self.left_output_board], self.done, hit, win
    
    
if __name__ == '__main__':

    QP = QuantumPong()
    done = False
    steps = 0
    while not done:
        a = QP.step(3,4)
        done = a[2]
        cv2.imshow("right_output_board",a[1][0]/255)
        cv2.imshow("left_output_board",a[1][1]/255)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        
                
                
                
                
        