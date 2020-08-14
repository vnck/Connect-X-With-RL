import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from collections import defaultdict
import math

def get_mcts_agent(player,model_path,num_sims=25):
    model = Connect4Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    mcts = MCTS(Game(),model)
    def agent(observation,config):
        board = reshape4(observation["board"])
        if player == 1:
            board = transform_board(board)
        elif player == 2:
            board = transform_board(board,f=-1,s=1)

        for i in range(num_sims):
            mcts.search(board)
        pi = np.array([mcts.Nsa[str(board)+str(a)]/mcts.Ns[str(board)] for a in range(7)])
        return pi.argmax().item()
    return agent

def get_greedy_agent(player,model_path):
    model = Connect4Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    def agent(observation,config):
        board = reshape4(observation["board"])
        if player == 1:
            board = transform_board(board)
        elif player == 2:
            board = transform_board(board,f=-1,s=1)
        board = torch.tensor(board).float()
        pi,v = model.predict2(board)
        a = pi.argmax()
        
        return a.item()
    return agent

class Game:
    def __init__(self):
        self.board = create_board()
        
    def get_action_size(self):
        return 7
    
    def get_next_state(self,board,player,action):
        board = drop_piece(board,action,player)
        return board,-player
    
    def get_valid_actions(self,board):
        return get_valid_actions(board)
    
    def get_game_ended(self, board, player):
        result = check_winner(board)
        # Win
        if result == player:
            return 1
        # Lose
        elif result == -player:
            return -1
        #Draw
        elif result == 'draw':
            return 1e-5
        # Continue playing game
        elif result == None:
            return 0
        else:
            print("error")
        
    def get_canonical_form(self, board, player):
        # Flip player from 1 to -1 so that NN plays from the 'same' board
        return (np.array(board) * player).tolist()

    def get_symmetry(self, board, pi):
        """Board is left/right board symmetric"""
        reverse_board = [reversed(row) for row in board]
        return [(board, pi), (reverse_board, pi[::-1])]

    def to_string(self, board):
        return str(board)


class MCTS:
    def __init__(self,game,nnet,device=None):
        """
        helper class for Monte Carlo Tree Search
        """
        self.visited = set()
        self.game = game
        self.nnet = nnet
        self.P = {}
        self.Qsa = defaultdict(int)
        self.Ns = defaultdict(int)
        self.Nsa = defaultdict(int)
        self.c_puct = 1
        if device == None:
            self.device = torch.device("cpu")
#             self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
        else:
            self.device = device
            
    def search(self,s):
        """
        Note negative reward is returned as 
        −v is the value of the current board from the perspective of the other player
        """
        self.nnet.eval()
        self.nnet.to(self.device)
        reward = self.game.get_game_ended(s,1)
        if reward!=0: 
            # if game has ended return reward
            return -reward
        
        if str(s) not in self.visited:
            self.visited.add(str(s))
            self.P[str(s)], v = self.nnet.predict(torch.tensor(s).float().to(self.device))
            return -v[0]

        max_u, best_a = -float("inf"), -1
        for a in self.game.get_valid_actions(s):
            sa = str(s) + str(a)
            if sa in self.Qsa:
                u = self.Qsa[sa] + self.c_puct*self.P[str(s)][a]*math.sqrt(self.Ns[str(s)])/(1+self.Nsa[str(s)+str(a)])
            else:
                u = self.c_puct*self.P[str(s)][a]*math.sqrt(self.Ns[str(s)]+1e-8)
                
            if u>max_u:
                max_u = u
                best_a = a
                
        assert best_a != -1
        
        a = best_a
        
        # Player makes move and changes player
        sp,p = self.game.get_next_state(s,1,a)
        # Inverse board so that NN 'thinks' it is playing as the same player 
        sp = self.game.get_canonical_form(sp, p)
        v = self.search(sp)

        self.Qsa[str(s)+str(a)] = (self.Nsa[str(s)+str(a)]*self.Qsa[str(s)+str(a)] + v)/(self.Nsa[str(s)+str(a)]+1)

        self.Nsa[str(s)+str(a)] += 1
        self.Ns[str(s)]+=1
        return -v


def create_board():
    return [[0 for i in range(7)] for j in range(6)]

def reshape4(arr):
    """
    reshape into 6x7 without numpy 
    """
    line1 = arr[0:7]
    line2 = arr[7:14]
    line3 = arr[14:21]
    line4 = arr[21:28]
    line5 = arr[28:35]
    line6 = arr[35:42]
    board = [line1, line2 , line3, line4, line5, line6] 
    return board

def drop_piece(board,col,mark):
    """
    drop piece at next position
    """
    board = copy.deepcopy(board)
    for row in range(6-1, -1, -1):
        if board[row][col] == 0:
            break
    board[row][col] = mark
    return board

def get_valid_actions(board):
    """
    get possible valid actions
    """
    return [c for c in range(0,7) if board[0][c]==0]

def transform_board(board,a=1,b=2,f=1,s=-1):
    """
    Map 1 -> -1 and 2 -> 1
    This is done so that we can easily get the canonical board easily
    """
    return_board = copy.deepcopy(board)
    for i in range(len(board)):
        for j in range(len(board[0])):
            curr = board[i][j]
            if curr == a:
                return_board[i][j] = f
            elif curr == b:
                return_board[i][j] = s
    return return_board

def inverse_transform(board):
    return transform_board(board,1,-1,1,2)

def check_winner(board):
    # Check rows for winner
    for row in range(6):
        for col in range(4):
            if (board[row][col] == board[row][col + 1] == board[row][col + 2] ==\
                board[row][col + 3]) and (board[row][col] != 0):
                return board[row][col]  #Return Number that match row

    # Check columns for winner
    for col in range(7):
        for row in range(3):
            if (board[row][col] == board[row + 1][col] == board[row + 2][col] ==\
                board[row + 3][col]) and (board[row][col] != 0):
                return board[row][col]  #Return Number that match column

    # Check diagonal (top-left to bottom-right) for winner

    for row in range(3):
        for col in range(4):
            if (board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] ==\
                board[row + 3][col + 3]) and (board[row][col] != 0):
                return board[row][col] #Return Number that match diagonal


    # Check diagonal (bottom-left to top-right) for winner

    for row in range(5, 2, -1):
        for col in range(4):
            if (board[row][col] == board[row - 1][col + 1] == board[row - 2][col + 2] ==\
                board[row - 3][col + 3]) and (board[row][col] != 0):
                return board[row][col] #Return Number that match diagonal
    c = 0
    for col in range(7):
        if board[0][col]!=0:
            c +=1
    if c == 7:
        # This is a draw
        return "draw"
    
    # No winner: return None
    return None

def get_random_agent():
    def ra(board):
        return random.choice(get_valid_actions(board))
    return ra

class Connect4Net(nn.Module):
    """
    input: bs x 6 x 7 
    output: pi bs x 7, value bs x 1
    """
    def __init__(self,action_size=7,hidden_size=128):
        super(Connect4Net, self).__init__()
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(1, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1,padding=1)
        self.conv4 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1,padding=1)

        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        self.bn4 = nn.BatchNorm2d(hidden_size)

        self.fc1 = nn.Linear(hidden_size*6*7, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, 6, 7)    
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.hidden_size*6*7)
        
        s = self.fc1(s)
        s = self.fc_bn1(s)
        s = F.relu(s)
        
        s = self.fc2(s)
        s = self.fc_bn2(s)
        s = F.relu(s)
        
        pi = self.fc3(s)
        v = self.fc4(s)
        
        return F.log_softmax(pi, dim=1), torch.tanh(v)
    
    @torch.no_grad()
    def predict(self,s):
        a,b = self.forward(s.unsqueeze(0))
        a = torch.exp(a)
        return a.squeeze(0).cpu().tolist(),b.squeeze(0).cpu().tolist()

    @torch.no_grad()
    def predict2(self,s):
        a,b = self.forward(s.unsqueeze(0))
        a = torch.exp(a)
        return a.squeeze(0).cpu().numpy(),b.squeeze(0).cpu().numpy()