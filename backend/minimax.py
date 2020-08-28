import numpy as np
import numba as nb

@nb.njit
def count_nb(arr, value):
    result = 0
    for x in arr:
        if x == value:
            result += 1
    return result

@nb.njit
def value_fn(board,player):
    if player == 1:
        opp_player = 2
    elif player ==2:
        opp_player =1
    return _value_fn(board,player)-5*_value_fn(board,opp_player)
    
@nb.njit
def _value_fn(board,player):
    score = 0
    
    # Value of center column
    score+= 101*count_nb(board[:,3],player)
    
    # Count score for each row
    for r in range(6):
        row_array = board[r,:]
        for c in range(4):
            window = row_array[c:c+4]
            score += evaluate_row(window, player,r)
    
    # Count score for each column
    for c in range(7):
        col_array = board[:,c]
        for r in range(3):
            window = col_array[r:r+4]
            score += evaluate_column(window, player)
    
    # Count score on each diagonal
    # Forward Diagonal
    for r in range(3):
        for c in range(4):
            window = np.array([board[r+i][c+i] for i in range(4)])
            score += evaluate_diagonal(window, player,r)
    
    # Backward Diagonal
    for r in range(3):
        for c in range(4):
            window = np.array([board[r+3-i][c+i] for i in range(4)])
            score += evaluate_diagonal(window, player,r)
            
    return score

@nb.njit
def evaluate_diagonal(window,player,r):
    score = 0
    # Lower diagonal is better
    # Doesnt check if empty place == 0 for given row but will do for now
    inverse_row = 7-r
    score += inverse_row #ranges from 1-7
    if count_nb(window,player) == 4:
        score += 10000
    elif count_nb(window,player) == 3 and count_nb(window,0)==1:
        score += 100
    elif count_nb(window,player) == 2 and count_nb(window,0)==2:
        score += 10
    return score

@nb.njit
def evaluate_row(window,player,r):
    score = 0
    
    # Lower row is better
    inverse_row = 7-r
    score += inverse_row #ranges from 1-7
    
    # Weighs higher on forming rows on odd number rows for player 1
    # and even rows for player 2 
    # See connect4 even odd strategy
    if player == 1 and r%2==0:
        score +=10
    if player == 2 and r%2!=0:
        score +=10
    
    if count_nb(window,player) == 4:
        score += 10000
    elif count_nb(window,player) == 3 and count_nb(window,0)==1:
        score += 100
    elif count_nb(window,player) == 2 and count_nb(window,0)==2:
        score += 10
    return score

@nb.njit
def evaluate_column(window,player):
    score = 0
    if count_nb(window,player) == 4:
        score += 10000
    elif count_nb(window,player) == 3 and count_nb(window,0)==1:
        score += 100
    elif count_nb(window,player) == 2 and count_nb(window,0)==2:
        score += 10
    return 0.5*score

@nb.njit
def get_valid_actions(board):
    """
    get possible valid actions
    """
    return [c for c in range(0,7) if board[0][c]==0]

@nb.njit
def drop_piece(board,col,mark):
    """
    drop piece at next position
    """
    board = board.copy()
    for row in range(6-1, -1, -1):
        if board[row][col] == 0:
            break
    board[row][col] = mark
    return board

@nb.njit
def check_winner(board):
    """
    Returns player that wins
    -1 if draws
    0 if game has not ended
    """
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
        return -1
    # No winner: return None
    return 0

@nb.njit
def alphabeta(node,depth,alpha,beta,max_player,player,ai_player):
    winner = check_winner(node)
    if depth == 0 or winner !=0:
        if winner == ai_player:
            return None, 999999999
        elif winner == player:
            return None, -999999999
        elif winner == -1:
            return None,0
        else:
            value = value_fn(node,ai_player)
            return None,value
    
    if max_player:
        best_value = -9999999999999
        best_action = 3
        for action in get_valid_actions(node):
            child = drop_piece(node,action,ai_player)
            score = alphabeta(child,depth-1,alpha,beta,False,player,ai_player)[1]
            if score > best_value:
                best_value = score
                best_action = action
            alpha = max(alpha,best_value)
            if alpha>=beta:
                break
#         print(best_action)
        return best_action,best_value
    
    else:
        worst_value = 9999999999999
        worst_action = 3
        for action in get_valid_actions(node):
            child = drop_piece(node,action,player)
            score = alphabeta(child,depth-1,alpha,beta,True,player,ai_player)[1]
            if score < worst_value:
                worst_value = score
                worst_action = action
            beta = min(beta,worst_value)
            if beta<=alpha:
                break
        return worst_action,worst_value

def my_agent(observation,config):
    board = np.array(observation["board"]).reshape(6,7)
    player = observation["mark"]
    if player == 1:
        opp_player =2
    else:
        opp_player =1
    action,value = alphabeta(board,4,-9999999999999,9999999999999,True,opp_player,player)
    return action

def get_minimax_agent(depth=7):
    def minimax_agent(observation,config):
        board = np.array(observation["board"]).reshape(6,7)
        player = observation["mark"]
        if player == 1:
            opp_player =2
        else:
            opp_player =1
        action,value = alphabeta(board,depth,-9999999999999,9999999999999,True,opp_player,player)
        return action
    return minimax_agent