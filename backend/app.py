from flask import Flask,request
from flask_cors import CORS
import gym
from kaggle_environments import make
from dqn_dense_model import get_dense_agent
from azero import get_mcts_agent, get_greedy_agent
from kaggle_environments.envs.connectx.connectx import negamax_agent
from kaggle_environments.envs.connectx.connectx import random_agent
import numpy as np

app = Flask(__name__)
env = None
observation = None
playing = None

class ConnectX(gym.Env):
    def __init__(self,opponent_agent,player):
        self.env = make('connectx', debug=True)
        if player == 1:
            self.pair = [None,opponent_agent]
        elif player == 2:
            self.pair = [opponent_agent,None]
        self.trainer = self.env.train(self.pair)

        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns
                * config.rows)

    def step(self, action):
        return self.trainer.step(action)

    def reset(self):
        return self.trainer.reset()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

@app.route("/createboard",methods=["POST"])
def create_board():
    global env
    global observation
    global playing
    global assist_agent
    json_post = request.get_json()
    # agent_name = json_post["agent_name"]
    player = json_post["player"]
    agents_dict = {
        -1: None,
        2: get_dense_agent('weights-DQN.pth'),
        3: get_mcts_agent(player, 'azero_60.pth'),
        4: get_greedy_agent(player, 'azero_60.pth'),
        5: negamax_agent,
        6: random_agent,
    }
    opponent_agent = agents_dict[json_post["opponent_agent"]]
    assist_agent = agents_dict[json_post["assist_agent"]]
    env = ConnectX(opponent_agent, player)
    observation = env.reset()
    playing = "playing"
    return 'ok',200

@app.route("/getboard")
def get_board():
    global observation
    global env
    global playing
    if playing == None:
        return 'Please call create board first',400
    board = observation["board"]
    valid_actions = [c for c in range(0,7) if board[c]==0]
    board = np.array(board).reshape(6,7).tolist()
    mark = observation["mark"]
    return {"board":board,"mark":mark,"playing":playing,"valid_actions":valid_actions, "assist": assist_agent is not None}

@app.route("/getpiece")
def get_piece():
    global observation
    global playing
    global env
    global assist_agent
    if playing == None:
        return 'Please call create board first',400
    action =  assist_agent(observation, env.env.configuration)
    return {"action": action}

@app.route("/setpiece",methods=["POST"])
def set_piece():
    json_post = request.get_json()
    action = json_post["action"]
    global env
    global playing
    global observation
    (obs, reward, done, _) = env.step(action)
    print(obs, reward, done)
    observation["board"] = obs["board"]
    observation["mark"] = obs["mark"]
    if done and reward ==1:
        playing = "win"
    elif done and reward == 0:
        playing = "draw"
    elif done and reward ==-1:
        playing = "lose"
    else:
        playing = playing
    return {"reward":reward}


if __name__ == "__main__":
    CORS(app)
    app.run(debug=True,host='localhost')

# env = ConnectX("negamax")
# observations = env.reset()
# done = False
# while not done:
#     curr_action = int(input("Enter your input: "))
#     prev_observations = observations
#     (observations, reward, done, _) = env.step(curr_action)
    
#     print(observations)
    
#     b = observations["board"]
#     print(np.array(b).reshape(6,7))
#     print(reward)
#     print(done)
#     env.render()