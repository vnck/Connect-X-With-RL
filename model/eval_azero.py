import numpy as np
from kaggle_environments import evaluate
import azero
from nega_agents import return_nega

def mean_reward(rewards):
    return np.round(rewards.count([1,-1])/len(rewards),2)


def evaluation(my_agent,agent_name,num_episodes=100):
    print(f"{agent_name} vs. Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=num_episodes)))
    print(f"{agent_name} vs. Negamax1 Agent:", mean_reward(evaluate("connectx", [my_agent, negamax_agent1], num_episodes=num_episodes)))
    print(f"{agent_name} vs. Negamax2 Agent:", mean_reward(evaluate("connectx", [my_agent, negamax_agent2], num_episodes=num_episodes)))
    print(f"{agent_name} vs. Negamax3 Agent:", mean_reward(evaluate("connectx", [my_agent, negamax_agent3], num_episodes=num_episodes)))
    print(f"{agent_name} vs. Negamax4 Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=num_episodes)))
    print(f"Random Agent vs. {agent_name}:", mean_reward(evaluate("connectx", ["random", my_agent], num_episodes=num_episodes)))
    print(f"Negamax1 Agent vs. {agent_name}:", mean_reward(evaluate("connectx", [negamax_agent1, my_agent], num_episodes=num_episodes)))
    print(f"Negamax2 Agent vs. {agent_name}:", mean_reward(evaluate("connectx", [negamax_agent2, my_agent], num_episodes=num_episodes)))
    print(f"Negamax3 Agent vs. {agent_name}:", mean_reward(evaluate("connectx", [negamax_agent3, my_agent], num_episodes=num_episodes)))
    print(f"Negamax4 Agent vs. {agent_name}:", mean_reward(evaluate("connectx", ["negamax", my_agent], num_episodes=num_episodes)))


if __name__ == "__main__":
    negamax_agent1 = return_nega(1)
    negamax_agent2 = return_nega(2)
    negamax_agent3 = return_nega(3)
    NUM_EPISODES = 10
    MCTS_agent = azero.get_mcts_agent("azero_final.pth")
    greedy_agent = azero.get_greedy_agent("azero_final.pth")
    print("Evaluation for AlphaZero")
    print('-'*10 + "Alpha Zero greedy policy action"+"-"*10)
    evaluation(greedy_agent,"AlphaZero_greedy")
    print('-'*10 + "Alpha Zero with Monte Carlo Tree Search"+"-"*10)
    evaluation(MCTS_agent,"AlphaZero_MCTS")