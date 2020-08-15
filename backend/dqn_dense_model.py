import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DeepModel(nn.Module):
    def __init__(self, num_states, num_actions, hidden_units):
        super(DeepModel, self).__init__()
        self.hidden_layers = nn.ModuleList([])
        for i in range(len(hidden_units)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(num_states, hidden_units[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
        self.output_layer = nn.Linear(hidden_units[-1], num_actions)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x
    
class DQN():
    def __init__(self, num_states, num_actions, hidden_units, gamma=0.99, max_experiences=10000, min_experiences=100, batch_size=32, lr=1e-2):
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = DeepModel(num_states, num_actions, hidden_units).to(self.device)
        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(self.device)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.loss_hist = []
    
    def predict(self, inputs):
        return self.model(torch.from_numpy(inputs).float().to(self.device))
    
    def train(self, net):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        # randomly select n experiences in buffer to form batch
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        # prepare labels
        states_next = np.asarray([self.preprocess(self.experience['s2'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        pred_values = np.max(net.predict(states_next).detach().cpu().numpy(),axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * pred_values)
        actual_values = torch.FloatTensor(actual_values).to(self.device)
        actions = np.expand_dims(actions, axis=1)
        actions_onehot = torch.FloatTensor(self.batch_size, self.num_actions).zero_()
        actions_onehot = actions_onehot.scatter_(1, torch.LongTensor(actions), 1).to(self.device)
        self.optimizer.zero_grad()
        selected_action_values = torch.sum(self.predict(states) * actions_onehot, dim=1).to(self.device)
        loss = self.criterion(selected_action_values, actual_values)
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return int(np.random.choice([c for c in range(self.num_actions) if state['board'][c] == 0]))
        else:
            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().cpu().numpy()
            for i in range(self.num_actions):
                if state['board'][i] != 0:
                    prediction[i] = -1e7
            return int(np.argmax(prediction))
    
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for (key, value) in exp.items():
            self.experience[key].append(value)
    
    def copy_weights(self, net):
        self.model.load_state_dict(net.model.state_dict())
    
    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
    def preprocess(self, state):
        board = (state['board'])[:]
        if state.mark == 1:
            board[board == 2] = -1
        else:
            board[board == 1] = -1
            board[board == 2] = 1
        return board

def get_dense_agent(model_path):
    num_states = 42
    num_actions = 7
    hidden_units = [128, 128, 128, 128]
    model = DQN(num_states,num_actions,hidden_units)
    model.load_weights(model_path)
    def dense_agent(observation, configuration):
        return model.get_action(observation, 0.0)
    return dense_agent