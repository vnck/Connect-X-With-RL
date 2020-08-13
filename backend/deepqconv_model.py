import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DeepModel(nn.Module):
    def __init__(
        self,
        num_states,
        num_actions,
        ):
        super(DeepModel, self).__init__()
        self.conv1 = nn.Conv2d(1,20,(1,1))
        self.conv2 = nn.Conv2d(1,20,(1,7))
        self.conv3 = nn.Conv2d(1,20,(6,1))
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(20*126,128)
        self.output_layer = nn.Linear(128, num_actions)

    def forward(self, x):
        x = x.view(-1,1,6,7)
        self.input_x = x
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x2 = x2.expand(-1,20,6,7)
        x3 = x3.expand(-1,20,6,7)
        x_cat = torch.cat((x1,x2,x3),1)
        x = x_cat.view(-1,20*126)
        x = torch.sigmoid(self.fc(x))
        x = self.output_layer(x)
        return x


class DQN:
    def __init__(
        self,
        num_states=0,
        num_actions=7,
        gamma=0,
        max_experiences=0,
        min_experiences=0,
        batch_size=0,
        lr=0,
        ):
        self.device = torch.device(('cuda'
                                    if torch.cuda.is_available() else 'cpu'
                                   ))
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = DeepModel(num_states,
                               num_actions).to(self.device)
        print(self.model)
#         self.model.conv1.register_backward_hook(self.backward_hook)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(self.device)
        self.experience = {
            's': [],
            'a': [],
            'r': [],
            's2': [],
            'done': [],
            }
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(torch.from_numpy(inputs).float().to(self.device))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            # only start training process if enough experiences in buffer
            return 0

        # randomly select n experiences in buffer to form batch
        ids = np.random.randint(low=0, high=len(self.experience['s']),
                                size=self.batch_size)
        states = np.asarray([self.preprocess(self.experience['s'][i])
                            for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])

        # prepare labels
        states_next = np.asarray([self.preprocess(self.experience['s2'
                                 ][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = \
            np.max(TargetNet.predict(states_next).detach().cpu().numpy(),
                   axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma
                                 * value_next)

        actions = np.expand_dims(actions, axis=1)
        actions_one_hot = torch.FloatTensor(self.batch_size,
                self.num_actions).zero_()
        actions_one_hot = actions_one_hot.scatter_(1,
                torch.LongTensor(actions), 1).to(self.device)
        selected_action_values = torch.sum(self.predict(states)
                * actions_one_hot, dim=1).to(self.device)
        actual_values = torch.FloatTensor(actual_values).to(self.device)

        self.optimizer.zero_grad()
        loss = self.criterion(selected_action_values, actual_values)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, epsilon):
        # to get an action by using epsilon-greedy
        if np.random.random() < epsilon:
            return int(np.random.choice([c for c in range(self.num_actions) if state['board'][c] == 0]))
        else:
            prediction = \
                self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().cpu().numpy()
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

    def copy_weights(self, TrainNet):
        self.model.load_state_dict(TrainNet.model.state_dict())

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def preprocess(self, state):
        # each state consists of overview of the board and the mark in the obsevations
        # results = (state['board'])[:]
        # results.append(state.mark)
        # return results
        board = (state['board'])[:]
        if state.mark == 1:
            board[board == 2] = -1
        else:
            board[board == 1] = -1
            board[board == 2] = 1
        return board

    def backward_hook(self, module, grad_in, grad_out):
        print(grad_out[0].shape)


model = DQN(num_actions=7)
model.load_weights('weights/weights-deepqconv-12082339.pth')

def my_agent(observation, configuration):
  return model.get_action(observation, 0.0)