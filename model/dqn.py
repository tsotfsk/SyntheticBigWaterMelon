import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import xavier_normal_
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class Net(nn.Module):
    def __init__(self, in_channels=1, num_actions=4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32, kernel_size=3, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=8, kernel_size=3, stride=1)

        self.pool2d = nn.MaxPool2d(kernel_size=4, stride=2)

        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_actions)

        self.ball_embed = nn.Embedding(5, 32)

        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, obs, ball_id):
        obs = self.relu(self.pool2d(self.conv1(obs)))
        obs = self.relu(self.pool2d(self.conv2(obs)))
        obs = self.relu(self.pool2d(self.conv3(obs)))
        obs = obs.view(obs.size(0), -1)
        obs = self.relu(self.fc1(obs))
        obs = self.fc2(obs)

        be = self.ball_embed(ball_id).squeeze(1)
        prob = (be * obs)
        return prob


class DQN():

    capacity = 2000
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 256
    gamma = 0.8
    update_count = 0

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net(), Net()
        self.memory = [None] * self.capacity
        self.optimizer = optim.Adam(
            self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.num_actions = 32

    def select_action(self, state, ball):
        state = state.unsqueeze(0)
        ball = torch.tensor(ball).long()
        value = self.act_net(state, ball)
        _, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= 0.9:  # epslion greedy
            action = np.random.choice(self.num_actions, 1)[0]
        return action

    def store_transition(self, transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        if self.memory_count >= self.capacity:
            state = torch.stack([t.state for t in self.memory]).float()
            ball = torch.tensor([t.ball for t in self.memory]).long()
            action = torch.LongTensor(
                [t.action for t in self.memory]).view(-1, 1).long()
            reward = torch.tensor([t.reward for t in self.memory]).float()
            next_state = torch.stack(
                [t.next_state for t in self.memory]).float()
            next_ball = torch.tensor(
                [t.next_ball for t in self.memory]).long()

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                target_v = reward + self.gamma * self.target_net(next_state, next_ball).max(1)[0]

            # Update...
            total_loss = 0
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))),
                                      batch_size=self.batch_size, drop_last=False):
                v = self.act_net(state[index], ball[index])
                v = v.gather(1, action[index])
                loss = self.loss_func(target_v[index].unsqueeze(
                    1), v)
                self.optimizer.zero_grad()
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.update_count += 1
                if self.update_count % 100 == 0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
            return total_loss
        else:
            print(f"Memory Buff is too less. {self.memory_count}/{self.capacity}")
            return None


if __name__ == '__main__':
    inp = torch.randn((1, 3, 450, 640))
    dqn = Net()
    dqn(inp)
