from CustomGame.channel_env import ZF_80211ax
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
from utils.MU_selection import Q_table_to_action, user_to_action, action_to_user
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
env = ZF_80211ax()

class DQN(nn.Module):
    def __init__(self, input_dims):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.n_actions = 385

        self.linear = nn.Sequential(nn.Linear(10 * 4, 385*4*4),
                                    nn.Linear(385*4*4, 385*4*4))
                                    # nn.Linear(10000, 10000), nn.LeakyReLU(),
                                    # nn.Linear(10000, 10000), nn.LeakyReLU(),
                                    # nn.Linear(10000, 385 * 4 * 4 * 2), nn.LeakyReLU(),)
        self.lrelu = nn.LeakyReLU()

    def forward(self, states):
        x = states.view(-1, 10 * 4)
        x = self.linear(x)
        x = x.reshape([-1,385,4,4])
        # x = x[:,:,:,:,0] + 1j * x[:,:,:,:,1]
        # x = x @ x.transpose(2,3)
        out = x.reshape([-1,385,4,4])

        return out

n_episode = 5000000
BATCH_SIZE = 128
GAMMA = 0.0
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = n_episode / 10
TARGET_UPDATE = 100
input_dims = 10 * 4
lr = args.lr

# Get number of actions from gym action space
n_actions = 385

policy_net = DQN(input_dims).to(device)
target_net = DQN(input_dims).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.SGD(policy_net.parameters(), lr=lr, momentum=0.9)
memory = ReplayMemory(10000)

steps_done = 0

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))

def select_action():
    return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).to(device).to(torch.int64)
    reward_batch = torch.cat(batch.reward).to(device)

    reward_batch = reward_batch.reshape([-1,4*4])

    state_action_values = policy_net(state_batch)[torch.arange(policy_net(state_batch).shape[0]).unsqueeze(-1), action_batch]
    state_action_values = state_action_values.reshape([-1,4*4])

    # reward_batch_reim = torch.cat([reward_batch.real, reward_batch.imag], dim=1)
    reward_batch_reim = reward_batch
    # state_action_values_rein = torch.cat([state_action_values.real, state_action_values.imag], dim=1)
    state_action_values_rein = state_action_values

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(reward_batch_reim, state_action_values_rein)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


def print_progress(i_episode, loss, acc):
    print(f'Episode step: {i_episode}, Avg. loss: {loss:.5f}')

# def print_accuracy(n_itr):
#

for i_episode in range(n_episode):
    # Initialize the environment and state

    obs = env.reset()
    state = torch.tensor(obs, dtype=torch.float).to(device)
    # Select and perform an action
    action = select_action()
    action = action_to_user(action)
    reward = env.step(torch.tensor([action]))

    # Store the transition in memory
    memory.push(state, user_to_action(torch.tensor(action)), reward)

    # Perform one step of the optimization (on the policy network)
    loss = optimize_model()

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        if loss:
            print_progress(i_episode, loss, reward)

print('Complete')
