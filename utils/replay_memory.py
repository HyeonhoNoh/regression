# import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity=20000, alpha=0.4, epsilon_per=0.001, beta=0.7):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon_per
        self.prob_bean = []
        self.alpha_decay = (self.alpha - 0.0) / 3000
        self.beta_increasing = (1.0 - self.beta) / 3000 / 30

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            if len(self.prob_bean) == 0:
                self.prob_bean.append(self.epsilon)
            else:
                try:
                    self.prob_bean.append(max(self.prob_bean))
                except ValueError:
                    print('a')

        # if len(self.memory) == 0:
        #     prob = 0.001
        # else:
        #     prob = max(self.memory, key=lambda k: k.loss_weight).loss_weight

        self.memory[self.position] = Transition(*args)

        if len(self.memory) == self.capacity:
            self.prob_bean[self.position] = max(self.prob_bean)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        tmp = (np.array(self.prob_bean, dtype=float) + 1e-5) ** self.alpha
        try:
            idx = np.random.choice(range(len(self.memory)), size=batch_size, replace=False, p=tmp / np.sum(tmp))
        except ValueError:
            print('a')

        batch = list()
        bean = list()
        for i in range(batch_size):
            batch.append(self.memory[idx[i]])
            bean.append(self.prob_bean[idx[i]])

        transition = Transition(*zip(*batch))

        bean = np.array(bean, dtype=float)
        loss_weight = (1 / bean ** self.alpha * np.sum(bean ** self.alpha) / len(self.prob_bean)) ** self.beta
        loss_weight = loss_weight / max(loss_weight)

        return transition, loss_weight, idx

    def update_weight(self, idx, TD_error):
        for i in range(idx.shape[0]):
            self.prob_bean[idx[i]] = TD_error[i] + self.epsilon

    def __len__(self):
        return len(self.memory)


def saveLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]

    train_scores = dict()
    train_scores['x'] = x
    train_scores['running_avg'] = running_avg

    sio.savemat(filename, train_scores)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        state = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return state, actions, rewards, states_, terminal
