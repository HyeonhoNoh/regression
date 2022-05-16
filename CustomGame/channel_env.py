import numpy as np
import torch
from utils.MU_selection import action_to_user

class ZF_80211ax():
    def __init__(self):
        self.states = np.zeros((1,2,10,4))

    def reset(self):
        channel_gain = np.random.lognormal(0.0, 0.1, size=[1,10,4,1])
        channel_phase = np.random.uniform(low=-np.pi, high=np.pi, size=[1, 10, 4, 1])
        # self.channel = channel_gain * np.cos(channel_phase) + 1j * channel_gain * np.sin(channel_phase)
        # self.states = np.concatenate([channel_gain * np.cos(channel_phase), channel_gain * np.sin(channel_phase)], axis=3)
        self.channel = channel_gain * np.cos(channel_phase)
        self.states = channel_gain * np.cos(channel_phase)

        # self.states = self.channel

        return self.states

    def step(self, act):
        act_ = []
        for i in act:
            act_.append(i.tolist())
        sel_mat = self.channel[:, np.array(act[0])].squeeze(0).squeeze(-1)
        sel_mat_w_zero = np.zeros([4,4])
        sel_mat_w_zero[:sel_mat.shape[0], :sel_mat.shape[1]] = sel_mat
        mul_mat = np.matrix(sel_mat_w_zero) * np.matrix(sel_mat_w_zero).getH()
        # inv_mat = np.linalg.inv(np.matrix(self.channel[:, np.array(act[0])].squeeze(0)) * np.matrix(self.channel[:, np.array(act[0])].squeeze(0)).getH())
        # results = (np.diag(inv_mat) ** -1).real
        # u, s, v = np.linalg.svd(self.states[:, np.array(act[0])].squeeze(0))

        # results = np.log2(1+results)
        # reward = np.zeros((10, 1))
        # reward[act[0]] = s.reshape(act[0].shape[0],1)
        # reward[act[0]] = results.reshape(act[0].shape[0],1)

        # done = False
        # new_states = self.states

        # return new_states, reward, done, None
        # results = np.zeros([4,4], dtype=complex)
        # results[:mul_mat.shape[0],:mul_mat.shape[1]] = mul_mat
        reward = torch.tensor(mul_mat).reshape([-1,4,4])

        return reward

    def get_max_reward(self):

        max_act = 0
        max_reward_sum = 0
        max_reward = np.zeros((10,1))

        for act_itr in range(385):
            act = [action_to_user(act_itr)]

            _, reward, done, _ = self.step(np.array(act))

            if reward.sum() > max_reward_sum:
                max_act = act
                max_reward = reward
                max_reward_sum = reward.sum()

        return max_reward, max_act

    def get_max_SINR(self):
        SINR = self.get_SINR()
        act = SINR.argsort()[-4:]
        act.sort()
        return act

    def get_SINR(self):
        SINR = self.channel.__pow__(2).sum(3).sum(2).sum(0) / self.states[0].shape[0]
        return SINR
