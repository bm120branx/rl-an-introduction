#######################################################################
# Copyright (C)                                                       #
# Branka Mirchevska                                                   #
#######################################################################
import numpy as np


class BanditAgent:

    def __init__(self, num_arms=10, step_size=0.1, epsilon=0.1, sample_avg=True):
        self.num_arms = num_arms
        self.step_size = step_size
        self.epsilon = epsilon
        self.actions = np.arange(self.num_arms)
        self.q_true = np.random.normal(0, 0.01, self.num_arms)
        self.action_count = np.zeros(self.num_arms)
        self.q_est = np.zeros(self.num_arms)
        self.sample_avg = sample_avg

    def act(self):
        # return either random action if the rand number is smaller than epsilon,
        # or the current best action if epsilon is set to 0.
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.q_est)

    def reset(self):
        self.q_true = np.random.normal(0, 0.01, self.num_arms)
        self.q_est = np.zeros(self.num_arms)
        self.action_count = np.zeros(self.num_arms)

    def step(self, action):
        # we need to perform random walk for q_true in each step
        self.q_true[action] = np.random.randn() + self.q_true[action]
        reward = self.q_true[action]
        self.action_count[action] += 1
        if self.sample_avg:
            self.q_est[action] += (reward - self.q_est[action]) / self.action_count[action]
        else:
            self.q_est[action] += self.step_size * (reward - self.q_est[action])
        return reward
