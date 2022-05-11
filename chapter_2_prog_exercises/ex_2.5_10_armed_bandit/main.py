#######################################################################
# Copyright (C)                                                       #
# Branka Mirchevska                                                   #
#######################################################################
from bandit_agent import BanditAgent

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def run(num_runs, num_time_steps, bandits):
    rewards = np.zeros((len(bandits), num_runs, num_time_steps))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in range(num_runs):
            bandit.reset()
            for t in range(num_time_steps):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == np.argmax(bandit.q_true):
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def plot(mean_best_action_counts, mean_rewards, num_agents, num_time_steps, labels):
    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax2 = f2.add_subplot(111)
    colors = ['green', 'red']
    for a in range(num_agents):
        mean_best_action_counts_in_percent = [int(x * 100) for x in mean_best_action_counts[a]]
        ax1.plot(range(num_time_steps), mean_best_action_counts_in_percent, colors[a], label=labels[a])
        ax2.plot(range(num_time_steps), mean_rewards[a], colors[a], label=labels[a])

    ax1.set_xlabel('Number of time-steps')
    ax2.set_xlabel('Number of time-steps')

    ax1.set_ylabel('Percentage of times the best action is chosen')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_ylabel('Average reward')

    ax1.set_title('Percentage of times the best action is chosen wrt. ts over all runs')
    ax2.set_title('Average reward per time-step over all runs')

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")
    f1.savefig('plots_ex_2.5/best_action.png')
    f2.savefig('plots_ex_2.5/avg_rew.png')


if __name__ == '__main__':

    bandit_1 = BanditAgent(num_arms=10, step_size=0.1, epsilon=0.1, sample_avg=True)
    bandit_2 = BanditAgent(num_arms=10, step_size=0.1, epsilon=0.1, sample_avg=False)

    n_runs = 2000
    num_time_steps_per_run = 10000
    bandits_to_run = [bandit_1, bandit_2]
    plot_labels = ['Sample average', 'Constant step-size ' + r'$\alpha$=0.1']

    final_best_action_counts, final_rewards = run(n_runs, num_time_steps_per_run, bandits_to_run)
    plot(final_best_action_counts, final_rewards, len(bandits_to_run), num_time_steps_per_run, plot_labels)
