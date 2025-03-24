#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using code structure from
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
from DQN import DQNAgent
import gym
from helper import LearningCurvePlot, smooth


def average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,
                 num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER,
                 use_TN, n_repetitions, smoothing_window, n_layers):
    reward_results = np.empty([n_repetitions, num_episodes])
    now = time.time()
    env = gym.make('CartPole-v1')
    env.render(mode='human')

    for rep in range(n_repetitions):
        if policy == 'egreedy':
            print("e-greedy Experiment Repetition: ", rep + 1)
            agent = DQNAgent(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER, use_TN, n_layers)

            # train the agent
            episode_rewards = agent.train(env)

        elif policy == 'annealing_egreedy':
            print("Annealing e-greedy Experiment Repetition: ", rep + 1)
            agent = DQNAgent(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER, use_TN, n_layers)
            # train the agent
            episode_rewards = agent.train(env)

        elif policy == 'softmax':
            print("softmax Experiment Repetition: ", rep + 1)
            agent = DQNAgent(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER, use_TN, n_layers)

            # train the agent
            episode_rewards = agent.train(env)

        reward_results[rep] = episode_rewards

    # print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    print("the experiment took {} minutes".format((time.time() - now) / 60))
    learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
    learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    return learning_curve


def experiment():
    n_repetitions = 20
    smoothing_window = 150
    num_episodes = 1000
    model = 'DQN'
    epsilon = 0.1
    n_states = 4
    n_actions = 2
    n_hidden = 64
    lr = 0.001
    gamma = 0.9
    memory_size = 10000
    batch_size = 64
    target_network_update = 100
    max_steps = 500
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.7
    temp = 0.2
    plot = False
    n_layers = 3
    use_ER = True
    use_TN = True


    Plot = LearningCurvePlot(title='Exploration strategies')

    # Annealing e-greedy policy
    policy = 'annealing_egreedy'
    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                                              target_network_update,
                                              num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                                              epsilon_end, epsilon_decay, temp, plot, use_ER,
                                              use_TN, n_repetitions, smoothing_window, n_layers)
    Plot.add_curve(learning_curve, label="annealing e-greedy")

    # e-greedy policy
    policy = 'egreedy'
    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                                              target_network_update,
                                              num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                                              epsilon_end, epsilon_decay, temp, plot, use_ER,
                                              use_TN, n_repetitions, smoothing_window, n_layers)
    Plot.add_curve(learning_curve, label="e-greedy")

    # Softmax policy
    policy = 'softmax'
    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                                              target_network_update,
                                              num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                                              epsilon_end, epsilon_decay, temp, plot, use_ER,
                                              use_TN, n_repetitions, smoothing_window, n_layers)
    Plot.add_curve(learning_curve, label="softmax")

    Plot.save('exploration_strategies.png')


if __name__ == '__main__':
    experiment()