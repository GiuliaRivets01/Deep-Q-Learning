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
        if use_ER and not use_TN:
            print('DQN without target network')
            print("Experiment Repetition: ", rep + 1)
            agent = DQNAgent(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER, use_TN, n_layers)

            # train the agent
            episode_rewards = agent.train(env)
        elif not use_ER and use_TN:
            print('DQN without experience replay')
            print("Experiment Repetition: ", rep + 1)
            agent = DQNAgent(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER, use_TN, n_layers)
            # train the agent
            episode_rewards = agent.train(env)
        elif use_ER and use_TN:
            print('DQN')
            print("Experiment Repetition: ", rep + 1)
            agent = DQNAgent(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER, use_TN, n_layers)

            # train the agent
            episode_rewards = agent.train(env)
        elif not use_ER and not use_TN:
            print('DQN without experience replay and target network')
            print("Experiment Repetition: ", rep + 1)
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
    policy = 'annealing_egreedy'
    model = 'DQN'
    epsilon = 0.9

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


    # Ablation study: DQN vs DQN-ER vs DQN-TN vs DQN-ER-TN
    Plot = LearningCurvePlot(title='Comparison of DQN with DQN-ER and DQN-TN and DQN-ER-TN')

    use_ER = True
    use_TN = True

    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,
                 num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER,
                 use_TN, n_repetitions, smoothing_window, n_layers)
    Plot.add_curve(learning_curve, label="DQN")

    use_ER = False
    use_TN = True

    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,
                 num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER,
                 use_TN, n_repetitions, smoothing_window, n_layers)
    Plot.add_curve(learning_curve, label="DQN-ER")

    use_ER = True
    use_TN = False

    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,
                 num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER,
                 use_TN, n_repetitions, smoothing_window, n_layers)
    Plot.add_curve(learning_curve, label="DQN-TN")

    use_ER = False
    use_TN = False

    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,
                 num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, temp, plot, use_ER,
                 use_TN, n_repetitions, smoothing_window, n_layers)
    Plot.add_curve(learning_curve, label="DQN-ER-TN")

    Plot.save('experiment_DQN_ER_TN.png')


if __name__ == '__main__':
    experiment()


