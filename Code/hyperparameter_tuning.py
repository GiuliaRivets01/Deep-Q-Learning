from DQN import DQNAgent
import numpy as np
import itertools
import gym
from joblib import parallel_backend


def main():
    # Defining search space
    hpSearchSpace = {
        'lr': [0.0001, 0.001, 0.01, 0.1],
        'n_hidden': [64, 128, 256],  # Number of neurons in the hidden layer
        'batch_size': [32, 64, 128],
        'epsilon': [0.1, 0.5, 1.0],
        'gamma': [0.9, 0.95, 0.99],
    }


    env = gym.make('CartPole-v1')
    env.render(mode='human')

    print("DQN")
    # Generate all possible combinations of hyperparameters
    bestParameters = grid_search(env, hpSearchSpace)
    print("Best parameters after hyperparameter tuning:")
    print(bestParameters)

    return


def grid_search(env, hpSearchSpace):
    bestAvgReward = float('-inf')
    bestParameters = None

    with parallel_backend('multiprocessing', n_jobs=-1):
        # Iterate over each combination
        num_combinations = len(list(itertools.product(*hpSearchSpace.values())))
        print(num_combinations)
        i = 1
        for parametersCombination in itertools.product(*hpSearchSpace.values()):
            parametersDictionary = dict(zip(hpSearchSpace.keys(), parametersCombination))

            agent = DQNAgent(n_states=4, n_actions=2, memory_size=10000, target_network_update=100,
                             num_episodes=1000, policy='egreedy', model='DQN', max_steps=500, epsilon_start=1.0,
                             epsilon_end=0.01, temp=0.2, plot=False, use_ER=True,
                             use_TN=True,
                             **parametersDictionary)

            print("Running with the following hyperparameters combin {}/{}:".format(i, num_combinations))
            print(parametersDictionary)
            i += 1
            # Train the agent
            episode_rewards = agent.train(env)
            # Evaluate the performance of the agent and save results
            avg_episode_reward = np.mean(episode_rewards)

            if avg_episode_reward > bestAvgReward:
                bestAvgReward = avg_episode_reward
                bestParameters = parametersDictionary
    return bestParameters


def experiment():
    n_states = 4
    n_actions = 2
    n_hidden = 256
    lr = 0.0001
    gamma = 0.95
    memory_size = 10000
    batch_size = 128
    target_network_update = 100
    num_episodes = 1000
    policy = 'annealing_egreedy'
    model = 'DQN'
    epsilon = 0.1
    max_steps = 500
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    temp = 0.2
    plot = False
    use_ER = True
    use_TN = True
    bestNum = None

    best = float('-inf')

    env = gym.make('CartPole-v1')
    env.render(mode='human')

    for n_layers in range(3, 7):
        print("Running n_layers: ", n_layers)
        agent = DQNAgent(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                         target_network_update,
                         num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                         epsilon_end, epsilon_decay, temp, plot, use_ER,
                         use_TN, n_layers)

        rewards = agent.train(env)
        reward = np.mean(rewards)
        if reward > best:
            best = reward
            bestNum = n_layers
    print("Best num = {} with reward = {}" .format(bestNum, best))


def experiment():
    n_states = 4
    n_actions = 2
    n_hidden = 256
    lr = 0.0001
    gamma = 0.95
    memory_size = 10000
    batch_size = 128
    target_network_update = 100
    num_episodes = 1000
    policy = 'annealing_egreedy'
    model = 'DQN'
    epsilon = 0.1
    max_steps = 500
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    temp = 0.2
    plot = False
    use_ER = True
    use_TN = True
    bestNum = None

    best = float('-inf')

    env = gym.make('CartPole-v1')
    env.render(mode='human')

    for n_layers in range(3, 4):
        print("Running n_layers: ", n_layers)
        agent = DQNAgent(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                         target_network_update,
                         num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                         epsilon_end, epsilon_decay, temp, plot, use_ER,
                         use_TN, n_layers)

        rewards = agent.train(env)
        reward = np.mean(rewards)
        if reward > best:
            best = reward
            bestNum = n_layers
    print("Best num = {} with reward = {}".format(bestNum, best))



if __name__ == "__main__":
    main()
    experiment()

# Best found parameters:
# {'lr': 0.0001, 'n_hidden': 256, 'batch_size': 128, 'epsilon': 0.1, 'gamma': 0.95}

# Best found number of layers: 4

