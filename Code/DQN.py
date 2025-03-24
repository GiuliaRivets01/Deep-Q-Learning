import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
from tqdm import tqdm
import torch.nn.functional as F
import sys
import time

class DQN(nn.Module):

    def __init__(self, n_states, n_actions, n_hidden, n_layers):
        super(DQN, self).__init__()

        self.n_layers = n_layers
        # Change number of layers
        if n_layers == 3:
            self.layer1 = nn.Linear(n_states, n_hidden)
            self.layer2 = nn.Linear(n_hidden, n_hidden)
            self.layer3 = nn.Linear(n_hidden, n_actions)
        elif n_layers == 4:
            self.layer1 = nn.Linear(n_states, n_hidden)
            self.layer2 = nn.Linear(n_hidden, n_hidden)
            self.layer3 = nn.Linear(n_hidden, n_hidden)
            self.layer4 = nn.Linear(n_hidden, n_actions)
        elif n_layers == 5:
            self.layer1 = nn.Linear(n_states, n_hidden)
            self.layer2 = nn.Linear(n_hidden, n_hidden)
            self.layer3 = nn.Linear(n_hidden, n_hidden)
            self.layer4 = nn.Linear(n_hidden, n_hidden)
            self.layer5 = nn.Linear(n_hidden, n_actions)
        elif n_layers == 6:
            self.layer1 = nn.Linear(n_states, n_hidden)
            self.layer2 = nn.Linear(n_hidden, n_hidden)
            self.layer3 = nn.Linear(n_hidden, n_hidden)
            self.layer4 = nn.Linear(n_hidden, n_hidden)
            self.layer5 = nn.Linear(n_hidden, n_hidden)
            self.layer6 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        if self.n_layers == 3:
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)
        elif self.n_layers == 4:
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            return self.layer4(x)
        elif self.n_layers == 5:
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            x = F.relu(self.layer4(x))
            return self.layer5(x)
        elif self.n_layers == 6:
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            x = F.relu(self.layer4(x))
            x = F.relu(self.layer5(x))
            return self.layer6(x)

class DuelingDQN(nn.Module):

    def __init__(self, n_states, n_actions, n_hidden):
        super(DuelingDQN, self).__init__()
        self.input_dim = n_states
        self.output_dim = n_actions
        self.n_hidden = n_hidden

        ''' Feature layer: process the input state to extract important features.
        It consists of two fully connected layers with ReLU activation functions.'''
        self.feauture_layer = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )

        ''' Stream for estimating the state-value'''
        self.value_stream = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

        ''' Stream for estimating state-dependent action advantages'''
        self.advantage_stream = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )

    def forward(self, state):
        # Combine the state-value and advantage outputs
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals


''' Convolutional Dueling DQN '''
class ConvDuelingDQN(nn.Module):

    def __init__(self, n_states, n_actions, n_hidden):
        super(ConvDuelingDQN, self).__init__()
        self.input_dim = n_states
        self.output_dim = n_actions
        self.n_hidden = n_hidden
        self.fc_input_dim = self.feature_size()

        self.conv = nn.Sequential(
            nn.Conv2d(n_states, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.output_dim)
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    def feature_size(self):
        return self.conv(torch.autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


# Define the DQN agent
class DQNAgent():

    # Initialize arguments of the DQN agent
    def __init__(self, n_states=4, n_actions=2, n_hidden=64, lr=0.001, gamma=0.99, memory_size=10000, batch_size=32,
                 target_network_update=100,
                 num_episodes=1000, policy='egreedy', model='DQN', epsilon=1.0, max_steps=500, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, temp=0.2, plot=False, use_ER=True,
                 use_TN=True, n_layers=3):
        self.n_states= n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.lr = lr
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_network_update = target_network_update
        self.policy = policy
        self.model = model
        self.use_ER = use_ER
        self.use_TN= use_TN
        self.batch = []
        self.n_layers = n_layers
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.temp = temp
        self.plot = plot
        self.steps = 0
        if self.use_ER:
            self.memory = deque(maxlen=self.memory_size)

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the networks based on the selected one
        if self.model == 'DQN' or self.model == 'DoubleDQN':
            if self.use_TN:
                self.initialize_DQN()
            else:
                self.initialize_DQN_noTN()
        elif self.model == 'DuelingDQN' or model == 'ConvDuelingDQN':
            self.initialize_dueling()


    def initialize_DQN(self):
        self.q_network = DQN(self.n_states, self.n_actions, self.n_hidden, self.n_layers).to(self.device)
        self.target_network = DQN(self.n_states, self.n_actions, self.n_hidden,self.n_layers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def initialize_DQN_noTN(self):
        self.q_network = DQN(self.n_states, self.n_actions, self.n_hidden, self.n_layers).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def initialize_dueling(self):
        self.q_network = DuelingDQN(self.n_states, self.n_actions, self.n_hidden).to(self.device)
        self.target_network = DuelingDQN(self.n_states, self.n_actions, self.n_hidden).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    # Select an action based on either e-greedy, annealing e-greedy or softmax policy
    def select_action(self, state):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        eps = self.epsilon

        # select action based on the selected policy
        if self.policy == "egreedy":
            action = self.egreedy_policy(eps, q_values)

        elif self.policy == "annealing_egreedy":
            action = self.annealing_egreedy_policy(eps, q_values)
        elif self.policy == "softmax":
            action = self.softmax_policy(q_values)

        return action

    def egreedy_policy(self, eps, q_values):
        if np.random.rand() < eps:
            action = np.random.randint(self.n_actions)
        else:
            action = torch.argmax(q_values).item()
        return action

    def annealing_egreedy_policy(self, eps, q_values):
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.epsilon_decay * self.steps)
        if np.random.rand() < eps:
            action = np.random.randint(self.n_actions)
        else:
            action = torch.argmax(q_values).item()
        return action

    def softmax_policy(self, q_values):
        action = torch.softmax(q_values / self.temp, dim=-1).data.numpy().squeeze()
        action = np.random.choice(self.n_actions, p=action)
        return action

    def update(self):
        if self.use_ER and len(self.memory) < self.batch_size:
            return

        # if the replay buffer is not used, use a batch
        if self.use_ER:
            batch = random.sample(self.memory, self.batch_size)
        else:
            batch = self.batch

        # Transition
        state, action, reward, next_state, done = zip(*batch)

        # Conversion to tensors
        state = torch.FloatTensor(np.float32(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Q-values computation
        q_values = self.q_network(state).gather(1, action)

        # Next Q-values computation
        if self.use_TN and self.model == 'DQN':
            next_q_values = self.target_network(next_state).max(1)[0].unsqueeze(1)
        elif self.use_TN and self.model == 'DoubleDQN':
            next_actions = self.q_network(next_state).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_state).gather(1, next_actions)
        elif self.use_TN and self.model == 'DuelingDQN' or self.model == 'ConvDuelingDQN':
            next_actions = self.q_network(next_state).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_state).gather(1, next_actions)
        else:
            next_q_values = self.q_network(next_state).max(1)[0].unsqueeze(1)



        # Expected Q-values computation
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        # Loss computation
        loss = nn.functional.smooth_l1_loss(q_values, expected_q_values.detach())

        # Update policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_network_update == 0 and self.use_TN:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, env):
        # initialize the list to store the rewards
        episode_rewards = []

        # loop through the episodes
        for i in tqdm(range(self.num_episodes)):
            state = env.reset()

            episode_reward = 0
            # loop through the steps in each episode
            for j in range(self.max_steps):
                # select action based on the selected policy
                action = self.select_action(state)
                # take the action and get the next state, reward, and done
                next_state, reward, done, _ = env.step(action)

                # Store data in replay buffer if we are using experience replay
                if self.use_ER:
                    self.memory.append((state, action, reward, next_state, done))
                # Otherwise store data in batch
                else:
                    self.batch = [(state, action, reward, next_state, done)]

                # Update the network
                self.update()

                # Go to next state
                state = next_state
                episode_reward += reward

                if self.plot:
                    env.render()

                # if the episode is done, break
                if done:
                    break

            episode_rewards.append(episode_reward)

        return episode_rewards

def experiment():

    # Parameters
    num_episodes = 1000 # Number of episodes
    policy = 'egreedy' # Policy can be 'egreedy', 'annealing_egreedy' or 'softmax'
    model = 'DQN' # Model can be 'DQN', 'DoubleDQN', 'DuelingDQN'
    epsilon = 0.9 # Exploration factor
    n_states = 4 # Number of neurons in input layer
    n_actions = 2 # Number of neurons in output layer
    n_hidden = 64 # Number of neurons in hidden layers
    lr = 0.001 # Learning rate
    gamma = 0.9 # Discount factor
    memory_size = 10000 # Capacity of the memory for experience replay
    batch_size = 64 # Size of each batch
    target_network_update = 100 # Target network update frequency
    max_steps = 500 # Maximum number of steps
    epsilon_start = 1.0 # Staring value for the exploration factor
    epsilon_end = 0.01 # Final value for the exploration factor
    epsilon_decay = 0.7 # Decay value for the exploration factor
    tau = 0.2 # Temperature
    plot = False # If True, plots the Cartpole evolution
    n_layers = 3 # Number of layers for the DQN network
    n_repetitions = 20 # Number of repetitions
    use_ER = True # If True, use experience replay
    use_TN = True # If True, use target network

    # To easily turn off model's components
    arguments = sys.argv[1:]
    if len(arguments)<2:
        for _ in range(len(arguments), 2):
            arguments.append('Nope')

    if arguments[0] == "--experience_replay" or arguments[1] == "--experience_replay":
        use_ER = False
    if arguments[0] == "--target_network" or arguments[1] == "--target_network":
        use_TN = False

    if arguments[0] == "double":
        model = 'DoubleDQN'
        use_ER = True
        use_TN = True

    if arguments[0] == "dueling":
        model = 'DuelingDQN'
        use_ER = True
        use_TN = True

    if arguments[0] == "conv_dueling":
        model = 'ConvDuelingDQN'
        use_ER = True
        use_TN = True


    returns_over_repetitions = []
    now = time.time()
    env = gym.make('CartPole-v1')
    env.render(mode='human')

    print("Running with experience replay: {}, target network: {}" .format(use_ER, use_TN))
    print("Model: ", model)

    for rep in range(n_repetitions):
        print("DQN repetition {}/{}".format(rep+1, n_repetitions))
        agent = DQNAgent(n_states, n_actions, n_hidden, lr, gamma, memory_size, batch_size,
                 target_network_update,num_episodes, policy, model, epsilon, max_steps, epsilon_start,
                 epsilon_end, epsilon_decay, tau, plot, use_ER, use_TN, n_layers)

        # train the agent
        episode_rewards = agent.train(env)
        returns_over_repetitions.append(episode_rewards)

    print("Average reward: ", np.mean(returns_over_repetitions))
    print("Running DQN took {} minutes".format((time.time() - now) / 60))

if __name__ == '__main__':
    experiment()
