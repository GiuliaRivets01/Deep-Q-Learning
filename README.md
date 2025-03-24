
# Deep Q-Learning README

### Requirements

## Python 3.8 is required

# Create a virtual environment 

```
 python3 -m venv rl_env

```

# Activate the environment

```
 rl_env\Scripts\activate

```

# Use the requirements file to add the needed libraries

```
 pip install -r requirements.txt

```

### DQN Implementation

The DQN agent was implemented in file *DQN.py*.

If you want to create a standard DQN agent you can run this file by command line in the following ways:

- *python3 DQN.py* if you want to use both experience replay and target network
- *python3 DQN.py --experience_replay* to run the algorithm without experience replay
- *python3 DQN.py --target_network* to run the algorithm without target network
- *python3 DQN.py --target_network --experience_replay*  or *python3 DQN.py --experience_replay --target_network* to run the algorithm without both target network and experience replay


Alternatively, you could also create a double DQN agent or dueling DQN agent (with or without convolutions) in the following way:
- *python3 DQN.py double*
- *python3 DQN.py dueling*
- *python3 DQN.py conv_dueling*
With these two models, both experience replay and target networks are used.


### Hyperparameter tuning
File *hyperparameter_tuning.py* performs the hyperparameter tuning part. By running this file you will obtain the best hyperparameters.

### Ablation study
File *ablation_study.py* performs the ablation study, where DQN is implemented with and/or without experience replay/target network. By running this file you will obtain the plot for the ablation study

### Explorayion stragies
In file *exploration_strategies.py* we have compared the perfromance of the DQN by emplying three different explorayion strategies: e-greedy, annealing e-greedy and softmax. By runnign the file you will obtain the plot showint the learning curves of these policies. 

### DQN improvements
File *DQN_improvements.py* performs the comarison between DQN, double DQN, dueling DQN and dueling DQN with convolutions. Runnign this file will give you the plot where the different models are compared.






