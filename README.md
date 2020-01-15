# neural_exploration

Contextual bandits are single-state decision processes with the assumption that the rewards for each arm at each step are generated from a (possibly noisy) function of observable features. Similarly, contextual MDPs offer a setting for reinforcement learning where rewards and transition probabilities can be inferred from vector features.

The literature has focused on optimistic exploration bounds under assumptions of linear dependency with the features, resulting in celebrated algorithms such as LinUCB (bandit) and LinUCBVI (fixed horizon RL with value iteration).

Recently, https://arxiv.org/pdf/1911.04462.pdf introduced NeuralUCB, an optimistic exploration bound algorithm that leverages the power of deep neural networks as universal function approximators to alleviate the constraint of linearity.

The goal of this repo is to implement these methods, design synthetic bandits and MDPs to test the various algorithms and introduce NeuralUCBVI, a value iteration algorithm based on neural approximators of rewards and transition kernel.
