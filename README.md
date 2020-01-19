# neural_exploration

Contextual bandits are single-state decision processes with the assumption that the rewards for each arm at each step are generated from a (possibly noisy) function of observable features. Similarly, contextual MDPs offer a setting for reinforcement learning where rewards and transition probabilities can be inferred from vector features.

The literature has focused on optimistic exploration bounds under assumptions of linear dependency with the features, resulting in celebrated algorithms such as LinUCB (bandit) and LinUCBVI (fixed horizon RL with value iteration).

Recently, https://arxiv.org/pdf/1911.04462.pdf introduced NeuralUCB, an optimistic exploration bound algorithm that leverages the power of deep neural networks as universal function approximators to alleviate the constraint of linearity.

The goal of this repo is to implement these methods, design synthetic bandits and MDPs to test the various algorithms and introduce NeuralUCBVI, a value iteration algorithm based on neural approximators of rewards and transition kernel for efficient exploration of fixed-horizon MDP.

## Experiments

All methods are tested on 3 types of contextual rewards : linear, quadratic, and highly nonlinear (cosine). 

For episodic MDPs, transition matrix are assumed to be linear in the features in all cases. While LinUCB and LinUCB-VI perform well in the linear case (sublinear or even no regret growth), they are slightly sub-optimal in the quadratic case and completely fail in presence of stronger nonlinearity. This is consistent with results from https://arxiv.org/abs/1907.05388 on approximate linear MDP and https://arxiv.org/pdf/1911.00567.pdf on low-rank MDPs, which give control on the performance of linear exploration as a function of the magnitude of the nonlinearity.

Neural exploration on the other hand rely on more sophisticated approximators, which are expressive enough to predict rewards or Q-functions generated by more complicated functions of the features (given wide or deep enough architecture, neural networks are universal approximators). NeuralUCB and NeuralUCB-VI efficiently explore and quickly reach optimality (no or very slow regret growth). 

### Stability

In some cases, neural exploration appears unstable in the sense that after an optimal policy is learned and regret stops growing, it may happen that a small perturbation pushes the network out of the optimal solution and regret starts growing again, something akin to catastrophic forgetting. 

![Unstable NeuralUCB-VI on linear rewards](https://github.com/sauxpa/neural_exploration/blob/master/neural_ucbvi_linear_unstable.pdf)
