import numpy as np
import itertools


class ContextualBandit():
    def __init__(self,
                 T,
                 n_arms,
                 n_features,
                 h,
                 noise_std=1.0,
                ):
        # number of rounds
        self.T = T
        # number of arms
        self.n_arms = n_arms
        # number of features for each arm
        self.n_features = n_features
        # average reward function
        # h : R^d -> R
        self.h = h

        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std
        
        # generate random features
        self.reset()

    @property
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)
        
    def reset(self):
        """Generate new features and new rewards.
        """
        self.reset_features()
        self.reset_rewards()

    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        x = np.random.randn(self.T, self.n_arms, self.n_features)
        x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
        self.features = x

    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        self.rewards = np.array(
            [
                self.h(self.features[t, k]) + self.noise_std*np.random.randn()\
                for t,k in itertools.product(range(self.T), range(self.n_arms))
            ]
        ).reshape(self.T, self.n_arms)

        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)
