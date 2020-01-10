import numpy as np
import abc
from tqdm import tqdm


class UCB(abc.ABC):
    """Base class for UBC methods.
    """
    def __init__(self,
                 bandit,
                 throttle=int(1e2),
                ):
        # bandit class, contains features and generated rewards
        self.bandit = bandit
        # throttle tqdm updates
        self.throttle = throttle
        
        self.reset()
        
    def reset_upper_confidence_bounds(self):
        """Initialize upper confidence bounds and related quantities.
        """
        self.exploration_bonus = np.empty((self.bandit.T, self.bandit.n_arms))
        self.mu_hat = np.empty((self.bandit.T, self.bandit.n_arms)) 
        self.upper_confidence_bounds = np.ones((self.bandit.T, self.bandit.n_arms))
        
    def reset_regrets(self):
        """Initialize regrets.
        """
        self.regrets = np.empty(self.bandit.T)

    def reset_actions(self):
        """Initialize cache of actions.
        """
        self.actions = np.empty(self.bandit.T)
    
    def sample_action(self):
        """Return the action to play based on current estimates
        """
        return np.argmax(self.upper_confidence_bounds[self.iteration])

    @abc.abstractmethod
    def reset(self):
        """Initialize variables of interest.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def update_confidence_bounds(self):
        """Update the confidence bounds for all arms at time t.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def update_chosen_arm(self):
        """Update the parameters for the arm chosen at time t.
        To be defined in children classes.
        """
        pass

    def run(self):
        """Run an episode of bandit.
        """
        postfix = {'cum_regret': 0.0}
        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):
                # update confidence of all arms based on observed features at time t
                self.update_confidence_bounds()
                # pick action with the highest boosted estimated reward
                self.action = self.sample_action()
                self.actions[t] = self.action
                # update A and b for chosen action
                self.update_chosen_arm()
                # compute regret
                self.regrets[t] = self.bandit.best_rewards_oracle[t]-self.bandit.rewards[t, self.action]
                # increment counter
                self.iteration += 1
                
                # log
                postfix['cum_regret'] += self.regrets[t]
                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)
            
class LinUCB(UCB):
    """Liner UCB.
    """
    def __init__(self,
                 bandit,
                 reg_factor=1.0,
                 delta=0.01,
                 bound_theta=1.0,
                 confidence_scaling_factor=0.0,
                ):

        # L2 regularization strength
        self.reg_factor = reg_factor
        # Confidence bound with probability 1-delta
        self.delta = delta
        # range of the linear predictors
        self.bound_theta = bound_theta
        # multiplier for the confidence bound
        # (default is bandit reward noise std dev)
        if confidence_scaling_factor == 0.0:
            confidence_scaling_factor = bandit.noise_std
        self.confidence_scaling_factor = confidence_scaling_factor
        
        # maximum L2 norm for the features across all arms and all rounds
        self.bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))

        super().__init__(bandit)

    def reset(self):
        """Return the internal estimates
        """
        self.reset_upper_confidence_bounds()
        self.reset_regrets()
        self.reset_actions()
        self.iteration = 0

        # randomly initialize linear predictors within their bounds
        self.theta = np.random.uniform(-1, 1, (self.bandit.n_arms, self.bandit.n_features)) * self.bound_theta

        # n_arms square matrices of size n_features*n_features
        self.A_inv = np.array(
            [
                np.eye(self.bandit.n_features)/self.reg_factor for _ in self.bandit.arms
            ]
        )

        # initialize reward-weighted features sum at zero
        self.b = np.zeros((self.bandit.n_arms, self.bandit.n_features))

    @property
    def alpha(self):
        """LinUCB confidence interval multiplier.
        """
        return self.confidence_scaling_factor \
    * np.sqrt(self.bandit.n_features*np.log((1+self.iteration*self.bound_features/self.reg_factor)/self.delta))\
    + np.sqrt(self.reg_factor)*np.linalg.norm(self.theta, ord=2)

    def update_confidence_bounds(self):
        """Update confidence bounds and related quantities for all arms.
        """
        # update theta
        self.theta = np.array(
            [
                np.matmul(self.A_inv[a], self.b[a]) for a in self.bandit.arms
            ]
        )
        
        # features at current time
        u = self.bandit.features[self.iteration]

        # estimated average reward
        self.mu_hat[self.iteration] = np.array(
            [
                np.dot(u[a], self.theta[a]) for a in self.bandit.arms
            ]
        )
        
        # UCB exploration bonus
        self.exploration_bonus[self.iteration] = np.array(
            [
                self.alpha * np.sqrt(np.dot(u[a], np.dot(self.A_inv[a], u[a].T))) for a in self.bandit.arms
            ]
        )

        # estimated combined bound for reward
        self.upper_confidence_bounds[self.iteration] = self.mu_hat[self.iteration] + self.exploration_bonus[self.iteration]

    def update_chosen_arm(self):
        u_at = self.bandit.features[self.iteration, self.action]
        # Sherman-Morrison formula
        Au = np.dot(self.A_inv[self.action], u_at)

        self.A_inv[self.action] -= np.outer(Au, Au)/(1+np.dot(u_at.T, Au))
        self.b[self.action] += u_at*self.bandit.rewards[self.iteration, self.action]
