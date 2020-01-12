import numpy as np
from .ucbvi import UCBVI


class LinUCBVI(UCBVI):
    """Value Iteration with LinUCB exploration.
    """
    def __init__(self,
                 mdp,
                 n_episodes=1,
                 init_state=0,
                 reg_factor=1.0,
                 confidence_scaling_factor=-1.0,
                 bound_theta=1.0,
                 throttle=int(1e2),
                ):

        # range of the linear predictors
        self.bound_theta = bound_theta
        
        super().__init__(mdp,
                         n_episodes=n_episodes,
                         init_state=init_state,
                         reg_factor=reg_factor,
                         confidence_scaling_factor=confidence_scaling_factor,
                         throttle=throttle,
                        )

    @property
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        return self.mdp.n_features
    
    def update_output_gradient(self):
        """For linear approximators, simply returns the features.
        """
        self.grad_approx = self.mdp.features
    
    def reset(self):
        """Return the internal estimates
        """
        self.reset_upper_confidence_bounds()
        self.reset_regrets()
        self.reset_policy()
        self.reset_state_action_reward_buffer()
        self.reset_A_inv()
        self.reset_grad_approx()

        # randomly initialize linear predictors within their bounds
        self.theta = np.random.uniform(-1, 1, (self.mdp.H, self.mdp.n_features)) * self.bound_theta
        
        # initialize reward-weighted features sum at zero
        self.b = np.zeros((self.mdp.H, self.mdp.n_features))

    @property
    def confidence_multiplier(self):
        """LinUCB confidence interval multiplier.
        """
        return self.confidence_scaling_factor
    
    def train(self):
        """Update linear predictor theta.
        """
        self.b[self.mdp.iteration] += self.mdp.features[self.state, self.action] * (self.reward + np.max(self.Q_hat[self.mdp.iteration+1, self.buffer_states[self.mdp.iteration+1]]))
        self.theta[self.mdp.iteration] = np.matmul(
            self.A_inv[self.mdp.iteration], 
            self.b[self.mdp.iteration]
        )
    
    def predict(self):
        """Predict reward.
        """
        self.Q_hat[self.mdp.iteration] = np.dot(self.mdp.features, self.theta[self.mdp.iteration])
