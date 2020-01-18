import numpy as np
import abc
from tqdm import tqdm
import itertools

from .utils import inv_sherman_morrison

class UCBVI(abc.ABC):
    """Value Iteration with UCB exploration.
    """
    def __init__(self,
                 mdp,
                 n_episodes=1,
                 init_state=0,
                 reg_factor=1.0,
                 confidence_scaling_factor=-1.0,
                 train_every=1,
                 throttle=int(1e2),
                ):
        # MDP object, contains transition kernel and rewards as functions of features.
        self.mdp = mdp
        # number of episodes of the fixed-horizon MDP to run
        self.n_episodes = n_episodes
        # initial state
        self.init_state = init_state
        # L2 regularization strength
        self.reg_factor = reg_factor
        # multiplier for the confidence bound (default is mdp reward noise std dev)
        if confidence_scaling_factor == -1.0:
            confidence_scaling_factor = mdp.noise_std
        self.confidence_scaling_factor = confidence_scaling_factor
        
        # train approximator only every few rounds
        self.train_every = train_every
        
        # throttle tqdm updates
        self.throttle = throttle
        
        self.reset()
        
    def reset_upper_confidence_bounds(self):
        """Initialize upper confidence bounds and related quantities.
        """
        self.Q_hat = np.concatenate(
            (
                np.empty((self.mdp.H, self.mdp.n_states, self.mdp.n_actions)), 
                np.zeros((1, self.mdp.n_states, self.mdp.n_actions))
            )
        )
        self.exploration_bonus = np.empty((self.mdp.H, self.mdp.n_states, self.mdp.n_actions))
        self.upper_confidence_bounds = np.ones((self.mdp.H, self.mdp.n_states, self.mdp.n_actions))
    
    def reset_regrets(self):
        """Initialize regrets and optimal value function 
        (to compute regret only, not to be used in the algorithm).
        """
        self.regrets = np.empty(self.n_episodes)
        self.V_star, self.pi_star = self.mdp.optimal_policy()
        
    def reset_policy(self):
        """Initialize policy.
        """
        self.policy = np.empty((self.mdp.H, self.mdp.n_states)).astype('int')
        
    def reset_state_action_reward_buffer(self):
        """Initialize cache of (state, action, reward).
        """
        self.buffer_states = np.empty(self.mdp.H+1).astype('int')
        self.buffer_actions = np.empty(self.mdp.H).astype('int')
        self.buffer_rewards = np.empty(self.mdp.H)
        
        # initial state
        self.state = self.init_state
    
    def reset_A_inv(self):
        """Initialize square matrice representing the inverse of exploration bonus matrice.
        """
        self.A_inv = np.array(
            [
                np.eye(self.approximator_dim)/self.reg_factor for _ in range(self.mdp.H)
            ]
        ).reshape(self.mdp.H, self.approximator_dim, self.approximator_dim)
        
    def reset_grad_approx(self):
        """Initialize the gradient of the approximator w.r.t its parameters.
        """
        self.grad_approx = np.zeros((self.mdp.n_states, self.mdp.n_actions, self.approximator_dim))

    def take_action(self):
        """Return the action to play based on current estimates.
        """
        self.policy[self.mdp.iteration] = np.argmax(
            self.upper_confidence_bounds[self.mdp.iteration],
            axis=1,
        ).astype('int')
        self.action = self.policy[self.mdp.iteration, self.state]
    
    @abc.abstractmethod
    def reset(self):
        """Initialize variables of interest.
        To be defined in children classes.
        """
        pass

    @property
    @abc.abstractmethod
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        pass
    
    @property
    @abc.abstractmethod
    def confidence_multiplier(self):
        """Multiplier for the confidence exploration bonus.
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
    def update_output_gradient(self):
        """Compute output gradient of the approximator w.r.t its parameters.
        """
        pass
    
    @abc.abstractmethod
    def train(self):
        """Update approximator.
        To be defined in children classes.
        """
        pass
    
    @abc.abstractmethod
    def predict(self):
        """Predict rewards based on an approximator.
        To be defined in children classes.
        """
        pass

    def update_confidence_bounds(self):
        """Update confidence bounds and related quantities for current (state,action).
        """
        self.update_output_gradient()
        
        # UCB exploration bonus
        g_flatT = self.grad_approx.flatten().reshape(self.approximator_dim, -1).T
        self.exploration_bonus[self.mdp.iteration] = self.confidence_multiplier * np.sqrt(
                np.einsum('...i,...i->...', g_flatT.dot(self.A_inv[self.mdp.iteration]), g_flatT)
            ).reshape(self.mdp.n_states, self.mdp.n_actions)
        # alternative writing:
        # bonus = (g.T.dot(A_inv)*g.T).sum(axis=1)

        # update reward prediction Q_hat
        self.predict()
        
        # estimated combined bound for reward
        self.upper_confidence_bounds[self.mdp.iteration] = np.clip(
            self.Q_hat[self.mdp.iteration] + self.exploration_bonus[self.mdp.iteration],
            None,
            self.mdp.H
        )
        
    def update_A_inv(self):
        self.A_inv[self.mdp.iteration] = inv_sherman_morrison(
            self.grad_approx[self.state, self.action], 
            self.A_inv[self.mdp.iteration],
        ) 
        
    def run(self):
        """Run an episode of MDP.
        """
        postfix = {
            'total regret': 0.0,
        }
        
        with tqdm(total=self.n_episodes, postfix=postfix) as pbar:
            for k in range(self.n_episodes):
                self.mdp.reset_iteration('backward')
                for h in reversed(range(self.mdp.H)):
                    if k > 0:
                        self.action = self.buffer_actions[h]
                        self.reward = self.buffer_rewards[h]
                        self.state = self.buffer_states[h]
                    
                        # update exploration indicator A_inv for chosen action
                        self.update_A_inv()
                        # update approximator
                        if k % self.train_every == 0:
                            self.train()
                    
                        # update confidence of all (state, action) pairs based on observed features 
                        self.update_confidence_bounds()

                    # decrement MDP counter
                    self.mdp.iteration -= 1
                
                self.mdp.reset_iteration('forward')
                self.state = self.init_state
                self.buffer_states[0] = self.state

                for h in range(self.mdp.H):
                    # udpate policy to be greedy w.r.t boosted Q function and pick action
                    self.take_action()
                            
                    # get reward and move on
                    self.reward = self.mdp.rewards[h, self.state, self.action]
                    self.state = self.mdp.new_state(self.state, self.action)
                    
                    # update buffers
                    self.buffer_actions[h] = self.action
                    self.buffer_rewards[h] = self.reward
                    self.buffer_states[h+1] = self.state
                    
                    # increment MDP counter
                    self.mdp.iteration += 1
                
                # compute regret
                V = self.mdp.evaluate_policy(self.policy)
                self.regrets[k] = self.V_star[0, self.init_state]-V[0, self.init_state]

                # log
                postfix['total regret'] += self.regrets[k]
                if k % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)
                