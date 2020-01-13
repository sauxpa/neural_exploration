import numpy as np
import itertools


class MDPFeatures():
    """MDP (stationary) with features kernel for transition and rewards.
    """
    def __init__(self,
                 H,
                 n_states=0,
                 n_features=0,
                 n_actions=0,
                 reward_func=None,
                 noise_std=1.0,
                ):
        # horizon
        self.H = H
        # number of states
        self.n_states = n_states
        # number of actions
        self.n_actions = n_actions
        # number of features for each (state, action) pair
        self.n_features = n_features
        # average reward function
        # h : R^{n_features} -> R
        self.reward_func = reward_func

        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std
        
        # generate random features
        self.reset()

    @property
    def states(self):
        """Return [0, ...,n_states-1]
        """
        return range(self.n_states)
        
    @property
    def actions(self):
        """Return [0, ...,n_actions-1]
        """
        return range(self.n_actions)
    
    def new_state(self, s, a):
        """Return state reached by the MDP after taking action a at state s.
        """
        return np.random.choice(self.states, p=self.transition_matrix[s, a])
        
    def reset(self):
        """Generate new features and new rewards.
        """
        self.reset_features()
        self.reset_rewards()
        self.reset_transition_matrix()
        
    def reset_iteration(self, direction):
        """Set the clock at the horizon.
        """
        if direction == 'backward':
            self.iteration = self.H-1
        elif direction == 'forward':
            self.iteration = 0
        else:
            raise Exception('Unknown direction {}'.format(direction))
    
    def reset_features(self):
        """Generate normalized random N(0,1) features phi(s,a)
        where s is the current state and a the action.
        """
        x = np.random.randn(self.n_states, self.n_actions, self.n_features)
        x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.n_states, self.n_actions, self.n_features)
        self.features = x
        self.features_flat = x.flatten().reshape(-1, self.n_features)

    def reset_rewards(self):
        """Generate rewards for each transition (state, action, next_state) and each round,
        following reward_function + Gaussian noise.
        This part is a priori not linear in the features, as reward_function may not be linear.
        """
        self.rewards = np.array(
            [
                self.reward_func(self.features[s, a]) + self.noise_std*np.random.randn()\
                for t, s, a in itertools.product(range(self.H), self.states, self.actions)
            ]
        ).reshape(self.H, self.n_states, self.n_actions)
    
    def reset_transition_matrix(self):
        """Generate random transition matrix from the features.
        This part is linear in the features.
        """
        psi = np.random.randn(self.n_states, self.n_features)
        P = np.array(
            [
                np.dot(self.features[s, a], psi[next_s]) \
                for s, a, next_s in itertools.product(self.states, self.actions, self.states)
            ]
        ).reshape(self.n_states, self.n_actions, self.n_states)

        # make it positive
        P = np.abs(P)
        # make it sum to 1
        for s, a in itertools.product(self.states, self.actions):
            P[s, a] /= np.sum(P[s,a])
        self.transition_matrix = P
        
    @property
    def apply_policy(self):
        """Helper to reduce transition and reward tensors
        along the action dimension using a policy.
        """
        return lambda X, policy, s: X[s, policy[s]]
    
    @property
    def apply_action(self):
        """Helper to reduce transition and reward tensors
        along the action dimension.
        """
        return lambda X, a: X[:, a]
    
    def evaluate_policy(self, policy):
        """Compute the value function of the provided policy
        by backward induction.
        For simplicity, dterministic policy only: (T, n_states) -> action spaces
        """
        # initialize value function, at terminal time value is zero
        V = np.concatenate((np.empty((self.H, self.n_states)), np.zeros((1, self.n_states))))

        for h in reversed(range(self.H)):
            # reduce P and R to square (n_states, n_states) matrices using the policy 
            P_reduced = self.apply_policy(self.transition_matrix, policy[h, :], self.states)
            R_reduced = self.apply_policy(self.rewards[h], policy[h, :], self.states)
            V[h, :] = np.matmul(P_reduced, V[h+1, :]) + R_reduced.T
        
        return V
    
    def optimal_policy(self):
        """Compute the optimal policy and its value function by backward induction.
        """
        # initialize value function, at terminal time value is zero
        V = np.concatenate((np.empty((self.H, self.n_states)), np.zeros((1, self.n_states))))
        policy = np.empty((self.H, self.n_states)).astype(int)

        for h in reversed(range(self.H)):
            # Bellman optimal equation
            v = np.array(
                [
                    np.matmul(self.apply_action(self.transition_matrix, a), V[h+1, :])\
                    +self.apply_action(self.rewards[h], a) for a in self.actions
                ]
            )            
            policy[h, :] = np.argmax(v, axis=0)
            V[h, :] = self.apply_policy(v.T, policy[h, :], self.states)

        return V, policy
    
    def sanity_policy(self):
        """Check that optimal value function is the value function
        of the optimal policy.
        """
        V_star, pi_star = self.optimal_policy()
        assert np.array_equal(self.evaluate_policy(pi_star), V_star), \
        'Check the implementation of backward induction for value iteration!'