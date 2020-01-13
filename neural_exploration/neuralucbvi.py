import numpy as np
import itertools
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Dense
from keras.regularizers import l2
from .ucbvi import UCBVI

import gc

class NeuralUCBVI(UCBVI):
    """Value Iteration with NeuralUCB exploration.
    """
    def __init__(self,
                 mdp,
                 hidden_size=20,
                 n_episodes=1,
                 init_state=0,
                 reg_factor=1.0,
                 confidence_scaling_factor=-1.0,
                 training_window=100,
                 learning_rate=0.01,
                 batch_size=1,
                 epochs=50,
                 nn_verbose=0,
                 throttle=1,
                ):

        # hidden size of the NN layers
        self.hidden_size = hidden_size
        
        # number of rewards in the training buffer
        self.training_window = training_window
        
        # NN parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.nn_verbose = nn_verbose
        
        # neural network
        self.model = Sequential()
        self.model.add(Dense(self.hidden_size, input_dim=mdp.n_features, activation='relu'))
        self.model.add(Dense(1, input_dim=self.hidden_size, kernel_regularizer=l2(reg_factor)))

        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        super().__init__(mdp,
                         n_episodes=n_episodes,
                         init_state=init_state,
                         reg_factor=reg_factor,
                         confidence_scaling_factor=confidence_scaling_factor,
                         throttle=throttle,
                        )

    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return self.model.count_params()
    
    @property
    def output_gradient_func(self):
        """Function to get and compute network gradient.
        """
        grads = K.gradients(self.model.output, self.model.trainable_weights)
        inputs = self.model.inputs
        return K.function(inputs, grads)
    
    def update_output_gradient(self):
        """For linear approximators, simply returns the features.
        """
        ### THIS IS SLOW (AND LEAKING MEMORY)
        func = self.output_gradient_func
        
        self.grad_approx = np.array(
            [
                np.concatenate(
                    [
                        g.flatten()/np.sqrt(self.hidden_size) for g in func([[
                            self.mdp.features[s, a]
                        ]])
                    ]) for s, a in itertools.product(self.mdp.states, self.mdp.actions)
            ]
        ).reshape(self.mdp.n_states, self.mdp.n_actions, self.approximator_dim)

    def reset(self):
        """Return the internal estimates
        """
        self.reset_upper_confidence_bounds()
        self.reset_regrets()
        self.reset_policy()
        self.reset_state_action_reward_buffer()
        self.reset_A_inv()
        self.reset_grad_approx()

    @property
    def confidence_multiplier(self):
        """LinUCB confidence interval multiplier.
        """
        return self.confidence_scaling_factor
    
    def train(self):
        """Update linear predictor theta.
        """
        x_train = np.array([self.mdp.features[self.state, self.action]])
        y_train = np.array([self.reward + np.max(
            self.Q_hat[self.mdp.iteration+1, self.buffer_states[self.mdp.iteration+1]]
        )])
        
        self.model.fit(x_train,
                       y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=self.nn_verbose,
                      )
    
    def predict(self):
        """Predict reward.
        """
#         self.Q_hat[self.mdp.iteration] = np.array(
#             [
#                 self.model.predict([[self.mdp.features[s, a]]]).squeeze() for s, a in itertools.product(self.mdp.states, self.mdp.actions)
#             ]
#         ).reshape(self.mdp.n_states, self.mdp.n_actions)

        self.Q_hat[self.mdp.iteration] = self.model.predict(
            [self.mdp.features_flat]
        ).reshape(self.mdp.n_states, self.mdp.n_actions)
