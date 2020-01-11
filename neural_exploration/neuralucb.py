import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Dense
from .ucb import UCB


class NeuralUCB(UCB):
    """Neural UCB.
    """
    def __init__(self,
                 bandit,
                 hidden_size=20,
                 reg_factor=1.0,
                 delta=0.01,
                 confidence_scaling_factor=-1.0,
                 training_window=100,
                 learning_rate=0.01,
                 batch_size=-1,
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
        self.model.add(Dense(self.hidden_size, input_dim=bandit.n_features, activation='relu'))
        self.model.add(Dense(1, input_dim=self.hidden_size))

        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

        super().__init__(bandit, 
                         reg_factor=reg_factor,
                         confidence_scaling_factor=confidence_scaling_factor,
                         delta=delta,
                         throttle=throttle,
                        )

    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return self.model.count_params()

    @property
    def confidence_multiplier(self):
        """Constant equal to confidence_scaling_factor
        """
        return self.confidence_scaling_factor
    
    @property
    def output_gradient_func(self):
        """Function to get and compute network gradient.
        """
        grads = K.gradients(self.model.output, self.model.trainable_weights)
        inputs = self.model.inputs
        return K.function(inputs, grads)

    def update_output_gradient(self):
        """Get gradient of network prediction w.r.t network weights.
        """
        func = self.output_gradient_func
        batch = self.bandit.features[self.iteration]
        self.grad_approx = np.array(
            [
                np.concatenate([g.flatten() for g in func([[x]])]) for x in batch
            ]
        )
   
    def reset(self):
        """Reset the internal estimates.
        """
        self.reset_upper_confidence_bounds()
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

    def train(self):
        """Train neural approximator.
        """
        iterations_so_far = range(np.max([0, self.iteration-self.training_window]), self.iteration+1)
        actions_so_far = self.actions[np.max([0, self.iteration-self.training_window]):self.iteration+1]

        x_train = self.bandit.features[iterations_so_far, actions_so_far]
        y_train = self.bandit.rewards[iterations_so_far, actions_so_far]
        
        self.model.fit(x_train,
                       y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=self.nn_verbose,
                      )
        
    def predict(self):
        """Predict reward.
        """
        self.mu_hat[self.iteration] = self.model.predict(
            self.bandit.features[self.iteration]
        ).squeeze()
