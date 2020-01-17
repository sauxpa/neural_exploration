import numpy as np
import itertools
import torch
import torch.nn as nn
from .ucbvi import UCBVI
from .utils import Model


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
                 p=0.0,
                 learning_rate=0.01,
                 batch_size=1,
                 epochs=50,
                 throttle=1,
                 use_cuda=False,
                ):

        # hidden size of the NN layers
        self.hidden_size = hidden_size
        
        # NN parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            raise Exception(
                'Not yet CUDA compatible : TODO for later (not necessary to obtain good results')
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
    
        # dropout rate
        self.p = p
        
        # neural network
        self.model = Model(mdp.n_features, self.hidden_size, self.p).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
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
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)    
    
    def update_output_gradient(self):
        """Get gradient of network prediction w.r.t network weights.
        """
        for s, a in itertools.product(self.mdp.states, self.mdp.actions):
            x = torch.FloatTensor(self.mdp.features[s, a].reshape(1,-1)).to(self.device)
            self.model.zero_grad()
            y = self.model(x)
            y.backward()

            self.grad_approx[s, a] = torch.cat(
                [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.model.parameters() if w.requires_grad]
            ).to(self.device)

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
        """Train neural approximator.
        """
        x_train = torch.FloatTensor(self.mdp.features[self.state, self.action]).to(self.device)
        y_train = torch.FloatTensor([self.reward + np.max(
            self.Q_hat[self.mdp.iteration+1, self.buffer_states[self.mdp.iteration+1]]
        )]).to(self.device)
        
        # train mode
        self.model.train()
        y_pred = self.model.forward(x_train).squeeze()
        
        loss = nn.MSELoss()(y_train, y_pred)
       
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def predict(self):
        """Predict reward.
        """
        # eval mode
        self.model.eval()
        self.Q_hat[self.mdp.iteration] = self.model.forward(
            torch.FloatTensor(self.mdp.features_flat).to(self.device)
        ).detach().reshape(self.mdp.n_states, self.mdp.n_actions)
        
