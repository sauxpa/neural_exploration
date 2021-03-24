import numpy as np
import torch.nn as nn


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv


class Model(nn.Module):
    """Template for fully connected neural network for scalar approximation.
    """
    def __init__(self,
                 input_size=1,
                 hidden_size=2,
                 n_layers=1,
                 activation='ReLU',
                 p=0.0,
                 ):
        super(Model, self).__init__()

        self.n_layers = n_layers

        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, 1)]
        else:
            size = [input_size] + [hidden_size, ] * (self.n_layers-1) + [1]
            self.layers = [nn.Linear(size[i], size[i+1]) for i in range(self.n_layers)]
        self.layers = nn.ModuleList(self.layers)

        # dropout layer
        self.dropout = nn.Dropout(p=p)

        # activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.dropout(self.activation(self.layers[i](x)))
        x = self.layers[-1](x)
        return x
