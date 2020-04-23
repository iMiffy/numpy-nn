import numpy as np
from nn.operators import softmax_cross_entropy


class Loss(object):

    def __init__(self):
        self.trainable = False  # Whether there are parameters in this layer that can be trained
        self.training = False  # The phrase, if for training then true

    def forward(self, input, labels):
        """Forward pass, reture output"""
        raise NotImplementedError

    def backward(self, input, labels):
        """Backward pass, return gradients to input"""
        raise NotImplementedError

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training


class SoftmaxCrossEntropy(Loss):
    def __init__(self, num_class):
        """Initialization

        # Arguments
            num_class: int, the number of category
        """
        super(SoftmaxCrossEntropy, self).__init__()
        self.num_class = num_class
        self.softmax_cross_entropy = softmax_cross_entropy()

    def forward(self, input, labels):
        output, probs = self.softmax_cross_entropy.forward(input, labels)
        return output, probs

    def backward(self, input, labels):
        in_grad = self.softmax_cross_entropy.backward(input, labels)
        return in_grad


class L2(Loss):
    def __init__(self, w=0.01):
        """Initialization

        # Arguments
            w: float, weight decay coefficient.
        """
        self.w = w

    def forward(self, params):
        """Forward pass

        # Arguments
            params: dictionary, store all weights of the whole model

        # Returns
            output: float, L2 regularization loss
        """
        loss = 0
        for n, v in params.items():
            # only decay the weights
            if 'weights' in n:
                loss += np.sum(v**2)
        output = 0.5 * self.w * loss
        return output

    def backward(self, params):
        """Backward pass

        # Arguments
            params: dictionary, store all weights of the whole model

        # Returns
            in_grad: dictionary, gradients to each weights in params 
        """
        in_grad = {}
        for k, v in params.items():
            if 'weights' in k:
                in_grad[k] = self.w * params[k]
            else:
                in_grad[k] = np.zeros(v.shape)
        return in_grad
