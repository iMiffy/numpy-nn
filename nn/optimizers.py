import numpy as np
import copy


class Optimizer():

    def __init__(self, lr):
        """Initialization

        # Arguments
            lr: float, learnig rate 
        """
        self.lr = lr

    def update(self, x, x_grad, iteration):
        """Update parameters with gradients"""
        raise NotImplementedError

    def sheduler(self, func, iteration):
        """learning rate sheduler, to change learning rate with respect to iteration

        # Arguments
            func: function, arguments are lr and iteration
            iteration: int, current iteration number in the whole training process (not in that epoch)

        # Returns
            lr: float, the new learning rate
        """
        lr = func(self.lr, iteration)
        return lr


class SGD(Optimizer):

    def __init__(self, lr=0.01, beta=0, decay=0, sheduler_func=None):
        """Initialization

        # Arguments
            lr: float, learnig rate 
            beta: float, the ratio of momentum
            decay: float, the learning rate decay ratio
        """
        super(SGD, self).__init__(lr)
        self.beta = beta
        self.momentum = None
        self.decay = decay
        self.sheduler_func = sheduler_func

    def update(self, w, w_grads, iteration):
        """Initialization

        # Arguments
            w: dictionary, all weights of model
            w_grads: dictionary, gradients to all weights of model, same keys with w
            iteration: int, current iteration number in the whole training process (not in that epoch)

        # Returns
            new_w: dictionary, new weights of model
        """
        new_w = {}
        if self.decay > 0:
            self.lr *= (1/(1+self.decay*iteration))
        if self.sheduler_func:
            self.lr = self.sheduler(self.sheduler_func, iteration)
        if not self.momentum:
            self.momentum = {}
            for k, v in w_grads.items():
                self.momentum[k] = np.zeros(v.shape)
        for k in list(w.keys()):
            self.momentum[k] = self.beta * self.momentum[k] + w_grads[k]
            new_w[k] = w[k] - self.lr * self.momentum[k]
        return new_w


class Adagrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=None, decay=0, sheduler_func=None):
        """Initialization

        # Arguments
            lr: float, learnig rate 
            epsilon: float, precision to avoid numerical error
            decay: float, the learning rate decay ratio
        """
        super(Adagrad, self).__init__(lr)
        self.epsilon = epsilon
        self.decay = decay
        if not self.epsilon:
            self.epsilon = 1e-8
        self.accumulators = None
        self.sheduler_func = sheduler_func

    def update(self, w, w_grads, iteration):
        """Initialization

        # Arguments
            w: dictionary, all weights of model
            w_grads: dictionary, gradients to all weights of model, same keys with w
            iteration: int, current iteration number in the whole training process (not in that epoch)

        # Returns
            new_w: dictionary, new weights of model
        """
        new_w = {}
        if self.decay > 0:
            self.lr *= (1/(1+self.decay*iteration))
        if self.sheduler_func:
            self.lr = self.sheduler(self.sheduler_func, iteration)
        if not self.accumulators:
            self.accumulators = {}
            for k, v in w.items():
                self.accumulators[k] = np.zeros(v.shape)
        for k in list(w.keys()):
            self.accumulators[k] += w_grads[k]**2
            new_w[k] = w[k] - self.lr * w_grads[k] / (np.sqrt(self.accumulators[k] + self.epsilon))
        return new_w


class RMSprop(Optimizer):
    def __init__(self, lr=0.001, bata=0.9, epsilon=None, decay=0, sheduler_func=None):
        """Initialization

        # Arguments
            lr: float, learnig rate 
            beta: float, the weight of moving average for second moment of gradient
            epsilon: float, precision to avoid numerical error
            decay: float, the learning rate decay ratio
        """
        super(RMSprop, self).__init__(lr)
        self.bata = bata
        self.epsilon = epsilon
        self.decay = decay
        if not self.epsilon:
            self.epsilon = 1e-8
        self.accumulators = None
        self.sheduler_func = sheduler_func

    def update(self, w, w_grads, iteration):
        """Initialization

        # Arguments
            w: dictionary, all weights of model
            w_grads: dictionary, gradients to all weights of model, same keys with w
            iteration: int, current iteration number in the whole training process (not in that epoch)

        # Returns
            new_w: dictionary, new weights of model
        """
        new_w = {}
        if self.decay > 0:
            self.lr *= (1/(1+self.decay*iteration))
        if self.sheduler_func:
            self.lr = self.sheduler(self.sheduler_func, iteration)
        if not self.accumulators:
            self.accumulators = {}
            for k, v in w.items():
                self.accumulators[k] = np.zeros(v.shape)
        for k in list(w.keys()):
            self.accumulators[k] = (self.bata*self.accumulators[k]) + (1-self.bata)*(w_grads[k]**2) 
            new_w[k] = w[k] - self.lr * w_grads[k] / (np.sqrt(self.epsilon + self.accumulators[k])) 
            
        return new_w


class Adam(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, bias_correction=False, sheduler_func=None):
        """Initialization

        # Arguments
            lr: float, learnig rate 
            beta_1: float
            beta_2: float
            epsilon: float, precision to avoid numerical error
            decay: float, the learning rate decay ratio
            bias_correction: bool
        """
        super(Adam, self).__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.bias_correction = bias_correction
        if not self.epsilon:
            self.epsilon = 1e-8

        self.momentum = None
        self.accumulators = None
        self.sheduler_func = sheduler_func

    def update(self, w, w_grads, iteration):
        """Initialization

        # Arguments
            w: dictionary, all weights of model
            w_grads: dictionary, gradients to all weights of model, same keys with w
            iteration: int, current iteration number in the whole training process (not in that epoch, starting from 0)

        # Returns
            new_w: dictionary, new weights of model
        """
        new_w = {}
        if self.decay > 0:
            self.lr *= (1/(1+self.decay*iteration))
        if self.sheduler_func:
            self.lr = self.sheduler(self.sheduler_func, iteration)
        if (self.accumulators is None) and (self.momentum is None):
            self.momentum = {}
            self.accumulators = {}
            for k, v in w.items():
                self.momentum[k] = np.zeros(v.shape)
                self.accumulators[k] = np.zeros(v.shape)
        for k in list(w.keys()):
            self.momentum[k] = self.beta_1 * self.momentum[k] + (1-self.beta_1) * w_grads[k]
            self.accumulators[k] = self.beta_2 * self.accumulators[k] + (1 - self.beta_2) * w_grads[k]**2
            if self.bias_correction:
                momentum_t = self.momentum[k] / (1 - self.beta_1**(iteration+1))
                accumulators_t = self.accumulators[k] / (1 - self.beta_2**(iteration+1))
                new_w[k] = w[k] - self.lr * momentum_t / (np.sqrt(accumulators_t+self.epsilon))
            else:
                new_w[k] = w[k] - self.lr * self.momentum[k] / (np.sqrt(self.accumulators[k] + self.epsilon))
        return new_w
