# -*- coding: utf-8 -*-
"""
Class representing a multi-layer perceptron neural network and holding the
logic to train it on a meteography dataset and make predictions.
"""

import logging
import time

import theano as t
import lasagne

logger = logging.getLogger(__name__)


class Perceptron:
    def __init__(self, insize, outsize, nb_units=[50], learn_rate=50,
                 update_func=lasagne.updates.momentum):
        """
        Create a Perceptron

        Parameters
        ----------
        insize : int
            The dimensionality of the network input
        outsize : int
            The dimensionality of the network output
        nb_units : list of int
            The number of neurons in the each hidden layer. The length of the
            list will thus be the number of hidden layers
        learn_rate : number
            The default learning rate to use during training
        update_func : function
            A function that accepts the same first 3 arguments as lasagne
            update functions, and return a theano update dictionary. The
            dictionary will be used to update the model during training
        """
        input_var = t.tensor.fmatrix('input_var')
        target_var = t.tensor.fmatrix('target_var')

        sigmoid = lasagne.nonlinearities.sigmoid
        dense_layer = lasagne.layers.DenseLayer
        layer = lasagne.layers.InputLayer((None, insize), input_var=input_var)
        for nu in nb_units:
            layer = dense_layer(layer, num_units=nu, nonlinearity=sigmoid)
        self.layer_out = dense_layer(layer, num_units=outsize,
                                     nonlinearity=sigmoid)

        prediction = lasagne.layers.get_output(self.layer_out)
        loss = lasagne.objectives.squared_error(prediction, target_var).mean()
        params = lasagne.layers.get_all_params(self.layer_out, trainable=True)
        updates = update_func(loss, params, learning_rate=learn_rate)

        self._predict = t.function([input_var], prediction, name='predict')
        self._loss = t.function([input_var, target_var], loss, name='loss')
        self._train = t.function([input_var, target_var], loss,
                                 updates=updates, name='train')

    def train(self, X, target, batch_size=None,
              min_loss_diff=.999, max_epoch=500):
        """
        Train the network with stochastic gradient descent on the input `X`
        and expected output `target`.

        Parameters
        ----------
        X : matrix of shape (any, insize)
            The training input, with the number of columns specified when
            instantiating this Perceptron
        target : matrix of shape (len(X), outsize)
            The expected output, with the same number of lines as X and the
            number of columns specified when instantiating this Perceptron
        batch_size : int in [1, len(X)] or None
            The number of examples to use in each batch of the gradient descent
            When None (the default value), the whole set of examples is used
            (plain gradient descent)
        min_loss_diff : float in [0, 1]
            The convergence criteria to stop the gradient descent. 0 means any
            any step is considered as convergence, 1 means real convergence
            (descent stopped)
        max_epoch : int > 0
            Maximum number of traning cycles on the whole set of examples

        Return
        ------
        list, float: The list of the loss values at the end of each training
            cycle and the time (in seconds) it took to perform the training
        """
        if batch_size is None:
            batch_size = len(X)
        costs = []
        start_time = time.time()

        for epoch in range(max_epoch):
            for b in range(0, len(X), batch_size):
                batch = X[b:b+batch_size]
                batch_target = target[b:b+batch_size]
                cost = self._train(batch, batch_target)
            if batch_size == len(X):
                costs.append(cost)
            else:
                costs.append(self.compute_loss(X, target))
            logger.info("Epoch %d: loss=%f" % (epoch, costs[-1]))
            if epoch > 0:
                conv_sample = costs[-3:]
                maxc, minc = max(conv_sample), min(conv_sample)
                #Stop when descent flattens or starts to reverse
                converging = minc / maxc >= min_loss_diff
                ascending = minc == conv_sample[0]
                if converging or ascending:
                    break

        elapsed = time.time() - start_time
        logger.info("Total training time: %.3fs in %d epochs, cost: %f"
                    % (elapsed, epoch+1, costs[-1]))
        return costs, elapsed

    def predict(self, X):
        return self._predict(X)

    def compute_loss(self, X, target):
        return self._loss(X, target)

    def get_params(self):
        return lasagne.layers.get_all_param_values(self.layer_out)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.layer_out, params)
