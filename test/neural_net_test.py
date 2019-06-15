import sys

sys.path.append('../src')
from neural_net import EzNet
import unittest
import numpy as np


class TestNeuralNet(unittest.TestCase):

    def test_noramlize_zeros(self):
        inp = np.zeros(shape=(3, 2))
        expected = np.zeros(shape=(3, 2))
        net = EzNet(np.size(inp, axis=1))
        net.set_normalization_params(inp)
        norm = net.normalize(inp)
        err_margin = np.full(np.shape(inp), 0.01)
        assert np.all(np.less(np.abs(norm - expected), err_margin))

    def test_noramlize_nonzero(self):
        inp = np.array([[1, 2, 3, 2, 1],
                        [2, 3, 4, 3, 2],
                        [3, 4, 5, 4, 3]])
        expected = np.array([[-1.225, -1.225, -1.225, -1.225, -1.225],
                             [0, 0, 0, 0, 0],
                             [1.225, 1.225, 1.225, 1.225, 1.225]])
        net = EzNet(np.size(inp, axis=1))
        net.set_normalization_params(inp)
        norm = net.normalize(inp)
        err_margin = np.full(np.shape(inp), 0.01)
        assert np.all(np.less(np.abs(norm - expected), err_margin))

    def test_feed_forward_single_row(self):
        inp = np.zeros(shape=(1, 3))
        expected = np.array([[0.5]])
        net = EzNet(np.size(inp, axis=1))
        net.set_normalization_params(inp)
        net._thetas[-1][:, 0] = 0
        prob_vals = net.predict(inp)[-1][-1]
        err_margin = np.full(np.shape(expected), 0.001)
        assert np.all(np.less(np.abs(prob_vals - expected), err_margin))

    def test_feed_forward_single_row(self):
        inp = np.zeros(shape=(4, 3))
        expected = np.full(shape=(4, 1), fill_value=0.5)
        net = EzNet(np.size(inp, axis=1))
        net.set_normalization_params(inp)
        net._thetas[-1][:, 0] = 0
        prob_vals = net.predict(inp)[-1][-1]
        err_margin = np.full(np.shape(expected), 0.001)
        assert np.all(np.less(np.abs(prob_vals - expected), err_margin))

    def test_compute_loss(self):
        x_inp = np.array([[1, 2],
                          [1, 3],
                          [2, 4]])
        y_inp = np.array([[1],
                          [3],
                          [4]])
        expected = 1.25
        net = EzNet(np.size(x_inp, axis=1), None, 10)
        for theta in net._thetas:
            theta[:, :] = 0
        loss = net.compute_loss(x_inp, y_inp)
        err_margin = 0.005
        assert abs(loss - expected) < err_margin
