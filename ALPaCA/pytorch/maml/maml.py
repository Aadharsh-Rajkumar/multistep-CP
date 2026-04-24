""" Code for the MAML algorithm and network definitions — PyTorch version. """
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


class MAML(nn.Module):
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5,
                 update_lr=1e-3, meta_lr=0.001, norm='None'):
        """Must call construct_model() after initializing MAML!"""
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.test_num_updates = test_num_updates
        self.norm = norm
        self.dim_hidden = [40, 40]
        self.loss_func = mse
        self.weights = None

    def construct_weights(self):
        weights = OrderedDict()
        weights['w1'] = nn.Parameter(torch.randn(self.dim_input, self.dim_hidden[0]) * 0.01)
        weights['b1'] = nn.Parameter(torch.zeros(self.dim_hidden[0]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = nn.Parameter(
                torch.randn(self.dim_hidden[i - 1], self.dim_hidden[i]) * 0.01)
            weights['b' + str(i + 1)] = nn.Parameter(torch.zeros(self.dim_hidden[i]))
        weights['w' + str(len(self.dim_hidden) + 1)] = nn.Parameter(
            torch.randn(self.dim_hidden[-1], self.dim_output) * 0.01)
        weights['b' + str(len(self.dim_hidden) + 1)] = nn.Parameter(torch.zeros(self.dim_output))
        return weights

    def forward_fc(self, inp, weights):
        hidden = F.relu(inp @ weights['w1'] + weights['b1'])
        for i in range(1, len(self.dim_hidden)):
            hidden = F.relu(hidden @ weights['w' + str(i + 1)] + weights['b' + str(i + 1)])
        return hidden @ weights['w' + str(len(self.dim_hidden) + 1)] + weights['b' + str(len(self.dim_hidden) + 1)]

    def construct_model(self):
        weights = self.construct_weights()
        self.weights = nn.ParameterDict(weights)
        self.meta_optimizer = optim.Adam(self.parameters(), lr=self.meta_lr)

    def task_metalearn(self, inputa, inputb, labela, labelb, num_updates):
        """Perform gradient descent for one task."""
        task_outputbs = []
        task_lossesb = []

        # Use OrderedDict for fast_weights so we can compute grads
        weights = OrderedDict(self.weights.items())

        # Forward on support set
        task_outputa = self.forward_fc(inputa, weights)
        task_lossa = self.loss_func(task_outputa, labela)

        # Inner loop gradient updates
        grads = torch.autograd.grad(task_lossa, list(weights.values()), create_graph=True)
        fast_weights = OrderedDict(
            (name, param - self.update_lr * grad)
            for (name, param), grad in zip(weights.items(), grads)
        )

        output = self.forward_fc(inputb, fast_weights)
        task_outputbs.append(output)
        task_lossesb.append(self.loss_func(output, labelb))

        for j in range(num_updates - 1):
            loss = self.loss_func(self.forward_fc(inputa, fast_weights), labela)
            grads = torch.autograd.grad(loss, list(fast_weights.values()), create_graph=True)
            fast_weights = OrderedDict(
                (name, param - self.update_lr * grad)
                for (name, param), grad in zip(fast_weights.items(), grads)
            )
            output = self.forward_fc(inputb, fast_weights)
            task_outputbs.append(output)
            task_lossesb.append(self.loss_func(output, labelb))

        return task_outputa, task_outputbs, task_lossa, task_lossesb

    def forward(self, inputa, inputb, labela, labelb, num_updates=None):
        """
        Meta-learning forward pass across a batch of tasks.
        All inputs: (meta_batch_size, n_samples, dim)
        """
        if num_updates is None:
            num_updates = max(self.test_num_updates, 1)

        meta_batch_size = inputa.shape[0]
        total_loss1 = 0.0
        total_losses2 = [0.0] * num_updates
        all_outputbs = [[] for _ in range(num_updates)]

        for i in range(meta_batch_size):
            outputa, outputbs, lossa, lossesb = self.task_metalearn(
                inputa[i], inputb[i], labela[i], labelb[i], num_updates
            )
            total_loss1 = total_loss1 + lossa
            for j in range(num_updates):
                total_losses2[j] = total_losses2[j] + lossesb[j]
                all_outputbs[j].append(outputbs[j])

        total_loss1 = total_loss1 / meta_batch_size
        total_losses2 = [l / meta_batch_size for l in total_losses2]
        all_outputbs = [torch.stack(obs) for obs in all_outputbs]

        return total_loss1, total_losses2, all_outputbs


## Loss functions
def mse(pred, label):
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    return torch.mean((pred - label) ** 2)


def xent(pred, label):
    return F.cross_entropy(pred, label)
