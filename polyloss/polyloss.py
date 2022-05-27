import torch
from torch.nn.functional import cross_entropy, one_hot, softmax


class PolyLoss(torch.nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    <https://arxiv.org/abs/2204.12511>
    """

    def __init__(self, epsilon=2.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        ce = cross_entropy(outputs, targets)
        pt = one_hot(targets, outputs.size()[1]) * softmax(outputs, 1)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()
