import torch
from torch.nn.functional import one_hot


class PolyLoss(torch.nn.Module):
    """
    Implementation of poly loss.
    Refers to `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
    <https://arxiv.org/abs/2204.12511>
    """

    def __init__(self, num_classes=1000, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes

    def forward(self, output, target):
        ce = self.criterion(output, target)
        pt = one_hot(target, num_classes=self.num_classes) * self.softmax(output)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=-1))).mean()
