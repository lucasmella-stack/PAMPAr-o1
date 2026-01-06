import torch.nn as nn

class PrintedVsHandwritten(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.classifier(x)
