import torch.nn as nn

class StyleSelector(nn.Module):
    def __init__(self, input_dim, num_styles):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_styles)

    def forward(self, x):
        return self.classifier(x)
