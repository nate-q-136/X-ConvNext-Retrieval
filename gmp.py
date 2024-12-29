from torch import nn

class GlobalMaxPooling2d(nn.Module):
    def forward(self, x):
        return x.max(dim=2, keepdim=False)[0].max(dim=2, keepdim=False)[0]
