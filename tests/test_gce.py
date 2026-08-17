import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, q=0.7, weight=None):
        super().__init__()
        self.q = q
        self.weight = weight

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,)
        probs = F.softmax(logits, dim=1)
        
        # Get the probability of the true class
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Compute GCE loss: (1 - p^q) / q
        loss = (1.0 - torch.pow(target_probs, self.q)) / self.q
        
        if self.weight is not None:
            weights = self.weight[targets]
            loss = loss * weights
            
        return loss.mean()
        
logits = torch.randn(5, 5)
targets = torch.randint(0, 5, (5,))
criterion = GeneralizedCrossEntropyLoss()
print(criterion(logits, targets))
