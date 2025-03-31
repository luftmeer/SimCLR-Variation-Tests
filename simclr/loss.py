import torch
import torch.nn as nn

class NTXentLoss(nn.Module):
    def __init__(self,  batch_size: int, device, temperature: float=0.5):
        super(NTXentLoss, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_fn = nn.CosineSimilarity(dim=2)
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # When distributed, the batch size is world_size times the given config size.
        N = 2 * self.batch_size
        
        # Dynamically calculate the mask
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, i + self.batch_size] = 0
            mask[i + self.batch_size, i] = 0
        
        z = torch.cat((z_i, z_j), dim=0).float()

        sim = self.similarity_fn(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros( N, dtype=torch.long, device=positive_samples.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1).float()
        loss = self.criterion(logits, labels)
        loss /= N
        return loss, logits.detach()
        