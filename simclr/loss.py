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
        #N = 2 * self.batch_size
        N = z_i.shape[0]
        
        z = torch.cat((z_i, z_j), dim=0)
        z = nn.functional.normalize(z, dim=1)
        full_N = z.shape[0]
        
        # Dynamically calculate the mask
        mask = torch.ones((full_N, full_N), dtype=bool, device=self.device)
        mask.fill_diagonal_(False)
        for i in range(N):
            mask[i, i + N] = 0
            mask[i + N, i] = 0
        

        #sim = self.similarity_fn(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, N)# // 2)
        sim_j_i = torch.diag(sim, -N)# // 2)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        #positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(-1, 1) #.reshape(N, 1)
        #negative_samples = sim[mask].reshape(N, -1).view(N, -1)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(full_N, 1)
        #positive_samples = torch.cat((torch.diag(sim, N), torch.diag(sim, -N)), dim=0).view(full_N, 1)
        negative_samples = sim[mask].view(full_N, -1)
        print(f'{positive_samples.mean().item()=}')
        #labels = torch.zeros(N, dtype=torch.long, device=positive_samples.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1).float()
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = self.criterion(logits, labels)
        loss /=full_N
        return loss, logits.detach().cpu(), sim.detach().cpu(), positive_samples.detach().cpu(), negative_samples.detach().cpu()