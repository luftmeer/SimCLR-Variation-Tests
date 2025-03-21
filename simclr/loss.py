import torch
import torch.nn as nn

class NTXentLoss(nn.Module):
    def __init__(self,  batch_size: int, device, temperature: float=0.5):
        super(NTXentLoss, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_fn = nn.CosineSimilarity(dim=2)
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.device = device
        
    def mask_correlated_samples(self, batch_size: int):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask    
    
    
    def forward(self, Zs):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size
        total_loss = 0
        for i in range(len(Zs)-1):
            for j in range(i+1, len(Zs)):
                z_i = Zs[i]
                z_j = Zs[j]
                z_i = z_i.to(self.device)
                z_j = z_j.to(self.device)
                
                z = torch.cat((z_i, z_j), dim=0)

                sim = self.similarity_fn(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
                sim_i_j = torch.diag(sim, self.batch_size)
                sim_j_i = torch.diag(sim, -self.batch_size)

                # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
                positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
                negative_samples = sim[self.mask].reshape(N, -1)

                labels = torch.zeros(N).to(positive_samples.device).long()
                logits = torch.cat((positive_samples, negative_samples), dim=1)
                loss = self.criterion(logits, labels)
                loss /= N
                total_loss += loss
        return total_loss
        
        """N = len(Zs) * self.batch_size
        
        logits = torch.empty(size=(self.batch_size,))
        for query in range(self.batch_size):
            pos_pair = torch.exp(self.similarity_fn(Zs[0][query], Zs[1][query]) / self.temperature)
            neg_pair_sum = torch.empty_like((pos_pair))
            for k in range(self.batch_size):
                if query != k:
                    neg_pair_sum += torch.exp(self.similarity_fn(Zs[0][query], Zs[1][k]) / self.temperature)
                    
            logits[query] = -torch.log(pos_pair / neg_pair_sum)
            
        labels = torch.arange(self.batch_size, dtype=torch.float)
        
        loss = self.criterion(logits, labels)
        loss /= N
        return loss"""