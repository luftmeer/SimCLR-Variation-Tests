import torch
import torch.nn as nn

class SimCLR(nn.Module):
    def __init__(self, encoder, n_features: int, projection_dim: int, image_size: int, batch_size: int, device: str='cuda'):
        super(SimCLR, self).__init__()
        
        self.encoder = encoder
        # Replace fc or classifier with Idendity function
        if hasattr(self.encoder, 'fc'):
            self.encoder.fc = nn.Identity()
        elif hasattr(self.encoder, 'classifier'):
            self.encoder.classifier = nn.Identity()
        else:
            raise AttributeError(f"Encoder not supported or of wrong type ({type(self.encoder)})")
        
        self.device = device
        
        self.image_size = image_size
        self.batch_size = batch_size
        
        self.projector = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_features, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=n_features, out_features=projection_dim, bias=False)
        )
    # Takes a list of size n (amount of augmentations), encoding and projecting each batch individually. Returning two lists with embeddings and projections
    def forward(self, Xs: list):
        hs = []
        zs = []
            
        for batch in Xs:
            batch = batch.to(self.device)
            h = self.encoder(batch)
            hs.append(h)
            
            h = h.to(self.device)
            z = self.projector(h)
            zs.append(z)
                
        return hs, zs