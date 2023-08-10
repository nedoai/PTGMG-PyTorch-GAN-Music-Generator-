import torch.nn as nn
import numpy as np

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(input_shape)),
            nn.Tanh()
        )
        
    def forward(self, z):
        generated_audio = self.fc(z)
        generated_audio = generated_audio.view(generated_audio.size(0), *self.input_shape)
        return generated_audio
    
# Don't touch here anything if you don't know how you can optimize this