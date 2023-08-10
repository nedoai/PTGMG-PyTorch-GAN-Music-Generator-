import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from torch.nn.utils.rnn import pad_sequence
from cfg import fine_tune_audio_path, fine_tune_desc_path


"""

THIS CODE USING FOR TUNING MODEL WITH TRANSFER METHOD. USING THIS IF YOU WANT GET BEST RESULT.
Transfer learning is the pre-training of a pre-trained model on new data to solve another, closely related problem.
In this code I will implement pre-training of the generator on new audio files to improve sound generation. 

"""

class AudioDataset(Dataset):
    def __init__(self, audio_files, descriptions, max_time_steps):
        self.audio_files = audio_files
        self.descriptions = descriptions
        self.max_time_steps = max_time_steps

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        audio, _ = librosa.load(file_path, sr=44100)
        normalized_audio = audio / np.max(np.abs(audio))
        n_mels = 128
        hop_length = 512
        mel_spec = librosa.feature.melspectrogram(y=normalized_audio, sr=44100, n_mels=n_mels, hop_length=hop_length)
        log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

        if log_mel_spec.shape[1] > self.max_time_steps:
            log_mel_spec = log_mel_spec[:, :self.max_time_steps]
        else:
            pad_width = ((0, 0), (0, self.max_time_steps - log_mel_spec.shape[1]))
            log_mel_spec = np.pad(log_mel_spec, pad_width, mode='constant')

        description = self.descriptions[idx]

        return torch.tensor(log_mel_spec, dtype=torch.float32), description

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

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.fc = nn.Sequential(
            nn.Linear(np.prod(input_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, audio):
        audio_flat = audio.view(audio.size(0), -1)
        validity = self.fc(audio_flat)
        return validity

# Params
latent_dim = 100
n_mels = 128
hop_length = 512
max_audio_duration = max([181.33, 143.55, 140.41, 169.82, 132.61])
time_steps = int(max_audio_duration * 44100 / hop_length)
input_shape = (1, n_mels, time_steps)

def_audio_path = fine_tune_audio_path
def_description_path = fine_tune_desc_path

audio_files = [os.path.join(def_audio_path, filename) for filename in os.listdir(def_audio_path) if filename.endswith('.wav')]

with open(def_description_path, 'r', encoding="utf-8") as f:
    descriptions = f.readlines()

dataset = AudioDataset(audio_files, descriptions, time_steps)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), [item[1] for item in batch]))

# Creating instances of the generator and discriminator
generator = Generator(latent_dim, input_shape)
discriminator = Discriminator(input_shape)

# Creating optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Criterion
criterion = nn.BCELoss()

# Transfer method training
num_epochs = 10  # Epochs
best_loss = float('inf')
best_generator_weights = None

for epoch in range(num_epochs):
    for i, (real_audio, descriptions) in enumerate(dataloader):
        batch_size = real_audio.size(0)
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)
        # Train G
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        fake_audio = generator(z)
        real_loss = criterion(discriminator(real_audio), valid)
        fake_loss = criterion(discriminator(fake_audio.detach()), fake)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # Train G
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        gen_audio = generator(z)
        g_loss = criterion(discriminator(gen_audio), valid)
        g_loss.backward()
        optimizer_G.step()
        if g_loss.item() < best_loss:
            best_loss = g_loss.item()
            best_generator_weights = generator.state_dict()
            torch.save(best_generator_weights, 'best_transferred_generator.pth')

        print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] D Loss: {d_loss.item()} G Loss: {g_loss.item()}")
if best_generator_weights is not None:
    torch.save(best_generator_weights, 'best_transferred_generator.pth') # Saving best model

# Saving just final model.
torch.save(generator.state_dict(), 'transferred_generator.pth')