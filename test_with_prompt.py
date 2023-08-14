import os
import numpy as np
import torch
import librosa
import soundfile as sf
import torch.nn as nn
import torchtext
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class Generator(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, np.prod(input_shape)),
            nn.Tanh()
        )

    def forward(self, z):
        generated_audio = self.fc(z)
        generated_audio = generated_audio.view(generated_audio.size(0), *self.input_shape)
        return generated_audio

model_path = 'best_generator.pth' # Or diffrent model
latent_dim = 100
n_mels = 128
hop_length = 512
max_audio_duration = max([181.33, 143.55, 140.41, 169.82, 132.61])
time_steps = int(max_audio_duration * 44100 / hop_length)
input_shape = (1, n_mels, time_steps)

generator = Generator(latent_dim, input_shape)
generator.load_state_dict(torch.load(model_path))
generator.eval()

predefined_vocab = ['<pad>', '<unk>', '<bos>', '<eos>']

# RU Tokenizer
tokenizer = torchtext.data.utils.get_tokenizer('moses', language='ru')

counter = Counter()
vocab = torchtext.vocab.Vocab(counter=counter, specials=predefined_vocab)

def text_to_sequence(text):
    tokens = tokenizer(text)
    sequence = [vocab[token] for token in tokens]
    return sequence

def generate_audio(request):
    description = [request]
    encoded_description = [torch.tensor(text_to_sequence(d)) for d in description]
    padded_description = pad_sequence(encoded_description, batch_first=True, padding_value=0)
    padded_description = padded_description.to("cpu")

    with torch.no_grad():
        z = torch.randn(padded_description.size(0), latent_dim)
        generated_audio = generator(z).squeeze().cpu().numpy()

    mel_spec = generated_audio * 0.5 + 0.5
    mel_spec = librosa.db_to_amplitude(mel_spec)
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=44100, hop_length=hop_length)

    audio /= np.max(np.abs(audio))

    return audio

request = "Мягкий фонк для отдыха"
generated_audio = generate_audio(request)

output_path = 'generated_audio.wav'
sf.write(output_path, generated_audio, 44100, format='WAV')
print("Audio saved in: ", output_path)
