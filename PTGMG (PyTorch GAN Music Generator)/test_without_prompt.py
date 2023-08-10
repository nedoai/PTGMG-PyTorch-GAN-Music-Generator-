import os
import numpy as np
import torch
from generator_definition import Generator
import soundfile as sf

latent_dim = 100
n_mels = 128
hop_length = 512
max_audio_duration = max([181.33, 143.55, 140.41, 169.82, 132.61])
time_steps = int(max_audio_duration * 44100 / hop_length)
input_shape = (1, n_mels, time_steps)

generator = Generator(latent_dim, input_shape)
generator.load_state_dict(torch.load('best_generator.pth')) # If you want test different model - replace transferred_generator.pth and paste path for your model
generator.eval()  # Switch mode to evaluation mode

def generate_audio(generator, num_samples):
    z = torch.randn(num_samples, latent_dim)
    with torch.no_grad():
        generated_audio = generator(z)
    return generated_audio

output_folder = 'generated_audio' # Or different folder.
os.makedirs(output_folder, exist_ok=True)

# Generating
num_samples_to_generate = 10
generated_audio = generate_audio(generator, num_samples_to_generate)
for i, audio_tensor in enumerate(generated_audio):
    audio_array = (audio_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16)
    audio_path = os.path.join(output_folder, f'generated_audio_{i}.wav')
    sf.write(audio_path, np.ravel(audio_array), 44100)

print(f"{num_samples_to_generate} audio files saved in {output_folder}")