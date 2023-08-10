import os
import numpy as np
import torch
from generator_definition import Generator
import soundfile as sf

"""

Test your final result

"""

latent_dim = 100
n_mels = 128
hop_length = 512
max_audio_duration = max([181.33, 143.55, 140.41, 169.82, 132.61])
time_steps = int(max_audio_duration * 44100 / hop_length)
input_shape = (1, n_mels, time_steps)

generator = Generator(latent_dim, input_shape)
generator.load_state_dict(torch.load('transferred_generator.pth')) # If you want test different model - replace transferred_generator.pth and paste path for your model
generator.eval()  # Switch mode to evaluation mode

output_folder = 'generated_audio' # Or paste different folder
os.makedirs(output_folder, exist_ok=True)

prompt = input("Write your prompt (what you want to hear): ")

z = torch.randn(1, latent_dim)
with torch.no_grad():
    generated_audio = generator(z).squeeze().cpu().numpy()

audio_path = os.path.join(output_folder, f"{prompt}.wav")
sf.write(audio_path, np.ravel((generated_audio * 32767.0).astype(np.int16)), 44100)

print(f"Audio generated succesfully! {audio_path}")