import os
import librosa

audio_path = r'audios/atmo_phonk' # Paste audio folder

def get_audio_parameters(audio_files):
    for file_path in audio_files:
        audio, sr = librosa.load(file_path, sr=None)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Sample Rate (Hz): {sr}")
        print(f"Duration (s): {len(audio) / sr:.2f}")
        print(f"Number of Samples: {len(audio)}")
        print(f"Shape of Audio Signal: {audio.shape}")
        print()

audio_files = [os.path.join(audio_path, filename) for filename in os.listdir(audio_path) if filename.endswith('.wav')]
get_audio_parameters(audio_files)
