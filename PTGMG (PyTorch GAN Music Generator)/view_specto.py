import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Замените на путь к вашему аудио-файлу
audio_file_path = r'generated_audio\атмосферный фонк.wav'

# Загрузка аудио
audio, sr = librosa.load(audio_file_path, sr=None)

# Вычисление спектрограммы
n_mels = 128
hop_length = 512
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

# Отображение спектрограммы
plt.figure(figsize=(10, 6))
plt.imshow(log_mel_spec, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.show()
