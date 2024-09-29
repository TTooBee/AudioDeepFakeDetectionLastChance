import librosa
import torch, torchaudio
import os
import glob
from scipy import signal

audio_path = "../AudioDeepFakeDetection/real_temp/wav/LJ001-0001_16k.wav"

waveform, Fs = torchaudio.load(audio_path)

waveform = waveform.reshape(-1)
print(f'shape of audio : {waveform.shape}')

frame = waveform[1000:1000+320]

print(f'shape of audio frame : {frame.shape}')

frame = frame.numpy()

a = librosa.lpc(y=frame, order=10)
print(f'lpc : {a}')

H, w = signal.freqz(1, a, 1024)

print(abs(H))