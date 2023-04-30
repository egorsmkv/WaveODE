import argparse
import os
import numpy as np
import librosa
from scipy.io import wavfile
import soundfile as sf

from utils import config_setup


def GenerateMELs(args):

    for wav_name in os.listdir(args.wav_folder):
        wav_path = os.path.join(args.wav_folder, wav_name)
        mel_path = wav_path.replace('/wavs/', '/mels/').replace('.wav', '.npy')

        print('Converting')
        print(wav_path)
        print('->')
        print(mel_path)
        print()

        y, sr = librosa.load(wav_path)

        # mel_dim: 80
        # hop_size: 256
        # win_size: 1024

        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80
        )

        np.save(mel_path, melspec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help='Path to output')
    parser.add_argument('--wav_folder', type=str, required=True, help='Path to WAVs')
    parser.add_argument('--mel_folder', type=str, required=True, help='Path to MELs')
   
    args = parser.parse_args()

    GenerateMELs(args)

if __name__ == "__main__":
    main() 
