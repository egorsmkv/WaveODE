import argparse
import os
import torch
import torch.utils.data
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn


MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def generate_mels(args):
    # Config 
    n_fft = 1024
    num_mels = 80
    hop_size = 256
    win_size = 1024
    fmin = 0.0
    fmax = 8000.0
    sampling_rate = 22050

    # Iterate over all files
    for wav_name in os.listdir(args.wav_folder):
        wav_path = os.path.join(args.wav_folder, wav_name)
        mel_path = wav_path.replace('/wavs/', '/mels/').replace('.wav', '.mel')

        input_audio, _ = load_wav(wav_path)
        input_audio = input_audio / MAX_WAV_VALUE

        output_audio = torch.FloatTensor(input_audio)
        output_audio = output_audio.unsqueeze(0)

        # Extract MEL spectrogram
        mel = mel_spectrogram(output_audio, n_fft, num_mels,
                              sampling_rate, hop_size, win_size, fmin, fmax,
                              center=False)

        mel_squeezed = mel.squeeze()

        # Log operation
        print('Converting')
        print(wav_path)
        print('->')
        print(mel_path)
        print()

        # Save it
        torch.save(mel_squeezed, mel_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, required=True, help='Path to output')
    parser.add_argument('--wav_folder', type=str, required=True, help='Path to WAVs')
    parser.add_argument('--mel_folder', type=str, required=True, help='Path to MELs')

    args = parser.parse_args()

    generate_mels(args)
