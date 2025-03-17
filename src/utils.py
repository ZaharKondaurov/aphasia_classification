import os
from io import BytesIO
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import random

from pydub import AudioSegment

import torch
import torchaudio
from torch import nn
from torch import hub
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

import librosa

from transformers import Wav2Vec2Processor


def get_speech_and_silence_timestamps(waveform: torch.Tensor,
                                      sr: int, return_seconds: bool = False,
                                      threshold: float = 0.6,
                                      min_speech_duration_ms: int = 500,
                                      min_silence_duration_ms: int = 1000):
    speech_model = load_silero_vad()
    duration = waveform.shape[0] // sr

    speech_timestamps = get_speech_timestamps(waveform, speech_model, threshold=threshold,
                                              min_speech_duration_ms=min_speech_duration_ms,
                                              min_silence_duration_ms=min_silence_duration_ms,
                                              sampling_rate=sr, return_seconds=return_seconds)
    silence_timestamps = []
    speech_duration = 0
    speech_end = 0

    for x in speech_timestamps:
        silence_timestamps.append({'start': speech_end, 'end': x['start'] - speech_end})
        speech_duration += x['end'] - x['start']

        speech_end = x['end']
    silence_timestamps.append({'start': speech_end, 'end': duration - speech_end})

    mean_speach_duration = 0
    if len(speech_timestamps) > 0:
        mean_speach_duration = speech_duration / len(speech_timestamps)
    mean_silence_duration = 0
    if len(silence_timestamps) > 0:
        mean_silence_duration = (duration - speech_duration) / len(silence_timestamps)

    return (speech_duration, len(speech_timestamps), speech_timestamps, mean_speach_duration,
            duration - speech_duration, len(silence_timestamps), silence_timestamps, mean_silence_duration)


def remove_silence(waveform: torch.Tensor,
                   sr: int,
                   return_seconds: bool = False,
                   threshold: float = 0.6,
                   min_speech_duration_ms: int = 500,
                   min_silence_duration_ms: int = 1000
                   ) -> torch.Tensor:
    _, _, speech_timestamps, _, _, _, _, _ = get_speech_and_silence_timestamps(waveform, sr,
                                                                               return_seconds=return_seconds,
                                                                               threshold=threshold,
                                                                               min_speech_duration_ms=min_speech_duration_ms,
                                                                               min_silence_duration_ms=min_silence_duration_ms)

    output = []
    for ts in speech_timestamps:
        output.append(waveform[ts['start'] * sr // 1000: ts['end'] * sr // 1000, ...])

    # print(len(output), output)
    if len(output) == 0 or len(output[0]) == 0:
        output = [waveform]
    output = torch.concatenate(output, dim=0)
    return output


class SignalWindowing(torch.nn.Module):

    def __init__(self,
                 window_size: int,
                 stride: int,
                 with_silence: bool = True,
                 sr: int = 8_000,
                 threshold: float = 0.6,
                 min_speech_duration_ms: int = 500,
                 min_silence_duration_ms: int = 1000):

        super(SignalWindowing, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.with_silence = with_silence
        self.sr = sr
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = x
        if not self.with_silence:
            signal = remove_silence(signal, sr=self.sr, threshold=self.threshold,
                                    min_speech_duration_ms=self.min_speech_duration_ms,
                                    min_silence_duration_ms=self.min_silence_duration_ms)
        # print(signal.shape)
        remainder = (signal.shape[-1] - self.window_size) % self.stride
        pad_count = 0

        if remainder != 0:
            pad_count = self.stride - remainder

        signal = torch.nn.functional.pad(signal, (0, pad_count), "constant", 0)
        chunks = signal.unfold(-1, self.window_size, self.stride)

        return chunks

class AphasiaDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_sample_rate=16000, fft_size=512,
                 hop_length=256, win_length=512, min_duration=10, max_duration=15, transforms=None):
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.min_duration = min_duration * 1000  # конвертируем в миллисекунды
        self.max_duration = max_duration * 1000
        self.data = []
        self.transforms = transforms

        # Загружаем список файлов из CSV
        df = pd.read_csv(csv_file)

        # Обработка и сегментация аудио
        for _, row in df.iterrows():
            file_name, label = row['file_name'], row['label']
            file_path = self.find_audio_file(file_name, label)
            if file_path:
                try:
                    segments = self.process_audio(file_path)
                    self.data.extend([(s, label) for s in segments])
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

        random.shuffle(self.data)

    def find_audio_file(self, file_name, label):
        """Ищем файл в соответствующей папке по метке"""
        folder = "Aphasia" if label == 1 else "Norm"
        file_name = file_name[:-4]
        file_path = os.path.join(self.root_dir, folder, f"{file_name}.3gp")  # Убрано ".wav"
        if os.path.exists(file_path):
            return file_path
        print(f"Warning: {file_name}.3gp not found in {folder} folder.")
        return None

    def process_audio(self, file_path):
        audio = AudioSegment.from_file(file_path, format="3gp")
        duration = len(audio)  # в миллисекундах
        segments = []

        if duration < self.min_duration:
            return [self.preprocess(audio)]

        start = 0
        while start + self.min_duration <= duration:
            segment_duration = min(random.randint(self.min_duration, self.max_duration), duration - start)
            end = start + segment_duration
            segment = audio[start:end]
            preprocessed_segment = self.preprocess(segment)
            if preprocessed_segment is not None:
                segments.append(preprocessed_segment)
            start = end
        return segments

    @abstractmethod
    def preprocess(self, segment):
        ...
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem, label = self.data[idx]

        return elem, torch.tensor(label, dtype=torch.long)


class AphasiaDatasetSpectrogram(AphasiaDataset):

    def __init__(self, csv_file, root_dir, target_sample_rate=16000, fft_size=512,
                 hop_length=256, win_length=512, min_duration=10, max_duration=15, transforms=None):
        super(AphasiaDatasetSpectrogram, self).__init__(csv_file, root_dir, target_sample_rate, fft_size,
                 hop_length, win_length, min_duration, max_duration, transforms)

    def preprocess(self, segment):
        try:
            buffer = BytesIO()
            segment.export(buffer, format="wav")
            buffer.seek(0)
            waveform, sample_rate = torchaudio.load(buffer)

            if sample_rate != self.target_sample_rate:
                resampler = Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[1] < self.fft_size:
                return None

            y = waveform.numpy().squeeze()
            spectrogram = librosa.stft(y, n_fft=self.fft_size, hop_length=self.hop_length, win_length=self.win_length)
            mag = np.abs(spectrogram).astype(np.float32)
            return torch.tensor(mag.T).unsqueeze(0)
        except Exception as e:
            print(f"Spectrogram error: {str(e)}")
            return None


class AphasiaDatasetMFCC(AphasiaDataset):

    def __init__(self, csv_file, root_dir, mfcc=128, n_mels=150, target_sample_rate=16000, fft_size=512,
                 hop_length=256, win_length=512, min_duration=10, max_duration=15, transforms=None):
        self.mfcc_class = torchaudio.transforms.MFCC(sample_rate=8_000, n_mfcc=mfcc,
                                                     log_mels=True, melkwargs={"n_fft": fft_size,
                                                                               "win_length": win_length,
                                                                               "hop_length": hop_length,
                                                                               "n_mels": n_mels})
        super(AphasiaDatasetMFCC, self).__init__(csv_file, root_dir, target_sample_rate, fft_size,
                 hop_length, win_length, min_duration, max_duration, transforms)

    def preprocess(self, segment):
        try:
            buffer = BytesIO()
            segment.export(buffer, format="wav")
            buffer.seek(0)
            waveform, sample_rate = torchaudio.load(buffer)

            if sample_rate != self.target_sample_rate:
                resampler = Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[1] < self.fft_size:
                return None

            y = waveform.squeeze()

            mfcc = self.mfcc_class(y)

            return torch.tensor(mfcc)
        except Exception as e:
            print(f"MFCC error: {str(e)}")
            return None


class AphasiaDatasetWaveform(AphasiaDataset):

    def __init__(self, csv_file, root_dir, target_sample_rate=16000,
                 hop_length=256, win_length=512, min_duration=10, max_duration=15, transforms=None):
        super(AphasiaDatasetWaveform, self).__init__(csv_file, root_dir, target_sample_rate,
                                                 hop_length=hop_length, win_length=win_length,
                                                     min_duration=min_duration, max_duration=max_duration,
                                                     transforms=transforms)

    def preprocess(self, segment):
        buffer = BytesIO()
        segment.export(buffer, format="wav")
        buffer.seek(0)
        waveform, sample_rate = torchaudio.load(buffer)

        if sample_rate != self.target_sample_rate:
            resampler = Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        return waveform


# class AphasiaDatasetWav2vec(AphasiaDataset):
#
#     def __init__(self, csv_file, root_dir, target_sample_rate=16000,
#                  hop_length=256, win_length=512, min_duration=10, max_duration=15):
#         super(AphasiaDatasetWaveform, self).__init__(csv_file, root_dir, target_sample_rate,
#                                                  hop_length=hop_length, win_length=win_length,
#                                                      min_duration=min_duration, max_duration=max_duration)
#
#         self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base",
#                                                            sampling_rate=target_sample_rate)
#
#     def preprocess(self, segment):
#         buffer = BytesIO()
#         segment.export(buffer, format="wav")
#         buffer.seek(0)
#         waveform, sample_rate = torchaudio.load(buffer)
#
#         if sample_rate != self.target_sample_rate:
#             resampler = Resample(sample_rate, self.target_sample_rate)
#             waveform = resampler(waveform)
#
#         y = self.processor(waveform, sampling_rate=sample_rate).input_values[0]
#
#         return y
