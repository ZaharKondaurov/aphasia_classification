import os
from io import BytesIO
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from scipy.signal import fftconvolve

import random
from random import randint, uniform, choice, shuffle

from pydub import AudioSegment

import torch
import torchaudio
from torch import nn
from torch import hub
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

import librosa
import pyroomacoustics as pra

from transformers import Wav2Vec2Processor

from functools import wraps

from typing import Union, Tuple


WALLS_KEYWORDS = ["hard_surface", "ceramic_tiles", "plasterboard", "wooden_lining", "glass_3mm"]
FLOOR_KEYWORDS = ["linoleum_on_concrete", "carpet_cotton"]
CEILING_KEYWORDS = ["ceiling_plasterboard", "ceiling_fissured_tile", "ceiling_metal_panel", ]

def get_speech_and_silence_timestamps(waveform: torch.Tensor,
                                      sr: int, return_seconds: bool = False,
                                      threshold: float = 0.6,
                                      min_speech_duration_ms: int = 500,
                                      min_silence_duration_ms: int = 1000):
    speech_model = load_silero_vad()
    duration = waveform.shape[-1] # // sr

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

        remainder = (signal.shape[-1] - self.window_size) % self.stride
        pad_count = 0

        if remainder != 0:
            pad_count = self.stride - remainder

        signal = torch.nn.functional.pad(signal, (0, pad_count), "constant", 0)
        chunks = signal.unfold(-1, self.window_size, self.stride)

        return chunks


class AphasiaDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_sample_rate=16000, fft_size=512,
                 hop_length=256, win_length=512, min_duration=10, max_duration=15,
                 add_noise: bool = False,
                 snr: Union[int, Tuple[int, int]] = 0,
                 room_square: Tuple[float, float] = (7., 14.),
                 room_height: Tuple[float, float] = (3., 4.),
                 noise_dir: str = None,
                 rirs_dir: str = None,
                 file_format: str = "3gp",
                 transforms=None,
                 triplet=False):
        self.file_format = file_format
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.min_duration = min_duration * 1000  # конвертируем в миллисекунды
        self.max_duration = max_duration * 1000
        self.data = []
        self.transforms = transforms

        self.add_noise = add_noise
        self.snr = snr
        self.room_height = room_height
        self.room_square = room_square
        self.noise_dir = noise_dir
        self.noise_files = None
        self.rirs_dir = rirs_dir
        self.triplet = triplet

        if self.noise_dir is not None:
            self.noise_files = os.listdir(self.noise_dir)

        if self.rirs_dir is not None:
            self.rirs_files = os.listdir(self.rirs_dir)

        # Загружаем список файлов из CSV
        df = pd.read_csv(csv_file)
        self.labels = []
        self.data_dict = {}
        # Обработка и сегментация аудио
        for _, row in df.iterrows():
            file_name, label = row['file_name'], row['label']
            file_path = self.find_audio_file(file_name, label)
            if file_path:
                try:
                    segments = self.process_audio(file_path)
                    self.data.extend([(s, label) for s in segments])
                    if label not in self.data_dict:
                        self.data_dict[label] = [(s, label) for s in segments]
                    else:
                        self.data_dict[label].extend([(s, label) for s in segments])
                    if label not in self.labels:
                        self.labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

        random.shuffle(self.data)

    @staticmethod
    def simulate_noise(signal: torch.Tensor, noise: torch.Tensor, snr_db: int) -> torch.Tensor:
        len_noise = noise.shape[-1]
        noise_cat = [noise]
        while len_noise < signal.shape[-1]:
            len_noise += noise.shape[-1]
            noise_cat.append(noise)
        noise = torch.cat(noise_cat, dim=-1)

        if noise.shape[-1] > signal.shape[-1]:
            noise = noise[..., :signal.shape[-1]]

        rms_signal = torch.sqrt(torch.mean(signal ** 2))
        rms_noise = torch.sqrt(torch.mean(noise ** 2))

        target_rms_noise = rms_signal / (10 ** (snr_db / 10))

        noise = noise * (target_rms_noise / rms_noise)

        return noise

    def simulate_rir_shoebox(self, signal: torch.Tensor) -> torch.Tensor:
        square = uniform(*self.room_square)
        width = uniform(2.5, square * 0.75)
        length = square / width
        height = uniform(*self.room_height)

        rt60 = uniform(0.3, 1.25)   # Делать длинный ревёрб
        room_dim = [length, width, height]

        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

        wall = pra.Material(choice(WALLS_KEYWORDS))
        ceil = pra.Material(choice(CEILING_KEYWORDS))
        floor = pra.Material(choice(FLOOR_KEYWORDS))

        material = {"east": wall, "west": wall, "north": wall, "south": wall, "ceiling": ceil, "floor": floor}

        room = pra.ShoeBox(room_dim, fs=self.target_sample_rate, materials=material, max_order=max_order,
                           use_rand_ism=True, max_rand_disp=0.05, ray_tracing=False)

        source_locs = [uniform(0.01, length), uniform(0.01, width), uniform(1.0, 2.0)]
        mic_locs = np.array([x * 0.98 for x in source_locs])[:, None]

        room.add_source(source_locs, signal=signal.squeeze(), delay=0.5)

        room.add_microphone_array(mic_locs)
        room.compute_rir()
        room.simulate()     # Внутри есть параметр snr, возможно, он пригодится

        return room.rir[0][0]   # [микрофон, источник]

    def add_noise_and_reverb(self, signal: torch.Tensor) -> torch.Tensor:
        if isinstance(self.snr, tuple):
            snr_db = randint(self.snr[0], self.snr[1])
        else:
            snr_db = self.snr

        if self.rirs_dir is not None:
            filename_rir = choice(self.rirs_files)
            rir, rir_sr = torchaudio.load(os.path.join(self.rirs_dir, filename_rir))

            if rir_sr != self.target_sample_rate:
                resampler = Resample(rir_sr, self.target_sample_rate)
                rir = resampler(rir)

            signal = torch.from_numpy(fftconvolve(signal, rir[None, :], mode='same', axes=-1))

        if self.noise_dir is not None:
            filename_noise = choice(self.noise_files)
            noise, noise_sr = torchaudio.load(os.path.join(self.noise_dir, filename_noise))

            if noise_sr != self.target_sample_rate:
                resampler = Resample(noise_sr, self.target_sample_rate)
                noise = resampler(noise)

            noise = self.simulate_noise(signal, noise, snr_db)

            output = signal + noise

            return output

        return signal

    def find_audio_file(self, file_name, label):
        """Ищем файл в соответствующей папке по метке"""
        folder = "Aphasia" if label > 0 else "Norm"
        file_name = file_name[:-4]
        file_path = os.path.join(self.root_dir, folder, f"{file_name}.{self.file_format}")  # Убрано ".wav"
        if os.path.exists(file_path):
            return file_path
        print(f"Warning: {file_name}.{self.file_format} not found in {folder} folder.")
        return None

    def process_audio(self, file_path):
        audio = AudioSegment.from_file(file_path, format=self.file_format)
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
        raise NotImplemented

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.triplet:
            elem, label = self.data[idx]

            pos_data = self.data_dict[label]
            neg_label = choice([x for x in self.labels if x != label])
            neg_data = self.data_dict[neg_label]

            pos_elem, _ = choice(pos_data)
            neg_elem, _ = choice(neg_data)

            return elem, pos_elem, neg_elem
        else:
            elem, label = self.data[idx]

            return elem, torch.tensor(label, dtype=torch.long)


class AphasiaDatasetSpectrogram(AphasiaDataset):

    def __init__(self, csv_file, root_dir, target_sample_rate=16000, fft_size=512,
                 hop_length=256, win_length=512, min_duration=10, max_duration=15,
                 add_noise: bool = False,
                 snr: Union[int, Tuple[int, int]] = 0,
                 room_square: Tuple[float, float] = (7., 14.),
                 room_height: Tuple[float, float] = (3., 4.),
                 noise_dir: str = None,
                 rirs_dir: str = None,
                 file_format: str = "3gp",
                 transforms=None,
                 triplet: bool = False,):
        super(AphasiaDatasetSpectrogram, self).__init__(csv_file, root_dir, target_sample_rate, fft_size,
                 hop_length, win_length, min_duration, max_duration, add_noise, snr, room_square,
                                                        room_height, noise_dir, rirs_dir, file_format, transforms,
                                                        triplet)

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

            if self.add_noise:
                waveform = self.add_noise_and_reverb(waveform)

            y = waveform.numpy().squeeze()

            if self.transforms is not None:
                y = self.transforms(samples=y, sample_rate=self.target_sample_rate)

            spectrogram = librosa.stft(y, n_fft=self.fft_size, hop_length=self.hop_length, win_length=self.win_length)
            mag = np.abs(spectrogram).astype(np.float32)
            return torch.tensor(mag.T).unsqueeze(0)
        except Exception as e:
            print(f"Spectrogram error: {str(e)}")
            return None


class AphasiaDatasetMFCC(AphasiaDataset):

    def __init__(self, csv_file, root_dir, mfcc=128, n_mels=150, target_sample_rate=16000, fft_size=512,
                 hop_length=256, win_length=512, min_duration=10, max_duration=15,
                 add_noise: bool = False,
                 snr: Union[int, Tuple[int, int]] = 0,
                 room_square: Tuple[float, float] = (7., 14.),
                 room_height: Tuple[float, float] = (3., 4.),
                 noise_dir: str = None,
                 rirs_dir: str = None,
                 file_format: str = "3gp",
                 transforms=None,
                 triplet: bool = False):
        self.mfcc_class = torchaudio.transforms.MFCC(sample_rate=8_000, n_mfcc=mfcc,
                                                     log_mels=True, melkwargs={"n_fft": fft_size,
                                                                               "win_length": win_length,
                                                                               "hop_length": hop_length,
                                                                               "n_mels": n_mels})
        super(AphasiaDatasetMFCC, self).__init__(csv_file, root_dir, target_sample_rate, fft_size,
                 hop_length, win_length, min_duration, max_duration, add_noise, snr, room_square,
                 room_height, noise_dir, rirs_dir, file_format, transforms, triplet)

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

            if self.add_noise:
                waveform = self.add_noise_and_reverb(waveform)

            if self.transforms is not None:
                waveform = torch.from_numpy(self.transforms(samples=waveform.numpy().squeeze(),
                                                            sample_rate=self.target_sample_rate))

            y = waveform.squeeze()

            mfcc = self.mfcc_class(y)

            return torch.tensor(mfcc)
        except Exception as e:
            print(f"MFCC error: {str(e)}")
            return None


class AphasiaDatasetWaveform(AphasiaDataset):

    def __init__(self, csv_file, root_dir, target_sample_rate=16000,
                 hop_length=256, win_length=512, min_duration=10, max_duration=15,
                 add_noise: bool = False,
                 snr: Union[int, Tuple[int, int]] = 0,
                 room_square: Tuple[float, float] = (7., 14.),
                 room_height: Tuple[float, float] = (3., 4.),
                 noise_dir: str = None,
                 rirs_dir: str = None,
                 file_format: str = "3gp",
                 transforms=None,
                 triplet: bool = False):
        super(AphasiaDatasetWaveform, self).__init__(csv_file, root_dir, target_sample_rate,
                                                 hop_length=hop_length, win_length=win_length,
                                                     min_duration=min_duration, max_duration=max_duration,
                                                     add_noise=add_noise, snr=snr, room_square=room_square,
                                                     room_height=room_height, noise_dir=noise_dir, rirs_dir=rirs_dir,
                                                     file_format=file_format, transforms=transforms, triplet=triplet)

    def preprocess(self, segment):
        buffer = BytesIO()
        segment.export(buffer, format="wav")
        buffer.seek(0)
        waveform, sample_rate = torchaudio.load(buffer)

        if sample_rate != self.target_sample_rate:
            resampler = Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        if self.add_noise:
            waveform = self.add_noise_and_reverb(waveform)

        if self.transforms is not None:
            waveform = torch.from_numpy(self.transforms(samples=waveform.numpy(), sample_rate=self.target_sample_rate))

        return waveform


def inference_preprocessing(func):
    @wraps(func)
    def wrapper(*args, audio_path, sr, **kwargs):
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != sr:
            resampler = Resample(sample_rate, sr)
            waveform = resampler(waveform)

        output = func(*args, waveform=waveform, sr=sr, **kwargs)

        return output

    return wrapper


@inference_preprocessing
def get_spectrogram(n_fft, hop_length, win_length, waveform, sr):
    spectrogram = librosa.stft(waveform.numpy(), n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(spectrogram).astype(np.float32)
    return torch.from_numpy(mag)


@inference_preprocessing
def get_waveform(waveform, sr):
    return waveform


@inference_preprocessing
def get_mfcc(n_mfcc, n_fft, hop_length, win_length, n_mels, waveform, sr):
    mfcc_class = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
                                                 log_mels=True, melkwargs={"n_fft": n_fft,
                                                                           "win_length": win_length,
                                                                           "hop_length": hop_length,
                                                                           "n_mels": n_mels})
    mfcc = mfcc_class(waveform)
    return mfcc
