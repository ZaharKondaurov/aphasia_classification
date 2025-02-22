import os

import torch
from torch import nn
from torch import hub
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

import librosa


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

    if len(output) == 0:
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
