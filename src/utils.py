import os

import torch
from torch import nn
from torch import hub

import librosa


def get_speech_and_silence_timestamps(waveform: torch.Tensor,
                                      sr: int, return_seconds: bool = False,
                                      threshold: float = 0.6,
                                      min_speech_duration_ms: int = 500,
                                      min_silence_duration_ms: int = 1000):
    speech_model, utils = hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

    (get_speech_timestamps, _, _, _, _) = utils

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
                   ):
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


class RemoveSilence(nn.Module):
    def __init__(self):
        super(RemoveSilence, self).__init__()

    def forward(self, signal, timestamps=None):
        ...
