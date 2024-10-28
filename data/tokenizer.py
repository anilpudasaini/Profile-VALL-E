#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from typing import Any, Dict, List, Optional, Pattern, Union
import numpy as np


class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        device: Any = None,
    ) -> None:
        # Instantiate a pretrained EnCodec model
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            if torch.backends.mps.is_available():
                device = torch.device("mps")

        self._device = device
        print(f"Allocated device: {self._device}")

        self.codec = model.to(device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(frames)


def tokenize_audio(tokenizer: AudioTokenizer, audio):
    # Load and pre-process the audio waveform
    if isinstance(audio, str):
        wav, sr = torchaudio.load(audio)
    else:
        wav, sr = audio
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames

def decode(tokenizer: AudioTokenizer, encoded_frames):
    # Decode the codes to reconstruct the waveform
    with torch.no_grad():
        decoded_wav = tokenizer.decode(encoded_frames)
    return decoded_wav

# for debug purposes
if __name__ == "__main__":
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    samples = torch.from_numpy(np.random.random([2, 1, 1600])).type(
        torch.float32
    )
    codes_raw = model.encode(samples)
    print(codes_raw)