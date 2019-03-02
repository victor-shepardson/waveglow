# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read as wavread
from librosa.core import load as audioread
import numpy as np

MAX_WAV_VALUE = 32768.0

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def load_audio_to_torch(full_path, sampling_rate):
    """
    Loads audio data into torch array
    """
    try:
        file_sampling_rate, data = wavread(full_path)
        assert sampling_rate==file_sampling_rate
    except Exception:
        data, _ = audioread(full_path, sr=sampling_rate, mono=True, res_type='kaiser_fast')
        data *= MAX_WAV_VALUE
    return torch.from_numpy(data).float()#, sampling_rate


class AudioDataset(torch.utils.data.Dataset):
    """
    This is Mel2Samp from mel2samp.py with all the mel removed -- just an identical audio loader
    Also adds ability to read more formats+convert sample rate with librosa
    """
    def __init__(self, training_files, segment_length, sampling_rate, sanity_check):
        random.seed(1234)
        self.sanity_check = sanity_check
        self.audio_files = files_to_list(training_files)
        random.shuffle(self.audio_files)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def __getitem__(self, index):
        # if sanity_check flag is set, return synthetic data
        if self.sanity_check=='sine':
        #     return ((torch.linspace(
        #         0, 440*self.segment_length/self.sampling_rate,
        #         self.segment_length
        #         )+torch.rand(1))*2*np.pi).sin()*(0.5*2**(torch.rand(1)*2-2))
            return (torch.linspace(
                0, 440*self.segment_length/self.sampling_rate,
                self.segment_length
                )*2*np.pi).sin()*0.2

        # Read audio
        filename = self.audio_files[index]
        audio = load_audio_to_torch(filename, self.sampling_rate)
        # if sampling_rate != self.sampling_rate:
        #     raise ValueError("{} SR doesn't match target {} SR".format(
        #         sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        audio = audio / MAX_WAV_VALUE

        return audio

    def __len__(self):
        return len(self.audio_files)
