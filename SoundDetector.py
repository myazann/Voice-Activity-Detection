#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
import torch
import torchaudio
import os
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from torch.nn import *
import torch.nn.functional as F


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SoundDetectorModel(Module):   
    def __init__(self):
        super(SoundDetectorModel, self).__init__()
          
        self.sound_detector_model =  Sequential(
            self.ConvBlock(1, 32, 3, 1, 1),
            self.ConvBlock(32, 32, 3, 1, 1, True),
            self.ConvBlock(32, 64, 3, 1, 1),
            self.ConvBlock(64, 64, 3, 1, 1, True),
            self.ConvBlock(64, 128, 3, 1, 1, True),
            Dropout(0.5),
            self.ConvBlock(128, 256, 3, 1, 1),
            self.ConvBlock(256, 256, 3, 1, 1, True),
            Dropout(0.5),
            self.ConvBlock(256, 512, 3, 1, 1),
            self.ConvBlock(512, 1024, 3, 1, 1, True),
            Dropout(0.5)
            )
        
        self.classifier = nn.Sequential(
            BidirectionalGRU(1024, 1024, 0.25, True),
            Flatten(),
            self.LinearBlock(32768, 4096),
            self.LinearBlock(4096, 2048),
            self.LinearBlock(2048, 256),
            Linear(256, 2)
        )

    def ConvBlock(self, input_channels, output_channels, kernel_size=3, stride=1, padding = 1, maxpool = False):

      if maxpool:
        return Sequential(
          Conv2d(input_channels, output_channels, kernel_size, stride, padding),
          ReLU(inplace=True),
          MaxPool2d(kernel_size=2, stride=2)
          )
      else:
        return Sequential(
          Conv2d(input_channels, output_channels, kernel_size, stride, padding),
          ReLU(inplace=True)
        )

    def LinearBlock(self, input_channels, output_channels):

      return Sequential(
            Linear(input_channels, output_channels), 
            ReLU(inplace=True),
            LayerNorm(output_channels)
      )
          
    def forward(self, x):

        x = self.sound_detector_model(x)
        
        sizes = x.size()
        x = x.view(sizes[0], sizes[2] * sizes[3], sizes[1])
        ##x = x.transpose(1, 2)
        x = self.classifier(x)

        return x
                   
    """
    def forward(self, x):
        x = self.sound_detector_model(x)
        return x        
    """

## Main class for model functions.
class SoundDetector():
    
    def __init__(self, device):
        
        self.device = device
        self.detector = SoundDetectorModel()
        
        self.detector = torch.load('SoundDetector.pt', map_location=torch.device(device))
        
        self.detector = self.detector.to(device)
        self.detector.eval()
        
 
    def predict(self, path):

        song = self.preprocess_audio(path)
        
        with torch.no_grad():
            out = self.detector(song)

        _, preds = torch.max(out, 1)

        
        if torch.sum(preds) > 0:
            print("People are speaking!")
        else:
            print("Nobody is speaking!")
        
        return torch.sum(preds)


    def listen(self, seconds, listening_time):
        elapsed_time = 0

        while elapsed_time < listening_time:
            print("Listening")
            fs = 16000  
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
            sd.wait()  
            self.predict(myrecording)
            elapsed_time += seconds


    def preprocess_audio(self, data):

        if isinstance(data, str):

            audio = AudioSegment.from_file(data)
            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)
            audio = audio.get_array_of_samples()


        elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):

            audio = np.array(data)

            if len(audio.shape) > 2:
                raise ValueError("Dimension cannot be bigger than 2!")
            elif len(audio.shape) == 2:
                if audio.shape[0] == 2:
                    torch.reshape((audio.shape[1], audio.shape[0]))
                if isinstance(data, torch.Tensor):
                    audio = np.mean(audio.numpy(), axis = 1)
                else:
                    audio = np.mean(audio, axis = 1)
                    
        else:
            raise TypeError("Only path, numpy array of torch tensors are allowed!")

        aud_tensor = torch.tensor(np.array(audio))

        if aud_tensor.shape[0] < 160000:
            pad_shape = 160000 - aud_tensor.shape[0]
            pad_tensor = torch.zeros(pad_shape)
            aud_tensor = torch.cat((aud_tensor, pad_tensor))

        elif aud_tensor.shape[0] > 160000:
            splitted_tensor = list(torch.split(aud_tensor, 160000))
            if splitted_tensor[-1].shape[0] < 160000:
                pad_shape = 160000 - splitted_tensor[-1].shape[0]
                if aud_tensor.dtype == torch.int16:
                    pad_tensor = torch.zeros(pad_shape, dtype = torch.int16)
                else:
                    pad_tensor = torch.zeros(pad_shape)
                splitted_tensor[-1] = torch.cat((splitted_tensor[-1], pad_tensor)) 
            aud_tensor = torch.stack(splitted_tensor)


        if aud_tensor.dtype == torch.int16:
            aud_tensor = torch.tensor(aud_tensor, dtype = torch.float)

        aud_tensor = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft = 2048)(aud_tensor)

        if len(aud_tensor.shape) == 2:
            aud_tensor = aud_tensor[None, None, ...]
        elif len(aud_tensor.shape) == 3:
            aud_tensor = aud_tensor[:,None,...]

        return aud_tensor
