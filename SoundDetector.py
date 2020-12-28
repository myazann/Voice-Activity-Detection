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
from torch import nn
import torch.nn.functional as F


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

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
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.5),
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Dropout(0.5),
            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Dropout(0.5),
            )
        
        self.classifier = nn.Sequential(
            BidirectionalGRU(1024, 1024, 0.25, True),
            Flatten(),
            Linear(32768, 4096), 
            ReLU(inplace=True),
            LayerNorm(4096),
            Linear(4096, 2048), 
            ReLU(inplace=True),
            LayerNorm(2048),
            Linear(2048, 256), 
            ReLU(inplace=True),
            LayerNorm(256),
            Linear(256, 2)
        )        
  
    def forward(self, x):

        x = self.sound_detector_model(x)
        
        sizes = x.size()
        x = x.view(sizes[0], sizes[2] * sizes[3], sizes[1])  # (batch, feature, time)
        ##x = x.transpose(1, 2)
        x = self.classifier(x)

        
        return x
                   
    """
    def forward(self, x):
        x = self.sound_detector_model(x)
        return x        
    """
    
class SoundDetector():
    
    def __init__(self, device):
        
        self.device = device
        self.detector = SoundDetectorModel()
        
        self.detector = torch.load('SoundDetector.pt', map_location=torch.device(device))
        
        self.detector = self.detector.to(device)
        self.detector.eval()
        
 
    def predict(self, path):

        song = preprocess_audio(path)
        
        with torch.no_grad():
            out = self.detector(song)

        _, preds = torch.max(out, 1)

        
        if torch.sum(preds) > 0:
            print("Konuşma var!")
        else:
            print("Konuşma yok!")
        
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



def preprocess_audio(data):
    
    if isinstance(data, bytes):
        
        data = np.frombuffer(data)
    
    if isinstance(data, str):
    
        audio = AudioSegment.from_file(data)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        audio = audio.get_array_of_samples()
        
        
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        
        audio = np.array(data)
        
        if len(audio.shape) > 2:
            raise ValueError("Ses dizisinin boyutu en fazla 2 olabilir!")
        elif len(audio.shape) == 2:
            if audio.shape[0] == 2:
                torch.reshape((audio.shape[1], audio.shape[0]))
            if isinstance(data, torch.Tensor):
                audio = np.mean(audio.numpy(), axis = 1)
            else:
                audio = np.mean(audio, axis = 1)
            
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

