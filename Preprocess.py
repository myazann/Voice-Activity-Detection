from torch.utils import data
import torchaudio
import os
from pydub import AudioSegment
import numpy as np
import pydub
import torch

## Given the path of the files, this function converts them to a suitable format for the model.

def prepare_audio(audio_path, save_path, file_names):
    
    j = 0

    for song in os.listdir(audio_path):

        song_path = os.path.join(audio_path, song)

        sound = AudioSegment.from_file(song_path)
        sound = sound.set_frame_rate(16000)
        sound = sound.set_channels(1)
        sound = sound.get_array_of_samples()

        aud_tensor = torch.tensor(np.array(sound))


        if aud_tensor.shape[0] < 160000:
            pad_shape = 160000 - aud_tensor.shape[0]
            pad_tensor = torch.zeros(pad_shape, dtype = torch.int16) 
            aud_tensor = torch.cat((aud_tensor, pad_tensor))

        elif aud_tensor.shape[0] > 160000:
            splitted_tensor = list(torch.split(aud_tensor, 160000))
            if splitted_tensor[-1].shape[0] < 160000:
                pad_shape = 160000 - splitted_tensor[-1].shape[0]
                pad_tensor = torch.zeros(pad_shape, dtype = torch.int16)
                splitted_tensor[-1] = torch.cat((splitted_tensor[-1], pad_tensor)) 
            aud_tensor = torch.stack(splitted_tensor)

        
        song_name = file_names + str(j) + ".wav"
        
        save_path = os.path.join(save_path, song_name)

        if len(aud_tensor.shape) != 1:
            i = 0
            while i < aud_tensor.shape[0]:
                torchaudio.save(save_path, aud_tensor[i], sample_rate = 16000)
                song_name = "war_" + str(j) + ".wav"
                i += 1
                j += 1
        else:
            torchaudio.save(save_path, aud_tensor, sample_rate = 16000)
            j += 1
    

## This function can be used to split the data into train, val, and test folders.
def data_split(sp_path):

    sp_list = os.listdir(sp_path)

    sp_list = list(np.random.permutation(np.array(sp_list)))

    sptest_len = int(len(sp_list)*0.2)

    sp_test = sp_list[0:sptest_len]
    sp_val = sp_list[sptest_len:(sptest_len*2)]
    sp_train = sp_list[(sptest_len)*2:]

    for file in os.listdir(sp_path):
        old_path = os.path.join(sp_path, file)
        if file in sp_test:    
            new_path = os.path.join(sp_path, "Test", file)
        elif file in sp_val:
            new_path = os.path.join(sp_path, "Val", file)
        else:
            new_path = os.path.join(sp_path, "Train", file)
        os.rename(old_path, new_path)




