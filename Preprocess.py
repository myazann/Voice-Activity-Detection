#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils import data
import torchaudio
import os
from pydub import AudioSegment
import numpy as np
import pydub
import torch


# ## !!İsimlendirmeyi değiştirmeyi unutma!!

# In[2]:


j = 0

for song in os.listdir("OneDrive\\Masaüstü\\yeniler\\Non_Speech"):
    print(song)
    
    song_path = "OneDrive\\Masaüstü\\yeniler\\Non_Speech\\" + song
    
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
        
    save_path = "D:\\Belgeler\\Audio_Pad\\Non_Speech"
    
    song_name = "war_" + str(j) + ".wav"
        
    if len(aud_tensor.shape) != 1:
        i = 0
        while i < aud_tensor.shape[0]:
            torchaudio.save(save_path + "\\" + song_name, aud_tensor[i], sample_rate = 16000)
            song_name = "war_" + str(j) + ".wav"
            i += 1
            j += 1
    else:
        torchaudio.save(save_path + "\\" + song_name, aud_tensor, sample_rate = 16000)
        j += 1
    


# In[2]:


j = 0

for song in os.listdir("OneDrive\\Masaüstü\\yeniler\\Speech"):
    print(song)
    
    song_path = "OneDrive\\Masaüstü\\yeniler\\Speech\\" + song
    
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
        
    save_path = "D:\\Belgeler\\Audio_Pad\\Speech"
    
    song_name = "uclu_" + str(j) + ".wav"
        
    if len(aud_tensor.shape) != 1:
        i = 0
        while i < aud_tensor.shape[0]:
            torchaudio.save(save_path + "\\" + song_name, aud_tensor[i], sample_rate = 16000)
            song_name = "uclu_" + str(j) + ".wav"
            i += 1
            j += 1
    else:
        torchaudio.save(save_path + "\\" + song_name, aud_tensor, sample_rate = 16000)
        j += 1
    


# In[3]:


sp_path = "D:\\Belgeler\\Audio_Pad\\Speech"

sp_list = os.listdir(sp_path)

sp_list = list(np.random.permutation(np.array(sp_list)))

sptest_len = int(len(sp_list)*0.3)

sp_test = sp_list[0:sptest_len]
sp_val = sp_list[sptest_len:(sptest_len*2)]
sp_train = sp_list[(sptest_len)*2:]

for file in os.listdir(sp_path):
    old_path = sp_path + "\\" + file
    if file in sp_test:    
        new_path = "D:\\Belgeler\\Audio_Pad\\Test\\Speech\\" + file
    elif file in sp_val:
        new_path = "D:\\Belgeler\\Audio_Pad\\Val\\Speech\\" + file
    else:
        new_path = "D:\\Belgeler\\Audio_Pad\\Train\\Speech\\" + file
    os.rename(old_path, new_path)


# In[4]:


nonsp_path = "D:\\Belgeler\\Audio_Pad\\Non_Speech"

nonsp_list = os.listdir(nonsp_path)    
nonsp_list = list(np.random.permutation(np.array(nonsp_list)))


nonsptest_len = int(len(nonsp_list)*0.3)

nonsp_test = nonsp_list[0:nonsptest_len]
nonsp_val = nonsp_list[nonsptest_len:(nonsptest_len)*2]
nonsp_train = nonsp_list[(nonsptest_len)*2:]


    
for file in os.listdir(nonsp_path):
    old_path = nonsp_path + "\\" + file
    if file in nonsp_test:    
        new_path = "D:\\Belgeler\\Audio_Pad\\Test\\Non_Speech\\" + file
    elif file in nonsp_val:
        new_path = "D:\\Belgeler\\Audio_Pad\\Val\\Non_Speech\\" + file
    else:
        new_path = "D:\\Belgeler\\Audio_Pad\\Train\\Non_Speech\\" + file
    os.rename(old_path, new_path)


# In[3]:


sp_path = "D:\\Belgeler\\Audio_Pad\\Speech"

sp_list = os.listdir(sp_path)

sp_list = list(np.random.permutation(np.array(sp_list)))

spval_len = int(len(sp_list)*0.3)

sp_val = sp_list[0:spval_len]
sp_train = sp_list[spval_len:]

for file in os.listdir(sp_path):
    old_path = sp_path + "\\" + file
    if file in sp_val:
        new_path = "D:\\Belgeler\\Eklenti\\Val\\Speech\\" + file
    else:
        new_path = "D:\\Belgeler\\Eklenti\\Train\\Speech\\" + file
    os.rename(old_path, new_path)
    
    
    
nonsp_path = "D:\\Belgeler\\Audio_Pad\\Non_Speech"

nonsp_list = os.listdir(nonsp_path)    
nonsp_list = list(np.random.permutation(np.array(nonsp_list)))


nonspval_len = int(len(nonsp_list)*0.3)

nonsp_val = nonsp_list[0:nonspval_len]
nonsp_train = nonsp_list[nonspval_len:]


    
for file in os.listdir(nonsp_path):
    old_path = nonsp_path + "\\" + file
    if file in nonsp_val:
        new_path = "D:\\Belgeler\\Eklenti\\Val\\Non_Speech\\" + file
    else:
        new_path = "D:\\Belgeler\\Eklenti\\Train\\Non_Speech\\" + file
    os.rename(old_path, new_path)

