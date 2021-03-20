
from SoundDetector import *



model = SoundDetector("cpu")


## Accepts path, numpy array or torch tensor as argument.


path = "AudioSet/Non_Speech"
len_dir = len(os.listdir(path))
true = 0
results = {}

for song in os.listdir(path):
  with torch.no_grad():
    res = model.predict(path+ "/" + song)    
    if res == 1:
        print(song)
    else:
        true += 1
        
nonsp_acc = true/len_dir

path = "AudioSet/Speech"
len_dir = len(os.listdir(path))
true = 0
results = {}

for song in os.listdir(path):
  with torch.no_grad():
    res = model.predict(path+ "/" + song)    
    if res == 0:
        print(song)
    else:
        true += 1
        
sp_acc = true/len_dir
print("Speech Accuracy:",sp_acc,"Non_Speech Accuracy:",nonsp_acc)


## Listens the environment from the computer mic and predicts if there is human voice or not. 
## First argument determines the length of the period, and second argument is for total listening seconds. 


model.listen(5, 60)




