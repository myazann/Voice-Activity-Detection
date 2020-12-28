
from SoundDetector import *



model = SoundDetector("cpu")


## Accepts path, numpy array or torch tensor as argument.


model.predict("bus_chatter.wav")


## Listens the environment from the computer mic and predicts if there is human voice or not. 
## First argument determines the length of the period, and second argument is for total listening seconds. 


model.listen(5, 60)




