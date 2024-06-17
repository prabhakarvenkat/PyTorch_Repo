import torch
import torch.nn as nn


### COMPLETE MODEL ###
torch.save(model, PATH)

# model class must be defined somewhere
model = torch.load(PATH)
model.eval()










##### STATE DICT #####
torch.save(model.state_dict(), PATH)


# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()