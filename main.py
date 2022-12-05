from types import SimpleNamespace
import json
from model_3fgl import Model3FGL

# Import the params from the json config file into a simple namespace
with open('params/params_3fgl.json') as json_file:
    params = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))

# Create the 3FGL model
model3fgl = Model3FGL(params)

# TODO: Get the data and create a dataloader
# dataloader = ...

# Train the 3FGL model
model3fgl.run_training(dataloader)