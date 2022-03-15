
import torch.nn as nn
from importlib import import_module


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        module = import_module('models.' + args.model.lower() + '.' + args.model.lower())
        self.model = module.make_model(args)

    def forward(self, tenInput, tenRef):
        return self.model(tenInput, tenRef)

    def load(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_state_dict(self):
        return self.model.state_dict()
