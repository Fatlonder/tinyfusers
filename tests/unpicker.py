import numpy as np
import torch
from torch import nn
from tinyfusers.storage.unpicker import load_weights

class SimplerModel(nn.Module):
    def __init__(self):
        super(SimplerModel, self).__init__()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc2(x)
        return x

def test_simplermodel_weight_load(weight_path, fc2):
    weight_dict = load_weights(weight_path)
    np.testing.assert_allclose(weight_dict['fc2.weight'], fc2, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    weight_path = "path/waddd.pth"
    model = SimplerModel()
    fc2 = model.state_dict()['fc2.weight'].numpy()
    torch.save(model.state_dict(), weight_path)
    test_simplermodel_weight_load(weight_path, fc2)