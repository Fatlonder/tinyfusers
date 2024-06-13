import numpy as np
from tinygrad import Tensor
from tinyfusers.variants.sd import StableDiffusion
from tinyfusers.storage.state import update_state
from tinyfusers.ff.linear import Linear

class SModel:
    def __init__(self):
        self.fc3 = Linear(5, 1, bias=False)
    def forward(self, x):
        x = self.fc3(x)
        return x
class SimplerModel:
    def __init__(self):
        self.fc1 = SModel()
        self.fc2 = Linear(5, 1, bias=False)
        self.blocks = [[SModel(), Tensor.silu, SModel()], SModel()]
        self.fc3 = SModel()
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x    
def test_simplermodel_state(model, state_dict):
    update_state(model, state_dict, prefix='')
    fc1fc3_weight = model.__dict__["fc1"].__dict__["fc3"].__dict__["weight"].numpy()
    np.testing.assert_allclose(fc1fc3_weight, state_dict['fc1.fc3.weight'].numpy(), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    model = SimplerModel()
    state_dict = {"fc1.fc3.weight": Tensor.zeros(1,5),"fc1.fc3.bias": Tensor.zeros(1), 
                  "fc2.weight": Tensor.zeros(1,5), "fc2.bias": Tensor.zeros(1), 
              "fc3.fc3.weight": Tensor.zeros(1,5), "fc3.fc3.bias": Tensor.zeros(1),
              "blocks.0.0.fc3.weight":Tensor.zeros(1,5), "blocks.0.0.fc3.bias": Tensor.zeros(1),
              "blocks.0.2.fc3.weight":Tensor.zeros(1,5), "blocks.0.2.fc3.bias": Tensor.zeros(1),
              "blocks.1.fc3.weight":Tensor.zeros(1,5), "blocks.1.fc3.bias": Tensor.zeros(1)}    
    test_simplermodel_state(model, state_dict)