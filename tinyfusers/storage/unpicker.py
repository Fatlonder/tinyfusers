import zipfile, pickle, struct, io
import collections
import numpy as np

def load_tensor_data(file_name):
  if zipfile.is_zipfile(file_name):
    myzip = zipfile.ZipFile(file_name, 'r')
    base_name = myzip.namelist()[0].split('/', 1)[0]
    for n in myzip.namelist():
      if n.startswith(f'{base_name}/data/'):
        with myzip.open(n, mode='r') as myfile:
          buffer.append(myfile.read())
  return buffer

def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
  #tensor = torch.as_strided(torch.Tensor(storage(), device=storage.device), size=size, stride=stride)
  tensor = np.reshape(storage(), size) #check other cases for stride inconsistency.
  tensor = tensor.astype(np.float32) #read dtype from storage
  return tensor

class TypedStorage():
  def __init__(self, dtype, file_index, device, num_elements):
    self.dtype = dtype
    self.file_index = int(file_index)
    self.device = device
    self.num_elements = num_elements
    self.dtype_dict = {"float32": "f", "float16": "f"} # expand latter: https://docs.python.org/3/library/struct.html#format-characters
  def __call__(self):
    return struct.unpack(self.dtype_dict[self.dtype]*self.num_elements, buffer[self.file_index])
  def nbytes(self):
    return 0
  def element_size(self):
    return 0
  def device(self):
    return self.device
  def file_index(self):
    return self.file_index

class TorchUnpickler(pickle.Unpickler):
    def persistent_load(self, saved_id):
        assert saved_id[0] == 'storage'
        _, type_class, file_index, device, num_elements = saved_id
        return TypedStorage(type_class, file_index, device, num_elements)
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return getattr(collections, name)
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return _rebuild_tensor_v2
        if module == 'torch' and name in ['FloatStorage', 'HalfStorage']:
            return "float32"
        if module == 'numpy.core.multiarray' and name == 'scalar':
            return np.core.multiarray.scalar
        if module == 'numpy' and name == 'dtype':
            return np.dtype
        raise pickle.UnpicklingError(f"global {module}.{name} is not supported")
    
def load_weights(weight_path):
    global buffer
    buffer = load_tensor_data(f"{weight_path}.pth")
    with open(f"{weight_path}/data.pkl", 'rb') as f: st = f.read()
    return TorchUnpickler(io.BytesIO(st)).load()