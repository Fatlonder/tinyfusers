import ctypes, ctypes.util
import numpy as np
import os
import sys

class Device:
    def __init__(self, device:str):
        CUDA_HOME = os.getenv('CUDA_HOME')
        if CUDA_HOME == None:
            CUDA_HOME = os.getenv('CUDA_PATH')
        if CUDA_HOME == None:
            raise RuntimeError('Environment variable CUDA_HOME or CUDA_PATH is not set')
        self.device = device
    
    def load_library_from_path(self, device, lib_path, lib_name = "cublas"):
        return 0

    def __str__(self):
        return self.device