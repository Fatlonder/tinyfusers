import ctypes

class Nvrtc:
    def __init__(self,):
        self.dll = ctypes.CDLL('libnvrtc.so')

    def nvrtcCreateProgram(self, prog, code_str):
        self.dll.nvrtcCreateProgram.restype = ctypes.c_uint32
        self.dll.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(_nvrtcProgram)), ctypes.POINTER(ctypes.c_char), 
                                         ctypes.POINTER(ctypes.c_char), ctypes.c_int32, 
                                         ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), 
                                         ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
        status = self.dll.nvrtcCreateProgram(ctypes.byref(prog), code_str.encode()
                                             , "<null>".encode(), 0, None, None)
        return status
    
    def nvrtcCompileProgram(self, prog, compile_options):
        self.dll.nvrtcCompileProgram.restype = ctypes.c_uint32
        self.dll.nvrtcCompileProgram.argtypes = [nvrtcProgram, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
        status = self.dll.nvrtcCompileProgram(prog, len(compile_options), bstr_to_ctype(compile_options, ctypes.c_char))
        return status

    def nvrtcGetCUBINSize(self, prog, data_size):
        self.dll.nvrtcGetCUBINSize.restype = ctypes.c_uint32
        self.dll.nvrtcGetCUBINSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
        status = self.dll.nvrtcGetCUBINSize(prog, data_size)
        return status

    def nvrtcGetCUBIN(self, prog, data):
        self.dll.nvrtcGetCUBIN.restype = ctypes.c_uint32
        self.dll.nvrtcGetCUBIN.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
        status = self.dll.nvrtcGetCUBIN(prog, data)
        return status

    def nvrtcGetPTXSize(self, prog):
        self.dll.nvrtcGetPTXSize.restype = ctypes.c_uint32
        self.dll.nvrtcGetPTXSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_uint64)]
        status = self.dll.nvrtcGetPTXSize(prog)
        return status

    def nvrtcGetPTX(self, prog, data):
        self.dll.nvrtcGetPTX.restype = ctypes.c_uint32
        self.dll.nvrtcGetPTX.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
        status = self.dll.nvrtcGetPTX(prog, data)
        return status

def bstr_to_ctype(args, c_type):
    return(ctypes.POINTER(ctypes.c_char) * len(args))(*[ctypes.cast(ctypes.create_string_buffer(o), ctypes.POINTER(c_type)) for o in args])     

class _nvrtcProgram(ctypes.Structure):
    pass
nvrtcProgram = ctypes.POINTER(_nvrtcProgram)

nvrtc = Nvrtc()
nvrtc.NVRTC_SUCCESS = 0
nvrtc.NVRTC_ERROR_OUT_OF_MEMORY = 1
nvrtc.NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
nvrtc.NVRTC_ERROR_INVALID_INPUT = 3
nvrtc.NVRTC_ERROR_INVALID_PROGRAM = 4
nvrtc.NVRTC_ERROR_INVALID_OPTION = 5
nvrtc.NVRTC_ERROR_COMPILATION = 6
nvrtc.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
nvrtc.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
nvrtc.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
nvrtc.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
nvrtc.NVRTC_ERROR_INTERNAL_ERROR = 11
nvrtc.NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12
nvrtc.nvrtcProgram = nvrtcProgram
nvrtc.nvrtcResult = ctypes.c_uint32