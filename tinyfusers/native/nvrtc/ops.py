import ctypes

class Nvrtc:
    def __init__(self,):
        self.dll = ctypes.CDLL('libnvrtc.so')

    def nvrtcCreateProgram(self, prog: nvrtcProgram, code_str: str):
        self.dll.nvrtcCreateProgram.restype = ctypes.c_uint32
        self.dll.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.POINTER(_nvrtcProgram)), ctypes.POINTER(ctypes.c_char), 
                                         ctypes.POINTER(ctypes.c_char), ctypes.c_int32, 
                                         ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), 
                                         ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
        status = self.dll.nvrtcCreateProgram(ctypes.byref(prog), code_str.encode()
                                             , "<null>".encode(), 0, None, None)
        return status
    
    def nvrtcCompileProgram(self, prog: nvrtcProgram, compile_options):
        self.dll.nvrtcCompileProgram.restype = ctypes.c_uint32
        self.dll.nvrtcCompileProgram.argtypes = [nvrtcProgram, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
        status = nvrtc.nvrtcCompileProgram(prog, len(compile_options), bstr_to_ctype(compile_options, ctypes.c_char))
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
nvrtc.nvrtcProgram

