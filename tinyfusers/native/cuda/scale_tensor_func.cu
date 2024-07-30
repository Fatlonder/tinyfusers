#include "math_functions.h"
#include "math_constants.h"

extern "C"
__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        inp[idx] *= scale;
    }
}