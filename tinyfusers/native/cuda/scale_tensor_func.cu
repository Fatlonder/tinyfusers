#include "math_functions.h"
#include "math_constants.h"

extern "C"
__global__ void scale_kernel(float* inp, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        inp[idx] *= scale;
    }
}