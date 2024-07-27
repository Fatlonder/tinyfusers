#include "math_functions.h"
#include "math_constants.h"

extern "C"
__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        int rest = idx % (NH * T * T);
        rest = rest % (T * T);
        int t2 = rest / T;
        int t = rest % T;
        inp[idx] *= scale;
    }
}