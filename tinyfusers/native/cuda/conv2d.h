#include <stdio.h>
#include <iostream>

void conv_depthwise2d_forward_kernel(
    const float *input,
    float *output,
    const float *weight,
    const float *bias,
    bool biasEnabled,
    int totalElements,
    const int outputChannels,
    const int depthwiseMultiplier,
    const int inputWidth, const int inputHeight,
    const int outputWidth, const int outputHeight,
    const int kernelWidth, const int kernelHeight,
    const int strideWidth, const int strideHeight,
    const int padWidth, const int padHeight,
    const int dilationWidth, const int dilationHeight);