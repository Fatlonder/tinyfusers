#include <stdio.h>
#include <iostream>

__global__ 
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
    const int dilationWidth, const int dilationHeight) {
  const int KW_LIMIT = kernelWidth;
  const int KH_LIMIT = kernelHeight;

  int _index = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i=_index; _index < (totalElements); _index+=blockDim.x * gridDim.x, i=_index)
  {
    const int batchStride = outputChannels * outputHeight * outputWidth;
    const int channelStride = outputHeight * outputWidth;
    const int n = i / batchStride; 
    const int c = (i / channelStride) % outputChannels;
    const int h = (i / outputWidth) % outputHeight;
    const int w = i % outputWidth;

    int inputChannel = c;
    int inputChannels = outputChannels;
    if (depthwiseMultiplier !=1) {
      inputChannel /= depthwiseMultiplier;
      inputChannels /= depthwiseMultiplier;
    }

    int weightOffset = c * kernelHeight * kernelWidth;

    float value = biasEnabled ? static_cast<float>(bias[c]) : 0;
    const int offset0 = (n * inputChannels + inputChannel) * inputHeight * inputWidth;
    for (int kH = 0; kH < KH_LIMIT; ++kH) {
      for (int kW = 0; kW < KW_LIMIT; ++kW) {
        const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
        const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;

        if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth)) {
          const int offset = offset0 + h_in * inputWidth + w_in;
          value += (static_cast<float>(weight[weightOffset]) *
                    static_cast<float>(input[offset]));
        }
        ++weightOffset;
      }
    }
    output[i] = static_cast<float>(value);
  }
}