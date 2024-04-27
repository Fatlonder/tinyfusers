#include "../native/cuda/conv2d.h"

float* fill_array(int lenght)
{
    float *val, *arr;
    cudaMalloc(&val, lenght*sizeof(float)); 
    arr = (float*)malloc(lenght*sizeof(float));

    for (int i = 0; i<lenght; i++) arr[i] = 1.0;

    cudaMemcpy(val, arr, lenght*sizeof(float), cudaMemcpyHostToDevice);
    return val;
}

void print_output(int output_elmnts, float *output)
{
  float *y_output;
  y_output = (float*)malloc(output_elmnts*sizeof(float));
  cudaMemcpy(y_output, output, output_elmnts*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i<output_elmnts; i++)
  {
    printf("%d, %f\n", i, (float)y_output[i]);
  }
  free(y_output);
}

int main(int argc, char* argv[]){
    const int strideWidth = 1;
    const int strideHeight = 1;
    const int dilationWidth = 1;
    const int dilationHeight = 1;
    const int padWidth = 0;
    const int padHeight = 0;

    float *input, *weight, *output, *bias; 
    const int outputWidth = atoi(argv[1]);
    const int outputHeight = atoi(argv[2]);
    const int inputWidth = atoi(argv[3]);
    const int inputHeight = atoi(argv[4]);
    const int kernelWidth = atoi(argv[5]);
    const int kernelHeight = atoi(argv[6]);
    const int _inputChannel = atoi(argv[7]);
    const int _outputChannel = atoi(argv[8]);
    const int n = atoi(argv[9]);
    printf("%d\n", outputWidth);
    printf("%d\n", outputHeight);
    printf("%d\n", inputWidth);
    printf("%d\n", inputHeight);
    printf("%d\n", kernelWidth);
    printf("%d\n", kernelHeight);
    printf("%d\n", _inputChannel);
    printf("%d\n", _outputChannel);
    printf("%d\n", n);

    bool biasEnabled = false;
    const int outputChannels = 1;
    const int depthwiseMultiplier = 1;

    int output_elmnts = _outputChannel * outputHeight * outputHeight; 
    int input_elmnts = _inputChannel * inputHeight * inputWidth; 
    int kernel_elmnts = _outputChannel * kernelHeight * kernelWidth; 

    input = fill_array(input_elmnts);
    weight =  fill_array(kernel_elmnts);
    output = fill_array(output_elmnts);
    bias = fill_array(1);

    conv_depthwise2d_forward_kernel<<<(sizeof(weight)+255)/256, 256>>>(input, output, 
        weight, bias, biasEnabled,
        input_elmnts, outputChannels, depthwiseMultiplier, inputWidth, inputHeight, 
        outputWidth, outputHeight, kernelWidth, kernelHeight,
        strideWidth, strideHeight, padWidth, padHeight, dilationWidth, dilationHeight);

    print_output(n, output);

    cudaFree(input);
    cudaFree(weight);
    cudaFree(output);
    cudaFree(bias);
    return 0;
}