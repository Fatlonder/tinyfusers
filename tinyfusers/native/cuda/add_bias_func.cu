extern "C"
__global__ 
void add_bias(float* out, const float* bias, int BT, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BT * OC){
      int col = idx % OC;
      out[((idx % OC) * BT) + idx/OC] += bias[col];
    }
}