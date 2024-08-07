extern "C"
__global__ void s_transpose(float *odata, const float *idata, int width, int height)
{
  int TILE_DIM = 4;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y; 
  extern __shared__ float tile[];
  
  if (x < width && y < height){
      tile[threadIdx.y * TILE_DIM + threadIdx.x] = idata[x*height + y];
  }
  __syncthreads();
    
    if (x < width && y < height)
      odata[y*width + x] = tile[threadIdx.y * TILE_DIM + threadIdx.x];
}