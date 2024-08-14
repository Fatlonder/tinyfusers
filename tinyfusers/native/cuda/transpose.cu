extern "C"
__global__ void s_transpose(float *odata, const float *idata, int d_x, int d_y, int d_z, int i, int j, int k)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = (k == -1) ? 0 : blockIdx.z * blockDim.z + threadIdx.z;
  int dims[3] = {d_x, d_y, d_z};
  int num_dims = (k == -1) ? 2 : 3;

  if (x >= dims[0] || y >= dims[1] || (num_dims == 3 && z >= dims[2])) return;

  int coords[3] = {x, y, z};
  int new_x = coords[i];
  int new_y = coords[j];
  int new_z = (num_dims == 3) ? coords[k] : 0;

  int input_idx, output_idx;
  if (num_dims == 3) {
      input_idx = x * (dims[1] * dims[2]) + y * dims[2] + z;
      output_idx = new_x * (dims[j] * dims[k]) + new_y * dims[k] + new_z;
  } else {
      input_idx = x * dims[1] + y;
      output_idx = new_x * dims[j] + new_y;
  }

  odata[output_idx] = idata[input_idx];
}
