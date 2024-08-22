extern "C"
__global__ void transpose(float *output, float *input, int d_x, int d_y, int d_z, int d_w, int i, int j, int k, int l) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int dims[4] = {d_x, d_y, d_z, d_w};
    
    int flat_index = y * (dims[0] * dims[1]) + x;

    int x_r = flat_index / (dims[1] * dims[2] * dims[3]);
    int y_r = (flat_index / (dims[2] * dims[3])) % dims[1];
    int z_r = (flat_index / dims[3]) % dims[2];
    int w_r = flat_index % dims[3];
    
    if (flat_index >= (d_x * d_y * d_z * d_w)) return;

    int coords[4] = {x_r, y_r, z_r, w_r};
    int new_x = coords[i];
    int new_y = coords[j];
    int new_z = coords[k];
    int new_w = coords[l];

    int output_idx = new_x * (dims[j] * dims[k] * dims[l]) + new_y * (dims[k] * dims[l]) + new_z * dims[l] + new_w;

    output[output_idx] = input[flat_index];
}