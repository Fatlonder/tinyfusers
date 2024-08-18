__global__ void transpose(float *output, float *input, int *dims, int i, int j, int k, int l) {
    //int x = blockIdx.x * (blockDim.x * blockDim.x) + threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //int flat_index = y * TILE_DIM + x;
    int flat_index = y * (dims[0] * dims[1]) + x;

    int x_r = flat_index / (dims[1] * dims[2] * dims[3]);
    int y_r = (flat_index / (dims[2] * dims[3])) % dims[1];
    int z_r = (flat_index / dims[3]) % dims[2];
    int w_r = flat_index % dims[3];
    
    //if (x >= dims[0] || y >= dims[1] || z >= dims[2] || w >= dims[3]) return;

    int coords[4] = {x_r, y_r, z_r, w_r};
    int new_x = coords[i];
    int new_y = coords[j];
    int new_z = coords[k];
    int new_w = coords[l];

    int output_idx = new_x * (dims[j] * dims[k] * dims[l]) +
                     new_y * (dims[k] * dims[l]) +
                     new_z * dims[l] +
                     new_w;

    output[flat_index] = w_r;
}