//
//  main.cpp
//  pi_with_cuda
//
//  Created by Mirco Meazzo on 21/10/2019.
//  Copyright © 2019 Mirco Meazzo. All rights reserved.
//

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <typeinfo>


#define NLIM 10000000


__global__ void compute_r(int *mem, double *rand_real, double *rand_imag ) {
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int total_blocks= gridDim.x;
    int stride= blockDim.x * total_blocks;

    for (int i=index; i<(int(NLIM)); i+=stride) {
        
        if ((sqrt(rand_real[i]*rand_real[i] + rand_imag[i]*rand_imag[i])) <= 1.0f) {
            mem[i] = 1;
        }
        else
            mem[i] = 0;
    }
}

__global__ void reduction(int *mem, int *res) {
    // Copy from global memory to shared memory the values
    __shared__ int mem_gpu[512];
    int tid = threadIdx.x;

    mem_gpu[tid] = mem[tid + blockDim.x*blockIdx.x];
    __syncthreads();
    
    // Wait all threads within the block
    // Start memory reduction process
    if (blockDim.x >= 512) {  
        if (tid < 256) {
            mem_gpu[tid] += mem_gpu[tid + 256];
        }
        __syncthreads(); 
    }
    if (blockDim.x >= 256) {
        if (tid < 128) {
            mem_gpu[tid] += mem_gpu[tid + 128];
        }
        __syncthreads();
    }
    if (blockDim.x >= 128) {  
        if (tid < 64) {
            mem_gpu[tid] += mem_gpu[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {                           // Instruction within warps scope
        volatile int *smem_gpu = mem_gpu;     // Volatile means no schedule optimization, we're freezing 
                                              // the status on these 64 threads
        smem_gpu[tid] += smem_gpu[tid + 32];  // Warps are synchronized, these rows are executed  
        smem_gpu[tid] += smem_gpu[tid + 16];  // one by one, no need of further sync
        smem_gpu[tid] += smem_gpu[tid + 8];
        smem_gpu[tid] += smem_gpu[tid + 4];
        smem_gpu[tid] += smem_gpu[tid + 2];
        smem_gpu[tid] += smem_gpu[tid + 1];
    }
    if (tid == 0) {
        res[blockIdx.x] = mem_gpu[tid];
    }
}

int main(int argc, const char * argv[]) {
    
    std::cout << "Refine Pi using " << NLIM << " iterations" << std::endl;
    
    double pi;
    int *gpu_inner;
    double *rand_imag, *rand_real;
//    gpu_inner = new int[NLIM];
//    rand_real = new double[NLIM];
//    rand_imag = new double[NLIM];
    cudaMallocManaged(&gpu_inner,int(NLIM)*sizeof(int));
    cudaMallocManaged(&rand_real,int(NLIM)*sizeof(double));
    cudaMallocManaged(&rand_imag,int(NLIM)*sizeof(double));

 
    for (int i=0; i<(int(NLIM )-1); i++) {
        rand_real[i] = double(rand()) / double(RAND_MAX);
        rand_imag[i] = double(rand()) / double(RAND_MAX);
    }

    int block_size = 128;
    int n_blocks = (int(NLIM) + block_size - 1) / block_size; 
    int *inner; 
    cudaMallocManaged(&inner, n_blocks*sizeof(int));
    std::cout << "Executing Kernel with " << block_size << " threads on " << n_blocks << " blocks" << std::endl;
    compute_r <<<n_blocks, block_size>>> (gpu_inner, rand_real, rand_imag);
    cudaDeviceSynchronize();
    reduction <<<n_blocks, block_size>>> (gpu_inner,inner);
// compute_r (gpu_inner,rand_real,rand_imag);
    cudaDeviceSynchronize();

    for (int i=1; i<n_blocks; i++) {
        inner[0] += inner[i];
    }

    pi= 4.0f* (inner[0]/double(NLIM));
    std::cout << "Pi is " << pi << std::endl;
    cudaFree(gpu_inner);
    cudaFree(rand_imag);
    cudaFree(rand_real);

    return 0;
}
