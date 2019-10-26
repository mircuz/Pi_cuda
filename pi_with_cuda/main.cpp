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


 void compute_r(int *mem, double *rand_real, double *rand_imag ) {
    
   int index=0, stride=1;

    for (int i=index; i<(int(NLIM)-1); i+=stride) {
        
        if ((sqrt(rand_real[i]*rand_real[i] + rand_imag[i]*rand_imag[i])) <= 1.0f) {
            mem[i] = 1;
        }
        else
            mem[i] = 0;
    }
}

int main(int argc, const char * argv[]) {
    
    std::cout << "Refine Pi using " << NLIM << " iterations" << std::endl;
    
    double pi;
    int inner=0;
    int *gpu_inner;
    double *rand_imag, *rand_real;
    gpu_inner = new int[NLIM];
    rand_real = new double[NLIM];
    rand_imag = new double[NLIM];
   
    
 
    for (int i=0; i<(int(NLIM )-1); i++) {
        rand_real[i] = double(rand()) / double(RAND_MAX);
        rand_imag[i] = double(rand()) / double(RAND_MAX);
    }

    int block_size = 256;
    int n_blocks = (int(NLIM) + block_size - 1) / block_size;   //Ensure (n_blocks * block_size) > NLIM
//    compute_r <<<n_blocks, block_size>>> (gpu_inner, rand_real, rand_imag);
    compute_r (gpu_inner,rand_real,rand_imag);
    
 //   cudaDeviceSynchronize();

    for (int i=0; i<int(NLIM); i++) {
        inner += gpu_inner[i];
    }

    pi= 4.0f* (inner/double(NLIM));
    std::cout << "Pi is " << pi << std::endl;


    return 0;
}
