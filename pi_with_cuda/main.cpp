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
#include <chrono>




#define NLIM 100000000

void compute_r(int *mem, double* rand_real, double* rand_imag ) {
    int iter=0;
    double r;
   
    for (iter=0; iter<int(NLIM); iter++) {
        
        r = sqrt(rand_real[iter]*rand_real[iter] + rand_imag[iter]*rand_imag[iter]);
        if (r <= 1.0f) {
            mem[iter] = 1;
        }
        else
            mem[iter] = 0;
    }
}

int main(int argc, const char * argv[]) {
    
    std::cout << "Refine Pi using " << NLIM << " iterations" << std::endl;
    
    double pi;
    int inner=0;
    int* gpu_inner;
    double *rand_imag; double *rand_real;
    gpu_inner = new int[NLIM];
    rand_real = new double[NLIM];
    rand_imag = new double[NLIM];
//    cudaMallocManaged(&gpu_inner,sizeof(int)*int(NLIM));
//    cudaMallocManaged(&rand_real,sizeof(double)*int(NLIM));
//    cudaMallocManaged(&rand_imag,sizeof(double)*int(NLIM));
   
    for (int i=0; i<int(NLIM); i++) {
        rand_real[i] = double(rand()) / double(RAND_MAX);
        rand_imag[i] = double(rand()) / double(RAND_MAX);
    }
    
//    compute_r<<1,1>> (gpu_inner,rand_real,rand_imag);
 compute_r (gpu_inner,rand_real,rand_imag);
    
//    cudaDeviceSynchronize();
    
    for (int i=0; i<int(NLIM); i++) {
        inner += gpu_inner[i];
    }
    
    pi= 4.0f* (inner/double(NLIM));
    std::cout << "Pi is " << pi << std::endl;
    
    return 0;
}
