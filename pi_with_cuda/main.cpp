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


#define NLIM 100000000

void compute_r(int &mem) {
    int inner=0, iter=0;
        double rand_real, rand_imag, r;
    
        for (iter=0; iter<int(NLIM); iter++) {
        
            rand_real = double(rand()) / double(RAND_MAX);
            rand_imag = double(rand()) / double(RAND_MAX);
            
            r = sqrt(rand_real*rand_real + rand_imag*rand_imag);
            
            if (r <= 1.0f) {
                inner++;
            }
            
        }

        mem = inner;
}

int main(int argc, const char * argv[]) {
    
    std::cout << "Refine Pi using " << NLIM << " iterations" << std::endl;
    
    double pi;
    int inner;
    compute_r(inner);
    
    pi= 4.0f* (inner/double(NLIM));
    std::cout << "Pi is " << pi << std::endl;
    
    return 0;
}
