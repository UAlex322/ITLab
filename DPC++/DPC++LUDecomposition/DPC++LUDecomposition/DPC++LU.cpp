#define TYPE float
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include "Matrix.h"
using namespace sycl;
using namespace std;

const int n = 4096;
const int block_size = 64;

int main(int argc, char *argv[]) {

    //freopen("dpc++_output.txt", "a+", stdout);

    //for (size_t n = 2000; n <= 16000; n += 2000) {
        std::cout << "Size: " << n << std::endl;
        for (size_t i = 0; i < 5; ++i) {
            Matrix A(n,n), B;
            A.generate_well_conditioned_matrix(2000.0, 0.04);
            B = A;

            auto begin = chrono::steady_clock::now();
            A.lu_block_parallel_dpc(block_size, block_size);
            auto end = chrono::steady_clock::now();
            std::cout << "DPC++ time: " << (chrono::duration_cast<chrono::milliseconds>(end - begin)).count() << std::endl;

            B.lu_trivial_sequential();
            check_correct(A,B,n,n);
        }
    //}

    return 0;
}