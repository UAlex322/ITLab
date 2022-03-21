#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include "Matrix.h"
using namespace sycl;
using namespace std;

#define TYPE float

void check_correct(const Matrix &M1, const Matrix &M2, int m, int n) {
    TYPE ae = 0.0, re = 0.0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            ae = std::max(ae, abs(M2(i,j) - M1(i,j)));
            re = std::max(re, (M1(i,j) == 0.0f) ? 0.0f : abs(M2(i,j)/M1(i,j) - 1.0f));
        }
    cout << "Absolute error: " << ae << '\n' << "Relative error: " << re << std::endl;
}

template <typename T>
void random_fill(vector<T> &vec) {
    uniform_real_distribution<float> fd(-10.0, 10.0);
    mt19937 gen(random_device{}());

    for (int i = 0; i < vec.size(); ++i)
        vec[i] = fd(gen);
}

const int N = 20;

int main(int argc, char *argv[]) {
    //#define DEBUG

    #ifndef DEBUG

    Matrix A(N,N), B(N,N);
    A.generate_well_conditioned_matrix(20.0);
    B = A;

    auto begin = chrono::steady_clock::now();
    A.LU3_block(4,4);
    auto end = chrono::steady_clock::now();
    B.LU2_sequential();
    std::cout << "DPC++ time : " << (chrono::duration_cast<chrono::milliseconds>(end - begin)).count() << std::endl;
    check_correct(A,B,N,N);

    #else
    {
        buffer<float,1> a_buf(a.data(), N);
        sycl::queue q{cpu_selector{}};
        q.submit([&](handler &cgh) {
            auto a_acc = a_buf.get_access<access::mode::read_write>(cgh);
            sycl::stream ostream(10000, 20, cgh);
            cgh.parallel_for(nd_range{range{32,11}, range{10,2}}, [=](nd_item<2> item) {
                ostream << item.get_global_id(0) << ' ' << item.get_local_id(0) << ' ' << item.get_local_range(0) << ' ' << 
                           item.get_global_id(1) << ' ' << item.get_local_id(1) << ' ' << item.get_local_range(1) << '\n';
            });
        });
    }
    #endif

    return 0;
}