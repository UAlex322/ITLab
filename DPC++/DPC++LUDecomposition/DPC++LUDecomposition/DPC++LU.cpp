#define MTX_TYPE float
//#define CHECK_CORRECT
#define GEN_WELL_COND

#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include "Matrix.h"
using namespace sycl;
using namespace std;

int64_t sz,
        lu_block,
        mtx_product_block;


int main(int argc, char **argv) {

    // Ïðè¸ì àðãóìåíòîâ èç êîìàíäíîé ñòðîêè:
    // argv[1] - äëèíà ìàòðèöû
    // argv[2] - äëèíà áëîêà â LU-ðàçëîæåíèè
    // argv[3] - äëèíà áëîêà â ìàòðè÷íîì óìíîæåíèè
    // Â óìíîæåíèè èä¸ò ðàçáèåíèå íà áëîêè 'argv[2] x argv[3]'
    if (argc > 1) {
        sz = stoll(argv[1]);
        lu_block = stoll(argv[2]);
        mtx_product_block = stoll(argv[3]);
    }
    else {
        sz = 8192;
        lu_block = mtx_product_block = 64;
    }

    //freopen("dpc++_output1.txt", "a+", stdout);

    cout << "DPC++,  ";
    cout << "sizes: (" << sz << ',' << lu_block << ',' << mtx_product_block << "), ";



    Matrix A(sz,sz)
#ifdef CHECK_CORRECT
    , B // A(n,n), B;
#endif
;

#ifdef GEN_WELL_COND
    A.generate_well_conditioned_matrix(0.5f * sz, 0.4f);
#else
    A.generate_random_matrix(10.0);
#endif

#ifdef CHECK_CORRECT
    B = A;
#endif
        
    auto begin = chrono::steady_clock::now();
    A.lu_block_parallel_dpc(lu_block, mtx_product_block);
    auto end = chrono::steady_clock::now();
    std::cout << "time: " << (chrono::duration_cast<chrono::milliseconds>(end - begin)).count() << " ms" << std::endl;

#ifdef CHECK_CORRECT
    B.lu_trivial_sequential();
    check_correct(A,B);
#endif

    return 0;
}
