// Îïðåäåëåíèÿ ïðåïðîöåññîðà äëÿ ñáîðêè::
// PARALLEL - èñïîëüçîâàòü ïàðàëëåëüíûé àëãîðèòì (ïî óìîë÷àíèþ - ïîñëåäîâàòåëüíûé)
// BLOCK - èñïîëüçîâàòü áëî÷íûé àëãîðèòì (ïî óìîë÷àíèþ - îáû÷íûé)

//#define PARALLEL
//#define BLOCK
#define MTX_TYPE float

#include <iostream>
#include <algorithm>
#include <string>
#include <limits>
#include <cstdlib>
#include <numeric>
#include <chrono>
#include <mkl.h>
#include "Matrix.h"
using namespace std;
using namespace chrono;


int main(int argc, char **argv) {
	size_t size,
		   lu_block,
		   mtx_product_block;

	// Ïðè¸ì àðãóìåíòîâ èç êîìàíäíîé ñòðîêè:
	// argv[1] - äëèíà ìàòðèöû
	// argv[2] - äëèíà áëîêà â LU-ðàçëîæåíèè
	// argv[3] - äëèíà áëîêà â ìàòðè÷íîì óìíîæåíèè
	// Â óìíîæåíèè èä¸ò ðàçáèåíèå íà áëîêè 'argv[2] x argv[3]'
	if (argc > 1) {
		size = stoll(argv[1]);
		
		if (argc > 2) {
			lu_block = stoll(argv[2]);
			mtx_product_block = stoll(argv[3]);
		}
		else
			lu_block = mtx_product_block = 128;
	}
	else
		size = 8192;

	cout << "OpenMP, ";

#if !defined(BLOCK) && !defined(PARALLEL)
	cout << "trivial, seq; ";
#elif !defined(BLOCK) && defined(PARALLEL)
	cout << "trivial, par; ";
#elif defined(BLOCK) && !defined(PARALLEL)
	cout << "block,   seq; ";
#elif defined(BLOCK) && defined(PARALLEL)
	cout << "block,   seq; ";
#endif

	cout << "sizes: (" << size << ',' << lu_block << ',' << mtx_product_block << "), ";

	Matrix<MTX_TYPE> A(size, size);
	A.generate_well_conditioned_matrix(0.25f * size, 0.04f);

	auto start_time = steady_clock::now();

#if !defined(BLOCK) && !defined(PARALLEL)
	A.lu_trivial_sequential();
#elif !defined(BLOCK) && defined(PARALLEL)
	A.lu_trivial_parallel_omp();
#elif defined(BLOCK)
	A.lu_block(lu_block, mtx_product_block);
#endif

	auto end_time = steady_clock::now();

	cout << "time: " << duration_cast<milliseconds>(end_time-start_time).count() << " ms" << endl;

	return 0;
}
