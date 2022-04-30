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

#define MTX_TYPE float

int main(int argc, char **argv) {
	// MULTIPLICATION TESTS
	//#define MULT_TEST

	// LU-DECOMPOSITION TESTS
	std::cout << "LU-DECOMPOSITION TESTS\n";
	int64_t	sequential_max_time,
			parallel_max_time,
			parallel_min_time;
	int		max_size,
			min_size,
			num_of_tests = 10,
			lu_block = 128,
			mtx_product_block = 128;

	sequential_max_time = 0;
	parallel_max_time = 0;
	parallel_min_time =	INT64_MAX;

	// ѕриЄм аргументов из командной строки:
	// argv[1] - пор€док матрицы
	// argv[2] - длина блока в LU-разложении
	// argv[3] - длина блока в матричном умножении
	// ¬ умножении идЄт разбиение на блоки 'argv[2] x argv[3]'
	if (argc > 1) {
		string s1(argv[1]), s2(argv[2]), s3(argv[3]);
		max_size = min_size = stoll(s1);
		lu_block = stoll(s2);
		mtx_product_block = stoll(s3);
	}
	else
		max_size = min_size = 8192;

	//freopen("openmp_trivial_parallel_output.txt", "a+", stdout);

	for (size_t size = 1000; size <= 8000; size += 1000) {

		cout << "Size: " << size << endl;

		for (size_t num_of_tests = 0; num_of_tests < 4; ++num_of_tests) {
			Matrix<MTX_TYPE> A(size, size); //B(n,n);
			A.generate_well_conditioned_matrix(500.0, 0.04);
			//B = A;

			auto start_time = steady_clock::now();

			//A.lu_trivial_sequential();
			//A.lu_trivial_parallel_omp();
			//A.lu_block_parallel_omp(lu_block, mtx_product_block);

			auto end_time = steady_clock::now();

			//parallel_max_time = max(parallel_max_time, duration_cast<milliseconds>(end_time-start_time).count());
			cout << "OpenMP Time: " << duration_cast<milliseconds>(end_time-start_time).count() << endl;
			//B.lu_trivial_sequential();
			//check_correct(A,B);

		}

		//}
	}
	//std::cout << "Max time: " << parallel_max_time << "ms\n\n";
/*
	// MKL LU TESTS
	std::cout << "MKL LU TESTS\n";
	parallel_max_time = 0;
	if (argc > 1) {
		string s(argv[1]);
		max_size = min_size = stoll(s);
	}
	else
		max_size = min_size = 10000;
	int info;

	for (size_t i = 0; i < num_of_tests; ++i) {
		int n = rand() % (max_size - min_size + 1) + min_size;
		Matrix<MTX_TYPE> A(n,n);
		int *pivots = new int[n];
		iota(pivots, pivots+n, 0);
		
		A.generate_random_matrix();
		//A.print();

		auto start_time = chrono::steady_clock::now();
		LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, &A(0,0), n, pivots);
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
		parallel_max_time = std::max(parallel_max_time, dur.count());
		sum_time += dur.count();
		std::cout << "Test " << i+1 << " : size - (" << n << "), time -" << dur.count() << "ms\n";
	}
	std::cout << "Max time: " << parallel_max_time << "ms, avg time - " << (MTX_TYPE)sum_time/num_of_tests << "\n\n";
	*/

	return 0;
}