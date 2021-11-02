#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mkl.h>
#include "Matrix.h"
using namespace std;

#define MTX_TYPE double

#define TEST

#ifndef TEST

int main() {
	size_t n;
	MTX_TYPE *A, *L, *U;

	cin >> n;
	A = new MTX_TYPE[n*n];
	L = new MTX_TYPE[n*n];
	U = new MTX_TYPE[n*n];

	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			cin >> A[i*n+j];

	LU(A, L, U, n);
	
	cout.precision(6);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j)
			cout << fixed << L[i*n+j] << " ";
		cout << '\n';
	}
	cout << '\n';
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j)
			cout <<	fixed << U[i*n+j] << " ";
		cout << '\n';
	}
	cout << '\n';

	delete[] A; 
	delete[] L; 
	delete[] U;
}
#endif

int main() {
	size_t max_size, min_size;
	// MULTIPLICATION TESTS
	//#define MULT_TEST
#ifdef MULT_TEST
	std::cout << "MULTIPLICATION TESTS\n";

	// MKL MULT TESTS
	std::cout << "MKL MULT TESTS" << '\n';
	// DEFINITE SIZES
	max_size = 4096; min_size = 2048;
	for (size_t size = 2048, i = 1; size < max_size; size += 64, ++i) {
		Matrix<double> m1(size, size), m2(size, size), prod1(size, size), cm1, cm2, prod2;
		m1.generate_random_matrix(); cm1 = m1;
		m2.generate_random_matrix(); cm2 = m2;

		auto start_time = chrono::steady_clock::now();
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0, &m1(0,0), size, &m2(0,0), size, 0.0, &prod1(0,0), size);
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

		prod2 = cm1 * cm2;

		double ae = 0.0, re = 0.0;
		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j) {
				ae = std::max(ae, abs(prod2(i, j) - prod1(i, j)));
				re = std::max(re, (prod1(i,j) == 0.0) ? 0.0 : abs(prod2(i,j)/prod1(i,j) - 1.0));
			}
		}

		std::cout << "Random, random sizes " << i+1 << " : size - (" << size << "), time - " << dur.count() << "ms, error: absolute - " 
			<< ae << " , relative - " << re << '\n';
	}

	// RANDOM TESTS
	// DEFINITE SIZES
	max_size = 4096; min_size = 2048;
	for (size_t size = 2048, i = 1; size < max_size; size += 64, ++i) {
		Matrix<double> m1(size, size), m2(size, size);
		m1.generate_random_matrix();
		m2.generate_random_matrix();

		auto start_time = chrono::steady_clock::now();
		Matrix<double> product = m1 * m2;
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

		std::cout << "Random, definite sizes " << i << " : sizes - (" << size << " " << size << " " << size << "), time - " << dur.count() << "ms" << '\n';
	}

	// RANDOM SIZES
	max_size = 2048;
	for (size_t i = 0; i < 50; ++i) {
		size_t m = rand() % (max_size - min_size + 1) + min_size, 
			n = rand() % (max_size - min_size + 1) + min_size, 
			p = rand() % (max_size - min_size + 1) + min_size;
		Matrix<double> m1(m,n), m2(n,p);
		m1.generate_random_matrix();
		m2.generate_random_matrix();

		auto start_time = chrono::steady_clock::now();
		Matrix<double> product = m1 * m2;
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

		std::cout << "Random, random sizes " << i+1 << " : sizes - (" << m << " " << n << " " << p << "), time - " << dur.count() << "ms" << '\n';
	}
#else

	// LU-DECOMPOSITION TESTS
	std::cout << "LU-DECOMPOSITION TESTS\n";
	long long sequential_max_time,
			  parallel_max_time,
			  num_of_tests,
			  sum_time;

	// BLOCK LU_DECOMPOSITION TESTS
	std::cout << "BLOCK LU-DECOMPOSITION TESTS\n";
	// RANDOM SIZES

	sequential_max_time = 0;
	parallel_max_time = 0;
	max_size = 5000, min_size = 5000;
	for (size_t i = 0; i < 50; ++i) {
		int n = rand() % (max_size - min_size + 1) + min_size, info;
		Matrix<double> A(n,n);
		A.generate_random_matrix();
		//A.print();

		auto start_time = chrono::steady_clock::now();
		A.LU3_block();
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
		parallel_max_time = std::max(parallel_max_time, dur.count());

		//mkl_dgetrfnp(&n, &n, &B(0,0), &n, &info);
		//B.print();
		/*
		for (int i = 0; i < n; ++i)
			L(i,i) = 1.0;
		for (int i = 1; i < n; ++i)
			for (int j = 0; j < i; ++j) {
				L(i, j) = A(i, j);
				A(i,j) = 0.0;
			}
		for (int i = 0; i < n; ++i)
			for (int j = i+1; j < n; ++j)
				L(i,j) = 0.0;
		P = L*A;

		double ae = 0.0, re = 0.0;
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				ae = std::max(ae, abs(P(i, j) - B(i, j)));
				re = std::max(re, (B(i,j) == 0.0) ? 0.0 : abs(P(i,j)/B(i,j) - 1.0));
			}
		}
		*/
		std::cout << "Random, random sizes " << i+1 << " : size - (" << n << "), time - " << dur.count() << "ms\n";
	}
	std::cout << "Max time: " << parallel_max_time << "ms\n\n";

	// MKL LU TESTS
	std::cout << "MKL LU TESTS\n";
	sequential_max_time = 0;
	parallel_max_time = 0;
	max_size = 4000, min_size = 4000;
	num_of_tests = 30, sum_time = 0;
	int info;

	for (size_t i = 0; i < num_of_tests; ++i) {
		int n = rand() % (max_size - min_size + 1) + min_size;
		Matrix<double> A(n,n);
		A.generate_random_matrix();
		//A.print();

		auto start_time = chrono::steady_clock::now();
		mkl_dgetrfnp(&n, &n, &A(0,0), &n, &info);
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
		parallel_max_time = std::max(parallel_max_time, dur.count());
		sum_time += dur.count();
		std::cout << "Random, random sizes " << i+1 << " : size - (" << n << "), time -" << dur.count() << "ms\n";
	}
	std::cout << "Max time: " << parallel_max_time << "ms, avg time - " << (double)sum_time/num_of_tests << "\n\n";

	
	/*

	// RANDOM SIZES (LU2)

	sequential_max_time = 0;
	parallel_max_time = 0;
	max_size = 4000, min_size = 4000;
	num_of_tests = 20;
	std::cout << "Sequential: \n";
	for (size_t i = 0; i < num_of_tests; ++i) {
		size_t n = rand() % (max_size - min_size + 1) + min_size;
		Matrix<double> A(n,n), U(n,n);
		A.generate_random_matrix();
		//A.print();

		auto start_time = chrono::steady_clock::now();
		A.LU2_sequential();
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
		sequential_max_time = std::max(sequential_max_time, dur.count());

		//A.print();
		std::cout << "Random, random sizes " << i+1 << " : size - (" << n << "), seq time - " << dur.count() << "ms\n";

	}

	std::cout << "Parallel: \n";
	for (size_t i = 0; i < num_of_tests; ++i) {
		size_t n = rand() % (max_size - min_size + 1) + min_size;
		Matrix<double> A(n, n), U(n, n);
		A.generate_random_matrix();
		//A.print();

		auto start_time = chrono::steady_clock::now();
		A.LU2_parallel();
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
		parallel_max_time = std::max(parallel_max_time, dur.count());

		std::cout << "Random, random sizes " << i+1 << " : size - (" << n << "), par time - " << dur.count() << "ms\n";
	}
	std::cout << "Max seq time: " << sequential_max_time << "ms, max par time - " << parallel_max_time << "\n\n";

	*/

	/*

	// DEFINITE SIZES (LU1)

	sequential_max_time = 0;
	parallel_max_time = 0;
	max_size = 2048;
	for (size_t n = 2, i = 1; n <= max_size; n += 2, ++i) {
		Matrix<double> A(n,n), L(n,n), U(n,n);
		A.generate_random_matrix();
		//A.print();

		auto start_time = chrono::steady_clock::now();
		A.LU1(L,U);
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
		sequential_max_time = std::max(sequential_max_time, dur.count());
		Matrix<double> C = L*U;
		//L.print();
		//U.print();
		//C.print();
		double ae = 0.0, re = 0.0;
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				ae = std::max(ae, abs(C(i, j) - A(i, j)));
				re = std::max(re, (A(i,j) == 0.0) ? 0.0 : abs(C(i,j)/A(i,j) - 1.0));
			}
		}
		std::cout << "Random, definite sizes " << i << " : size - (" << n << "), error: absolute - " << ae << " , relative - " <<
			re << " ,time - " << dur.count() << "ms" << '\n';
	}
	std::cout << "Maximal time: " << sequential_max_time << "\n\n";

	*/



	/*

	// WELL-CONDITIONED TESTS
	// 
	// RANDOM SIZES
	max_size = 500, min_size = 500;
	for (size_t i = 0; i < 1; ++i) {
		size_t n = rand() % (max_size - min_size + 1) + min_size;
		Matrix<double> A(n,n), L(n,n), U(n,n);
		A.generate_well_conditioned_matrix(3000.0);
		//A.print();

		auto start_time = chrono::steady_clock::now();
		A.LU1(L,U);
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
		sequential_max_time = std::max(sequential_max_time, dur.count());
		Matrix<double> C = L*U;

		//C.print();
		double ae = 0.0, re = 0.0;
		//for (int i = 0; i < n; ++i) {
		//	for (int j = 0; j < n; ++j) {
		//		std::cout << std::fixed << A(i, j) << " ";
		//	}
		//	std::cout << '\n';
		//}
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				ae = std::max(ae, abs(C(i,j) - A(i,j)));
				re = std::max(re, (A(i,j) == 0.0) ? 0.0 : abs(C(i,j)/A(i,j) - 1.0));
				//std::cout << std::fixed << C(i,j)/A(i,j) - 1.0 << " ";
			}
			//std::cout << '\n';
		}

		std::cout << std::defaultfloat << "Well-conditioned, random sizes " << i+1 << " : size - (" << n << "), error : absolute - " << ae << ", relative - " <<
			re << " ,time - " << dur.count() << "ms" << '\n';
	}
	std::cout << "Maximal time: " << sequential_max_time << "\n\n";

	*/

	/*

	// RANDOM TESTS 
	// 
	// RANDOM SIZES

	sequential_max_time = 0;
	max_size = 2048, min_size = 2047;
	for (size_t i = 0; i < 50; ++i) {
		size_t n = rand() % (max_size - min_size) + min_size;
		Matrix<double> A(n,n), L(n,n), U(n,n);
		A.generate_random_matrix();
		//A.print();

		auto start_time = chrono::steady_clock::now();
		A.LU1(L,U);
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
		sequential_max_time = std::max(sequential_max_time, dur.count());
		Matrix<double> C = L*U;
		//C.print();
		double ae = 0.0, re = 0.0;
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j) {
				ae = std::max(ae, abs(C(i,j) - A(i,j)));
				re = std::max(re, (A(i,j) == 0.0) ? 0.0 : abs(C(i,j)/A(i,j) - 1.0));
			}

		std::cout << "Random, random sizes " << i+1 << " : size - (" << n << "), error: absolute - " << ae << " , relative - " <<
			re << " ,time - " << dur.count() << "ms" << '\n';
	}
	std::cout << "Maximal time: " << sequential_max_time << "\n\n";

	*/

#endif

	return 0;
}
