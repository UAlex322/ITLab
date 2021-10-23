#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include "LU.h"
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
#else

int main() {
	srand(0);

	// MULTIPLICATION TESTS
#ifdef MULT_TEST
	cout << "MULTIPLICATION TESTS\n";

	// RANDOM TESTS
	// DEFINITE SIZES
	size_t max_size = 2048;
	for (size_t size = 128, i = 1; size < max_size; size += 64, ++i) {
		Matrix<double> m1(size, size), m2(size, size);
		m1.fill_matrix_random_elements();
		m2.fill_matrix_random_elements();

		auto start_time = chrono::steady_clock::now();
		Matrix<double> product = m1 * m2;
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

		std::cout << "Random test, definite sizes " << i << " : sizes - (" << size << " " << size << " " << size << "), time - " << dur.count() << '\n';
	}

	// RANDOM SIZES
	max_size = 2048;
	for (size_t i = 0; i < 50; ++i) {
		size_t m = rand() % max_size + 1, n = rand() % max_size + 1, p = rand() % max_size + 1;
		Matrix<double> m1(m,n), m2(n,p);
		m1.fill_matrix_random_elements();
		m2.fill_matrix_random_elements();

		auto start_time = chrono::steady_clock::now();
		Matrix<double> product = m1 * m2;
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

		std::cout << "Random test, random sizes " << i+1 << " : sizes - (" << m << " " << n << " " << p << "), time - " << dur.count() << '\n';
	}

#endif
	// LU-DECOMPOSITION TESTS
	cout << "LU-DECOMPOSITION TESTS\n";

	size_t max_size = 50;
	for (size_t i = 0; i < 50; ++i) {
		size_t n = rand() % max_size + 1;
		Matrix<double> A(n,n), L(n,n), U(n,n);
		A.fill_matrix_random_elements();

		auto start_time = chrono::steady_clock::now();
		LU1(A, L, U);
		auto end_time = chrono::steady_clock::now();
		auto dur = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

		Matrix<double> C = L*U;
		double ae = 0.0, re = 0.0;
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j) {
				ae = std::max(ae, abs(C(i,j) - A(i,j)));
				re = std::max(re, abs(C(i,j) - A(i,j))/A(i,j));
			}

		std::cout << "Random test, random sizes " << i+1 << " : size - (" << n << "), error: absolute - " << ae << " , relative - " <<
			re << " ,time - " << dur.count() << '\n';
	}

	return 0;
}


#endif
