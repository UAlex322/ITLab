#include <iostream>
#include "LU.h"
using namespace std;

#define MTX_TYPE double

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